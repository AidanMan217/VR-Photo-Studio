"""GPU discovery: platform behaviour, parsers, DXGI enumeration logic (with a fake API)."""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys

import pytest

from vrphoto.models import gpu
from vrphoto.models.gpu import DXGI_ADAPTER_DESC1, Device


@pytest.mark.skipif(sys.platform == "win32", reason="non-Windows behaviour")
def test_list_devices_empty_off_windows():
    assert gpu.list_devices() == []
    assert gpu.list_devices(refresh=True) == []
    choices = gpu.device_choices()
    assert choices[:2] == [("auto", "Automatic (fastest GPU)"), ("cpu", "CPU only")]


def test_vendor_names():
    assert gpu.vendor_name(0x10DE) == "nvidia"
    assert gpu.vendor_name(0x1002) == "amd"
    assert gpu.vendor_name(0x8086) == "intel"
    assert gpu.vendor_name(0x1234) == "other"


def test_dxgi_struct_layout_matches_windows_x64():
    if ctypes.sizeof(ctypes.c_size_t) != 8:
        pytest.skip("64-bit layout only")
    assert ctypes.sizeof(DXGI_ADAPTER_DESC1) == 312
    assert DXGI_ADAPTER_DESC1.VendorId.offset == 256
    assert DXGI_ADAPTER_DESC1.DedicatedVideoMemory.offset == 272
    assert DXGI_ADAPTER_DESC1.AdapterLuid.offset == 296
    assert DXGI_ADAPTER_DESC1.Flags.offset == 304


def _desc(
    name: str,
    vendor: int,
    vram_mb: int,
    flags: int = 0,
    *,
    luid: tuple[int, int] = (0, 0),
    device_id: int = 0,
    subsys: int = 0,
) -> DXGI_ADAPTER_DESC1:
    d = DXGI_ADAPTER_DESC1()
    raw = name.encode("utf-16-le")
    ctypes.memmove(d.Description, raw, len(raw))
    d.VendorId = vendor
    d.DeviceId = device_id
    d.SubSysId = subsys
    d.DedicatedVideoMemory = vram_mb * 1024 * 1024
    d.AdapterLuid.HighPart, d.AdapterLuid.LowPart = luid
    d.Flags = flags
    return d


class FakeDxgi:
    def __init__(self, descs):
        self.descs = descs
        self.closed = False

    def desc(self, i):
        return self.descs[i] if i < len(self.descs) else None

    def close(self):
        self.closed = True


def test_dxgi_enumeration_skips_software_and_keeps_dxgi_index():
    api = FakeDxgi(
        [
            _desc("Intel(R) UHD Graphics 770", 0x8086, 128),
            _desc("Microsoft Basic Render Driver", 0x1414, 0, flags=gpu.DXGI_ADAPTER_FLAG_SOFTWARE),
            _desc("NVIDIA GeForce RTX 4070", 0x10DE, 12282),
        ]
    )
    devs = gpu._dxgi_devices(api)
    assert devs == [
        Device(0, "Intel(R) UHD Graphics 770", 128, "intel"),
        Device(2, "NVIDIA GeForce RTX 4070", 12282, "nvidia"),  # index = DirectML device_id
    ]
    assert devs[1].label == "NVIDIA GeForce RTX 4070 (12 GB)"


def test_description_decoding():
    assert _desc("AMD Radeon RX 7800 XT", 0x1002, 16368).description == "AMD Radeon RX 7800 XT"


def test_parse_cim_json():
    text = json.dumps(
        [
            {
                "Name": "NVIDIA GeForce RTX 3060",
                "AdapterRAM": 4293918720,
                "PNPDeviceID": "PCI\\VEN_10DE&DEV_2504",
            },
            {
                "Name": "Intel(R) Iris(R) Xe Graphics",
                "AdapterRAM": 1073741824,
                "PNPDeviceID": "PCI\\VEN_8086&DEV_9A49",
            },
            {
                "Name": "Microsoft Basic Display Adapter",
                "AdapterRAM": None,
                "PNPDeviceID": "ROOT\\BasicDisplay",
            },
            {"Name": "Radeon Something", "AdapterRAM": -1, "PNPDeviceID": ""},
        ]
    )
    devs = gpu.parse_cim_json(text)
    assert [d.vendor for d in devs] == ["nvidia", "intel", "amd"]
    assert devs[0].vram_mb is None  # uint32-capped value treated as unknown
    assert devs[1].vram_mb == 1024
    assert [d.index for d in devs] == [0, 1, 2]
    single = gpu.parse_cim_json(
        json.dumps({"Name": "X", "AdapterRAM": 2147483648, "PNPDeviceID": "PCI\\VEN_1002&"})
    )
    assert single == [Device(0, "X", 2048, "amd")]
    assert gpu.parse_cim_json("") == []
    assert gpu.parse_cim_json("not json") == []


def test_parse_nvidia_smi():
    assert gpu.parse_nvidia_smi("12282\n") == [12282]
    assert gpu.parse_nvidia_smi("24564\r\n8192\r\n") == [24564, 8192]
    assert gpu.parse_nvidia_smi("[N/A]\n") == []
    assert gpu.parse_nvidia_smi("") == []


def test_nvidia_vram_mb(monkeypatch):
    monkeypatch.setattr(gpu, "_find_nvidia_smi", lambda: "/usr/bin/nvidia-smi")
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout="12282\n", stderr="")

    monkeypatch.setattr(gpu.subprocess, "run", fake_run)
    assert gpu.nvidia_vram_mb() == 12282
    cmd, kw = calls[0]
    assert cmd[1:] == ["--query-gpu=memory.total", "--format=csv,noheader,nounits"]
    assert kw["timeout"] > 0 and kw["stdin"] is subprocess.DEVNULL
    if sys.platform == "win32":
        assert kw["creationflags"] & 0x08000000


def test_nvidia_vram_mb_failures(monkeypatch):
    monkeypatch.setattr(gpu, "_find_nvidia_smi", lambda: None)
    assert gpu.nvidia_vram_mb() is None
    monkeypatch.setattr(gpu, "_find_nvidia_smi", lambda: "nvidia-smi")

    def boom(*a, **k):
        raise subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(gpu.subprocess, "run", boom)
    assert gpu.nvidia_vram_mb() is None
    monkeypatch.setattr(
        gpu.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(cmd, 9, stdout="", stderr="err")
    )
    assert gpu.nvidia_vram_mb() is None


def test_enumerate_windows_fallback_chain():
    def dxgi_fail():
        raise OSError("CreateDXGIFactory1 failed")

    cim = [Device(0, "NVIDIA GeForce RTX 4070", None, "nvidia"), Device(1, "Intel", 512, "intel")]
    devs = gpu._enumerate_windows(dxgi_fail, lambda: [Device(**d.__dict__) for d in cim], lambda: [12282])
    assert devs[0].vram_mb == 12282 and devs[1].vram_mb == 512
    # DXGI ok with exact VRAM -> nvidia-smi not consulted
    called = []
    devs = gpu._enumerate_windows(
        lambda: [Device(0, "NVIDIA", 8192, "nvidia")], lambda: [], lambda: called.append(1) or [1]
    )
    assert devs[0].vram_mb == 8192 and not called
    # everything fails -> []
    assert gpu._enumerate_windows(dxgi_fail, lambda: [], lambda: []) == []


def test_system_info_is_jsonable():
    info = gpu.system_info()
    json.dumps(info)
    assert "providers" in info and "CPUExecutionProvider" in info["providers"]


def test_cim_fallback_command_is_safe_and_utf8(monkeypatch):
    seen = {}

    def fake_run(cmd, timeout):
        seen["cmd"], seen["timeout"] = cmd, timeout
        return '{"Name":"NVIDIA GeForce RTX 4070","AdapterRAM":4293918720,"PNPDeviceID":"PCI\\\\VEN_10DE&DEV_2786"}'

    monkeypatch.setattr(gpu, "_run_quiet", fake_run)
    devs = gpu._cim_devices()
    assert devs == [Device(0, "NVIDIA GeForce RTX 4070", None, "nvidia")]
    cmd = seen["cmd"]
    assert isinstance(cmd, list) and cmd[1:4] == ["-NoProfile", "-NonInteractive", "-Command"]
    assert "UTF8" in cmd[4] and "Win32_VideoController" in cmd[4]
    assert seen["timeout"] > 0


def test_run_quiet_uses_no_window_and_no_shell(monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(gpu.subprocess, "run", fake_run)
    monkeypatch.setattr(gpu.sys, "platform", "win32")
    assert gpu._run_quiet(["tool", "--flag"], timeout=5) == "ok"
    cmd, kw = calls[0]
    assert kw.get("shell", False) is False and kw["creationflags"] & 0x08000000
    assert kw["encoding"] == "utf-8" and kw["errors"] == "replace"


# --------------------------------------------------------------------------- duplicate adapters (v0.1.1)
GTX1080 = "NVIDIA GeForce GTX 1080"


def _gtx1080(luid, **kw):
    return _desc(GTX1080, 0x10DE, 8192, luid=luid, device_id=0x1B80, subsys=0x33011462, **kw)


def _no_smi():
    raise AssertionError("nvidia-smi must not be queried")


def test_same_luid_listed_twice_is_one_gpu_with_its_first_dxgi_index():
    api = FakeDxgi(
        [
            _desc("Intel(R) UHD Graphics 630", 0x8086, 128, luid=(0, 0x1111)),
            _desc("Microsoft Basic Render Driver", 0x1414, 0, flags=gpu.DXGI_ADAPTER_FLAG_SOFTWARE),
            _gtx1080((0, 0xC0DE)),
            _gtx1080((0, 0xC0DE)),
        ]
    )
    devs = gpu._dxgi_devices(api, nvsmi=_no_smi)
    assert devs == [
        Device(0, "Intel(R) UHD Graphics 630", 128, "intel"),
        Device(2, GTX1080, 8192, "nvidia", aliases=(3,)),  # index 2 = DirectML device_id
    ]
    assert devs[1].label == "NVIDIA GeForce GTX 1080 (8 GB)"  # one GPU: no "GPU n:" prefix


def test_negative_luid_high_part_is_compared_correctly():
    devs = gpu._dxgi_devices(FakeDxgi([_gtx1080((-1, 5)), _gtx1080((-1, 5))]), nvsmi=_no_smi)
    assert [(d.index, d.aliases) for d in devs] == [(0, (1,))]


def test_phantom_nvidia_entry_dropped_when_nvidia_smi_reports_fewer_gpus():
    """The user's PC: one GTX 1080, listed twice by DXGI with different LUIDs."""
    calls = []
    api = FakeDxgi([_gtx1080((0, 0xA)), _gtx1080((0, 0xB))])
    devs = gpu._dxgi_devices(api, nvsmi=lambda: calls.append(1) or [8192])
    assert devs == [Device(0, GTX1080, 8192, "nvidia", aliases=(1,))]
    assert calls == [1]


def test_two_identical_real_gpus_are_both_kept_with_unique_labels():
    api = FakeDxgi([_gtx1080((0, 0xA)), _gtx1080((0, 0xB))])
    devs = gpu._dxgi_devices(api, nvsmi=lambda: [8192, 8192])
    assert [d.index for d in devs] == [0, 1]
    assert [d.label for d in devs] == [
        "GPU 0: NVIDIA GeForce GTX 1080 (8 GB)",
        "GPU 1: NVIDIA GeForce GTX 1080 (8 GB)",
    ]
    # nvidia-smi unavailable: nothing can be proven, so both stay (and stay distinguishable)
    devs = gpu._dxgi_devices(FakeDxgi([_gtx1080((0, 0xA)), _gtx1080((0, 0xB))]), nvsmi=list)
    assert [d.label.split(":")[0] for d in devs] == ["GPU 0", "GPU 1"]


def test_phantom_dropped_next_to_a_different_nvidia_gpu():
    api = FakeDxgi(
        [
            _desc("NVIDIA GeForce RTX 4090", 0x10DE, 24564, luid=(0, 1), device_id=0x2684, subsys=1),
            _gtx1080((0, 2)),
            _gtx1080((0, 3)),
            _gtx1080((0, 4)),
        ]
    )
    devs = gpu._dxgi_devices(api, nvsmi=lambda: [24564, 8192, 8192])  # 3 real GPUs, 4 entries
    assert [(d.index, d.aliases) for d in devs] == [(0, ()), (1, (3,)), (2, ())]  # first ones kept
    assert [d.label for d in devs][1:] == [
        "GPU 1: NVIDIA GeForce GTX 1080 (8 GB)",
        "GPU 2: NVIDIA GeForce GTX 1080 (8 GB)",
    ]
    assert devs[0].label == "NVIDIA GeForce RTX 4090 (24 GB)"


def test_nvidia_smi_not_queried_without_duplicate_hardware_ids():
    api = FakeDxgi(
        [
            _desc("NVIDIA GeForce RTX 3060", 0x10DE, 12288, luid=(0, 1), device_id=0x2504),
            _desc("NVIDIA GeForce RTX 4070", 0x10DE, 12282, luid=(0, 2), device_id=0x2786),
        ]
    )
    assert [d.index for d in gpu._dxgi_devices(api, nvsmi=_no_smi)] == [0, 1]


def test_adapter_details_logged_at_debug(caplog):
    api = FakeDxgi([_gtx1080((0, 0xC0DE)), _gtx1080((0, 0xC0DE))])
    with caplog.at_level("DEBUG", logger="vrphoto.models.gpu"):
        gpu._dxgi_devices(api, nvsmi=_no_smi)
    debug = [r.getMessage() for r in caplog.records if r.levelname == "DEBUG"]
    assert len(debug) == 2
    msg = debug[1]
    for part in ("adapter 1", "LUID=00000000:0000C0DE", "VendorId=0x10DE", "DeviceId=0x1B80"):
        assert part in msg
    assert "SubSysId=0x33011462" in msg and "Flags=0x0" in msg
    assert any("same LUID" in r.getMessage() for r in caplog.records if r.levelname == "INFO")


def test_repeated_wmi_names_get_unique_labels():
    item = {"Name": GTX1080, "AdapterRAM": 4293918720, "PNPDeviceID": "PCI\\VEN_10DE&DEV_1B80"}
    devs = gpu.parse_cim_json(json.dumps([item, item]))
    assert [d.label for d in devs] == ["GPU 0: NVIDIA GeForce GTX 1080", "GPU 1: NVIDIA GeForce GTX 1080"]


def test_canonical_index_maps_hidden_duplicates(monkeypatch):
    devs = [Device(0, "Intel UHD", 128, "intel"), Device(2, GTX1080, 8192, "nvidia", aliases=(1, 3))]
    monkeypatch.setattr(gpu, "list_devices", lambda **k: devs)
    assert gpu.canonical_index(2) == 2
    assert gpu.canonical_index(1) == 2 and gpu.canonical_index(3) == 2
    assert gpu.canonical_index(0) == 0 and gpu.canonical_index(7) == 7


def test_canonical_index_logs_a_hidden_duplicate_once(monkeypatch, caplog):
    """It runs on every model load: the mapping is logged once, not on every call."""
    devs = [Device(0, GTX1080, 8192, "nvidia", aliases=(1,))]
    monkeypatch.setattr(gpu, "list_devices", lambda **k: devs)
    monkeypatch.setattr(gpu, "_alias_logged", set())
    with caplog.at_level("INFO", logger="vrphoto.models.gpu"):
        assert [gpu.canonical_index(1) for _ in range(3)] == [0, 0, 0]
    assert sum("duplicate entry" in r.getMessage() for r in caplog.records) == 1


def test_list_devices_dedupes_and_runs_nvidia_smi_once(monkeypatch):
    """Windows path end to end (fake DXGI): one GTX 1080 listed twice -> one device."""
    smi_calls = []
    api = FakeDxgi(
        [
            _desc(GTX1080, 0x10DE, 0, luid=(0, 0xA), device_id=0x1B80),
            _desc(GTX1080, 0x10DE, 0, luid=(0, 0xB), device_id=0x1B80),
        ]
    )
    monkeypatch.setattr(gpu.sys, "platform", "win32")
    monkeypatch.setattr(gpu, "_DxgiApi", lambda: api)
    monkeypatch.setattr(gpu, "nvidia_vram_list", lambda: smi_calls.append(1) or [8192])
    monkeypatch.setattr(gpu, "_cached", None)
    devs = gpu.list_devices(refresh=True)
    assert devs == [Device(0, GTX1080, 8192, "nvidia", aliases=(1,))]  # VRAM from nvidia-smi
    assert smi_calls == [1] and api.closed
    assert gpu.device_choices()[2:] == [("0", "NVIDIA GeForce GTX 1080 (8 GB)")]
