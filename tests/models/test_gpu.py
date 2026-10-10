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


def _desc(name: str, vendor: int, vram_mb: int, flags: int = 0) -> DXGI_ADAPTER_DESC1:
    d = DXGI_ADAPTER_DESC1()
    raw = name.encode("utf-16-le")
    ctypes.memmove(d.Description, raw, len(raw))
    d.VendorId = vendor
    d.DedicatedVideoMemory = vram_mb * 1024 * 1024
    d.Flags = flags
    return d


class FakeDxgi:
    def __init__(self, descs):
        self.descs = descs

    def desc(self, i):
        return self.descs[i] if i < len(self.descs) else None


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
