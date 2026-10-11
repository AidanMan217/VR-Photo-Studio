"""GPU adapter discovery (Windows DXGI) for the device picker and DirectML ``device_id``.

On Windows the hardware adapters are enumerated with DXGI
(``CreateDXGIFactory1`` -> ``IDXGIFactory1::EnumAdapters1`` -> ``IDXGIAdapter1::GetDesc1``)
through raw COM vtable calls via ``ctypes``.  :attr:`Device.index` is the DXGI
enumeration ordinal, which is exactly the ``device_id`` the DirectML execution provider
passes to ``EnumAdapters1``; software adapters (Microsoft Basic Render Driver) are
skipped.  An adapter that DXGI lists more than once (same ``AdapterLuid``, or more
NVIDIA entries with identical hardware ids than ``nvidia-smi`` reports GPUs) is listed
once, under the index of its first entry; the dropped indexes are kept as
:attr:`Device.aliases`.  If DXGI fails, ``Get-CimInstance Win32_VideoController``
(PowerShell) is used as a fallback (its order is *usually* the DXGI order; its AdapterRAM
is capped at 4 GB, so NVIDIA VRAM is taken from ``nvidia-smi``).  Other platforms return
``[]``.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from ..core.log import get_logger

log = get_logger(__name__)

VENDOR_IDS = {0x10DE: "nvidia", 0x1002: "amd", 0x1022: "amd", 0x8086: "intel"}
MICROSOFT_VENDOR_ID = 0x1414  # Basic Render Driver / WARP
DXGI_ADAPTER_FLAG_SOFTWARE = 0x2
DXGI_ERROR_NOT_FOUND = 0x887A0002
_CREATE_NO_WINDOW = 0x08000000


@dataclass
class Device:
    """A GPU adapter.  ``index`` is the DirectML device_id (DXGI enumeration order)."""

    index: int
    name: str
    vram_mb: int | None
    vendor: str  # "nvidia" | "amd" | "intel" | "other" | "cpu"
    # several adapters share ``name``: the label starts with "GPU <index>: " to tell them apart
    show_index: bool = False
    # DXGI indexes of duplicate entries of this adapter that were dropped from the list
    aliases: tuple[int, ...] = field(default=())

    @property
    def label(self) -> str:
        """Human-readable label for a device picker (unique within :func:`list_devices`)."""
        text = f"{self.name} ({self.vram_mb / 1024:.0f} GB)" if self.vram_mb else self.name
        return f"GPU {self.index}: {text}" if self.show_index else text


def vendor_name(vendor_id: int) -> str:
    """Map a PCI vendor id to ``"nvidia" | "amd" | "intel" | "other"``."""
    return VENDOR_IDS.get(int(vendor_id), "other")


# --------------------------------------------------------------------------- DXGI via ctypes
class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class DXGI_ADAPTER_DESC1(ctypes.Structure):  # noqa: N801 - Windows SDK name
    """Layout of ``DXGI_ADAPTER_DESC1`` (WCHAR stored as uint16 so it is portable)."""

    _fields_ = [
        ("Description", ctypes.c_uint16 * 128),
        ("VendorId", ctypes.c_uint32),
        ("DeviceId", ctypes.c_uint32),
        ("SubSysId", ctypes.c_uint32),
        ("Revision", ctypes.c_uint32),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", ctypes.c_uint32),
    ]

    @property
    def description(self) -> str:
        raw = bytes(self.Description)
        return raw.decode("utf-16-le", errors="replace").split("\x00", 1)[0].strip()


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_uint8 * 8),
    ]


def _guid(d1: int, d2: int, d3: int, d4: tuple[int, ...]) -> _GUID:
    return _GUID(d1, d2, d3, (ctypes.c_uint8 * 8)(*d4))


# IDXGIFactory1 vtable: IUnknown(0-2) IDXGIObject(3-6) IDXGIFactory(7-11) EnumAdapters1=12
# IDXGIAdapter1 vtable: IUnknown(0-2) IDXGIObject(3-6) IDXGIAdapter(7-9) GetDesc1=10
_VT_RELEASE = 2
_VT_ENUM_ADAPTERS1 = 12
_VT_GET_DESC1 = 10


def _vcall(obj: ctypes.c_void_p, index: int, argtypes: list[Any], *args: Any) -> int:
    """Call COM method number ``index`` of ``obj``; returns the HRESULT as unsigned."""
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    proto = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)  # type: ignore[attr-defined]
    hr = proto(vtbl[index])(obj, *args)
    return hr & 0xFFFFFFFF


class _DxgiApi:
    """Real DXGI access (Windows only)."""

    def __init__(self) -> None:
        dxgi = ctypes.WinDLL("dxgi.dll")  # type: ignore[attr-defined]
        create = dxgi.CreateDXGIFactory1
        create.restype = ctypes.c_long
        create.argtypes = [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
        iid = _guid(0x770AAE78, 0xF26F, 0x4DBA, (0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
        self.factory = ctypes.c_void_p()
        hr = create(ctypes.byref(iid), ctypes.byref(self.factory)) & 0xFFFFFFFF
        if hr & 0x80000000 or not self.factory:
            raise OSError(f"CreateDXGIFactory1 failed (0x{hr:08X})")

    def desc(self, i: int) -> DXGI_ADAPTER_DESC1 | None:
        """Descriptor of adapter ``i`` or None past the last adapter."""
        adapter = ctypes.c_void_p()
        hr = _vcall(
            self.factory,
            _VT_ENUM_ADAPTERS1,
            [ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)],
            i,
            ctypes.byref(adapter),
        )
        if hr == DXGI_ERROR_NOT_FOUND:
            return None
        if hr & 0x80000000 or not adapter:
            raise OSError(f"EnumAdapters1({i}) failed (0x{hr:08X})")
        try:
            d = DXGI_ADAPTER_DESC1()
            hr = _vcall(adapter, _VT_GET_DESC1, [ctypes.POINTER(DXGI_ADAPTER_DESC1)], ctypes.byref(d))
            if hr & 0x80000000:
                raise OSError(f"GetDesc1({i}) failed (0x{hr:08X})")
            return d
        finally:
            _vcall(adapter, _VT_RELEASE, [])

    def close(self) -> None:
        if self.factory:
            _vcall(self.factory, _VT_RELEASE, [])
            self.factory = ctypes.c_void_p()


def _dxgi_devices(
    api: Any = None, max_adapters: int = 64, nvsmi: Callable[[], list[int]] | None = None
) -> list[Device]:
    """Enumerate hardware adapters through ``api.desc(i)`` (default: real DXGI).

    Each physical adapter is listed once, under its first DXGI index (the DirectML
    device_id): entries repeating an ``AdapterLuid`` are dropped, and so are surplus NVIDIA
    entries with identical VendorId/DeviceId/SubSysId when ``nvsmi()`` (default:
    :func:`nvidia_vram_list`, only called in that case) reports fewer NVIDIA GPUs.
    """
    own = api is None
    if own:
        api = _DxgiApi()
    found: list[tuple[Device, tuple[int, int, int]]] = []
    by_luid: dict[tuple[int, int], Device] = {}
    try:
        for i in range(max_adapters):
            d = api.desc(i)
            if d is None:
                break
            luid = (int(d.AdapterLuid.HighPart) & 0xFFFFFFFF, int(d.AdapterLuid.LowPart) & 0xFFFFFFFF)
            log.debug(
                "DXGI adapter %d: %r LUID=%08X:%08X VendorId=0x%04X DeviceId=0x%04X SubSysId=0x%08X "
                "Flags=0x%X VRAM=%d MB",
                i,
                d.description,
                luid[0],
                luid[1],
                d.VendorId,
                d.DeviceId,
                d.SubSysId,
                d.Flags,
                int(d.DedicatedVideoMemory) // (1024 * 1024),
            )
            if d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE or d.VendorId == MICROSOFT_VENDOR_ID:
                continue
            first = by_luid.get(luid) if luid != (0, 0) else None  # 0 = unknown (never a real LUID)
            if first is not None:
                log.info(
                    "GPU adapter %d (%s) is listed again as adapter %d (same LUID); ignoring the duplicate",
                    first.index,
                    first.name,
                    i,
                )
                first.aliases += (i,)
                continue
            vram = int(d.DedicatedVideoMemory) // (1024 * 1024)
            dev = Device(i, d.description or f"GPU {i}", vram or None, vendor_name(d.VendorId))
            if luid != (0, 0):
                by_luid[luid] = dev
            found.append((dev, (int(d.VendorId), int(d.DeviceId), int(d.SubSysId))))
    finally:
        if own:
            api.close()
    devs = _drop_surplus_nvidia(found, nvsmi or nvidia_vram_list)
    return _mark_repeated_names(devs)


def _drop_surplus_nvidia(
    found: list[tuple[Device, tuple[int, int, int]]], nvsmi: Callable[[], list[int]]
) -> list[Device]:
    """Drop NVIDIA entries that repeat another entry's hardware ids (VendorId, DeviceId,
    SubSysId) while DXGI lists more NVIDIA adapters than ``nvidia-smi`` reports GPUs (a
    leftover / virtual adapter).  The first entries (lowest DXGI index) are kept."""
    devs = [dev for dev, _ in found]
    groups: dict[tuple[int, int, int], list[Device]] = {}
    for dev, hw in found:
        if dev.vendor == "nvidia":
            groups.setdefault(hw, []).append(dev)
    if all(len(g) < 2 for g in groups.values()):
        return devs
    try:
        real = len(nvsmi())
    except Exception as e:  # pragma: no cover - nvidia_vram_list never raises
        log.debug("nvidia-smi query failed: %s", e)
        real = 0
    excess = sum(len(g) for g in groups.values()) - real
    if real <= 0 or excess <= 0:  # unknown count, or every entry is a real GPU
        return devs
    dropped: set[int] = set()
    for g in groups.values():
        for extra in reversed(g[1:]):
            if excess <= 0:
                break
            log.info(
                "GPU adapter %d (%s) has the same hardware id as adapter %d, but nvidia-smi reports "
                "only %d NVIDIA GPU(s); ignoring it",
                extra.index,
                extra.name,
                g[0].index,
                real,
            )
            g[0].aliases += (extra.index, *extra.aliases)
            dropped.add(extra.index)
            excess -= 1
    return [d for d in devs if d.index not in dropped]


def _mark_repeated_names(devs: list[Device]) -> list[Device]:
    """Make labels unique: adapters sharing a name are shown as "GPU <index>: <name>"."""
    counts = Counter(d.name.casefold() for d in devs)
    for d in devs:
        d.show_index = counts[d.name.casefold()] > 1
    return devs


# --------------------------------------------------------------------------- PowerShell / nvidia-smi
def _run_quiet(cmd: list[str], timeout: float) -> str | None:
    kw: dict[str, Any] = {}
    if sys.platform == "win32":
        kw["creationflags"] = _CREATE_NO_WINDOW
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            check=False,
            **kw,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.debug("%s failed: %s", cmd[0], e)
        return None
    if res.returncode != 0:
        log.debug("%s exited with %s: %s", cmd[0], res.returncode, (res.stderr or "").strip()[:200])
        return None
    return res.stdout


def parse_cim_json(text: str) -> list[Device]:
    """Parse ``Get-CimInstance Win32_VideoController | ConvertTo-Json`` output."""
    try:
        data = json.loads(text) if text and text.strip() else []
    except ValueError:
        return []
    if isinstance(data, dict):
        data = [data]
    devs: list[Device] = []
    idx = 0
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("Name") or "").strip()
        pnp = str(item.get("PNPDeviceID") or "").upper()
        vid = None
        if "VEN_" in pnp:
            try:
                vid = int(pnp.split("VEN_", 1)[1][:4], 16)
            except ValueError:
                vid = None
        if vid == MICROSOFT_VENDOR_ID or name.lower().startswith("microsoft basic"):
            continue
        ram = item.get("AdapterRAM")
        vram: int | None = None
        if isinstance(ram, (int, float)) and ram > 0:
            ram = int(ram) & 0xFFFFFFFF
            vram = None if ram >= 0xFFF00000 else ram // (1024 * 1024)  # uint32 cap -> unknown
        vendor = vendor_name(vid) if vid is not None else _vendor_from_name(name)
        devs.append(Device(idx, name or f"GPU {idx}", vram, vendor))
        idx += 1
    return _mark_repeated_names(devs)


def _vendor_from_name(name: str) -> str:
    n = name.lower()
    if "nvidia" in n or "geforce" in n or "quadro" in n:
        return "nvidia"
    if "amd" in n or "radeon" in n:
        return "amd"
    if "intel" in n:
        return "intel"
    return "other"


def _cim_devices() -> list[Device]:
    ps = shutil.which("powershell") or os.path.join(
        os.environ.get("SystemRoot", r"C:\Windows"), "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
    )
    out = _run_quiet(
        [
            ps,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            # force UTF-8 on the pipe (Windows PowerShell 5.1 defaults to the OEM code page)
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,PNPDeviceID "
            "| ConvertTo-Json -Compress",
        ],
        timeout=20,
    )
    return parse_cim_json(out or "")


def _find_nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    if sys.platform == "win32":
        root = os.environ.get("SystemRoot", r"C:\Windows")
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        for cand in (
            os.path.join(root, "System32", "nvidia-smi.exe"),
            os.path.join(pf, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"),
        ):
            if os.path.isfile(cand):
                return cand
    return None


def parse_nvidia_smi(text: str) -> list[int]:
    """Parse ``--query-gpu=memory.total --format=csv,noheader,nounits`` output (MiB)."""
    vals: list[int] = []
    for line in (text or "").splitlines():
        tok = line.strip().split(",")[0].strip()
        try:
            vals.append(int(float(tok)))
        except ValueError:
            continue
    return vals


def nvidia_vram_list() -> list[int]:
    """VRAM in MiB of each NVIDIA GPU (nvidia-smi order); [] if nvidia-smi is unavailable."""
    exe = _find_nvidia_smi()
    if not exe:
        return []
    out = _run_quiet([exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"], timeout=10)
    return parse_nvidia_smi(out or "")


def nvidia_vram_mb() -> int | None:
    """Largest NVIDIA GPU memory in MiB via ``nvidia-smi`` (None if not available)."""
    vals = nvidia_vram_list()
    return max(vals) if vals else None


# --------------------------------------------------------------------------- public
_lock = threading.Lock()
_cached: list[Device] | None = None


def list_devices(*, refresh: bool = False) -> list[Device]:
    """Hardware GPU adapters (Windows); ``[]`` on other platforms or on failure.

    Results are cached for the process; pass ``refresh=True`` to re-enumerate.
    """
    global _cached
    if sys.platform != "win32":
        return []
    with _lock:
        if _cached is not None and not refresh:
            return list(_cached)
        smi: list[list[int]] = []

        def nvsmi() -> list[int]:  # nvidia-smi is slow: run it at most once per enumeration
            if not smi:
                smi.append(nvidia_vram_list())
            return smi[0]

        devs = _enumerate_windows(lambda: _dxgi_devices(nvsmi=nvsmi), _cim_devices, nvsmi)
        _cached = devs
        return list(devs)


def _enumerate_windows(
    dxgi: Callable[[], list[Device]],
    cim: Callable[[], list[Device]],
    nvsmi: Callable[[], list[int]],
) -> list[Device]:
    devs: list[Device] = []
    source = "dxgi"
    try:
        devs = dxgi()
    except Exception as e:
        log.warning("DXGI adapter enumeration failed (%s); trying WMI", e)
    if not devs:
        source = "wmi"
        try:
            devs = cim()
        except Exception as e:  # pragma: no cover - defensive
            log.warning("WMI adapter enumeration failed: %s", e)
            devs = []
    nv = [d for d in devs if d.vendor == "nvidia"]
    if nv and (source == "wmi" or any(d.vram_mb is None for d in nv)):
        smi = nvsmi()
        for d, mb in zip(nv, smi):
            d.vram_mb = mb
    log.info("GPU adapters (%s): %s", source, ", ".join(f"{d.index}:{d.label}" for d in devs) or "none")
    return devs


def canonical_index(index: int) -> int:
    """The listed adapter that DXGI index ``index`` refers to.

    A duplicate entry dropped from :func:`list_devices` (see :attr:`Device.aliases`) maps
    to the adapter it duplicates, so a device chosen before the duplicate was hidden keeps
    working; any other index is returned unchanged.
    """
    devs = list_devices()
    if any(d.index == index for d in devs):
        return index
    for d in devs:
        if index in d.aliases:
            if index not in _alias_logged:  # called for every model load: say it once
                _alias_logged.add(index)
                log.info(
                    "GPU %d is a duplicate entry of GPU %d (%s); using GPU %d",
                    index,
                    d.index,
                    d.name,
                    d.index,
                )
            return d.index
    return index


_alias_logged: set[int] = set()


def device_choices() -> list[tuple[str, str]]:
    """``(value, label)`` pairs for a Settings device picker (value fits ``Settings.device``)."""
    out = [("auto", "Automatic (fastest GPU)"), ("cpu", "CPU only")]
    out += [(str(d.index), d.label) for d in list_devices()]
    return out


def system_info() -> dict[str, Any]:
    """JSON-able summary of GPUs and ONNX Runtime providers (for diagnostics / selftest)."""
    from .ort import available_providers

    return {
        "platform": sys.platform,
        "devices": [asdict(d) for d in list_devices()],
        "providers": available_providers(),
        "nvidia_vram_mb": nvidia_vram_mb(),
    }


__all__ = [
    "Device",
    "list_devices",
    "canonical_index",
    "nvidia_vram_mb",
    "nvidia_vram_list",
    "device_choices",
    "system_info",
    "vendor_name",
    "parse_cim_json",
    "parse_nvidia_smi",
]
