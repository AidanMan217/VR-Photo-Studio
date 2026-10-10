"""Network helpers for serving the viewer to a Quest on the local network.

* :func:`lan_ips` / :func:`primary_ip` - the PC's IPv4 addresses (no extra dependencies:
  UDP "connect" trick for the primary route + ``getaddrinfo(hostname)``, falling back to
  ``ipconfig`` / ``ip`` / ``ifconfig`` output only when those find nothing).
* :func:`qr_png` - a QR code PNG so the URL can be scanned instead of typed.
* :func:`firewall_status` / :func:`add_firewall_rule` - Windows Defender Firewall checks;
  inbound TCP on the *Private* profile must be allowed or the Quest cannot connect.

Everything here is safe to call from any thread and never raises for environmental
problems (missing tools, no network, user declining UAC): it degrades to empty results,
``"unknown"`` or ``False`` and logs why.
"""

from __future__ import annotations

import io
import ipaddress
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..core.errors import InputError
from ..core.log import get_logger

log = get_logger(__name__)

__all__ = [
    "FIREWALL_RULE_NAME",
    "lan_ips",
    "primary_ip",
    "qr_png",
    "firewall_status",
    "firewall_report",
    "add_firewall_rule",
]

FIREWALL_RULE_NAME = "VR Photo Studio"
#: Executables of the installed app (GUI and CLI live in the same folder).
_APP_EXES = ("VRPhotoStudio.exe", "vrphoto-cli.exe")
_ERROR_CANCELLED = 1223  # user declined the UAC prompt

# --------------------------------------------------------------------------- subprocess helpers


def _no_window_kwargs() -> dict[str, Any]:
    """Popen kwargs that keep console windows from flashing up on Windows."""
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def _run(args: list[str], timeout: float = 10.0) -> subprocess.CompletedProcess[str] | None:
    """Run a short command, returning None if it cannot be started or times out."""
    try:
        return subprocess.run(
            args,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            **_no_window_kwargs(),
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        log.debug("Command %s failed to run: %s", args[0] if args else "?", exc)
        return None


def _powershell() -> str | None:
    """Path of Windows PowerShell (5.1 ships with every Windows 10/11)."""
    root = os.environ.get("SystemRoot") or os.environ.get("windir") or r"C:\Windows"
    cand = Path(root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    if cand.is_file():
        return str(cand)
    return shutil.which("powershell") or shutil.which("pwsh")


# PowerShell treats the typographic quotes U+2018..U+201B exactly like ``'`` (see
# CharTraits.IsSingleQuote in the PowerShell parser): an unescaped one would end the
# literal early, e.g. for a photo named "Mom’s birthday.jpg".
_PS_SINGLE_QUOTES = re.compile("([\u0027\u2018\u2019\u201a\u201b])")


def _ps_quote(value: str) -> str:
    """Single-quoted PowerShell string literal.

    Inside single quotes PowerShell only interprets quote characters, and each one is
    escaped by doubling it - including the typographic single quotes it also accepts.
    NUL characters (never valid in paths) are dropped.
    """
    text = str(value).replace("\x00", "")
    return "'" + _PS_SINGLE_QUOTES.sub(r"\1\1", text) + "'"


def _write_temp_script(text: str, prefix: str) -> Path:
    """Write a PowerShell script as UTF-8 *with BOM* (PowerShell 5.1 needs it for non-ASCII)."""
    fd, name = tempfile.mkstemp(prefix=prefix, suffix=".ps1")
    with os.fdopen(fd, "w", encoding="utf-8-sig", newline="\r\n") as fh:
        fh.write(text)
    return Path(name)


def _ps_args(script: Path, *, sta: bool = False) -> list[str] | None:
    """Command line running ``script`` with Windows PowerShell (None if unavailable)."""
    exe = _powershell()
    if exe is None:
        return None
    args = [exe, "-NoProfile", "-NonInteractive"]
    if sta:
        args.append("-Sta")  # Shell.Application COM objects want a single-threaded apartment
    return args + ["-ExecutionPolicy", "Bypass", "-File", str(script)]


def _run_powershell(script_text: str, timeout: float = 30.0) -> subprocess.CompletedProcess[str] | None:
    """Run a generated PowerShell script from a temp ``.ps1`` file (deleted afterwards)."""
    try:
        path = _write_temp_script(script_text, "vrphoto_")
    except OSError as exc:
        log.debug("Cannot write temporary PowerShell script: %s", exc)
        return None
    try:
        args = _ps_args(path)
        if args is None:
            log.info("PowerShell is not available")
            return None
        return _run(args, timeout=timeout)
    finally:
        try:
            path.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------- addresses


def _usable(ip: str) -> bool:
    """True for a unicast, non-loopback, non-link-local IPv4 address."""
    try:
        a = ipaddress.IPv4Address(ip)
    except ValueError:
        return False
    return not (a.is_loopback or a.is_link_local or a.is_multicast or a.is_unspecified or a.is_reserved
                or str(a) == "255.255.255.255")


def _udp_primary() -> str | None:
    """Local address of the interface that routes to the internet / private networks.

    ``connect()`` on a UDP socket only selects a route - no packet is sent.
    """
    # Non-broadcast targets: connect() to a broadcast address fails without SO_BROADCAST.
    for target in (("8.8.8.8", 80), ("192.168.255.254", 1), ("10.255.255.254", 1), ("172.31.255.254", 1)):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(target)
                ip = s.getsockname()[0]
        except OSError:
            continue
        if _usable(ip):
            return ip
    return None


def _hostname_ips() -> list[str]:
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET, socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return []
    return [str(info[4][0]) for info in infos]


_IPCONFIG_RE = re.compile(r"IPv4[^:\r\n]*:\s*(\d{1,3}(?:\.\d{1,3}){3})")
_INET_RE = re.compile(r"\binet\s+(?:addr:)?(\d{1,3}(?:\.\d{1,3}){3})")


def _command_ips() -> list[str]:
    """Parse the OS network tool output (used only when the cheap methods found nothing)."""
    if sys.platform == "win32":
        cp = _run(["ipconfig"], timeout=10)
        return _IPCONFIG_RE.findall(cp.stdout) if cp and cp.stdout else []
    for args in (["ip", "-o", "-4", "addr", "show", "up"], ["ifconfig"]):
        if shutil.which(args[0]) is None:
            continue
        cp = _run(args, timeout=5)
        if cp and cp.returncode == 0 and cp.stdout:
            return _INET_RE.findall(cp.stdout)
    return []


def _rank(ip: str) -> int:
    a = ipaddress.IPv4Address(ip)
    if a in ipaddress.IPv4Network("192.168.0.0/16"):
        return 0
    if a in ipaddress.IPv4Network("10.0.0.0/8"):
        return 1
    if a in ipaddress.IPv4Network("172.16.0.0/12"):
        return 2  # often Docker / WSL / Hyper-V virtual switches
    return 3


def lan_ips() -> list[str]:
    """All non-loopback, non-link-local IPv4 addresses of this PC, best candidate first.

    The address of the default route comes first, then private addresses (192.168/16,
    10/8, 172.16/12), then any others.  Returns ``[]`` when the PC has no network.
    """
    primary = _udp_primary()
    found = [ip for ip in _hostname_ips() if _usable(ip)]
    if primary is None and not found:
        found = [ip for ip in _command_ips() if _usable(ip)]
    uniq: list[str] = []
    for ip in found:
        if ip not in uniq and ip != primary:
            uniq.append(ip)
    uniq.sort(key=_rank)  # stable: keeps OS order within a class
    return ([primary] if primary else []) + uniq


def primary_ip() -> str:
    """The most likely LAN address of this PC (``"127.0.0.1"`` when offline)."""
    ip = _udp_primary()
    if ip:
        return ip
    ips = lan_ips()
    return ips[0] if ips else "127.0.0.1"


# --------------------------------------------------------------------------- QR code


def qr_png(text: str, box_size: int = 8) -> bytes:
    """Encode ``text`` (usually a URL) as a QR code and return PNG bytes.

    Uses error-correction level M and the standard 4-module quiet zone.  Raises
    :class:`~vrphoto.core.errors.InputError` if the text does not fit in a QR code.
    """
    import qrcode
    from qrcode.exceptions import DataOverflowError
    from qrcode.image.pil import PilImage

    if box_size < 1:
        raise InputError(f"QR box size must be at least 1 pixel (got {box_size})")
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=int(box_size),
        border=4,
        image_factory=PilImage,
    )
    qr.add_data(text)
    try:
        qr.make(fit=True)
    except (DataOverflowError, ValueError) as exc:
        raise InputError("The text is too long for a QR code", hint="Use a shorter URL.") from exc
    img = qr.make_image(fill_color="black", back_color="white").get_image()
    buf = io.BytesIO()
    img.convert("L").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# --------------------------------------------------------------------------- Windows firewall

_FW_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$port = __PORT__
$exe = __EXE__
try { $fw = New-Object -ComObject HNetCfg.FwPolicy2 } catch { Write-Output 'RESULT|unknown||no-firewall-api'; exit 0 }
$cur = [int]$fw.CurrentProfileTypes
$names = @()
if ($cur -band 1) { $names += 'domain' }
if ($cur -band 2) { $names += 'private' }
if ($cur -band 4) { $names += 'public' }
$prof = $names -join ','
$on = $false; $defAllow = $false
foreach ($p in 1, 2, 4) {
  if (($cur -band $p) -and $fw.FirewallEnabled($p)) {
    $on = $true
    if ($fw.BlockAllInboundTraffic($p)) { Write-Output "RESULT|blocked|$prof|block-all"; exit 0 }
    if ([int]$fw.DefaultInboundAction($p) -eq 1) { $defAllow = $true }
  }
}
if (-not $on) { Write-Output "RESULT|ok|$prof|firewall-off"; exit 0 }
function PortMatch([string]$spec) {
  if (-not $spec -or $spec -eq '*') { return $true }
  foreach ($part in $spec.Split(',')) {
    $t = $part.Trim()
    if ($t -match '^(\d+)-(\d+)$') { if ($port -ge [int]$Matches[1] -and $port -le [int]$Matches[2]) { return $true } }
    elseif ($t -match '^\d+$') { if ([int]$t -eq $port) { return $true } }
  }
  return $false
}
$allow = $false; $block = $false
foreach ($r in $fw.Rules) {
  try {
    if (-not $r.Enabled) { continue }
    if ([int]$r.Direction -ne 1) { continue }
    if ((([int]$r.Profiles) -band $cur) -eq 0) { continue }
    $proto = [int]$r.Protocol
    if ($proto -ne 6 -and $proto -ne 256) { continue }
    if (-not (PortMatch ([string]$r.LocalPorts))) { continue }
    $app = [string]$r.ApplicationName
    if ($app) {
      $app = [Environment]::ExpandEnvironmentVariables($app)
      if (-not $exe -or ($app -ne $exe)) { continue }
    }
    if ([int]$r.Action -eq 1) { $allow = $true } else { $block = $true }
  } catch { continue }
}
if ($block) { Write-Output "RESULT|blocked|$prof|block-rule" }
elseif ($allow) { Write-Output "RESULT|ok|$prof|allow-rule" }
elseif ($defAllow) { Write-Output "RESULT|ok|$prof|default-allow" }
else { Write-Output "RESULT|blocked|$prof|no-rule" }
"""

_FW_REASONS = {
    "firewall-off": "Windows Firewall is off for the current network.",
    "allow-rule": "An inbound rule allows the port.",
    "default-allow": "The firewall allows inbound connections by default.",
    "block-all": "Windows Firewall blocks all incoming connections on this network.",
    "block-rule": "A firewall rule blocks VR Photo Studio or this port.",
    "no-rule": "No firewall rule allows incoming connections on this port.",
    "no-firewall-api": "The Windows Firewall settings could not be read.",
}


def _program_path() -> str:
    """Executable that owns the listening socket (the frozen exe or python.exe)."""
    return str(Path(sys.executable).resolve()) if sys.executable else ""


def firewall_report(port: int) -> dict[str, str]:
    """Detailed firewall check: ``{"status", "reason", "profile"}``.

    ``status`` is ``"ok"`` / ``"blocked"`` / ``"unknown"``; ``reason`` is a sentence
    for the GUI; ``profile`` lists the active network profiles (e.g. ``"public"`` - then
    the user should switch the network to *Private* in Windows settings).  Only Windows
    Defender Firewall is inspected; third-party firewalls are not detected.
    """
    if sys.platform != "win32":
        return {"status": "unknown", "reason": "Firewall checks are only available on Windows.", "profile": ""}
    try:
        port = int(port)
    except (TypeError, ValueError):
        return {"status": "unknown", "reason": f"Invalid port {port!r}.", "profile": ""}
    script = _FW_SCRIPT.replace("__PORT__", str(port)).replace("__EXE__", _ps_quote(_program_path()))
    cp = _run_powershell(script, timeout=45)
    if cp is None:
        return {"status": "unknown", "reason": "The firewall check could not be run.", "profile": ""}
    for line in (cp.stdout or "").splitlines():
        if line.startswith("RESULT|"):
            parts = (line.strip().split("|") + ["", "", ""])[:4]
            _, status, profile, code = parts
            if status not in ("ok", "blocked", "unknown"):
                status = "unknown"
            reason = _FW_REASONS.get(code, code)
            if status == "blocked" and profile == "public":
                reason += " Your network is set to Public - set it to Private in Windows network settings."
            return {"status": status, "reason": reason, "profile": profile}
    log.debug("Unexpected firewall check output: %r / %r", cp.stdout, cp.stderr)
    return {"status": "unknown", "reason": "The firewall check gave an unexpected answer.", "profile": ""}


def firewall_status(port: int) -> str:
    """``"ok"`` if Windows Firewall lets inbound TCP ``port`` through on the active
    network profile, ``"blocked"`` if not, ``"unknown"`` on other OSes / on failure."""
    rep = firewall_report(port)
    log.debug("Firewall status for port %s: %s (%s)", port, rep["status"], rep["reason"])
    return rep["status"]


def _rule_programs() -> list[str]:
    """Executables the allow rule is limited to: when frozen, this exe plus the app's other
    exe next to it (GUI / CLI, so ``vrphoto-cli serve`` keeps working).  Empty when running
    from source (the rule must not cover every Python program)."""
    if not (getattr(sys, "frozen", False) and sys.executable):
        return []
    me = _program_path()
    progs = [me]
    for name in _APP_EXES:
        other = Path(me).parent / name
        if other.name.lower() != Path(me).name.lower() and other.is_file():
            progs.append(str(other))
    return progs


def _firewall_command(ports: list[int]) -> str:
    """``cmd.exe /c`` parameters: replace our named rule(s) (and, when frozen, stale
    per-program *block* rules Windows creates if the user dismissed its prompt) with allow
    rules.  When frozen the rules only cover the app's own executables, so other programs
    listening on the same ports are not exposed to the network."""
    plist = ",".join(str(p) for p in ports)
    programs = _rule_programs()
    cmds = [f'netsh advfirewall firewall delete rule name="{FIREWALL_RULE_NAME}" dir=in >nul 2>&1']
    cmds += [f'netsh advfirewall firewall delete rule name=all dir=in program="{p}" >nul 2>&1' for p in programs]
    add = (
        f'netsh advfirewall firewall add rule name="{FIREWALL_RULE_NAME}" dir=in action=allow '
        f"protocol=TCP localport={plist} profile=private,domain"
    )
    adds = [f'{add} program="{p}"' for p in programs] or [add]
    # cmd.exe returns the exit code of the last command; "&&" makes a failed add count
    return "/c " + " & ".join(cmds) + " & " + " && ".join(adds)


def _shell_execute_runas(file: str, params: str, timeout_s: float = 120.0) -> int | None:
    """Run ``file params`` elevated (UAC prompt) and wait; return its exit code.

    Returns None when the user declined UAC or elevation is impossible.  Windows only.
    """
    import ctypes
    from ctypes import wintypes

    class SHELLEXECUTEINFOW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("fMask", ctypes.c_ulong),
            ("hwnd", wintypes.HWND),
            ("lpVerb", wintypes.LPCWSTR),
            ("lpFile", wintypes.LPCWSTR),
            ("lpParameters", wintypes.LPCWSTR),
            ("lpDirectory", wintypes.LPCWSTR),
            ("nShow", ctypes.c_int),
            ("hInstApp", wintypes.HINSTANCE),
            ("lpIDList", ctypes.c_void_p),
            ("lpClass", wintypes.LPCWSTR),
            ("hkeyClass", wintypes.HKEY),
            ("dwHotKey", wintypes.DWORD),
            ("hIconOrMonitor", wintypes.HANDLE),
            ("hProcess", wintypes.HANDLE),
        ]

    see_mask_nocloseprocess, see_mask_noasync = 0x00000040, 0x00000100
    sw_hide = 0
    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32 = ctypes.WinDLL("user32")
    ole32 = ctypes.WinDLL("ole32")
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wintypes.HWND
    shell32.ShellExecuteExW.argtypes = [ctypes.POINTER(SHELLEXECUTEINFOW)]
    shell32.ShellExecuteExW.restype = wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    # ShellExecuteEx wants COM initialised on the calling thread (STA, no OLE1 DDE).
    hr = ole32.CoInitializeEx(None, 0x2 | 0x4)
    try:
        info = SHELLEXECUTEINFOW()
        info.cbSize = ctypes.sizeof(info)
        info.fMask = see_mask_nocloseprocess | see_mask_noasync
        # Owner window = the window the user just clicked in: without one the UAC prompt
        # (requested from a worker thread) often only flashes in the taskbar.
        info.hwnd = user32.GetForegroundWindow()
        info.lpVerb = "runas"
        info.lpFile = file
        info.lpParameters = params
        info.nShow = sw_hide
        if not shell32.ShellExecuteExW(ctypes.byref(info)):
            err = ctypes.get_last_error()
            if err == _ERROR_CANCELLED:
                log.info("The administrator prompt was declined; firewall rule not added")
            else:
                log.warning("Could not start elevated command (Windows error %d)", err)
            return None
        if not info.hProcess:
            return None
        try:
            kernel32.WaitForSingleObject(info.hProcess, int(timeout_s * 1000))
            code = wintypes.DWORD(1)
            if not kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(code)):
                return None
            return int(code.value)
        finally:
            kernel32.CloseHandle(info.hProcess)
    finally:
        if hr in (0, 1):  # S_OK / S_FALSE -> balanced uninit
            ole32.CoUninitialize()


def add_firewall_rule(ports: list[int]) -> bool:
    """Allow inbound TCP ``ports`` on the Private/Domain profiles (Windows only).

    Only the HTTPS viewer port needs this: the HTTP listener is bound to 127.0.0.1, and
    loopback traffic is never filtered.  In the installed app the rule is limited to the
    app's executables.  Shows one UAC prompt (``netsh advfirewall`` needs administrator rights) and waits for
    it.  Returns True when the rule was added; False when declined, failed or not on
    Windows.  Never raises.
    """
    try:
        clean = sorted({int(p) for p in ports if 0 < int(p) < 65536})
    except (TypeError, ValueError):
        log.warning("Invalid firewall ports: %r", ports)
        return False
    if not clean:
        log.warning("No valid ports given for the firewall rule")
        return False
    if sys.platform != "win32":
        log.info("Firewall rules are only managed on Windows")
        return False
    root = os.environ.get("SystemRoot") or os.environ.get("windir") or r"C:\Windows"
    cmd = str(Path(root) / "System32" / "cmd.exe")
    try:
        code = _shell_execute_runas(cmd, _firewall_command(clean))
    except Exception as exc:  # ctypes / COM failures must never crash the GUI
        log.warning("Adding the firewall rule failed: %s", exc)
        return False
    if code is None:
        return False
    if code != 0:
        log.warning("netsh could not add the firewall rule (exit code %d)", code)
        return False
    log.info("Added Windows Firewall rule %r for TCP %s", FIREWALL_RULE_NAME, clean)
    return True
