"""Tests for vrphoto.webxr.net (addresses, QR codes, firewall helpers)."""

from __future__ import annotations

import io
import ipaddress
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from vrphoto.core.errors import InputError
from vrphoto.webxr import net


# --------------------------------------------------------------------------- addresses
def _fake_env(monkeypatch, *, primary, hostname_ips, command_ips=()):
    monkeypatch.setattr(net, "_udp_primary", lambda: primary)
    monkeypatch.setattr(net, "_hostname_ips", lambda: list(hostname_ips))
    calls = []

    def cmd():
        calls.append(1)
        return list(command_ips)

    monkeypatch.setattr(net, "_command_ips", cmd)
    return calls


def test_lan_ips_filters_and_orders(monkeypatch):
    calls = _fake_env(
        monkeypatch,
        primary="10.0.0.5",
        hostname_ips=["127.0.1.1", "169.254.10.2", "172.18.0.1", "192.168.1.20", "10.0.0.5",
                      "192.168.1.20", "8.8.4.4", "224.0.0.1", "0.0.0.0"],
    )
    ips = net.lan_ips()
    assert ips == ["10.0.0.5", "192.168.1.20", "172.18.0.1", "8.8.4.4"]
    assert not calls  # OS tool output is parsed only when needed


def test_lan_ips_falls_back_to_os_tools(monkeypatch):
    calls = _fake_env(monkeypatch, primary=None, hostname_ips=["127.0.0.1"],
                      command_ips=["127.0.0.1", "192.168.0.9", "169.254.1.1"])
    assert net.lan_ips() == ["192.168.0.9"]
    assert calls


def test_lan_ips_empty_when_offline(monkeypatch):
    _fake_env(monkeypatch, primary=None, hostname_ips=[], command_ips=[])
    assert net.lan_ips() == []
    assert net.primary_ip() == "127.0.0.1"


def test_primary_ip_prefers_route(monkeypatch):
    _fake_env(monkeypatch, primary="192.168.7.7", hostname_ips=["10.1.1.1"])
    assert net.primary_ip() == "192.168.7.7"
    _fake_env(monkeypatch, primary=None, hostname_ips=["10.1.1.1"])
    assert net.primary_ip() == "10.1.1.1"


def test_real_lan_ips_are_valid():
    for ip in net.lan_ips():
        a = ipaddress.IPv4Address(ip)
        assert not a.is_loopback and not a.is_link_local
    ip = net.primary_ip()
    ipaddress.IPv4Address(ip)


def test_ipconfig_and_ip_parsers(monkeypatch):
    ipconfig = (
        "Ethernet adapter Ethernet:\r\n"
        "   IPv4 Address. . . . . . . . . . . : 192.168.1.23(Preferred)\r\n"
        "   Autoconfiguration IPv4 Address. . : 169.254.3.4\r\n"
        "Drahtlos-LAN-Adapter WLAN:\r\n"
        "   IPv4-Adresse  . . . . . . . . . . : 10.0.0.8\r\n"
    )
    assert net._IPCONFIG_RE.findall(ipconfig) == ["192.168.1.23", "169.254.3.4", "10.0.0.8"]
    ip_out = (
        "1: lo    inet 127.0.0.1/8 scope host lo\\       valid_lft forever\n"
        "2: eth0    inet 192.168.5.4/24 brd 192.168.5.255 scope global eth0\n"
    )
    assert net._INET_RE.findall(ip_out) == ["127.0.0.1", "192.168.5.4"]


def test_command_ips_windows_branch(monkeypatch):
    monkeypatch.setattr(net.sys, "platform", "win32")
    out = "   IPv4 Address. . . . . . . . . . . : 192.168.1.23\r\n"
    monkeypatch.setattr(net, "_run", lambda args, timeout=10: subprocess.CompletedProcess(args, 0, out, ""))
    assert net._command_ips() == ["192.168.1.23"]


def test_run_returns_none_for_missing_program():
    assert net._run(["definitely-not-a-real-program-xyz"]) is None


# --------------------------------------------------------------------------- QR
def test_qr_png_is_png_and_decodes():
    url = "https://192.168.1.5:8443/"
    data = net.qr_png(url, box_size=6)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    img = Image.open(io.BytesIO(data))
    assert img.format == "PNG"
    w, h = img.size
    assert w == h and w % 6 == 0 and w >= 21 * 6
    import cv2

    arr = np.array(img.convert("L"))
    text, points, _ = cv2.QRCodeDetector().detectAndDecode(arr)
    if points is None:  # pragma: no cover - very old OpenCV builds
        pytest.skip("OpenCV QR detector unavailable")
    assert text == url


def test_qr_png_box_size_scales():
    small = Image.open(io.BytesIO(net.qr_png("hello", box_size=2))).size[0]
    big = Image.open(io.BytesIO(net.qr_png("hello", box_size=8))).size[0]
    assert big == small * 4


def test_qr_png_errors():
    with pytest.raises(InputError):
        net.qr_png("x", box_size=0)
    with pytest.raises(InputError):
        net.qr_png("x" * 5000)


# --------------------------------------------------------------------------- firewall
def test_firewall_unknown_off_windows(monkeypatch):
    monkeypatch.setattr(net.sys, "platform", "linux")
    assert net.firewall_status(8443) == "unknown"
    rep = net.firewall_report(8443)
    assert rep["status"] == "unknown" and rep["reason"]


def test_add_firewall_rule_false_off_windows(monkeypatch):
    monkeypatch.setattr(net.sys, "platform", "linux")
    assert net.add_firewall_rule([8443, 8080]) is False


@pytest.mark.parametrize("bad", [[], [0], [70000], ["abc"], None])
def test_add_firewall_rule_rejects_bad_ports(bad):
    assert net.add_firewall_rule(bad) is False


@pytest.mark.parametrize(
    "line,status,profile",
    [
        ("RESULT|ok|private|allow-rule", "ok", "private"),
        ("RESULT|blocked|public|no-rule", "blocked", "public"),
        ("RESULT|ok|domain,private|firewall-off", "ok", "domain,private"),
        ("RESULT|weird|x|y", "unknown", "x"),
    ],
)
def test_firewall_report_parses_script_output(monkeypatch, line, status, profile):
    monkeypatch.setattr(net.sys, "platform", "win32")
    seen = {}

    def fake_ps(script, timeout=30.0):
        seen["script"] = script
        return subprocess.CompletedProcess(["ps"], 0, "noise\n" + line + "\n", "")

    monkeypatch.setattr(net, "_run_powershell", fake_ps)
    rep = net.firewall_report(8443)
    assert rep["status"] == status and rep["profile"] == profile
    assert "$port = 8443" in seen["script"]
    if profile == "public" and status == "blocked":
        assert "Private" in rep["reason"]
    assert net.firewall_status(8443) == status


def test_firewall_report_handles_failure(monkeypatch):
    monkeypatch.setattr(net.sys, "platform", "win32")
    monkeypatch.setattr(net, "_run_powershell", lambda script, timeout=30.0: None)
    assert net.firewall_status(8443) == "unknown"
    monkeypatch.setattr(
        net, "_run_powershell", lambda script, timeout=30.0: subprocess.CompletedProcess(["ps"], 1, "", "boom")
    )
    assert net.firewall_status(8443) == "unknown"


def test_firewall_script_escapes_program_path(monkeypatch):
    monkeypatch.setattr(net.sys, "platform", "win32")
    monkeypatch.setattr(net, "_program_path", lambda: r"C:\Users\O'Brien\VR Photo Studio\vrphoto.exe")
    seen = {}

    def fake_ps(script, timeout=30.0):
        seen["script"] = script
        return subprocess.CompletedProcess(["ps"], 0, "RESULT|ok|private|allow-rule\n", "")

    monkeypatch.setattr(net, "_run_powershell", fake_ps)
    net.firewall_status(8080)
    assert r"$exe = 'C:\Users\O''Brien\VR Photo Studio\vrphoto.exe'" in seen["script"]


def test_add_firewall_rule_windows_success_and_command(monkeypatch):
    monkeypatch.setattr(net.sys, "platform", "win32")
    calls = []

    def fake_runas(file, params, timeout_s=120.0):
        calls.append((file, params))
        return 0

    monkeypatch.setattr(net, "_shell_execute_runas", fake_runas)
    assert net.add_firewall_rule([8443, 8080, 8443]) is True
    file, params = calls[0]
    assert file.lower().endswith("cmd.exe")
    assert params.startswith("/c ")
    assert ('netsh advfirewall firewall add rule name="VR Photo Studio" dir=in action=allow '
            "protocol=TCP localport=8080,8443 profile=private,domain") in params
    assert 'delete rule name="VR Photo Studio"' in params
    assert "program=" not in params  # not frozen: never touch python.exe rules


def test_firewall_command_removes_stale_program_rules_when_frozen(monkeypatch):
    monkeypatch.setattr(net.sys, "frozen", True, raising=False)
    monkeypatch.setattr(net, "_program_path", lambda: r"C:\Program Files\VR Photo Studio\VRPhotoStudio.exe")
    cmd = net._firewall_command([8443])
    assert r'program="C:\Program Files\VR Photo Studio\VRPhotoStudio.exe"' in cmd
    assert cmd.index("delete rule name=all") < cmd.index("add rule")


def test_frozen_firewall_rule_is_limited_to_the_app_executables(monkeypatch, tmp_path):
    """SEC-3: a port-wide rule would expose any other program listening on that port."""
    (tmp_path / "VRPhotoStudio.exe").write_bytes(b"MZ")
    (tmp_path / "vrphoto-cli.exe").write_bytes(b"MZ")
    gui, cli = str(tmp_path / "VRPhotoStudio.exe"), str(tmp_path / "vrphoto-cli.exe")
    monkeypatch.setattr(net.sys, "frozen", True, raising=False)
    monkeypatch.setattr(net, "_program_path", lambda: gui)
    cmd = net._firewall_command([8443])
    adds = [part for part in cmd.split("&") if "add rule" in part]
    assert len(adds) == 2
    assert all("localport=8443 " in a and "action=allow" in a for a in adds)
    assert f'program="{gui}"' in adds[0] and f'program="{cli}"' in adds[1]
    assert f'delete rule name=all dir=in program="{cli}"' in cmd
    assert " && " in cmd  # a failed first add makes the command fail
    # only the running exe when the other one is not installed
    (tmp_path / "vrphoto-cli.exe").unlink()
    cmd = net._firewall_command([8443])
    assert [p for p in cmd.split("&") if "add rule" in p][0].strip().endswith(f'program="{gui}"')
    assert cli not in cmd


@pytest.mark.parametrize("result", [None, 1, "raise"])
def test_add_firewall_rule_windows_failures_never_raise(monkeypatch, result):
    monkeypatch.setattr(net.sys, "platform", "win32")

    def fake_runas(file, params, timeout_s=120.0):
        if result == "raise":
            raise OSError("ctypes exploded")
        return result

    monkeypatch.setattr(net, "_shell_execute_runas", fake_runas)
    assert net.add_firewall_rule([8443]) is False


def test_ps_quote_and_temp_script(tmp_path):
    assert net._ps_quote("it's") == "'it''s'"
    p = net._write_temp_script("Write-Output 'ü'\n", "vrphoto_test_")
    try:
        raw = p.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf")  # BOM for Windows PowerShell 5.1
        assert "ü".encode() in raw
    finally:
        p.unlink()


def test_no_window_kwargs():
    kw = net._no_window_kwargs()
    if sys.platform == "win32":
        assert kw["creationflags"] & 0x08000000
    else:
        assert kw == {}


# --------------------------------------------------------------------------- review additions
def test_ps_quote_escapes_typographic_single_quotes():
    """PowerShell also ends single-quoted strings at U+2018..U+201B (common in file names
    such as "Mom\u2019s birthday.jpg"); unescaped they broke the script or injected code."""
    q = net._ps_quote("Mom\u2019s \u2018best\u2019 \u201aday\u201b $(calc) 'x'\x00")
    assert q == "'Mom\u2019\u2019s \u2018\u2018best\u2019\u2019 \u201a\u201aday\u201b\u201b $(calc) ''x'''"


def test_udp_primary_never_targets_broadcast(monkeypatch):
    targets = []

    class FakeSock:
        def __init__(self, *a):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def connect(self, addr):
            targets.append(addr[0])
            raise OSError("unreachable")

    monkeypatch.setattr(net.socket, "socket", FakeSock)
    assert net._udp_primary() is None
    assert targets and not any(t.endswith(".255") for t in targets)


def _find_powershell() -> str | None:
    import os
    import shutil

    return os.environ.get("VRPHOTO_PWSH") or shutil.which("pwsh") or (
        shutil.which("powershell") if sys.platform == "win32" else None)


_FW_MOCK = r"""
function MkRule($dir, $prof, $proto, $ports, $app, $action) {
  [pscustomobject]@{ Enabled = $true; Direction = $dir; Profiles = $prof; Protocol = $proto;
                     LocalPorts = $ports; ApplicationName = $app; Action = $action }
}
$rules = @()
switch ($env:VRPHOTO_FW_CASE) {
  'allow' { $rules = @( (MkRule 1 6 6 '80,8000-9000' '' 1), (MkRule 1 2 17 '8443' '' 0) ) }
  'block' { $rules = @( (MkRule 1 2 6 '8443' '' 1), (MkRule 1 2 256 '*' $env:VRPHOTO_FW_APP 0) ) }
  'none'  { $rules = @( (MkRule 1 2 6 'RPC' '' 1), (MkRule 2 2 6 '8443' '' 1) ) }
}
$global:FwMock = [pscustomobject]@{ CurrentProfileTypes = [int]$env:VRPHOTO_FW_PROFILE; Rules = $rules }
$global:FwMock | Add-Member ScriptMethod FirewallEnabled { param($p) $true }
$global:FwMock | Add-Member ScriptMethod BlockAllInboundTraffic { param($p) $false }
$global:FwMock | Add-Member ScriptMethod DefaultInboundAction { param($p) 0 }
"""


@pytest.mark.parametrize(
    "case,profile,expected",
    [("allow", 2, "RESULT|ok|private|allow-rule"), ("block", 2, "RESULT|blocked|private|block-rule"),
     ("none", 2, "RESULT|blocked|private|no-rule"), ("allow", 4, "RESULT|ok|public|allow-rule"),
     ("allow", 1, "RESULT|blocked|domain|no-rule"), ("block", 4, "RESULT|blocked|public|no-rule")],
)
def test_firewall_script_logic_in_real_powershell(tmp_path, monkeypatch, case, profile, expected):
    """Runs the generated firewall script in PowerShell with a mocked HNetCfg.FwPolicy2
    (skipped when no PowerShell is installed; set VRPHOTO_PWSH to a pwsh binary)."""
    ps = _find_powershell()
    if not ps:
        pytest.skip("PowerShell not available")
    exe = "C:\\Users\\Zo\u00eb\u2019s \u2018VR\u2019\\VR Photo Studio\\App.exe"
    script = net._FW_SCRIPT.replace("__PORT__", "8443").replace("__EXE__", net._ps_quote(exe))
    script = script.replace("New-Object -ComObject HNetCfg.FwPolicy2", "$global:FwMock")
    path = tmp_path / "fw.ps1"
    path.write_text(_FW_MOCK + script, encoding="utf-8-sig")
    env = dict(__import__("os").environ, VRPHOTO_FW_CASE=case, VRPHOTO_FW_PROFILE=str(profile),
               VRPHOTO_FW_APP=exe.upper())  # rule paths compare case-insensitively
    cp = subprocess.run([ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(path)],
                        capture_output=True, text=True, encoding="utf-8", timeout=60, env=env)
    assert expected in cp.stdout, (cp.stdout, cp.stderr)
