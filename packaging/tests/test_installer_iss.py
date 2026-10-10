"""Static checks of packaging/installer.iss and its consistency with the app and CI.

Inno Setup only runs on Windows, so these tests guard the script's key settings on every
platform; the real compile + silent install happens in .github/workflows/build.yml.
"""

from __future__ import annotations

import re
import sys

import pytest
import spec_helpers as H


@pytest.fixture(scope="module")
def iss(packaging_dir_module) -> str:
    return (packaging_dir_module / "installer.iss").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def packaging_dir_module():
    from pathlib import Path

    return Path(__file__).resolve().parents[1]


def _sections(text: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    cur = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(";") or line.startswith("#"):
            continue
        m = re.fullmatch(r"\[(\w+)\]", line)
        if m:
            cur = m.group(1)
            out.setdefault(cur, [])
        elif cur:
            out[cur].append(line)
    return out


def _setup(text: str) -> dict[str, str]:
    d = {}
    for line in _sections(text)["Setup"]:
        k, _, v = line.partition("=")
        d[k.strip()] = v.strip()
    return d


def _define(text: str, name: str) -> str:
    m = re.search(rf'^\s*#define\s+{name}\s+"([^"]*)"', text, re.M)
    assert m, f"#define {name} missing"
    return m.group(1)


def test_known_sections(iss):
    assert set(_sections(iss)) <= {
        "Setup",
        "Languages",
        "CustomMessages",
        "Tasks",
        "InstallDelete",
        "Files",
        "Icons",
        "Run",
        "UninstallDelete",
        "UninstallRun",
        "Code",
        "Registry",
        "Dirs",
    }


def test_core_setup_directives(iss):
    s = _setup(iss)
    assert s["PrivilegesRequired"] == "lowest"
    assert s["PrivilegesRequiredOverridesAllowed"] == "dialog"
    assert s["DefaultDirName"] == r"{autopf}\{#AppName}"
    assert s["OutputBaseFilename"] == "VRPhotoStudio-Setup-{#AppVersion}"
    assert s["Compression"].startswith("lzma2")
    assert s["SolidCompression"] == "yes"
    assert s["WizardStyle"].split()[0] == "modern"
    assert s["LicenseFile"].endswith(r"\LICENSE.txt")
    assert s["ArchitecturesAllowed"] == "x64compatible"
    assert s["SetupIconFile"] == "icon.ico"
    assert re.fullmatch(r"\{\{[0-9A-F]{8}(-[0-9A-F]{4}){3}-[0-9A-F]{12}\}", s["AppId"])


def test_version_comes_from_define(iss):
    assert re.search(r"^#ifndef AppVersion\s*$", iss, re.M)
    assert "AppVersion={#AppVersion}" in iss


def test_names_match_bundle(iss):
    assert _define(iss, "AppExe") == f"{H.GUI_EXE}.exe"
    assert _define(iss, "CliExe") == f"{H.CLI_EXE}.exe"
    assert _define(iss, "AppName") == H.APP_NAME
    assert _define(iss, "BundleDir").endswith(H.COLLECT_NAME)


def test_ports_match_settings(iss):
    """Only the Wi-Fi (https) port is opened: the http listener is localhost-only (SEC-3)."""
    from vrphoto.core.config import Settings

    s = Settings()
    ports = {int(p) for p in _define(iss, "ViewerPorts").split(",")}
    assert ports == {s.https_port}
    assert s.http_port not in ports


def test_data_dir_matches_app(iss, tmp_path, monkeypatch):
    import vrphoto
    from vrphoto.core import config

    monkeypatch.delenv("VRPHOTO_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(sys, "platform", "win32")
    assert config.data_root().name == _define(iss, "DataDirName") == vrphoto.APP_ID


def test_firewall_task_only_when_elevated(iss):
    tasks = _sections(iss)["Tasks"]
    fw = [t for t in tasks if t.startswith('Name: "firewall"')]
    assert len(fw) == 1 and "Check: IsAdmin" in fw[0]
    desktop = [t for t in tasks if t.startswith('Name: "desktopicon"')]
    assert desktop and "Flags: unchecked" in desktop[0]
    run = "\n".join(_sections(iss)["Run"])
    assert "advfirewall firewall add rule" in run
    assert "profile=private,domain" in run
    assert "localport={#ViewerPorts}" in run
    assert "dir=in" in run and "protocol=TCP" in run


def test_shortcuts(iss):
    icons = _sections(iss)["Icons"]
    assert any(i.startswith(r'Name: "{autoprograms}\{#AppName}"') for i in icons)
    assert any(i.startswith(r'Name: "{autodesktop}\{#AppName}"') and "Tasks: desktopicon" in i for i in icons)


def test_upgrade_cleans_old_runtime(iss):
    assert any(r"{app}\_internal" in line for line in _sections(iss)["InstallDelete"])


def test_uninstall_asks_before_deleting_data(iss):
    code = "\n".join(_sections(iss)["Code"])
    assert "usPostUninstall" in code
    assert "SuppressibleMsgBox" in code and "IDNO" in code and "MB_DEFBUTTON2" in code
    assert "{localappdata}" in code and "DelTree" in code
    assert "delete rule" in code  # firewall rule removed on uninstall


def test_build_workflow_uses_same_appid_and_names(iss, root):
    wf = (root / ".github" / "workflows" / "build.yml").read_text(encoding="utf-8")
    guid = _setup(iss)["AppId"][2:-1]
    assert f"{{{guid}}}_is1" in wf
    assert "VRPhotoStudio-Setup-$v.exe" in wf
    for flag in ("/VERYSILENT", "/CURRENTUSER", "/SUPPRESSMSGBOXES", "/DIR="):
        assert flag in wf
    assert "/DAppVersion=" in wf and "/DBundleDir=" in wf


def test_build_script_passes_defines(root):
    ps1 = (root / "packaging" / "build_windows.ps1").read_text(encoding="utf-8")
    for token in (
        "/DAppVersion=",
        "/DAppVersionNumeric=",
        "/DBundleDir=",
        "installer.iss",
        "vrphoto.spec",
        "--selftest",
        "make_icon.py",
        "Inno Setup 6",
    ):
        assert token in ps1


def test_firewall_rule_name_matches_app(iss):
    net = pytest.importorskip("vrphoto.webxr.net")
    name = getattr(net, "FIREWALL_RULE_NAME", None)
    if name is None:
        pytest.skip("webxr.net has no FIREWALL_RULE_NAME")
    assert _define(iss, "FirewallRuleName") == name


def test_requires_inno_setup_6_3(iss):
    """x64compatible and BOM-less UTF-8 licence files need Inno Setup 6.3+; fail with a clear message."""
    m = re.search(r"^#if VER < EncodeVer\((\d+), (\d+), (\d+)\)\s*\n\s*#error (.+)$", iss, re.M)
    assert m, "missing Inno Setup version guard"
    assert tuple(int(g) for g in m.groups()[:3]) >= (6, 3, 0)
    assert "jrsoftware.org" in m.group(4)


def test_firewall_task_removes_stale_program_block_rules(iss):
    """Windows' own per-program block rules (prompt dismissed) beat our allow rule - Setup and the
    uninstaller remove them, like webxr.net.add_firewall_rule does."""
    run = _sections(iss)["Run"]
    delete_prog = [r for r in run if "delete rule name=all dir=in program=" in r]
    # one for the GUI and one for the CLI executable (both get a program-scoped allow rule)
    assert len(delete_prog) == 2 and all("Tasks: firewall" in r for r in delete_prog)
    assert any(r"{app}\{#AppExe}" in r for r in delete_prog)
    assert any(r"{app}\{#CliExe}" in r for r in delete_prog)
    add = next(i for i, r in enumerate(run) if "firewall add rule" in r)
    assert all(run.index(r) < add for r in delete_prog), "stale rules must be removed before allow rules"
    adds = [r for r in run if "firewall add rule" in r]
    assert len(adds) == 2 and all("program=" in r and "localport={#ViewerPorts}" in r for r in adds)
    code = "\n".join(_sections(iss)["Code"])
    assert code.count("delete rule name=all program=") == 2  # GUI and CLI executables
    assert "{#AppExe}" in code and "{#CliExe}" in code
