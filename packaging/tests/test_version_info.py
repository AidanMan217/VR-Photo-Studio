"""Tests for packaging/version_info.py (Windows version resource)."""

from __future__ import annotations

import importlib
import sys
import types

import pytest
import spec_helpers as H
import version_info as V

pytest.importorskip("PyInstaller")


def _import_versioninfo():
    """PyInstaller's version-resource module imports ``pefile`` and ``compat.win32api``
    (Windows only); placeholders are enough for parsing the text form on other platforms."""
    import PyInstaller.compat as compat

    added_pefile = added_win32api = False
    try:
        import pefile  # noqa: F401
    except ImportError:
        sys.modules["pefile"] = types.ModuleType("pefile")
        added_pefile = True
    if not hasattr(compat, "win32api"):
        compat.win32api = types.ModuleType("win32api")
        added_win32api = True
    try:
        return importlib.import_module("PyInstaller.utils.win32.versioninfo")
    finally:
        if added_pefile:
            sys.modules.pop("pefile", None)
        if added_win32api:
            del compat.win32api


versioninfo = _import_versioninfo()


def _load(path):
    return versioninfo.load_version_info_from_text_file(str(path))


def _strings(info) -> dict[str, str]:
    out: dict[str, str] = {}
    for kid in info.kids:
        if isinstance(kid, versioninfo.StringFileInfo):
            for table in kid.kids:
                for s in table.kids:
                    out[s.name] = s.val
    return out


def test_render_is_loadable_by_pyinstaller(tmp_path):
    p = V.write_version_file(tmp_path / "v.txt", "1.2.3", exe_name="vrphoto-cli.exe", description="CLI")
    info = _load(p)
    assert isinstance(info, versioninfo.VSVersionInfo)
    assert info.ffi.fileVersionMS == (1 << 16) | 2
    assert info.ffi.fileVersionLS == (3 << 16) | 0
    assert info.ffi.productVersionMS == info.ffi.fileVersionMS
    assert info.ffi.fileFlags == 0
    s = _strings(info)
    assert s["OriginalFilename"] == "vrphoto-cli.exe"
    assert s["InternalName"] == "vrphoto-cli"
    assert s["FileDescription"] == "CLI"
    assert s["ProductName"] == H.APP_NAME
    assert s["CompanyName"] == H.PUBLISHER
    assert s["FileVersion"] == "1.2.3"
    assert "MIT" in s["LegalCopyright"]
    raw = info.toRaw()  # the binary resource PyInstaller embeds into the .exe
    assert isinstance(raw, bytes) and len(raw) > 200


def test_prerelease_flag(tmp_path):
    info = _load(V.write_version_file(tmp_path / "v.txt", "0.2.0rc1"))
    assert info.ffi.fileFlags == 0x2
    assert info.ffi.fileVersionMS == 2  # 0.2
    assert _strings(info)["ProductVersion"] == "0.2.0rc1"


def test_quotes_are_escaped(tmp_path):
    p = V.write_version_file(tmp_path / "v.txt", "1.0.0", description='It\'s "quoted" \\ text')
    assert _strings(_load(p))["FileDescription"] == 'It\'s "quoted" \\ text'


def test_default_version_comes_from_package(tmp_path):
    import vrphoto

    info = _load(V.write_version_file(tmp_path / "sub" / "v.txt"))
    assert _strings(info)["FileVersion"] == vrphoto.__version__


def test_copyright_year_range():
    assert V.version_strings("1.0", exe_name="a.exe", description="d", year=2026)[
        "LegalCopyright"
    ].startswith("Copyright (c) 2026 ")
    assert (
        "2026-2028"
        in V.version_strings("1.0", exe_name="a.exe", description="d", year=2028)["LegalCopyright"]
    )


def test_main(tmp_path, capsys):
    out = tmp_path / "ver.txt"
    assert V.main([str(out), "--exe", "X.exe", "--description", "Desc", "--version", "3.4.5"]) == 0
    assert "wrote" in capsys.readouterr().out
    s = _strings(_load(out))
    assert s["OriginalFilename"] == "X.exe" and s["FileVersion"] == "3.4.5"
