# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 6.x spec for VR Photo Studio (onedir).
#
#   pyinstaller packaging/vrphoto.spec --noconfirm [--distpath DIST] [--workpath BUILD]
#
# Produces ONE folder "VRPhotoStudio" containing TWO executables that share the same
# _internal/ runtime:
#   VRPhotoStudio.exe  - windowed GUI   (packaging/entry_gui.py -> vrphoto.__main__.main)
#   vrphoto-cli.exe    - console CLI    (packaging/entry_cli.py -> vrphoto.__main__.cli_entry)
# The logic deciding what goes in lives in packaging/spec_helpers.py (unit-tested).
# Builds on Windows (release) and Linux (CI smoke tests); Windows-only bits are guarded.
# No UPX: it breaks Qt/onnxruntime DLLs and triggers antivirus false positives.

import os
import sys
from pathlib import Path

HERE = Path(SPECPATH).resolve()  # noqa: F821 - provided by PyInstaller
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import make_icon  # noqa: E402
import spec_helpers as H  # noqa: E402
import version_info as V  # noqa: E402

ROOT = HERE.parent
SRC = ROOT / "src"
# Analyse the source tree we are building from, never a stale installed copy.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PyInstaller.utils.hooks import collect_submodules  # noqa: E402

IS_WIN = sys.platform == "win32"
VERSION = H.checked_version(ROOT)  # __version__, verified against pyproject.toml
WORK = Path(workpath)  # noqa: F821 - provided by PyInstaller
DIST = Path(DISTPATH)  # noqa: F821 - provided by PyInstaller

print(f"[vrphoto.spec] building VR Photo Studio {VERSION} for {sys.platform}")

# ---------------------------------------------------------------- inputs
hiddenimports = sorted(
    set(collect_submodules("vrphoto", filter=H.want_vrphoto_module)) | set(H.EXPLICIT_HIDDEN_IMPORTS)
)
datas = H.package_datas(SRC / "vrphoto") + H.license_datas()  # -> _internal/licenses/
binaries = [H.uv_binary_entry()]  # -> _internal/bin/uv(.exe)
excludes = H.all_excludes()

icon_path = HERE / "icon.ico"
if not icon_path.is_file():
    make_icon.write_ico(icon_path)

gui_version_file = cli_version_file = None
if IS_WIN:
    gui_version_file = V.write_version_file(
        WORK / "version_gui.txt", VERSION, exe_name=f"{H.GUI_EXE}.exe", description=H.APP_NAME
    )
    cli_version_file = V.write_version_file(
        WORK / "version_cli.txt",
        VERSION,
        exe_name=f"{H.CLI_EXE}.exe",
        description=f"{H.APP_NAME} (command line)",
    )

analysis_kwargs = dict(
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(HERE / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,  # keep asserts/docstrings: some libraries rely on docstrings
)


def _analysis(script):
    a = Analysis([str(HERE / script)], **analysis_kwargs)  # noqa: F821 - provided by PyInstaller
    keep_qml = H.uses_qml(a.binaries)  # QML runtime only if the GUI itself uses QtQuick/QtQml
    a.datas = H.filter_toc(a.datas, keep_qml=keep_qml)
    a.binaries = H.filter_toc(a.binaries, keep_qml=keep_qml)
    a.datas, a.binaries = H.promote_executables(a.datas, a.binaries)
    return a


a_gui = _analysis("entry_gui.py")
a_cli = _analysis("entry_cli.py")

exe_common = dict(
    exclude_binaries=True,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory="_internal",
    icon=[str(icon_path)] if IS_WIN else None,
)

exe_gui = EXE(  # noqa: F821
    PYZ(a_gui.pure),  # noqa: F821
    a_gui.scripts,
    [],
    name=H.GUI_EXE,
    console=False,
    disable_windowed_traceback=False,  # show a crash dialog instead of failing silently
    version=str(gui_version_file) if gui_version_file else None,
    **exe_common,
)

exe_cli = EXE(  # noqa: F821
    PYZ(a_cli.pure),  # noqa: F821
    a_cli.scripts,
    [],
    name=H.CLI_EXE,
    console=True,
    version=str(cli_version_file) if cli_version_file else None,
    **exe_common,
)

coll = COLLECT(  # noqa: F821
    exe_gui,
    a_gui.binaries,
    a_gui.datas,
    exe_cli,
    a_cli.binaries,
    a_cli.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=H.COLLECT_NAME,
)

# ---------------------------------------------------------------- extras next to the exes
_dist_dir = DIST / H.COLLECT_NAME
for _p in H.write_dist_extras(_dist_dir, root=ROOT, version=VERSION):
    print(f"[vrphoto.spec] wrote {_p.relative_to(DIST)}")

_problems = H.check_bundle(_dist_dir, pkg_dir=SRC / "vrphoto")
for _msg in _problems:
    print(f"[vrphoto.spec] WARNING: {_msg}")
if _problems and os.environ.get("VRPHOTO_STRICT_BUNDLE") == "1":
    raise SystemExit(f"Bundle check failed with {len(_problems)} problem(s); see warnings above.")
print(f"[vrphoto.spec] bundle: {_dist_dir} ({H.bundle_size_mb(_dist_dir):.0f} MB)")
