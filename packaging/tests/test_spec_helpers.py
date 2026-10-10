"""Tests for packaging/spec_helpers.py: what goes into (and stays out of) the Windows bundle."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import spec_helpers as H

# --------------------------------------------------------------------------- version


def test_read_version_matches_package():
    import vrphoto

    assert H.read_version() == vrphoto.__version__


def test_read_version_from_custom_package(tmp_path):
    (tmp_path / "__init__.py").write_text('"""doc"""\nAPP = 1\n__version__ = "2.5.1"\n', encoding="utf-8")
    assert H.read_version(tmp_path) == "2.5.1"


@pytest.mark.parametrize(
    "content", ["APP = 1\n", "__version__ = get()\n", "__version__ = 3\n"], ids=["missing", "call", "int"]
)
def test_read_version_errors(tmp_path, content):
    (tmp_path / "__init__.py").write_text(content, encoding="utf-8")
    with pytest.raises(H.BuildError, match="__version__"):
        H.read_version(tmp_path)


def test_read_version_missing_file(tmp_path):
    with pytest.raises(H.BuildError, match="Cannot read"):
        H.read_version(tmp_path / "nope")


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.1.0", (0, 1, 0, 0)),
        ("1.2.3.4", (1, 2, 3, 4)),
        ("1.2", (1, 2, 0, 0)),
        ("3", (3, 0, 0, 0)),
        ("1.2.3rc1", (1, 2, 3, 0)),
        ("0.1.0.dev5", (0, 1, 0, 0)),
        ("v2.0.1", (2, 0, 1, 0)),
        ("1.2.3.4.5", (1, 2, 3, 4)),
    ],
)
def test_version_tuple(version, expected):
    assert H.version_tuple(version) == expected


@pytest.mark.parametrize("bad", ["", "abc", "dev", "70000.1"])
def test_version_tuple_rejects(bad):
    with pytest.raises(H.BuildError):
        H.version_tuple(bad)


# --------------------------------------------------------------------------- data files


def _touch(p: Path, text: str = "x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def fake_pkg(tmp_path) -> Path:
    pkg = tmp_path / "vrphoto"
    for rel in [
        "__init__.py",
        "core/__init__.py",
        "core/config.py",
        "webxr/viewer/index.html",
        "webxr/viewer/scene.schema.json",
        "webxr/viewer/js/app.js",
        "webxr/viewer/vendor/three/build/three.module.js",
        "webxr/viewer/vendor/three/LICENSE",
        "webxr/viewer/vendor/profiles/left.glb",
        "addons/__init__.py",
        "addons/protocol.py",
        "addons/manager.py",
        "addons/manifests/trellis.toml",
        "addons/workers/__init__.py",
        "addons/workers/trellis_worker.py",
        "addons/workers/common/util.py",
        "third_party/__init__.py",
        "third_party/spatialmedia/__init__.py",
        "third_party/spatialmedia/LICENSE",
        "models/models.lock.json",
        "gui/resources/logo.png",
        # junk that must never be bundled
        "core/__pycache__/config.cpython-312.pyc",
        "addons/workers/__pycache__/trellis_worker.cpython-312.pyc",
        "webxr/viewer/node_modules/foo/index.js",
        "webxr/viewer/.DS_Store",
        "gui/resources/Thumbs.db",
        "models/partial.onnx.part",
        "core/stray.pyc",
    ]:
        _touch(pkg / rel)
    return pkg


def _dests(datas) -> set[str]:
    return {f"{dest}/{Path(src).name}" for src, dest in datas}


def test_package_datas_includes_assets_and_sources(fake_pkg):
    got = _dests(H.package_datas(fake_pkg))
    expected = {
        "vrphoto/webxr/viewer/index.html",
        "vrphoto/webxr/viewer/scene.schema.json",
        "vrphoto/webxr/viewer/js/app.js",
        "vrphoto/webxr/viewer/vendor/three/build/three.module.js",
        "vrphoto/webxr/viewer/vendor/three/LICENSE",
        "vrphoto/webxr/viewer/vendor/profiles/left.glb",
        "vrphoto/addons/protocol.py",
        "vrphoto/addons/manifests/trellis.toml",
        "vrphoto/addons/workers/__init__.py",
        "vrphoto/addons/workers/trellis_worker.py",
        "vrphoto/addons/workers/common/util.py",
        "vrphoto/third_party/__init__.py",
        "vrphoto/third_party/spatialmedia/__init__.py",
        "vrphoto/third_party/spatialmedia/LICENSE",
        "vrphoto/models/models.lock.json",
        "vrphoto/gui/resources/logo.png",
    }
    assert got == expected


def test_package_datas_excludes_junk_and_modules(fake_pkg):
    got = _dests(H.package_datas(fake_pkg))
    for bad in ("config.py", "manager.py", "__init__.py"):
        assert f"vrphoto/core/{bad}" not in got
    assert "vrphoto/addons/manager.py" not in got
    assert not any("__pycache__" in d or d.endswith((".pyc", ".part")) for d in got)
    assert not any("node_modules" in d or "DS_Store" in d or "Thumbs.db" in d for d in got)


def test_package_datas_dest_is_posix_and_rooted(fake_pkg):
    for src, dest in H.package_datas(fake_pkg):
        assert "\\" not in dest
        assert dest == "vrphoto" or dest.startswith("vrphoto/")
        assert Path(src).is_file()


def test_package_datas_custom_dest_root(fake_pkg):
    dests = {d for _, d in H.package_datas(fake_pkg, dest_root="pkg")}
    assert all(d.startswith("pkg") for d in dests)


def test_package_datas_rejects_non_package(tmp_path):
    with pytest.raises(H.BuildError, match="not the vrphoto package"):
        H.package_datas(tmp_path)


def test_package_datas_real_tree():
    got = _dests(H.package_datas())
    assert "vrphoto/webxr/viewer/scene.schema.json" in got
    assert "vrphoto/models/models.lock.json" in got
    assert "vrphoto/addons/protocol.py" in got
    allowed_py = ("vrphoto/addons/workers/", "vrphoto/third_party/", "vrphoto/addons/protocol.py")
    for d in got:
        if d.endswith(".py"):
            assert d.startswith(allowed_py), d


@pytest.mark.parametrize(
    ("name", "wanted"),
    [
        ("vrphoto", True),
        ("vrphoto.gui.app", True),
        ("vrphoto.addons.manager", True),
        ("vrphoto.addons.workersx", True),
        ("vrphoto.addons.workers", False),
        ("vrphoto.addons.workers.trellis_worker", False),
    ],
)
def test_want_vrphoto_module(name, wanted):
    assert H.want_vrphoto_module(name) is wanted


# --------------------------------------------------------------------------- licences


def test_license_texts_are_complete():
    for name, marker in [
        ("LGPL-3.0.txt", "GNU LESSER GENERAL PUBLIC LICENSE"),
        ("GPL-3.0.txt", "GNU GENERAL PUBLIC LICENSE"),
        ("GPL-2.0.txt", "GNU GENERAL PUBLIC LICENSE"),
        ("LGPL-2.1.txt", "GNU LESSER GENERAL PUBLIC LICENSE"),
        ("Apache-2.0.txt", "Apache License"),
    ]:
        text = (H.LICENSE_TEXTS_DIR / name).read_text(encoding="utf-8")
        assert marker in text[:400], name
        assert len(text) > 7000, name


def test_license_datas_real_environment():
    datas = H.license_datas()
    dests = {f"{d}/{Path(s).name}" for s, d in datas}
    assert "licenses/LGPL-3.0.txt" in dests and "licenses/GPL-3.0.txt" in dests
    # a few distributions that always ship licence files
    for dist in ("pillow", "cryptography", "uv", "imageio-ffmpeg"):
        assert any(d.startswith(f"licenses/{dist}/") for d in dests), dist
    assert len(dests) == len(datas), "duplicate destinations"
    for src, dest in datas:
        assert Path(src).is_file()
        assert dest.startswith("licenses") and "\\" not in dest and ".." not in dest.split("/")


def test_license_datas_unknown_distribution_ignored():
    datas = H.license_datas(["definitely-not-installed-xyz"])
    assert all(d == "licenses" for _, d in datas)  # only the bundled GPL/LGPL texts


def test_license_datas_custom_root():
    assert all(d.startswith("lic") for _, d in H.license_datas(["uv"], dest_root="lic"))


# --------------------------------------------------------------------------- Qt filtering

DROPPED = [
    "PySide6/Qt6WebEngineCore.dll",
    "PySide6/QtWebEngineProcess.exe",
    "PySide6/resources/qtwebengine_resources.pak",
    "PySide6/translations/qtwebengine_locales/de.pak",
    "PySide6/Qt/lib/libQt6WebEngineCore.so.6",
    "PySide6/Qt63DCore.dll",
    "PySide6/Qt6Quick3D.dll",
    "PySide6/Qt6Charts.dll",
    "PySide6/Qt6DataVisualization.dll",
    "PySide6/Qt6Multimedia.dll",
    "PySide6/Qt6MultimediaWidgets.dll",
    "PySide6/Qt6Pdf.dll",
    "PySide6/Qt/lib/libQt6Sql.so.6",
    "PySide6/Qt6Test.dll",
    "PySide6/Qt6Bluetooth.dll",
    "PySide6/Qt6Designer.dll",
    "PySide6/Qt6Help.dll",
    "PySide6/QtSql.pyd",
    "PySide6/QtWebEngineWidgets.pyd",
    "PySide6/QtCharts.abi3.so",
    "PySide6/Qt3DRender.pyd",
    "PySide6/plugins/sqldrivers/qsqlite.dll",
    "PySide6/Qt/plugins/sqldrivers/libqsqlite.so",
    "PySide6/plugins/multimedia/ffmpegmediaplugin.dll",
    "PySide6/plugins/sceneparsers/gltfsceneimport.dll",
    "PySide6/plugins/imageformats/qpdf.dll",
    "PySide6/Qt/plugins/imageformats/libqpdf.so",
    "PySide6/plugins/platforminputcontexts/qtvirtualkeyboardplugin.dll",
    "PySide6/Qt6VirtualKeyboard.dll",
    "PySide6/translations/designer_de.qm",
    "PySide6/translations/qtmultimedia_fr.qm",
    "PySide6/designer.exe",
    "PySide6/lupdate.exe",
    "PySide6/Qt/libexec/rcc",
    # QML runtime when the app itself does not use QtQuick/QtQml
    "PySide6/Qt6Qml.dll",
    "PySide6/Qt6Quick.dll",
    "PySide6/Qt/lib/libQt6QmlModels.so.6",
    "PySide6/qml/QtQuick/Controls/qtquickcontrols2plugin.dll",
]

KEPT = [
    "PySide6/Qt6Core.dll",
    "PySide6/Qt6Gui.dll",
    "PySide6/Qt6Widgets.dll",
    "PySide6/Qt6Svg.dll",
    "PySide6/Qt6Network.dll",
    "PySide6/Qt6OpenGL.dll",
    "PySide6/Qt6TextToSpeech.dll",
    "PySide6/Qt/lib/libQt6Core.so.6",
    "PySide6/Qt/lib/libQt6XcbQpa.so.6",
    "PySide6/QtCore.pyd",
    "PySide6/QtWidgets.abi3.so",
    "PySide6/QtSvg.pyd",
    "PySide6/pyside6.abi3.dll",
    "PySide6/plugins/platforms/qwindows.dll",
    "PySide6/plugins/imageformats/qjpeg.dll",
    "PySide6/plugins/imageformats/qsvg.dll",
    "PySide6/plugins/iconengines/qsvgicon.dll",
    "PySide6/plugins/styles/qmodernwindowsstyle.dll",
    "PySide6/translations/qtbase_de.qm",
    "PySide6/opengl32sw.dll",
    "shiboken6/shiboken6.abi3.dll",
    "onnxruntime/capi/DirectML.dll",
    "onnxruntime/capi/onnxruntime.dll",
    "imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe",
    "bin/uv.exe",
    "cv2/cv2.pyd",
    "vrphoto/webxr/viewer/index.html",
    "vrphoto/webxr/viewer/vendor/three/build/three.module.js",
    "pillow_heif.libs/libheif.so",
    "libheif-30fc1b3ceee9088934615cbf4948eba3.dll",
    "python312.dll",
]


@pytest.mark.parametrize("dest", DROPPED)
def test_is_unwanted_drops(dest):
    assert H.is_unwanted(dest), dest
    assert H.is_unwanted(dest.replace("/", "\\")), "Windows separators"


@pytest.mark.parametrize("dest", KEPT)
def test_is_unwanted_keeps(dest):
    assert not H.is_unwanted(dest), dest
    assert not H.is_unwanted(dest.replace("/", "\\"))


@pytest.mark.parametrize("dest", ["PySide6/Qt6Qml.dll", "PySide6/Qt6Quick.dll", "PySide6/qml/QtQuick/x.dll"])
def test_qml_kept_when_app_uses_it(dest):
    assert not H.is_unwanted(dest, keep_qml=True)


def test_vkb_dropped_even_with_qml():
    assert H.is_unwanted("PySide6/plugins/platforminputcontexts/qtvirtualkeyboardplugin.dll", keep_qml=True)


def test_uses_qml():
    assert H.uses_qml([("PySide6/QtQml.pyd", "x", "EXTENSION")])
    assert H.uses_qml([("PySide6\\QtQuick.abi3.so", "x", "EXTENSION")])
    assert not H.uses_qml(
        [("PySide6/QtWidgets.pyd", "x", "EXTENSION"), ("PySide6/Qt6Qml.dll", "x", "BINARY")]
    )


def test_filter_toc_keeps_order_and_entries():
    toc = [
        ("PySide6/Qt6Core.dll", "/s/a", "BINARY"),
        ("PySide6/Qt6Sql.dll", "/s/b", "BINARY"),
        ("vrphoto/x.json", "/s/c", "DATA"),
        ("PySide6/Qt6Quick.dll", "/s/d", "BINARY"),
    ]
    assert H.filter_toc(toc) == [toc[0], toc[2]]
    assert H.filter_toc(toc, keep_qml=True) == [toc[0], toc[2], toc[3]]


def test_promote_executables():
    datas = [
        ("imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2", "/src/ffmpeg", "DATA"),
        ("imageio_ffmpeg/binaries/README.md", "/src/readme", "DATA"),
        ("vrphoto/x.json", "/src/x", "DATA"),
    ]
    bins = [("cv2/cv2.so", "/src/cv2", "EXTENSION")]
    new_datas, new_bins = H.promote_executables(datas, bins, platform="linux")
    assert new_datas == datas[1:]
    assert new_bins == [
        bins[0],
        ("imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2", "/src/ffmpeg", "BINARY"),
    ]
    # Windows: untouched (ffmpeg.exe stays a plain data file, copied byte for byte)
    assert H.promote_executables(datas, bins, platform="win32") == (datas, bins)


def test_qt_excluded_modules_expands_installed():
    installed = ["QtCore", "QtGui", "QtWidgets", "QtSvg", "QtNetwork", "QtWebEngineFoo", "Qt3DNew", "QtSql"]
    mods = H.qt_excluded_modules(installed)
    assert "PySide6.QtWebEngineFoo" in mods
    assert "PySide6.Qt3DNew" in mods
    for keep in ("QtCore", "QtGui", "QtWidgets", "QtSvg", "QtNetwork"):
        assert f"PySide6.{keep}" not in mods
    for must in ("QtWebEngineCore", "QtMultimedia", "QtPdf", "QtSql", "QtTest", "QtCharts", "QtBluetooth"):
        assert f"PySide6.{must}" in mods or any(m.startswith(f"PySide6.{must}") for m in mods)


def test_qt_excluded_modules_uses_installed_pyside():
    mods = H.qt_excluded_modules()
    assert "PySide6.QtWebEngineCore" in mods
    assert "PySide6.QtCore" not in mods and "PySide6.QtWidgets" not in mods


def test_all_excludes():
    ex = H.all_excludes()
    for name in ("torch", "torchvision", "tkinter", "matplotlib", "IPython", "pytest", "PyInstaller"):
        assert name in ex
    for name in ("onnxruntime", "numpy", "cv2", "PIL", "PySide6", "PySide6.QtWidgets", "vrphoto"):
        assert name not in ex


def test_explicit_hidden_imports_not_excluded():
    ex = set(H.all_excludes())
    assert not ex & set(H.EXPLICIT_HIDDEN_IMPORTS)


# --------------------------------------------------------------------------- uv


def test_find_uv_binary_found():
    p = H.find_uv_binary()
    assert p.is_file()
    assert p.name.lower().startswith("uv")
    src, dest = H.uv_binary_entry()
    assert Path(src) == p and dest == "bin"


def test_find_uv_binary_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "uv", None)  # makes "import uv" raise ImportError
    monkeypatch.setattr(H.shutil, "which", lambda name: None)
    with pytest.raises(H.BuildError, match="pip install uv"):
        H.find_uv_binary()


def test_uv_exe_name():
    assert H.uv_exe_name() == ("uv.exe" if sys.platform == "win32" else "uv")


# --------------------------------------------------------------------------- dist extras


def _fake_root(tmp_path: Path, notices: bool = True) -> Path:
    root = tmp_path / "root"
    _touch(root / "LICENSE", "MIT License\n\nCopyright (c) 2026 AidanMan217\n")
    if notices:
        _touch(root / "THIRD_PARTY_NOTICES.md", "# Third-party notices\n\nQt LGPLv3\n")
    _touch(root / "src" / "vrphoto" / "__init__.py", '__version__ = "9.8.7"\n')
    return root


def test_combined_license_text(tmp_path):
    text = H.combined_license_text(_fake_root(tmp_path))
    assert text.index("MIT License") < text.index("Third-party notices")
    assert "Qt LGPLv3" in text


def test_combined_license_text_without_notices(tmp_path):
    text = H.combined_license_text(_fake_root(tmp_path, notices=False))
    assert text.strip().endswith("AidanMan217")


def test_combined_license_real_repo():
    text = H.combined_license_text()
    assert "MIT License" in text and "AidanMan217" in text


def test_write_dist_extras(tmp_path):
    root = _fake_root(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    written = H.write_dist_extras(dist, root=root)
    names = sorted(p.name for p in written)
    assert names == ["LICENSE.txt", "README.txt", "THIRD_PARTY_NOTICES.txt"]
    raw = (dist / "LICENSE.txt").read_bytes()
    assert b"\r\n" in raw and b"\r\r\n" not in raw  # Windows line endings, not doubled
    assert "9.8.7" in (dist / "README.txt").read_text(encoding="utf-8")


def test_dist_readme_links_follow_the_default_branch():
    """The repository has no 'main' branch: blob/HEAD follows the default branch."""
    assert "/blob/main/" not in H.DIST_README
    assert "/blob/HEAD/docs/USER_GUIDE.md" in H.DIST_README


def test_write_dist_extras_missing_dir(tmp_path):
    with pytest.raises(H.BuildError, match="does not exist"):
        H.write_dist_extras(tmp_path / "nope", root=_fake_root(tmp_path))


# --------------------------------------------------------------------------- bundle check


def _fake_bundle(tmp_path: Path, pkg: Path) -> Path:
    exe = ".exe" if sys.platform == "win32" else ""
    dist = tmp_path / "VRPhotoStudio"
    internal = dist / "_internal"
    for name in (H.GUI_EXE, H.CLI_EXE):
        _touch(dist / f"{name}{exe}")
    for name in ("LICENSE.txt", "THIRD_PARTY_NOTICES.txt"):
        _touch(dist / name)
    _touch(internal / "bin" / H.uv_exe_name())
    _touch(internal / "imageio_ffmpeg" / "binaries" / f"ffmpeg-test{exe}")
    _touch(internal / "onnxruntime" / "capi" / "onnxruntime.dll")
    _touch(internal / "onnxruntime" / "capi" / "DirectML.dll")
    _touch(internal / "PySide6" / "Qt6Core.dll")
    _touch(internal / "licenses" / "LGPL-3.0.txt")
    _touch(internal / "licenses" / "GPL-3.0.txt")
    for src, dest in H.package_datas(pkg):
        _touch(internal / dest / Path(src).name)
    return dist


@pytest.fixture
def fake_pkg_full(fake_pkg) -> Path:
    # the required files the real package always has
    _touch(fake_pkg / "webxr" / "viewer" / "scene.schema.json")
    return fake_pkg


def test_check_bundle_ok(tmp_path, fake_pkg_full):
    dist = _fake_bundle(tmp_path, fake_pkg_full)
    assert H.check_bundle(dist, pkg_dir=fake_pkg_full) == []


def test_check_bundle_reports_problems(tmp_path, fake_pkg_full):
    dist = _fake_bundle(tmp_path, fake_pkg_full)
    internal = dist / "_internal"
    (internal / "vrphoto" / "webxr" / "viewer" / "js" / "app.js").unlink()
    (internal / "bin" / H.uv_exe_name()).unlink()
    (dist / "LICENSE.txt").unlink()
    _touch(internal / "PySide6" / "Qt6WebEngineCore.dll")
    problems = "\n".join(H.check_bundle(dist, pkg_dir=fake_pkg_full))
    assert "vrphoto/webxr/viewer/js/app.js" in problems
    assert "bin/uv*" in problems
    assert "LICENSE.txt" in problems
    assert "Qt6WebEngineCore.dll" in problems


def test_check_bundle_missing_internal(tmp_path, fake_pkg_full):
    problems = H.check_bundle(tmp_path / "empty", pkg_dir=fake_pkg_full)
    assert any("_internal" in p for p in problems)
    assert any(H.GUI_EXE in p for p in problems)


def test_bundle_size_mb_ignores_symlinks(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"x" * 2_000_000)
    try:
        (tmp_path / "link.bin").symlink_to(f)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available")
    assert H.bundle_size_mb(tmp_path) == pytest.approx(2.0, abs=0.01)


# --------------------------------------------------------------------------- pyproject version


def _versioned_root(tmp_path: Path, init_version: str, project_version: str | None) -> Path:
    pkg = tmp_path / "src" / "vrphoto"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(f'__version__ = "{init_version}"\n', encoding="utf-8")
    project = f'version = "{project_version}"\n' if project_version else 'dynamic = ["version"]\n'
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "vrphoto"\n{project}', encoding="utf-8")
    return tmp_path


def test_repository_versions_agree():
    import vrphoto

    # pyproject uses a dynamic version read from vrphoto.__version__ (single source of truth)
    assert H.read_pyproject_version(H.ROOT) in (None, vrphoto.__version__)
    assert H.checked_version() == vrphoto.__version__


def test_checked_version_ok_and_dynamic(tmp_path):
    assert H.checked_version(_versioned_root(tmp_path / "a", "1.2.3", "1.2.3")) == "1.2.3"
    # a dynamic (setuptools-scm style) version is not compared
    assert H.read_pyproject_version(_versioned_root(tmp_path / "b", "2.0.0", None)) is None
    assert H.checked_version(tmp_path / "b") == "2.0.0"


def test_checked_version_mismatch_is_a_friendly_error(tmp_path):
    root = _versioned_root(tmp_path, "0.2.0", "0.1.0")
    with pytest.raises(H.BuildError, match=r"__init__\.py.*'0\.2\.0'.*pyproject\.toml.*'0\.1\.0'"):
        H.checked_version(root)


def test_read_pyproject_version_bad_toml(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project\n", encoding="utf-8")
    with pytest.raises(H.BuildError, match="Cannot read the project version"):
        H.read_pyproject_version(tmp_path)


def test_main_version_fails_on_mismatch(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(H, "ROOT", _versioned_root(tmp_path, "0.2.0", "0.1.0"))
    assert H.main(["version"]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "Version mismatch" in captured.err
    assert H.main(["numeric-version"]) == 1


# --------------------------------------------------------------------------- CLI


def test_main_version(capsys):
    import vrphoto

    assert H.main(["version"]) == 0
    assert capsys.readouterr().out.strip() == vrphoto.__version__


def test_main_numeric_version(capsys):
    assert H.main(["numeric-version"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.count(".") == 3 and all(p.isdigit() for p in out.split("."))


def test_main_check(tmp_path, capsys, fake_pkg_full):
    dist = _fake_bundle(tmp_path, fake_pkg_full)
    assert H.main(["check", str(dist), "--pkg-dir", str(fake_pkg_full)]) == 0
    assert "Bundle OK" in capsys.readouterr().out
    (dist / "LICENSE.txt").unlink()
    assert H.main(["check", str(dist), "--pkg-dir", str(fake_pkg_full)]) == 1
    assert "BUNDLE PROBLEM" in capsys.readouterr().err
