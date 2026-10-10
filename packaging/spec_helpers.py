"""Build-time helpers for the PyInstaller spec, the version resource and the build scripts.

Everything here is a plain function with no PyInstaller import at module level, so the
logic that decides *what goes into the Windows bundle* can be unit-tested on any OS
(see ``packaging/tests``).  ``vrphoto.spec`` imports this module from ``SPECPATH``.

Bundle layout produced by ``vrphoto.spec`` (PyInstaller 6 onedir)::

    VRPhotoStudio/
      VRPhotoStudio.exe          windowed GUI (no args -> GUI, args -> CLI)
      vrphoto-cli.exe            console CLI
      LICENSE.txt                MIT licence + third-party notices (also the installer licence page)
      THIRD_PARTY_NOTICES.txt
      README.txt
      _internal/                 == sys._MEIPASS
        vrphoto/...              package data (viewer, add-on manifests + workers, model lock, ...)
        bin/uv.exe               uv, used to create add-on virtual environments
        imageio_ffmpeg/binaries/ffmpeg-*.exe
        onnxruntime/capi/{onnxruntime,DirectML}.dll
        PySide6/...
"""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import shutil
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
PKG_DIR = SRC / "vrphoto"

APP_NAME = "VR Photo Studio"
APP_ID = "VRPhotoStudio"
PUBLISHER = "AidanMan217"
GUI_EXE = "VRPhotoStudio"
CLI_EXE = "vrphoto-cli"
COLLECT_NAME = "VRPhotoStudio"
UV_DEST_DIR = "bin"  # relative to sys._MEIPASS (the "_internal" folder)

#: Sub-trees of the vrphoto package that are copied as plain files *including* ``.py``
#: sources: add-on workers run inside separate virtual environments (not the frozen
#: interpreter), and ``third_party`` may hold vendored scripts/assets used the same way.
SOURCE_TREES: tuple[str, ...] = ("addons/workers", "third_party")

#: Single source files that external (add-on) interpreters may need on disk.
SOURCE_FILES: tuple[str, ...] = ("addons/protocol.py",)

#: Files/directories never bundled from the source tree.
_SKIP_DIRS = {"__pycache__", "node_modules", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".git"}
_SKIP_SUFFIXES = {".pyc", ".pyo", ".part", ".tmp", ".orig", ".rej"}
_SKIP_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini", ".gitignore", ".gitkeep", ".npmrc", ".gitattributes"}

#: Python modules that must never end up in the bundle (huge or dev-only).
EXCLUDES: list[str] = [
    # deep-learning frameworks live in add-on venvs, never in the app
    "torch",
    "torchvision",
    "torchaudio",
    "tensorflow",
    "jax",
    "xformers",
    # GUI toolkits / notebook stack we do not use
    "tkinter",
    "_tkinter",
    "turtle",
    "turtledemo",
    "idlelib",
    "matplotlib",
    "IPython",
    "ipykernel",
    "ipywidgets",
    "jupyter",
    "jupyter_client",
    "jupyter_core",
    "notebook",
    # build / test tooling
    "pytest",
    "_pytest",
    "pytestqt",
    "PyInstaller",
    # ONNX graph tooling is only used to build test stub models; onnxruntime does not need it
    "onnx",
    # optional backends pulled in by try-imports (qrcode -> lxml, numpy.__config__ -> yaml)
    "lxml",
    "bs4",
    "soupsieve",
    "yaml",
]

#: PySide6 / Qt modules we do not ship (big, unused).  Prefix entries end with ``*``.
QT_EXCLUDED: tuple[str, ...] = (
    "QtWebEngine*",
    "QtWebView",
    "Qt3D*",
    "QtQuick3D*",
    "QtCharts",
    "QtDataVisualization",
    "QtGraphs*",
    "QtMultimedia*",
    "QtSpatialAudio",
    "QtPdf*",
    "QtSql",
    "QtTest",
    "QtQuickTest",
    "QtBluetooth",
    "QtDesigner",
    "QtHelp",
    "QtUiTools",
)

#: The concrete modules behind :data:`QT_EXCLUDED` in PySide6 6.11 (always excluded, even
#: when PySide6 cannot be inspected at build time).
QT_EXCLUDED_KNOWN: tuple[str, ...] = (
    "QtWebEngineCore",
    "QtWebEngineWidgets",
    "QtWebEngineQuick",
    "QtWebView",
    "Qt3DAnimation",
    "Qt3DCore",
    "Qt3DExtras",
    "Qt3DInput",
    "Qt3DLogic",
    "Qt3DRender",
    "QtQuick3D",
    "QtCharts",
    "QtDataVisualization",
    "QtGraphs",
    "QtGraphsWidgets",
    "QtMultimedia",
    "QtMultimediaWidgets",
    "QtSpatialAudio",
    "QtPdf",
    "QtPdfWidgets",
    "QtSql",
    "QtTest",
    "QtQuickTest",
    "QtBluetooth",
    "QtDesigner",
    "QtHelp",
    "QtUiTools",
)

#: Tokens that, following ``Qt6`` in a Qt library name, mark it as excluded.
_QT_LIB_TOKENS = (
    "webengine",
    "webview",
    "3d",
    "quick3d",
    "charts",
    "datavisualization",
    "graphs",
    "multimedia",
    "spatialaudio",
    "pdf",
    "sql",
    "test",
    "quicktest",
    "bluetooth",
    "designer",
    "help",
    "uitools",
)
_QT_TOKEN_RE = "|".join(_QT_LIB_TOKENS)
# Qt shared libraries: Qt6Sql.dll, libQt6Sql.so.6, Qt63DCore.dll, ...
_QT_LIB_RE = re.compile(rf"^(lib)?qt6({_QT_TOKEN_RE})", re.IGNORECASE)
# PySide6 binding extension modules: PySide6/QtSql.pyd, PySide6/QtSql.abi3.so
_PYSIDE_MOD_RE = re.compile(rf"^qt({_QT_TOKEN_RE})[^.]*\.(pyd|abi3\.so|so|pyi)$", re.IGNORECASE)
# Qt plugin folders that only serve excluded modules
_QT_PLUGIN_DIRS = (
    "sqldrivers",
    "multimedia",
    "designer",
    "sceneparsers",
    "geometryloaders",
    "renderers",
    "renderplugins",
    "assetimporters",
    "qmltooling",
    "qmllint",
    "webview",
)
_QT_PLUGIN_RE = re.compile(r"(^|/)plugins/(" + "|".join(_QT_PLUGIN_DIRS) + r")(/|$)", re.IGNORECASE)
_QT_PDF_PLUGIN_RE = re.compile(r"(^|/)plugins/imageformats/(lib)?qpdf", re.IGNORECASE)
_QT_TRANSLATION_RE = re.compile(
    r"(^|/)translations/(qt(webengine|multimedia|designer|help|linguist|assistant)|designer|assistant|linguist)",
    re.IGNORECASE,
)
# On-screen virtual keyboard input plugin (pulls in Qt Quick/QML; useless on a desktop PC)
_QT_VKB_RE = re.compile(
    r"(^|/)(lib)?qt6virtualkeyboard|platforminputcontexts/(lib)?qtvirtualkeyboard", re.IGNORECASE
)
# Qt Quick / QML runtime: only kept when the app itself imports PySide6.QtQml/QtQuick
_QT_QML_LIB_RE = re.compile(r"^(lib)?qt6(qml|quick|labs)", re.IGNORECASE)
_QT_QML_DIR_RE = re.compile(r"^pyside6/(qt/)?qml/", re.IGNORECASE)
_PYSIDE_QML_MOD_RE = re.compile(r"^pyside6/qt(qml|quick)[^/]*\.(pyd|abi3\.so|so)$", re.IGNORECASE)
# Qt developer tools shipped inside the PySide6 wheel (designer.exe, lupdate, qmlls, ...)
_QT_TOOL_RE = re.compile(
    r"^pyside6/(qt/libexec/|[^/]+\.exe$|(designer|assistant|linguist|lupdate|lrelease|qmlformat|qmllint|qmlls|"
    r"balsam|balsamui|qsb|svgtoqml|uic|rcc)$)",
    re.IGNORECASE,
)


class BuildError(RuntimeError):
    """A friendly, actionable build-configuration error."""


# --------------------------------------------------------------------------- version
def read_version(pkg_dir: Path = PKG_DIR) -> str:
    """Return ``vrphoto.__version__`` by parsing ``__init__.py`` (no import needed)."""
    init = Path(pkg_dir) / "__init__.py"
    try:
        tree = ast.parse(init.read_text(encoding="utf-8"), filename=str(init))
    except OSError as e:
        raise BuildError(f"Cannot read the package version from {init}: {e}") from e
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets
        ):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    raise BuildError(f"No __version__ = '...' string found in {init}")


def read_pyproject_version(root: Path | None = None) -> str | None:
    """``[project] version`` from ``pyproject.toml`` (None when absent or dynamic)."""
    import tomllib

    path = Path(root if root is not None else ROOT) / "pyproject.toml"
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise BuildError(f"Cannot read the project version from {path}: {e}") from e
    project = data.get("project")
    version = project.get("version") if isinstance(project, dict) else None
    return str(version) if version else None


def checked_version(root: Path | None = None) -> str:
    """``vrphoto.__version__``, after checking that ``pyproject.toml`` says the same.

    A release needs both bumped (the tag check in ``build.yml`` reads ``__version__``, pip
    reads ``pyproject.toml``); a mismatch fails the build early with a clear message.
    """
    root = Path(root if root is not None else ROOT)
    version = read_version(root / "src" / "vrphoto")
    project_version = read_pyproject_version(root)
    if project_version is not None and project_version != version:
        raise BuildError(
            f"Version mismatch: src/vrphoto/__init__.py has __version__ = {version!r} but "
            f"pyproject.toml has version = {project_version!r}. Set both to the same version."
        )
    return version


def version_tuple(version: str) -> tuple[int, int, int, int]:
    """``"1.2.3rc1"`` -> ``(1, 2, 3, 0)``: the numeric 4-tuple Windows resources need."""
    m = re.match(r"^\s*v?(\d+(?:\.\d+){0,3})", version)
    if not m:
        raise BuildError(f"Version {version!r} does not start with a number like 1.2.3")
    parts = [int(p) for p in m.group(1).split(".")]
    if any(p > 0xFFFF for p in parts):
        raise BuildError(f"Version {version!r} has a component above 65535 (Windows limit)")
    return tuple((parts + [0, 0, 0, 0])[:4])  # type: ignore[return-value]


# --------------------------------------------------------------------------- data files
def _skipped(path: Path) -> bool:
    if path.name in _SKIP_NAMES or path.suffix.lower() in _SKIP_SUFFIXES:
        return True
    return any(part in _SKIP_DIRS for part in path.parts)


def package_datas(pkg_dir: Path = PKG_DIR, *, dest_root: str = "vrphoto") -> list[tuple[str, str]]:
    """``(source_file, dest_dir)`` pairs for every bundled data file of the package.

    * every non-Python file anywhere in the package (viewer HTML/JS/CSS, vendored JS,
      scene schema, add-on TOML manifests, ``models.lock.json``, GUI resources, ...),
      keeping the directory tree;
    * every file (``.py`` included) below :data:`SOURCE_TREES`;
    * the single files in :data:`SOURCE_FILES`.

    Files are listed one by one (not whole directories) so junk such as ``__pycache__``
    or ``node_modules`` can never sneak in.
    """
    pkg_dir = Path(pkg_dir)
    if not (pkg_dir / "__init__.py").is_file():
        raise BuildError(f"{pkg_dir} is not the vrphoto package directory (no __init__.py)")
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    source_roots = [pkg_dir / Path(t) for t in SOURCE_TREES]
    source_files = {(pkg_dir / Path(f)).resolve() for f in SOURCE_FILES}
    for f in sorted(pkg_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(pkg_dir)
        if _skipped(rel):
            continue
        in_source_tree = any(_is_relative_to(f, r) for r in source_roots)
        if f.suffix.lower() == ".py" and not in_source_tree and f.resolve() not in source_files:
            continue  # ordinary module: goes into the PYZ archive, not as a data file
        dest = "/".join((dest_root, *rel.parent.parts))
        key = f"{dest}/{f.name}"
        if key not in seen:
            seen.add(key)
            out.append((str(f), dest))
    return out


def _is_relative_to(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
    except ValueError:
        return False
    return True


def want_vrphoto_module(name: str) -> bool:
    """Filter for ``collect_submodules('vrphoto')``.

    Add-on worker scripts run in the add-on's own virtual environment (with torch etc.)
    and are bundled as source files, so they must not be analysed as app modules (that
    would drag their imports into the bundle).
    """
    return not (name == "vrphoto.addons.workers" or name.startswith("vrphoto.addons.workers."))


EXPLICIT_HIDDEN_IMPORTS: tuple[str, ...] = (
    # imported lazily inside functions throughout vrphoto; listed so a refactor that moves
    # an import into importlib/string form can never silently drop a runtime dependency
    "numpy",
    "cv2",
    "PIL.Image",
    "PIL.ImageCms",
    "pillow_heif",
    "onnxruntime",
    "imageio_ffmpeg",
    "cryptography",
    "cryptography.x509",
    "cryptography.hazmat.primitives.asymmetric.rsa",
    "cryptography.hazmat.primitives.asymmetric.ec",
    "qrcode",
    "qrcode.image.pil",
    "qrcode.image.svg",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtSvg",
    "tomllib",
    "multiprocessing",
    "concurrent.futures",
    "http.server",
    "ssl",
)


# --------------------------------------------------------------------------- licences
#: Distributions whose licence files are copied to ``_internal/licenses/<name>/``.
LICENSE_DISTS: tuple[str, ...] = (
    "PySide6",
    "PySide6_Essentials",
    "PySide6_Addons",
    "shiboken6",
    "numpy",
    "opencv-python-headless",
    "pillow",
    "pillow-heif",
    "onnxruntime-directml",
    "onnxruntime",
    "imageio-ffmpeg",
    "cryptography",
    "cffi",
    "pycparser",
    "qrcode",
    "uv",
)
#: Full GPL/LGPL/Apache texts shipped with the bundle (required by the LGPL for Qt etc.).
LICENSE_TEXTS_DIR = HERE / "licenses"
_LICENSE_FILE_RE = re.compile(
    r"(^|/)(licen[cs]e|copying|notice|thirdpartynotices)[^/]*$|\.dist-info/licen[cs]es/", re.IGNORECASE
)


def _dist_license_files(dist_name: str) -> list[tuple[Path, str]]:
    """``(file, sub_dir)`` licence files listed in an installed distribution's RECORD."""
    from importlib import metadata

    try:
        dist = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        return []
    out: list[tuple[Path, str]] = []
    for f in dist.files or ():
        rel = str(f).replace("\\", "/")
        if ".." in rel.split("/") or not _LICENSE_FILE_RE.search(rel):
            continue
        if rel.lower().endswith((".py", ".pyc", ".pyi")):
            continue
        m = re.search(r"\.dist-info/licen[cs]es/(.+)$", rel, re.IGNORECASE)
        if m:  # PEP 639 licence folder: keep its internal structure
            sub = Path(m.group(1)).parent.as_posix()
        elif re.search(r"\.dist-info/[^/]+$", rel, re.IGNORECASE):
            sub = ""
        else:  # e.g. onnxruntime/ThirdPartyNotices.txt, numpy/ma/LICENSE
            sub = Path(rel).parent.as_posix()
        try:
            src = Path(f.locate())
        except (OSError, NotImplementedError):  # pragma: no cover - exotic distributions
            continue
        if src.is_file():
            out.append((src, "" if sub == "." else sub))
    return out


def license_datas(
    dists: Iterable[str] = LICENSE_DISTS, *, dest_root: str = "licenses"
) -> list[tuple[str, str]]:
    """``(source, dest_dir)`` pairs that copy third-party licence files into the bundle."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    if LICENSE_TEXTS_DIR.is_dir():
        for f in sorted(LICENSE_TEXTS_DIR.glob("*.txt")):
            out.append((str(f), dest_root))
            seen.add(f"{dest_root}/{f.name}".lower())
    for name in dists:
        folder = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).lower()
        for src, sub in _dist_license_files(name):
            dest = "/".join(p for p in (dest_root, folder, sub) if p)
            key = f"{dest}/{src.name}".lower()
            if key not in seen:
                seen.add(key)
                out.append((str(src), dest))
    return out


# --------------------------------------------------------------------------- Qt filtering
def _pyside6_dir() -> Path | None:
    try:
        spec = importlib.util.find_spec("PySide6")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations)))


def _match_qt(pattern: str, name: str) -> bool:
    if pattern.endswith("*"):
        return name.startswith(pattern[:-1])
    return name == pattern


def qt_excluded_modules(installed: Iterable[str] | None = None) -> list[str]:
    """Fully-qualified ``PySide6.Qt*`` module names to exclude.

    The static list is expanded with whatever matching modules the installed PySide6
    actually ships (so a new ``QtWebEngineFoo`` in a future PySide6 is excluded too).
    """
    names = set(QT_EXCLUDED_KNOWN) | {p for p in QT_EXCLUDED if not p.endswith("*")}
    if installed is None:
        d = _pyside6_dir()
        installed = []
        if d is not None and d.is_dir():
            installed = [f.name.split(".")[0] for f in d.iterdir() if f.name.startswith("Qt")]
    for mod in installed:
        if any(_match_qt(p, mod) for p in QT_EXCLUDED):
            names.add(mod)
    return sorted(f"PySide6.{n}" for n in names)


def all_excludes() -> list[str]:
    return EXCLUDES + qt_excluded_modules()


def uses_qml(toc: Sequence[tuple]) -> bool:
    """True if the analysed app imports PySide6.QtQml / QtQuick (then QML must be kept)."""
    return any(_PYSIDE_QML_MOD_RE.match(str(e[0]).replace("\\", "/")) for e in toc)


def is_unwanted(dest: str, *, keep_qml: bool = False) -> bool:
    """True if a collected file (by its destination path in the bundle) must be dropped.

    Removes Qt libraries, plugins, translations, resources and developer tools that only
    serve excluded Qt modules (WebEngine, 3D, Multimedia, Pdf, Sql, ...), the virtual
    keyboard plugin and - unless ``keep_qml`` - the Qt Quick/QML runtime it drags in.
    """
    d = dest.replace("\\", "/")
    low = d.lower()
    base = low.rsplit("/", 1)[-1]
    if _QT_VKB_RE.search(low):
        return True
    if not keep_qml and (_QT_QML_LIB_RE.match(base) or _QT_QML_DIR_RE.match(low)):
        return True
    if "qtwebengine" in low or base in ("icudtl.dat", "v8_context_snapshot.bin"):
        return True
    if _QT_LIB_RE.match(base):
        return True
    if low.startswith("pyside6/") and _PYSIDE_MOD_RE.match(base):
        return True
    if _QT_PLUGIN_RE.search(low) or _QT_PDF_PLUGIN_RE.search(low):
        return True
    if _QT_TRANSLATION_RE.search(low):
        return True
    return bool(_QT_TOOL_RE.match(low))


def filter_toc(toc: Sequence[tuple], *, keep_qml: bool = False) -> list[tuple]:
    """Drop :func:`is_unwanted` entries from a PyInstaller TOC (``(dest, src, typecode)``)."""
    return [entry for entry in toc if not is_unwanted(str(entry[0]), keep_qml=keep_qml)]


def promote_executables(
    datas: Sequence[tuple], binaries: Sequence[tuple], *, platform: str = sys.platform
) -> tuple[list[tuple], list[tuple]]:
    """Move helper executables collected as *data* (imageio-ffmpeg's ffmpeg) to *binaries*.

    Older PyInstaller 6 releases copy data files without the executable bit on Linux/macOS,
    which would make the bundled ffmpeg unusable there.  On Windows nothing is changed (the
    file stays plain data and is copied byte for byte).
    """
    if platform.startswith("win"):
        return list(datas), list(binaries)
    new_datas: list[tuple] = []
    new_bins = list(binaries)
    for entry in datas:
        dest = str(entry[0]).replace("\\", "/")
        if dest.startswith("imageio_ffmpeg/binaries/") and dest.rsplit("/", 1)[-1].startswith("ffmpeg"):
            new_bins.append((entry[0], entry[1], "BINARY"))
        else:
            new_datas.append(entry)
    return new_datas, new_bins


# --------------------------------------------------------------------------- uv
def uv_exe_name() -> str:
    return "uv.exe" if sys.platform == "win32" else "uv"


def find_uv_binary() -> Path:
    """Locate the ``uv`` executable to bundle (``uv.find_uv_bin()``, then ``PATH``)."""
    candidates: list[str] = []
    try:
        import uv  # type: ignore[import-not-found]

        candidates.append(uv.find_uv_bin())
    except (ImportError, FileNotFoundError, OSError):
        pass
    which = shutil.which("uv")
    if which:
        candidates.append(which)
    for c in candidates:
        p = Path(c)
        if p.is_file():
            return p.resolve()
    raise BuildError(
        "Could not find the 'uv' executable to bundle. Install it into the build environment "
        "with:  python -m pip install uv"
    )


def uv_binary_entry() -> tuple[str, str]:
    """``(source, dest_dir)`` for the ``binaries`` list; lands at ``_internal/bin/uv(.exe)``."""
    return (str(find_uv_binary()), UV_DEST_DIR)


# --------------------------------------------------------------------------- dist extras
def combined_license_text(root: Path = ROOT) -> str:
    """MIT licence followed by the third-party notices (shown on the installer licence page)."""
    lic = (Path(root) / "LICENSE").read_text(encoding="utf-8").strip()
    notices_path = Path(root) / "THIRD_PARTY_NOTICES.md"
    parts = [lic]
    if notices_path.is_file():
        parts += ["", "", "=" * 78, "", notices_path.read_text(encoding="utf-8").strip()]
    return "\n".join(parts) + "\n"


DIST_README = """VR Photo Studio {version}
========================

Start the app with VRPhotoStudio.exe.

vrphoto-cli.exe is the command-line version (for scripting and troubleshooting):
    vrphoto-cli.exe --help
    vrphoto-cli.exe --selftest

Your settings, downloaded AI models and add-ons are stored in
%LOCALAPPDATA%\\VRPhotoStudio  (they can grow to many GB; delete the folder to reset).

User guide: https://github.com/AidanMan217/VR-Photo-Studio/blob/HEAD/docs/USER_GUIDE.md
Licences: LICENSE.txt and THIRD_PARTY_NOTICES.txt in this folder.
"""


def write_dist_extras(dist_dir: Path, *, root: Path = ROOT, version: str | None = None) -> list[Path]:
    """Write LICENSE.txt / THIRD_PARTY_NOTICES.txt / README.txt next to the executables.

    Uses Windows line endings so Notepad on old Windows builds shows them correctly.
    """
    dist_dir = Path(dist_dir)
    if not dist_dir.is_dir():
        raise BuildError(f"Bundle folder {dist_dir} does not exist - did PyInstaller fail?")
    version = version or read_version(Path(root) / "src" / "vrphoto")
    written: list[Path] = []

    def _w(name: str, text: str) -> None:
        p = dist_dir / name
        with open(p, "w", encoding="utf-8", newline="\r\n") as fh:
            fh.write(text)
        written.append(p)

    _w("LICENSE.txt", combined_license_text(root))
    notices = Path(root) / "THIRD_PARTY_NOTICES.md"
    if notices.is_file():
        _w("THIRD_PARTY_NOTICES.txt", notices.read_text(encoding="utf-8"))
    _w("README.txt", DIST_README.format(version=version))
    return written


# --------------------------------------------------------------------------- bundle check
#: Files (glob patterns relative to the bundle's ``_internal`` folder) that MUST exist.
REQUIRED_IN_BUNDLE: tuple[str, ...] = (
    "vrphoto/webxr/viewer/scene.schema.json",
    "vrphoto/models/models.lock.json",
    "vrphoto/addons/protocol.py",
    "bin/uv*",
    "imageio_ffmpeg/binaries/ffmpeg*",
    "onnxruntime/capi/*onnxruntime*",
    "licenses/LGPL-3.0.txt",
    "licenses/GPL-3.0.txt",
)


def check_bundle(dist_dir: Path, *, pkg_dir: Path = PKG_DIR) -> list[str]:
    """Return a list of human-readable problems with a built bundle (empty = OK).

    Verifies both executables, the required files, that every source data file made it
    in, that the add-on manifests/workers are present, and that no excluded Qt module
    leaked in.  Used by the tests and by ``build_windows.ps1`` / CI.
    """
    dist_dir = Path(dist_dir)
    problems: list[str] = []
    exe = ".exe" if sys.platform == "win32" else ""
    for name in (GUI_EXE, CLI_EXE):
        if not (dist_dir / f"{name}{exe}").is_file():
            problems.append(f"missing executable {name}{exe}")
    internal = dist_dir / "_internal"
    if not internal.is_dir():
        problems.append("missing _internal folder")
        return problems
    for pattern in REQUIRED_IN_BUNDLE:
        if not any(internal.glob(pattern)):
            problems.append(f"missing {pattern}")
    if sys.platform == "win32" and not (internal / "onnxruntime" / "capi" / "DirectML.dll").is_file():
        problems.append(
            "missing onnxruntime/capi/DirectML.dll (install onnxruntime-directml, not onnxruntime)"
        )
    for src, dest in package_datas(pkg_dir):
        target = internal / dest / Path(src).name
        if not target.is_file():
            problems.append(f"missing data file {dest}/{Path(src).name}")
    files = [f.relative_to(internal).as_posix() for f in internal.rglob("*") if f.is_file()]
    keep_qml = uses_qml([(rel,) for rel in files])
    for rel in files:
        if is_unwanted(rel, keep_qml=keep_qml):
            problems.append(f"excluded Qt file leaked into bundle: {rel}")
    for name in ("LICENSE.txt", "THIRD_PARTY_NOTICES.txt"):
        if not (dist_dir / name).is_file():
            problems.append(f"missing {name}")
    return problems


def bundle_size_mb(dist_dir: Path) -> float:
    total = 0
    for root, _dirs, files in os.walk(dist_dir):
        for f in files:
            p = os.path.join(root, f)
            try:
                if not os.path.islink(p):  # PyInstaller symlinks .so versions on Linux
                    total += os.path.getsize(p)
            except OSError:
                pass
    return total / 1e6


def main(argv: Sequence[str] | None = None) -> int:
    """``python packaging/spec_helpers.py check DIST_DIR | version | numeric-version``."""
    import argparse

    ap = argparse.ArgumentParser(description="VR Photo Studio build helpers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="verify a built onedir bundle")
    c.add_argument("dist_dir", type=Path)
    c.add_argument("--pkg-dir", type=Path, default=PKG_DIR, help="vrphoto source package to compare with")
    sub.add_parser("version", help="print the package version (e.g. 0.1.0); fails if pyproject.toml differs")
    sub.add_parser("numeric-version", help="print the 4-part numeric version (e.g. 0.1.0.0)")
    args = ap.parse_args(argv)
    if args.cmd in ("version", "numeric-version"):
        try:
            version = checked_version()
            text = version if args.cmd == "version" else ".".join(str(p) for p in version_tuple(version))
        except BuildError as e:
            sys.stderr.write(f"ERROR: {e}\n")
            return 1
        sys.stdout.write(text + "\n")
        return 0
    problems = check_bundle(args.dist_dir, pkg_dir=args.pkg_dir)
    for p in problems:
        sys.stderr.write(f"BUNDLE PROBLEM: {p}\n")
    if not problems:
        sys.stdout.write(f"Bundle OK: {args.dist_dir} ({bundle_size_mb(args.dist_dir):.0f} MB)\n")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
