"""PyInstaller hook for pillow-heif (HEIC/HEIF photos from phones).

pillow-heif is a pure-Python package ``pillow_heif`` plus a *top-level* C extension
``_pillow_heif`` that links libheif, libde265, libx265 (and on Windows the MinGW runtime).
Where those shared libraries live depends on the wheel:

* Windows (delvewheel, "platlib" mode): ``site-packages/libheif-<hash>.dll`` etc., next to
  ``_pillow_heif.pyd``; ``pillow_heif/__init__.py`` calls ``os.add_dll_directory`` on its
  parent folder - which is ``sys._MEIPASS`` in the frozen app.
* Linux (auditwheel): ``site-packages/pillow_heif.libs/*.so*`` found through the
  extension's RPATH (``$ORIGIN/pillow_heif.libs``).
* macOS (delocate): ``pillow_heif/.dylibs``.

PyInstaller's binary dependency analysis usually finds these, but they are resolved
indirectly, so this hook collects every shared library listed in the distribution's
RECORD explicitly, at the same relative location, so the loader patch / RPATH keeps working.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import PurePosixPath

from PyInstaller.utils.hooks import collect_delvewheel_libs_directory, collect_dynamic_libs

hiddenimports = ["_pillow_heif", "pillow_heif.as_plugin", "pillow_heif.HeifImagePlugin"]

binaries = collect_dynamic_libs("pillow_heif")
datas: list[tuple[str, str]] = []

_LIB_SUFFIXES = (".dll", ".so", ".dylib")


def _is_shared_lib(name: str) -> bool:
    low = name.lower()
    return low.endswith(_LIB_SUFFIXES) or ".so." in low


def _record_libs() -> list[tuple[str, str]]:
    """Shared libraries listed in the installed pillow-heif RECORD as (src, dest_dir)."""
    try:
        dist = metadata.distribution("pillow-heif")
    except metadata.PackageNotFoundError:
        return []
    out: list[tuple[str, str]] = []
    for f in dist.files or ():
        rel = PurePosixPath(str(f).replace("\\", "/"))
        if not _is_shared_lib(rel.name) or rel.name.startswith("_pillow_heif"):
            continue  # the extension module itself is collected via hiddenimports
        if ".." in rel.parts:
            continue  # files installed outside site-packages (should not happen)
        src = f.locate()
        try:
            exists = src.is_file()
        except OSError:
            exists = False
        if exists:
            dest = str(rel.parent) if str(rel.parent) not in ("", ".") else "."
            out.append((str(src), dest))
    return out


_known = {b[0] for b in binaries}
for _src, _dest in _record_libs():
    if _src not in _known:
        binaries.append((_src, _dest))
        _known.add(_src)

# Windows wheels built with delvewheel in the default (non-platlib) layout use a
# pillow_heif.libs folder next to the package; harmless no-op elsewhere.
datas, binaries = collect_delvewheel_libs_directory("pillow_heif", datas=datas, binaries=binaries)
