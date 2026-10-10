#!/usr/bin/env python3
"""Vendor Google's spatial-media library (MP4 spherical / stereo metadata) into the app.

Copies the ``spatialmedia`` Python package (minus the Tk GUI) and its LICENSE from a
pinned commit of https://github.com/google/spatial-media into
``src/vrphoto/third_party/spatialmedia/`` so it is importable as
``vrphoto.third_party.spatialmedia``, then applies small recorded patches:

* absolute ``spatialmedia.`` imports -> relative imports (and the ``sys.path`` hack in
  ``__init__`` removed), so the package works from any location / inside PyInstaller;
* ``print`` is shadowed by a logging shim (``_log.py``) so library diagnostics go to the
  application log instead of stdout (which is ``None`` in a windowed exe);
* a few upstream bugs on code paths we use are fixed (see ``PATCHES``).

``VENDORED.txt`` records the upstream commit, its date, the patches and the sha256 of
every vendored file.

Usage::

    python scripts/vendor_spatialmedia.py           # (re)vendor from the pinned commit
    python scripts/vendor_spatialmedia.py --check   # verify files against VENDORED.txt (offline)

Only github.com (git) is needed.  Re-running produces byte-identical output.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "src" / "vrphoto" / "third_party" / "spatialmedia"

UPSTREAM = "https://github.com/google/spatial-media"
COMMIT = "2308497c39bb604a7c3ba56ac17e302f2b1b6a96"
COMMIT_DATE = "2026-07-27"
VENDORED_ON = "2026-10-10"

#: (upstream path, vendored path relative to DEST)
FILES: list[tuple[str, str]] = [
    ("LICENSE", "LICENSE"),
    ("spatialmedia/__init__.py", "__init__.py"),
    ("spatialmedia/__main__.py", "__main__.py"),
    ("spatialmedia/metadata_utils.py", "metadata_utils.py"),
    ("spatialmedia/mpeg/__init__.py", "mpeg/__init__.py"),
    ("spatialmedia/mpeg/box.py", "mpeg/box.py"),
    ("spatialmedia/mpeg/constants.py", "mpeg/constants.py"),
    ("spatialmedia/mpeg/container.py", "mpeg/container.py"),
    ("spatialmedia/mpeg/mpeg4_container.py", "mpeg/mpeg4_container.py"),
    ("spatialmedia/mpeg/sa3d.py", "mpeg/sa3d.py"),
    ("spatialmedia/mpeg/sv3d.py", "mpeg/sv3d.py"),
]
EXCLUDED = [
    "spatialmedia/gui.py (Tk GUI)",
    "spatialmedia/spatial_media_metadata_injector.spec",
    "spatialmedia/README.md",
    "spatialmedia/LICENSE (identical to the root LICENSE)",
]

LOG_SHIM = '''"""Logging shim for the vendored spatial-media library (added by vrphoto).

Upstream reports problems with ``print()``.  Every vendored module imports
``log_print as print`` from here so those messages go to the application log.
"""

import logging

_logger = logging.getLogger("vrphoto.third_party.spatialmedia")


def log_print(*args, **kwargs):
    """Drop-in replacement for ``print`` that logs instead of writing to stdout."""
    msg = kwargs.get("sep", " ").join(str(a) for a in args)
    low = msg.lstrip().lower()
    if low.startswith(("error", "warning", "unsupported", "unknown")):
        _logger.warning(msg)
    else:
        _logger.debug(msg)
'''

#: Exact-text patches: (vendored path, old, new, reason).  Each must match exactly once.
PATCHES: list[tuple[str, str, str, str]] = [
    (
        "__init__.py",
        "# Ensure the package is available on the current path or is installed.\n"
        "import os\nimport sys\nsys.path.append(os.path.dirname(os.path.abspath(__file__)))\n",
        "",
        "remove sys.path manipulation (package is imported relatively)",
    ),
    (
        "__init__.py",
        "import spatialmedia.metadata_utils\nimport spatialmedia.mpeg\n",
        "from . import metadata_utils\nfrom . import mpeg\n",
        "relative imports",
    ),
    (
        "mpeg/__init__.py",
        "import spatialmedia.mpeg.sa3d\nimport spatialmedia.mpeg.box\nimport spatialmedia.mpeg.constants\n"
        "import spatialmedia.mpeg.container\nimport spatialmedia.mpeg.mpeg4_container\n",
        "from . import sa3d\nfrom . import box\nfrom . import constants\nfrom . import container\n"
        "from . import mpeg4_container\nfrom . import sv3d\n",
        "relative imports (sv3d imported explicitly; upstream relied on the absolute import side effect)",
    ),
    (
        "mpeg/box.py",
        "        self.content_size = len(contents)\n",
        "        self.content_size = len(new_contents)\n",
        "Box.set(): NameError (undefined 'contents')",
    ),
    (
        "mpeg/container.py",
        "                if isinstance(content, container_leaf):\n",
        "                if isinstance(content, Container):\n",
        "Container.add(): NameError (undefined 'container_leaf')",
    ),
    (
        "mpeg/container.py",
        "        assert(isinstance(element, container_box))\n",
        "        assert(isinstance(element, Container))\n",
        "Container.merge(): NameError (undefined 'container_box')",
    ),
    (
        "mpeg/sv3d.py",
        "        self.pose_yaw_degress = struct.unpack",
        "        self.pose_yaw_degrees = struct.unpack",
        "PRHDBox.load_content(): yaw stored under a misspelled attribute",
    ),
    (
        "metadata_utils.py",
        "            if child.tag[:len(spherical_prefix)] == spherical_prefix:\n"
        "                tag = child.tag[len(spherical_prefix):]\n",
        "            if child.tag[:len(SPHERICAL_PREFIX)] == SPHERICAL_PREFIX:\n"
        "                tag = child.tag[len(SPHERICAL_PREFIX):]\n",
        "parse_spherical_xml(): NameError (undefined 'spherical_prefix')",
    ),
]

_IMPORT_RES = [
    (re.compile(r"^from spatialmedia\.mpeg import (\w+)$", re.M), "from . import \\1"),  # inside mpeg/
    (re.compile(r"^from spatialmedia import (\w+)$", re.M), "from . import \\1"),  # package root
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, stdout=subprocess.DEVNULL)


def fetch(workdir: Path) -> Path:
    """Fetch the pinned commit into ``workdir/src`` and return that path."""
    src = workdir / "src"
    src.mkdir()
    _run(["git", "init", "--quiet"], src)
    _run(["git", "remote", "add", "origin", UPSTREAM], src)
    try:
        _run(["git", "fetch", "--quiet", "--depth", "1", "origin", COMMIT], src)
    except subprocess.CalledProcessError:  # server refuses fetch-by-sha: full clone
        shutil.rmtree(src)
        _run(["git", "clone", "--quiet", UPSTREAM, str(src)])
        _run(["git", "checkout", "--quiet", COMMIT], src)
        return src
    _run(["git", "checkout", "--quiet", "FETCH_HEAD"], src)
    return src


def transform(rel: str, text: str) -> str:
    """Apply import rewriting, the print shim and the exact-text patches to one file."""
    if not rel.endswith(".py"):
        return text
    for path, old, new, reason in PATCHES:
        if path != rel:
            continue
        count = text.count(old)
        if count != 1:
            raise SystemExit(f"patch for {rel} ({reason}) matched {count} times; upstream changed?")
        text = text.replace(old, new)
    for rx, repl in _IMPORT_RES:
        text = rx.sub(repl, text)
    leftovers = re.findall(r"^\s*(?:import|from) spatialmedia\b.*$", text, re.M)
    if leftovers:
        raise SystemExit(f"{rel}: unconverted absolute imports: {leftovers}")
    if re.search(r"\bprint\(", text):
        shim = (
            "from .._log import log_print as print\n"
            if rel.startswith("mpeg/")
            else ("from ._log import log_print as print\n")
        )
        # insert after the module docstring / leading comments, before the first import
        m = re.search(r"^(import|from) ", text, re.M)
        pos = m.start() if m else len(text)
        text = text[:pos] + shim + text[pos:]
    return text


def build(src: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for up, rel in FILES:
        raw = (src / up).read_bytes()
        out[rel] = transform(rel, raw.decode("utf-8")).encode("utf-8") if rel.endswith(".py") else raw
    out["_log.py"] = LOG_SHIM.encode("utf-8")
    return out


def vendored_txt(files: dict[str, bytes]) -> str:
    lines = [
        "Vendored third-party code: Google spatial-media (Python package 'spatialmedia')",
        f"Upstream:     {UPSTREAM}",
        f"Commit:       {COMMIT}",
        f"Commit date:  {COMMIT_DATE}",
        f"Vendored on:  {VENDORED_ON}",
        "License:      Apache License 2.0 (see LICENSE)",
        "Reproduce:    python scripts/vendor_spatialmedia.py   (verify: --check)",
        "Import as:    vrphoto.third_party.spatialmedia",
        "",
        "Excluded upstream files:",
        *[f"  - {e}" for e in EXCLUDED],
        "",
        "Local modifications:",
        "  - absolute 'spatialmedia.' imports rewritten as relative imports",
        "  - '_log.py' added; modules that call print() import 'log_print as print' from it",
        *[f"  - {p}: {reason}" for p, _o, _n, reason in PATCHES],
        "",
        "Files (sha256):",
        *[f"{_sha256(files[k])}  {k}" for k in sorted(files)],
        "",
    ]
    return "\n".join(lines)


def write(files: dict[str, bytes]) -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    for rel, data in files.items():
        p = DEST / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (DEST / "VENDORED.txt").write_bytes(vendored_txt(files).encode("utf-8"))


def check() -> int:
    """Verify vendored files against the hashes in VENDORED.txt (no network)."""
    meta = DEST / "VENDORED.txt"
    if not meta.is_file():
        print(f"missing {meta}", file=sys.stderr)
        return 1
    bad = 0
    listed: set[str] = set()
    for line in meta.read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"([0-9a-f]{64})  (\S+)", line)
        if not m:
            continue
        digest, rel = m.groups()
        listed.add(rel)
        p = DEST / rel
        if not p.is_file() or _sha256(p.read_bytes()) != digest:
            print(f"MISMATCH {rel}", file=sys.stderr)
            bad += 1
    if f"Commit:       {COMMIT}" not in meta.read_text(encoding="utf-8"):
        print("VENDORED.txt commit differs from the pinned COMMIT", file=sys.stderr)
        bad += 1
    extra = {
        str(p.relative_to(DEST)).replace("\\", "/")
        for p in DEST.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != "VENDORED.txt"
    } - listed
    for rel in sorted(extra):
        print(f"UNLISTED {rel}", file=sys.stderr)
        bad += 1
    print("ok" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="verify vendored files against VENDORED.txt")
    ap.add_argument("--source", type=Path, help="use an existing checkout of the pinned commit")
    args = ap.parse_args(argv)
    if args.check:
        return check()
    with tempfile.TemporaryDirectory(prefix="vendor-spatialmedia-") as tmp:
        src = args.source if args.source else fetch(Path(tmp))
        files = build(src)
    write(files)
    print(f"vendored spatial-media {COMMIT[:12]} -> {DEST.relative_to(REPO_ROOT)} ({len(files)} files)")
    return check()


if __name__ == "__main__":
    sys.exit(main())
