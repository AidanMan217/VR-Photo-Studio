"""Build the GitHub Actions job matrix for ``.github/workflows/addons.yml``.

Reads the add-on manifests (``src/vrphoto/addons/manifests/*.toml``) so the workflow
never needs editing when an add-on is added, and prints / writes::

    {"include": [{"id": "capture", "title": "...", "experimental": false}, ...]}

Usage::

    python packaging/ci/addon_matrix.py all                 # every add-on
    python packaging/ci/addon_matrix.py trellis             # one add-on
    python packaging/ci/addon_matrix.py capture,sharp       # several
    python packaging/ci/addon_matrix.py all --github-output "$GITHUB_OUTPUT"

The manifest id is ``id`` at the top level or inside an ``[addon]`` table, falling back
to the file name; ``experimental`` is read the same way (default false).
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_DIR = ROOT / "src" / "vrphoto" / "addons" / "manifests"


class MatrixError(ValueError):
    """Bad selection or unreadable manifests."""


def _field(data: dict, name: str, default=None):
    if name in data:
        return data[name]
    for table in ("addon", "metadata", "meta"):
        sub = data.get(table)
        if isinstance(sub, dict) and name in sub:
            return sub[name]
    return default


def read_manifests(manifest_dir: Path = MANIFEST_DIR) -> list[dict]:
    """Return ``[{"id", "title", "experimental"}]`` sorted with stable add-ons first."""
    manifest_dir = Path(manifest_dir)
    if not manifest_dir.is_dir():
        raise MatrixError(f"Add-on manifest folder not found: {manifest_dir}")
    out: list[dict] = []
    seen: set[str] = set()
    for path in sorted(manifest_dir.glob("*.toml")):
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as e:
            raise MatrixError(f"Cannot read add-on manifest {path.name}: {e}") from e
        addon_id = str(_field(data, "id", path.stem)).strip()
        if not addon_id:
            raise MatrixError(f"Add-on manifest {path.name} has an empty id")
        if addon_id in seen:
            raise MatrixError(f"Duplicate add-on id {addon_id!r} ({path.name})")
        seen.add(addon_id)
        out.append(
            {
                "id": addon_id,
                "title": str(_field(data, "title", addon_id)),
                "experimental": bool(_field(data, "experimental", False)),
            }
        )
    out.sort(key=lambda m: (m["experimental"], m["id"]))
    return out


def select(manifests: list[dict], selection: str) -> list[dict]:
    """Filter by ``all`` or a comma/space separated list of ids (unknown ids are an error)."""
    wanted = [s for s in selection.replace(",", " ").split() if s]
    if not wanted or wanted == ["all"]:
        if not manifests:
            raise MatrixError("No add-on manifests found - nothing to test.")
        return list(manifests)
    by_id = {m["id"]: m for m in manifests}
    unknown = [w for w in wanted if w not in by_id]
    if unknown:
        known = ", ".join(sorted(by_id)) or "(none)"
        raise MatrixError(f"Unknown add-on id(s): {', '.join(unknown)}. Known: {known}")
    return [by_id[w] for w in dict.fromkeys(wanted)]


def build_matrix(selection: str = "all", manifest_dir: Path = MANIFEST_DIR) -> dict:
    return {"include": select(read_manifests(manifest_dir), selection)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Add-on CI matrix from the manifests.")
    ap.add_argument("selection", nargs="?", default="all", help='"all" or add-on ids (comma separated)')
    ap.add_argument("--manifests", type=Path, default=MANIFEST_DIR, help="manifest folder")
    ap.add_argument("--github-output", type=Path, default=None, help="append matrix=<json> to this file")
    args = ap.parse_args(argv)
    try:
        matrix = build_matrix(args.selection, args.manifests)
    except MatrixError as e:
        sys.stderr.write(f"::error::{e}\n")
        return 2
    text = json.dumps(matrix, separators=(",", ":"))
    sys.stdout.write(json.dumps(matrix, indent=2) + "\n")
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as fh:
            fh.write(f"matrix={text}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
