#!/usr/bin/env python3
"""Pin the model registry: resolve exact upstream revisions, file names, sizes and
checksums and write them to ``src/vrphoto/models/models.lock.json``.

The registry (``vrphoto.models.registry``) overlays this lock file at import time, so
pins can be refreshed without code edits.  Run where huggingface.co / api.github.com are
reachable (CI job)::

    python scripts/pin_models.py                 # refresh the lock file
    python scripts/pin_models.py --check         # exit 1 if upstream differs from the lock
    python scripts/pin_models.py --models da2-small,lama
    python scripts/pin_models.py --lock other.json

Hugging Face models: ``GET /api/models/<repo>/revision/<rev>`` gives the commit sha, then
``GET /api/models/<repo>/tree/<sha>/<dir>`` lists files with sizes and LFS sha256.  A file
named in the registry that does not exist upstream is resolved to the single ``.onnx``
in the same folder (the registry carries TODO guesses for some community exports);
external-data siblings (``<file>_data`` / ``<file>.data``) are added automatically.
GitHub release assets: ``GET /repos/<owner>/<repo>/releases/tags/<tag>`` gives the size
and (for assets uploaded since 2025) a ``sha256:`` digest; otherwise the registry's md5
is kept.

Environment: ``HF_TOKEN`` (gated repos), ``GITHUB_TOKEN`` (API rate limit), ``HF_ENDPOINT``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote, urlparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from vrphoto.models.registry import (  # noqa: E402
    BASE_MODELS,
    LOCK_FILE_NAME,
    LOCK_SCHEMA,
    ModelSpec,
    hf_endpoint,
)

log = logging.getLogger("pin_models")
DEFAULT_LOCK = ROOT / "src" / "vrphoto" / "models" / LOCK_FILE_NAME
GITHUB_API = "https://api.github.com"

Fetch = Callable[[str], Any]


class PinError(Exception):
    """A model could not be pinned (message is human readable)."""


# --------------------------------------------------------------------------- HTTP
def make_fetch(hf_token: str | None = None, gh_token: str | None = None, timeout: float = 30.0) -> Fetch:
    """JSON GET with the right token per host."""
    hf_host = urlparse(hf_endpoint()).hostname

    def fetch(url: str) -> Any:
        headers = {"User-Agent": "vrphoto-pin-models", "Accept": "application/json"}
        host = urlparse(url).hostname
        if hf_token and host == hf_host:
            headers["Authorization"] = f"Bearer {hf_token}"
        if gh_token and host == "api.github.com":
            headers["Authorization"] = f"Bearer {gh_token}"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise PinError(f"GET {url} -> HTTP {e.code}") from e
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise PinError(f"GET {url} failed: {e}") from e

    return fetch


# --------------------------------------------------------------------------- Hugging Face
def hf_revision_sha(repo: str, revision: str, fetch: Fetch) -> str:
    data = fetch(f"{hf_endpoint()}/api/models/{repo}/revision/{quote(revision, safe='')}")
    sha = data.get("sha") if isinstance(data, dict) else None
    if not isinstance(sha, str) or len(sha) < 7:
        raise PinError(f"{repo}: no commit sha for revision {revision}")
    return sha


def hf_tree(repo: str, sha: str, folder: str, fetch: Fetch) -> dict[str, dict]:
    """Files directly inside ``folder`` at commit ``sha`` -> {path: entry}."""
    url = f"{hf_endpoint()}/api/models/{repo}/tree/{sha}"
    if folder:
        url += "/" + quote(folder, safe="/")
    data = fetch(url)
    if not isinstance(data, list):
        raise PinError(f"{repo}: unexpected tree listing for '{folder}'")
    return {e["path"]: e for e in data if isinstance(e, dict) and e.get("type") == "file" and "path" in e}


def _sha256_of(entry: dict) -> str | None:
    lfs = entry.get("lfs")
    if isinstance(lfs, dict):
        oid = lfs.get("oid") or lfs.get("sha256")
        if isinstance(oid, str) and len(oid) == 64:
            return oid.lower()
    return None  # plain git blob: only a sha1 is published -> recorded on first download


def _size_of(entry: dict) -> int:
    lfs = entry.get("lfs")
    if isinstance(lfs, dict) and isinstance(lfs.get("size"), int):
        return int(lfs["size"])
    return int(entry.get("size", 0))


def _folder(path: str) -> str:
    return path.rsplit("/", 1)[0] if "/" in path else ""


def resolve_entry(want: str, listing: dict[str, dict], model_id: str) -> str:
    """Return ``want`` if listed, else the single ``.onnx`` in its folder (TODO guesses)."""
    if want in listing:
        return want
    folder = _folder(want)
    cands = sorted(p for p in listing if p.endswith(".onnx") and _folder(p) == folder)
    if len(cands) == 1:
        log.warning("%s: '%s' not found upstream, using '%s'", model_id, want, cands[0])
        return cands[0]
    stem_hits = [c for c in cands if PurePosixPath(c).stem.replace("_", "-") in model_id]
    if len(stem_hits) == 1:
        return stem_hits[0]
    raise PinError(f"{model_id}: '{want}' not found upstream; candidates: {cands or 'none'}")


def external_data(entry: str, listing: dict[str, dict]) -> list[str]:
    """Sibling external-data files of an ONNX file (``model.onnx_data`` etc.)."""
    return sorted(p for p in (entry + "_data", entry + ".data") if p in listing)


def pin_hf(spec: ModelSpec, fetch: Fetch) -> dict[str, Any]:
    sha = hf_revision_sha(spec.repo, "main" if len(spec.revision) < 40 else spec.revision, fetch)
    folders = {_folder(f) for f in spec.files}
    if spec.fp16_file:
        folders.add(_folder(spec.fp16_file))
    listing: dict[str, dict] = {}
    for folder in sorted(folders):
        listing.update(hf_tree(spec.repo, sha, folder, fetch))
    entry = resolve_entry(spec.files[0], listing, spec.id)
    default = [entry, *external_data(entry, listing)]
    for extra in spec.files[1:]:
        if extra in listing and extra not in default:
            default.append(extra)
    fp16 = spec.fp16_file if spec.fp16_file and spec.fp16_file in listing else None
    fp16_data = external_data(fp16, listing) if fp16 else []
    # fp16 external data rides in ``files`` (download.variant_files splits the variants)
    files = default + [f for f in fp16_data if f not in default]
    pinned = files + ([fp16] if fp16 else [])
    sizes = {f: _size_of(listing[f]) for f in pinned}
    sums = {f: s for f in pinned if (s := _sha256_of(listing[f]))}
    return {
        "revision": sha,
        "files": files,
        "sizes": dict(sorted(sizes.items())),
        "sha256": dict(sorted(sums.items())),
        "size_mb": round(sum(sizes[f] for f in default) / 1e6, 1),
        "fp16_file": fp16,
    }


# --------------------------------------------------------------------------- GitHub releases
def pin_github(spec: ModelSpec, fetch: Fetch) -> dict[str, Any]:
    u = urlparse(spec.url or "")
    parts = u.path.strip("/").split("/")
    # /<owner>/<repo>/releases/download/<tag>/<asset>
    if u.hostname != "github.com" or len(parts) != 6 or parts[2:4] != ["releases", "download"]:
        raise PinError(f"{spec.id}: not a GitHub release URL: {spec.url}")
    owner, repo, tag, asset = parts[0], parts[1], parts[4], parts[5]
    data = fetch(f"{GITHUB_API}/repos/{owner}/{repo}/releases/tags/{quote(tag, safe='')}")
    assets = {a.get("name"): a for a in (data.get("assets") or [])} if isinstance(data, dict) else {}
    a = assets.get(asset)
    if a is None:
        raise PinError(f"{spec.id}: asset '{asset}' not in release {owner}/{repo}@{tag}")
    f = spec.files[0]
    out: dict[str, Any] = {"revision": tag, "sizes": {f: int(a.get("size", 0))}}
    digest = a.get("digest")
    if isinstance(digest, str) and digest.startswith("sha256:") and len(digest) == 71:
        out["sha256"] = {f: digest[7:].lower()}
    elif spec.sha256.get(f):
        out["sha256"] = {f: spec.sha256[f]}
    out["size_mb"] = round(int(a.get("size", 0)) / 1e6, 1)
    return out


# --------------------------------------------------------------------------- lock document
def pin_table(spec: ModelSpec) -> dict[str, Any]:
    """The registry table's own pins of a URL model that is already pinned there and is not a
    GitHub release (e.g. YuNet: a Git LFS file at an exact opencv_zoo commit); nothing to resolve."""
    out: dict[str, Any] = {"revision": spec.revision, "sha256": dict(spec.sha256)}
    if spec.sizes:
        out["sizes"] = dict(spec.sizes)
    if spec.size_mb:
        out["size_mb"] = spec.size_mb
    return out


def pin_spec(spec: ModelSpec, fetch: Fetch) -> dict[str, Any]:
    if spec.url:
        if spec.pinned and "/releases/download/" not in spec.url:
            return pin_table(spec)
        return pin_github(spec, fetch)
    if spec.repo:
        return pin_hf(spec, fetch)
    raise PinError(f"{spec.id}: no repo or url")


def build_lock(
    specs: dict[str, ModelSpec], fetch: Fetch, *, ids: list[str] | None = None, old: dict | None = None
) -> tuple[dict[str, Any], list[str]]:
    """Return ``(lock_document, errors)``; models that fail keep their ``old`` entry."""
    old_models = (old or {}).get("models") or {}
    models: dict[str, Any] = dict(old_models) if ids else {}
    errors: list[str] = []
    for mid in ids or list(specs):
        spec = specs.get(mid)
        if spec is None:
            errors.append(f"unknown model id '{mid}'")
            continue
        try:
            models[mid] = pin_spec(spec, fetch)
            log.info("pinned %s @ %s", mid, models[mid].get("revision"))
        except PinError as e:
            errors.append(str(e))
            if mid in old_models:
                models[mid] = old_models[mid]
    doc = {
        "schema": LOCK_SCHEMA,
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "generator": "scripts/pin_models.py",
        "note": "Exact model pins overlaid onto models/registry.py. Regenerate with scripts/pin_models.py.",
        "models": dict(sorted(models.items())),
    }
    return doc, errors


def diff_locks(old: dict, new: dict) -> list[str]:
    """Human-readable differences between two lock documents (ignores timestamps)."""
    out: list[str] = []
    om, nm = (old or {}).get("models") or {}, (new or {}).get("models") or {}
    for mid in sorted(set(om) | set(nm)):
        a, b = om.get(mid), nm.get(mid)
        if a == b:
            continue
        if a is None:
            out.append(f"{mid}: missing from lock")
            continue
        if b is None:
            out.append(f"{mid}: no longer pinned upstream")
            continue
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                out.append(f"{mid}.{k}: {a.get(k)!r} -> {b.get(k)!r}")
    return out


def write_lock(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None, *, fetch: Fetch | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--lock", type=Path, default=DEFAULT_LOCK, help="lock file path")
    ap.add_argument("--check", action="store_true", help="fail if upstream differs from the lock")
    ap.add_argument("--models", default="", help="comma-separated model ids (default: all)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    fetch = fetch or make_fetch(os.environ.get("HF_TOKEN"), os.environ.get("GITHUB_TOKEN"))
    ids = [s.strip() for s in args.models.split(",") if s.strip()] or None
    old: dict = {}
    if args.lock.is_file():
        try:
            old = json.loads(args.lock.read_text(encoding="utf-8"))
        except ValueError as e:
            sys.stderr.write(f"warning: cannot parse {args.lock}: {e}\n")
    new, errors = build_lock(BASE_MODELS, fetch, ids=ids, old=old)
    for e in errors:
        sys.stderr.write(f"error: {e}\n")
    if args.check:
        if ids:
            new["models"] = {k: v for k, v in new["models"].items() if k in ids}
            old = {"models": {k: v for k, v in (old.get("models") or {}).items() if k in ids}}
        diffs = diff_locks(old, new)
        for d in diffs:
            sys.stdout.write(f"DIFF {d}\n")
        if diffs or errors:
            sys.stdout.write(f"{len(diffs)} difference(s), {len(errors)} error(s)\n")
            return 1
        sys.stdout.write("lock file is up to date\n")
        return 0
    write_lock(new, args.lock)
    sys.stdout.write(f"wrote {args.lock} ({len(new['models'])} models, {len(errors)} error(s))\n")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
