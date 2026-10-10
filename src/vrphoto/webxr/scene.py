"""Scene library for the WebXR viewer: export, validate, list and delete viewer scenes.

A *scene* is a folder ``<library>/<id>/`` holding ``scene.json`` (the manifest described in
``webxr/viewer/scene.schema.json`` and docs/CONTRACTS.md section 8), its asset files (meshes,
images, videos, splats) and an optional ``thumb.jpg``.  The viewer server exposes the library
at ``/scenes/<id>/...`` and the list from :func:`list_scenes` at ``/api/scenes``.

:func:`validate_manifest` is a dependency-free re-implementation of the JSON schema (the
``jsonschema`` package is a development-only dependency and is not bundled in the app); the
tests cross-check both on good and bad manifests.  It is slightly stricter than the schema in
one respect: every file a manifest references must be a plain file name inside the scene folder.
"""

from __future__ import annotations

import copy
import json
import os
import re
import secrets
import shutil
import stat
import sys
import unicodedata
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from ..core.errors import InputError
from ..core.log import get_logger
from ..core.types import RGB

log = get_logger(__name__)

SCHEMA_ID = "vrphoto.scene/1"
MANIFEST_NAME = "scene.json"
THUMB_NAME = "thumb.jpg"
THUMB_MAX_PX = 512
MODES = ("photo3d", "stereo", "sphere", "object", "splat")

_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,80}")
_SLUG_MAX = 48
_SUFFIX_BYTES = 3  # 6 hex chars
_RESERVED_ASSETS = {MANIFEST_NAME}
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

# --------------------------------------------------------------------------- schema (mirror)
# Kept in sync with webxr/viewer/scene.schema.json (tests compare the two).
_STR = ("string",)
_NUM = ("number",)

# block -> (required keys, {key: spec}); spec = ("type", ...) | ("enum", values) | ("posnum",)
#   | ("strarray", min_items) | ("str_or_null",)
_BLOCKS: dict[str, tuple[tuple[str, ...], dict[str, tuple]]] = {
    "photo3d": (
        ("meshes",),
        {"meshes": ("strarray", 1), "anchor_distance_m": ("posnum",), "eye_height_m": _NUM},
    ),
    "stereo": (
        ("media", "kind", "layout"),
        {
            "media": _STR,
            "kind": ("enum", ("image", "video")),
            "layout": ("enum", ("sbs", "tb", "mono")),
            "aspect": ("posnum",),
            "screen_width_m": ("posnum",),
            "distance_m": ("posnum",),
        },
    ),
    "sphere": (
        ("media", "kind", "coverage", "layout"),
        {
            "media": _STR,
            "kind": ("enum", ("image", "video")),
            "coverage": ("enum", (180, 360)),
            "layout": ("enum", ("mono", "sbs", "tb")),
            "depth": ("str_or_null",),
            "radius_m": ("posnum",),
            "depth_near_m": ("posnum",),
            "depth_far_m": ("posnum",),
        },
    ),
    "object": (
        ("asset", "format"),
        {
            "asset": _STR,
            "format": ("enum", ("glb", "ply", "splat", "spz")),
            "height_m": ("posnum",),
            "pivot": ("enum", ("base", "center")),
            "placement": ("enum", ("hit-test", "front-floor")),
            "distance_m": ("posnum",),
            "alt_assets": ("strarray", 0),
            "up": ("enum", ("y", "-y", "z")),
        },
    ),
    "splat": (
        ("asset", "format"),
        {
            "asset": _STR,
            "format": ("enum", ("ply", "splat", "spz")),
            "up": ("enum", ("y", "-y", "z")),
            "scale": ("posnum",),
        },
    ),
}

#: Manifest fields that name files inside the scene folder: block -> keys (lists allowed).
_FILE_FIELDS: dict[str, tuple[str, ...]] = {
    "photo3d": ("meshes",),
    "stereo": ("media",),
    "sphere": ("media", "depth"),
    "object": ("asset", "alt_assets"),
    "splat": ("asset",),
}


def _is_json_number(v: Any) -> bool:
    """JSON-schema "number": ints and floats, never booleans."""
    return isinstance(v, int | float) and not isinstance(v, bool)


def _enum_ok(v: Any, values: tuple) -> bool:
    """JSON-schema ``enum`` equality (1 == 1.0, but True != 1 and "180" != 180)."""
    for x in values:
        if isinstance(x, str):
            if isinstance(v, str) and v == x:
                return True
        elif isinstance(x, bool) or isinstance(v, bool):
            if isinstance(x, bool) and isinstance(v, bool) and v == x:
                return True
        elif _is_json_number(v) and v == x:
            return True
    return False


def _fmt_enum(values: tuple) -> str:
    return ", ".join(str(v) for v in values)


def _check_value(path: str, value: Any, spec: tuple, errors: list[str]) -> None:
    kind = spec[0]
    if kind == "string":
        if not isinstance(value, str):
            errors.append(f"{path} must be text")
    elif kind == "number":
        if not _is_json_number(value):
            errors.append(f"{path} must be a number")
    elif kind == "posnum":
        if not _is_json_number(value) or not value > 0:
            errors.append(f"{path} must be a number greater than 0")
    elif kind == "enum":
        if not _enum_ok(value, spec[1]):
            errors.append(f"{path} must be one of: {_fmt_enum(spec[1])} (got {value!r})")
    elif kind == "strarray":
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            errors.append(f"{path} must be a list of file names")
        elif len(value) < spec[1]:
            errors.append(f"{path} must list at least {spec[1]} file(s)")
    elif kind == "str_or_null":
        if value is not None and not isinstance(value, str):
            errors.append(f"{path} must be a file name or null")
    else:  # pragma: no cover - programming error
        raise AssertionError(kind)


def is_safe_asset_name(name: Any) -> bool:
    """True if ``name`` is a plain file name that cannot escape the scene folder."""
    if not isinstance(name, str) or not name or len(name) > 200:
        return False
    if name in (".", "..") or name.startswith("..") or name != name.strip():
        return False
    if any(c in name for c in '/\\:*?"<>|') or any(ord(c) < 32 for c in name):
        return False
    if name.endswith(".") or name.split(".")[0].lower() in _WINDOWS_RESERVED:
        return False
    return True


def _manifest_errors(manifest: Any) -> list[str]:
    if not isinstance(manifest, Mapping):
        return ["the manifest must be a JSON object"]
    errors: list[str] = []
    for key in ("schema", "id", "title", "mode"):
        if key not in manifest:
            errors.append(f"'{key}' is missing")
    if "schema" in manifest and manifest["schema"] != SCHEMA_ID:
        errors.append(f"schema must be {SCHEMA_ID!r} (got {manifest['schema']!r})")
    if "id" in manifest:
        sid = manifest["id"]
        if not isinstance(sid, str) or not _ID_RE.fullmatch(sid):
            errors.append(
                "id must be 1-81 characters of lowercase letters, digits, '.', '_' or '-', starting with a letter or digit"
            )
    if "title" in manifest:
        title = manifest["title"]
        if not isinstance(title, str):
            errors.append("title must be text")
        elif len(title) > 200:
            errors.append("title must be at most 200 characters")
    for key in ("created", "thumbnail", "source"):
        if key in manifest and not isinstance(manifest[key], str):
            errors.append(f"{key} must be text")
    if "units" in manifest and manifest["units"] != "m":
        errors.append("units must be 'm' (metres)")
    mode = manifest.get("mode")
    if "mode" in manifest and not (isinstance(mode, str) and mode in MODES):
        errors.append(f"mode must be one of: {', '.join(MODES)} (got {mode!r})")
    for block, (required, props) in _BLOCKS.items():
        if block not in manifest:
            if mode == block:
                errors.append(f"mode '{block}' needs a '{block}' section")
            continue
        value = manifest[block]
        if not isinstance(value, Mapping):
            errors.append(f"'{block}' must be an object")
            continue
        for key in required:
            if key not in value:
                errors.append(f"{block}.{key} is missing")
        for key, spec in props.items():
            if key in value:
                _check_value(f"{block}.{key}", value[key], spec, errors)
    env = manifest.get("env")
    if "env" in manifest:
        if not isinstance(env, Mapping):
            errors.append("'env' must be an object")
        else:
            if "prefer" in env and not _enum_ok(env["prefer"], ("ar", "vr")):
                errors.append(f"env.prefer must be one of: ar, vr (got {env['prefer']!r})")
            if "background" in env and not isinstance(env["background"], str):
                errors.append("env.background must be text")
    if "notices" in manifest:
        notes = manifest["notices"]
        if not isinstance(notes, list) or not all(isinstance(n, str) for n in notes):
            errors.append("notices must be a list of text")
    return errors


def _referenced_files(manifest: Mapping[str, Any]) -> list[tuple[str, str]]:
    """(field path, file name) for every file the manifest references."""
    refs: list[tuple[str, str]] = []
    thumb = manifest.get("thumbnail")
    if isinstance(thumb, str):
        refs.append(("thumbnail", thumb))
    for block, keys in _FILE_FIELDS.items():
        b = manifest.get(block)
        if not isinstance(b, Mapping):
            continue
        for key in keys:
            v = b.get(key)
            if isinstance(v, str):
                refs.append((f"{block}.{key}", v))
            elif isinstance(v, list):
                refs.extend((f"{block}.{key}[{i}]", x) for i, x in enumerate(v) if isinstance(x, str))
    return refs


def validate_manifest(manifest: dict) -> None:
    """Validate a scene manifest; raises :class:`InputError` listing every problem found.

    Pure Python (mirrors ``scene.schema.json``) plus one extra rule: referenced files must be
    plain names inside the scene folder (no ``/``, ``\\``, ``..`` or drive letters).
    """
    errors = _manifest_errors(manifest)
    if isinstance(manifest, Mapping):
        for path, name in _referenced_files(manifest):
            if not is_safe_asset_name(name):
                errors.append(f"{path} must be a plain file name inside the scene folder (got {name!r})")
    if errors:
        err = InputError(
            "The VR scene description is invalid: " + "; ".join(errors) + ".",
            hint="This is usually a bug in the exporter - please report it with the log file.",
        )
        err.problems = errors  # type: ignore[attr-defined]
        raise err


# --------------------------------------------------------------------------- ids
def _slugify(title: str) -> str:
    text = unicodedata.normalize("NFKD", str(title or "")).encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    if len(slug) > _SLUG_MAX:
        slug = slug[:_SLUG_MAX].rstrip("-")
    return slug or "scene"


def new_scene_id(title: str) -> str:
    """A URL-safe id like ``beach-at-dusk-3f9a1c`` (slug of ``title`` + random suffix)."""
    sid = f"{_slugify(title)}-{secrets.token_hex(_SUFFIX_BYTES)}"
    assert _ID_RE.fullmatch(sid), sid
    return sid


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _library(library: Path | None) -> Path:
    if library is not None:
        lib = Path(library)
        lib.mkdir(parents=True, exist_ok=True)
        return lib
    from ..core.config import get_settings

    return get_settings().library_dir


# --------------------------------------------------------------------------- file helpers
def _temp_name(dst: Path) -> Path:
    """Hidden sibling temp file (normal umask permissions, unlike tempfile.mkstemp's 0600)."""
    return dst.with_name(f".{dst.name}.{secrets.token_hex(4)}.tmp")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = _temp_name(path)
    try:
        with open(tmp, "xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_copy(src: Path, dst: Path) -> None:
    tmp = _temp_name(dst)
    try:
        shutil.copyfile(src, tmp)
        _replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _replace(tmp: str | Path, dst: Path) -> None:
    """os.replace with a short retry (Windows: the viewer server may be reading the old file)."""
    import time

    for attempt in range(10):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.05 * (attempt + 1))


def _to_rgb_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr.astype(np.float32), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        raise InputError(f"Thumbnail must be an HxWx3 RGB image (got shape {arr.shape}).")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise InputError("Thumbnail image is empty.")
    return np.ascontiguousarray(arr)


def _thumbnail_jpeg(img: RGB) -> bytes:
    import io

    from PIL import Image

    pil = Image.fromarray(_to_rgb_uint8(img), "RGB")
    pil.thumbnail((THUMB_MAX_PX, THUMB_MAX_PX), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, "JPEG", quality=85, optimize=True)
    return buf.getvalue()


# --------------------------------------------------------------------------- public API
def export_scene(
    manifest: dict,
    assets: Mapping[str, Path | bytes],
    *,
    library: Path | None = None,
    thumbnail: RGB | None = None,
) -> Path:
    """Write a viewer scene to ``<library>/<id>/`` and return that folder.

    ``manifest`` is completed with defaults (``schema``, ``id`` from the title, ``created``,
    ``units``, ``thumbnail`` when one is given), validated, and written last and atomically as
    ``scene.json`` - a half-written scene is never listed.  ``assets`` maps plain file names to
    a source ``Path`` (copied) or ``bytes`` (written).  ``thumbnail`` (RGB array) is saved as
    ``thumb.jpg`` with its long side limited to 512 px.  Re-exporting an existing id replaces
    the scene's files.  ``library`` defaults to ``Settings.library_dir``.
    """
    if not isinstance(manifest, Mapping):
        raise InputError("The scene description must be a dictionary.")
    m: dict[str, Any] = copy.deepcopy(dict(manifest))
    m.setdefault("schema", SCHEMA_ID)
    if not isinstance(m.get("title"), str) or not m["title"].strip():
        m["title"] = "Untitled scene"
    m["title"] = m["title"].strip()[:200]
    if not m.get("id"):
        m["id"] = new_scene_id(m["title"])
    m.setdefault("created", _now_iso())
    m.setdefault("units", "m")

    if not isinstance(assets, Mapping):
        raise InputError("Scene assets must be a mapping of file name to path or bytes.")
    folded: dict[str, str] = {}  # Windows / macOS file names are case-insensitive
    for name, src in assets.items():
        if not is_safe_asset_name(name):
            raise InputError(
                f"Scene file name {name!r} is not allowed: use a plain name without folders, '..' or special characters."
            )
        key = name.casefold()
        if key in {r.casefold() for r in _RESERVED_ASSETS}:
            raise InputError(f"'{name}' is reserved for the scene description and cannot be an asset.")
        if thumbnail is not None and key == THUMB_NAME.casefold():
            raise InputError(
                f"'{THUMB_NAME}' is written from the thumbnail image; do not pass it as an asset too."
            )
        if key in folded:
            raise InputError(
                f"Scene files {folded[key]!r} and {name!r} differ only in upper/lower case, which "
                "Windows treats as the same file. Rename one of them."
            )
        folded[key] = name
        if isinstance(src, bytes | bytearray | memoryview):
            continue
        if not isinstance(src, str | os.PathLike):
            raise InputError(f"Scene asset {name!r} must be a file path or bytes (got {type(src).__name__}).")
        if not Path(src).is_file():
            raise InputError(f"Scene asset {name!r} not found: {src}")
    if thumbnail is not None:
        m["thumbnail"] = THUMB_NAME

    validate_manifest(m)

    try:
        manifest_bytes = json.dumps(m, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as e:
        raise InputError(f"The VR scene description contains a value that cannot be saved: {e}") from e

    lib = _library(library)
    scene_dir = lib / m["id"]
    present = set(assets) | ({THUMB_NAME} if thumbnail is not None else set())
    existing = {p.name for p in scene_dir.iterdir() if p.is_file()} if scene_dir.is_dir() else set()
    missing = [f"{path} ({name})" for path, name in _referenced_files(m) if name not in present | existing]
    if missing:
        raise InputError("The scene refers to files that were not provided: " + ", ".join(missing) + ".")

    scene_dir.mkdir(parents=True, exist_ok=True)
    try:
        for name, src in assets.items():
            dst = scene_dir / name
            if isinstance(src, bytes | bytearray | memoryview):
                _atomic_write_bytes(dst, bytes(src))
            else:
                src_path = Path(src)
                if dst.exists() and os.path.samefile(src_path, dst):
                    continue
                _atomic_copy(src_path, dst)
        if thumbnail is not None:
            _atomic_write_bytes(scene_dir / THUMB_NAME, _thumbnail_jpeg(thumbnail))
        _atomic_write_bytes(scene_dir / MANIFEST_NAME, manifest_bytes)
    except OSError as e:
        raise InputError(f"Could not write the VR scene to {scene_dir}: {e}") from e
    log.info("Exported %s scene %r to %s", m["mode"], m["id"], scene_dir)
    return scene_dir


def _parse_time(text: Any) -> float | None:
    if not isinstance(text, str) or not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


def read_manifest(scene_dir: Path) -> dict:
    """Read and validate ``<scene_dir>/scene.json`` (raises InputError)."""
    path = Path(scene_dir) / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise InputError(f"No scene found in {scene_dir}") from e
    except (OSError, ValueError) as e:
        raise InputError(f"The scene file {path} is damaged: {e}") from e
    validate_manifest(manifest)
    return manifest


def list_scenes(library: Path | None = None) -> list[dict]:
    """Scenes in the library, newest first: ``[{id, title, mode, thumbnail, url, created}]``.

    ``url`` is ``/view.html?scene=<id>``; ``thumbnail`` is ``/scenes/<id>/thumb.jpg`` (or the
    manifest's thumbnail file) when it exists, else ``None``.  Folders without a valid
    ``scene.json`` (e.g. an export still in progress) are skipped.
    """
    lib = _library(library)
    items: list[tuple[float, dict]] = []
    try:
        entries = list(lib.iterdir())
    except OSError as e:
        log.warning("Cannot read scene library %s: %s", lib, e)
        return []
    for d in entries:
        if not d.is_dir() or d.name.startswith(".") or not _ID_RE.fullmatch(d.name):
            continue
        if not (d / MANIFEST_NAME).is_file():
            continue
        try:
            m = read_manifest(d)
        except InputError as e:
            log.warning("Skipping scene folder %s: %s", d.name, e)
            continue
        sid = d.name
        if m.get("id") != sid:
            log.debug("Scene folder %s has manifest id %r; using the folder name", sid, m.get("id"))
        thumb_name = m.get("thumbnail") if isinstance(m.get("thumbnail"), str) else THUMB_NAME
        thumb = None
        for name in dict.fromkeys([thumb_name, THUMB_NAME]):
            if is_safe_asset_name(name) and (d / name).is_file():
                thumb = f"/scenes/{quote(sid)}/{quote(name)}"
                break
        created = m.get("created")
        ts = _parse_time(created)
        if ts is None:
            try:
                ts = (d / MANIFEST_NAME).stat().st_mtime
            except OSError:
                ts = 0.0
            created = datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        items.append(
            (
                ts,
                {
                    "id": sid,
                    "title": m.get("title") or sid,
                    "mode": m["mode"],
                    "thumbnail": thumb,
                    "url": f"/view.html?scene={quote(sid)}",
                    "created": created,
                },
            )
        )
    items.sort(key=lambda t: (t[0], t[1]["id"]), reverse=True)
    return [item for _, item in items]


def _on_rm_error(func, path, exc) -> None:  # noqa: ANN001 - shutil callback signature
    """Clear the read-only flag (Windows) and retry once."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        raise exc if isinstance(exc, BaseException) else exc[1] from None


def delete_scene(scene_id: str, library: Path | None = None) -> None:
    """Delete a scene folder.  Deleting a scene that does not exist is not an error."""
    if not isinstance(scene_id, str) or not _ID_RE.fullmatch(scene_id):
        raise InputError(f"Invalid scene id: {scene_id!r}")
    lib = _library(library)
    target = lib / scene_id
    if not target.exists():
        log.info("Scene %r already deleted", scene_id)
        return
    if target.is_symlink() or not target.is_dir() or target.resolve().parent != lib.resolve():
        raise InputError(f"Refusing to delete {target}: it is not a scene folder in the library.")
    if not (target / MANIFEST_NAME).is_file() and any(target.iterdir()):
        raise InputError(
            f"Refusing to delete {target}: it does not look like a VR scene (no {MANIFEST_NAME})."
        )
    try:
        if sys.version_info >= (3, 12):
            shutil.rmtree(target, onexc=_on_rm_error)
        else:  # pragma: no cover - Python 3.11
            shutil.rmtree(target, onerror=_on_rm_error)
    except OSError as e:
        raise InputError(
            f"Could not delete the scene {scene_id!r}: {e}",
            hint="Close any program that has files from this scene open (or stop the viewer) and try again.",
        ) from e
    log.info("Deleted scene %r", scene_id)


__all__ = [
    "SCHEMA_ID",
    "MANIFEST_NAME",
    "THUMB_NAME",
    "new_scene_id",
    "export_scene",
    "validate_manifest",
    "list_scenes",
    "delete_scene",
    "read_manifest",
    "is_safe_asset_name",
]
