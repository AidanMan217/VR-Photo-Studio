"""Binary glTF 2.0 (GLB) and Wavefront OBJ export of textured meshes, written from scratch.

:func:`write_glb` produces a self-contained ``.glb``:

* one node, one mesh (one triangle primitive) and one material per :class:`Mesh`;
* attributes ``POSITION`` (float32 VEC3 with exact ``min``/``max``), ``TEXCOORD_0``
  (float32 VEC2) and optional ``NORMAL`` (normalised float32 VEC3); ``uint32`` indices;
* materials use ``KHR_materials_unlit`` (photo textures carry their own lighting and must
  not be relit; listed in ``extensionsUsed`` - viewers without the extension fall back to
  plain PBR with metallic 0 / roughness 1), ``alphaMode``/``alphaCutoff``/``doubleSided``
  come from the mesh;
* the base-colour texture is embedded as JPEG, or PNG when the texture has an alpha
  channel or ``alpha_mode != "OPAQUE"`` (or ``texture_format="png"``), downscaled so its
  long side is <= ``max_texture``; identical texture arrays are embedded once;
* a single sampler: linear magnification, trilinear (linear-mipmap-linear)
  minification, clamp-to-edge wrapping;
* every buffer view starts on a 4-byte boundary, the JSON chunk is padded with spaces
  and the BIN chunk with zeros to a multiple of 4 bytes.

Degenerate triangles and unreferenced vertices are removed before writing (they would
only raise validator notices).  :func:`read_glb_meshes` is a small reader (GLB and
``.gltf`` with embedded / relative buffers) used by tests and for re-export.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import struct
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..core.errors import InputError, VRPhotoError
from ..core.log import get_logger
from ..core.types import CancelToken, Mesh, ProgressReporter

log = get_logger(__name__)

GLB_MAGIC = 0x46546C67  # b"glTF"
CHUNK_JSON = 0x4E4F534A  # b"JSON"
CHUNK_BIN = 0x004E4942  # b"BIN\0"

BYTE, UNSIGNED_BYTE, SHORT, UNSIGNED_SHORT, UNSIGNED_INT, FLOAT = 5120, 5121, 5122, 5123, 5125, 5126
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
LINEAR, LINEAR_MIPMAP_LINEAR, CLAMP_TO_EDGE = 9729, 9987, 33071
TRIANGLES = 4
UNLIT = "KHR_materials_unlit"
GENERATOR = "VR Photo Studio"

_JPEG_QUALITY = 90
_OBJ_MAX_TEXTURE = 8192

_COMPONENT_DTYPES = {
    BYTE: np.dtype("<i1"),
    UNSIGNED_BYTE: np.dtype("<u1"),
    SHORT: np.dtype("<i2"),
    UNSIGNED_SHORT: np.dtype("<u2"),
    UNSIGNED_INT: np.dtype("<u4"),
    FLOAT: np.dtype("<f4"),
}
_TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


class GltfError(VRPhotoError):
    """A glTF/GLB/OBJ file could not be written or parsed."""


# --------------------------------------------------------------------------- helpers
def _check(cancel: CancelToken | None) -> None:
    if cancel is not None:
        cancel.check()


def _as_reporter(progress: Any) -> ProgressReporter:
    if progress is None:
        return ProgressReporter.null()
    if isinstance(progress, ProgressReporter):
        return progress
    if callable(progress):
        return ProgressReporter(progress)
    raise InputError("progress must be a ProgressReporter or a callable(frac, msg).")


def _pad4(n: int) -> int:
    return (-n) % 4


def _texture_u8(tex: Any, name: str) -> np.ndarray:
    t = np.asarray(tex)
    if t.ndim == 2:
        t = np.repeat(t[..., None], 3, axis=2)
    if t.ndim != 3 or t.shape[2] not in (1, 3, 4) or t.shape[0] < 1 or t.shape[1] < 1:
        raise InputError(f"Mesh '{name}' has an invalid texture of shape {t.shape}; expected HxWx3 or HxWx4.")
    if t.shape[2] == 1:
        t = np.repeat(t, 3, axis=2)
    if t.dtype != np.uint8:
        if np.issubdtype(t.dtype, np.floating):
            t = (np.clip(np.nan_to_num(t), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        elif t.dtype == np.uint16:  # 16-bit texture: rescale, do not saturate
            t = ((t.astype(np.uint32) * 255 + 32767) // 65535).astype(np.uint8)
        else:
            t = np.clip(t, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(t)


def _limit_texture(t: np.ndarray, max_side: int) -> np.ndarray:
    h, w = t.shape[:2]
    if max(h, w) <= max_side:
        return t
    s = max_side / max(h, w)
    size = (max(1, min(max_side, int(round(w * s)))), max(1, min(max_side, int(round(h * s)))))
    return cv2.resize(t, size, interpolation=cv2.INTER_AREA)


def _encode_image(t: np.ndarray, fmt: str) -> bytes:
    buf = io.BytesIO()
    if fmt == "png":
        Image.fromarray(t, "RGBA" if t.shape[2] == 4 else "RGB").save(buf, "PNG", compress_level=6)
    else:
        Image.fromarray(np.ascontiguousarray(t[..., :3]), "RGB").save(
            buf, "JPEG", quality=_JPEG_QUALITY, subsampling=0, optimize=True
        )
    return buf.getvalue()


def _normalize_format(texture_format: str) -> str:
    f = str(texture_format).lower().strip(". ")
    if f in ("jpeg", "jpg"):
        return "jpeg"
    if f == "png":
        return "png"
    raise InputError(f"Unsupported texture format {texture_format!r}; use 'jpeg' or 'png'.")


def _wants_png(tex: np.ndarray, alpha_mode: str, fmt: str) -> bool:
    return fmt == "png" or tex.shape[2] == 4 or alpha_mode != "OPAQUE"


def _prepare(mesh: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Validate and clean one mesh: (vertices f32, faces u32, uvs f32, normals f32|None) or None if empty."""
    name = mesh.name
    v = np.asarray(mesh.vertices, dtype=np.float32)
    if v.ndim != 2 or v.shape[1] != 3:
        raise InputError(f"Mesh '{name}' vertices must be an Nx3 array (got shape {v.shape}).")
    uv = np.asarray(mesh.uvs, dtype=np.float32)
    if uv.shape != (len(v), 2):
        raise InputError(f"Mesh '{name}' UVs must be an Nx2 array matching the vertices (got {uv.shape}).")
    f = np.asarray(mesh.faces)
    if f.size == 0:
        return None
    if f.ndim != 2 or f.shape[1] != 3:
        raise InputError(f"Mesh '{name}' faces must be an Mx3 array (got shape {f.shape}).")
    f = f.astype(np.int64)
    if f.min() < 0 or f.max() >= len(v):
        raise InputError(f"Mesh '{name}' has face indices outside its {len(v)} vertices.")
    nrm = None
    if mesh.normals is not None:
        nrm = np.asarray(mesh.normals, dtype=np.float64)
        if nrm.shape != v.shape:
            raise InputError(f"Mesh '{name}' normals must match the vertices (got shape {nrm.shape}).")
    finite = np.isfinite(v).all(1) & np.isfinite(uv).all(1)
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2]) & finite[f].all(1)
    if not keep.all():
        dropped = int((~keep).sum())
        log.debug("Mesh '%s': dropping %d degenerate / non-finite faces", name, dropped)
        f = f[keep]
    if len(f) == 0:
        return None
    used = np.zeros(len(v), bool)
    used[f.ravel()] = True
    remap = np.cumsum(used, dtype=np.int64) - 1
    f = remap[f]
    v, uv = v[used], uv[used]
    if nrm is not None:
        nrm = nrm[used]
        ln = np.linalg.norm(nrm, axis=1)
        bad = ~np.isfinite(ln) | (ln < 1e-12)
        nrm[bad] = (0.0, 0.0, 1.0)
        ln[bad] = 1.0
        nrm = (nrm / ln[:, None]).astype(np.float32)
    return np.ascontiguousarray(v), f.astype(np.uint32), np.ascontiguousarray(uv), nrm


class _Builder:
    """Accumulates the binary buffer (4-byte aligned views) and the JSON objects."""

    def __init__(self) -> None:
        self.parts: list[bytes] = []
        self.length = 0
        self.views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []

    def view(self, data: bytes, target: int | None = None) -> int:
        pad = _pad4(self.length)
        if pad:
            self.parts.append(b"\x00" * pad)
            self.length += pad
        bv: dict[str, Any] = {"buffer": 0, "byteOffset": self.length, "byteLength": len(data)}
        if target is not None:
            bv["target"] = target
        self.parts.append(data)
        self.length += len(data)
        self.views.append(bv)
        return len(self.views) - 1

    def accessor(self, arr: np.ndarray, ctype: int, atype: str, target: int, minmax: bool = False) -> int:
        dt = _COMPONENT_DTYPES[ctype]
        data = np.ascontiguousarray(arr, dtype=dt)
        acc: dict[str, Any] = {
            "bufferView": self.view(data.tobytes(), target),
            "componentType": ctype,
            "count": int(data.shape[0]),
            "type": atype,
        }
        if minmax:
            flat = data.reshape(data.shape[0], -1)
            acc["min"] = [float(x) for x in flat.min(axis=0)]
            acc["max"] = [float(x) for x in flat.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def binary(self) -> bytes:
        pad = _pad4(self.length)
        return b"".join(self.parts) + b"\x00" * pad


def _glb_bytes(doc: dict[str, Any], binary: bytes) -> bytes:
    js = json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    js += b" " * _pad4(len(js))
    binary = binary + b"\x00" * _pad4(len(binary))
    total = 12 + 8 + len(js) + (8 + len(binary) if binary else 0)
    out = [struct.pack("<III", GLB_MAGIC, 2, total), struct.pack("<II", len(js), CHUNK_JSON), js]
    if binary:
        out += [struct.pack("<II", len(binary), CHUNK_BIN), binary]
    return b"".join(out)


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".part")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except OSError as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise GltfError(f"Cannot write {path.name}: {e}") from e


# --------------------------------------------------------------------------- writer
def write_glb(
    meshes: list[Mesh],
    path: str | Path,
    *,
    texture_format: str = "jpeg",
    max_texture: int = 4096,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> Path:
    """Write ``meshes`` (glTF frame) as one binary glTF 2.0 file and return its path.

    See the module docstring for the exact layout.  Empty meshes (no faces) are skipped
    with a warning; :class:`InputError` is raised when nothing remains or a mesh is
    malformed.  ``texture_format`` is ``"jpeg"`` (PNG is still used for textures with
    alpha / non-opaque materials) or ``"png"``.
    """
    rep = _as_reporter(progress)
    fmt = _normalize_format(texture_format)
    try:
        max_side = int(max_texture)
    except (TypeError, ValueError, OverflowError):
        max_side = 0
    if max_side < 1:
        raise InputError(f"max_texture must be a positive pixel size (got {max_texture!r}).")
    if isinstance(meshes, Mesh):
        meshes = [meshes]
    path = Path(path)
    b = _Builder()
    nodes: list[dict[str, Any]] = []
    gl_meshes: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    textures: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    tex_cache: dict[tuple[int, bool], int] = {}
    keepalive: list[Any] = []  # keep texture objects alive so id() keys stay unique
    n = max(1, len(meshes))
    for i, mesh in enumerate(meshes):
        _check(cancel)
        rep(i / n, f"Writing mesh {i + 1}/{n}")
        prep = _prepare(mesh)
        if prep is None:
            log.warning("Skipping empty mesh '%s'", mesh.name)
            continue
        v, f, uv, nrm = prep
        alpha_mode = str(mesh.alpha_mode).upper()
        if alpha_mode not in ("OPAQUE", "MASK", "BLEND"):
            raise InputError(f"Mesh '{mesh.name}' has an invalid alpha mode {mesh.alpha_mode!r}.")
        raw_tex = mesh.texture
        keepalive.append(raw_tex)
        tex8 = _texture_u8(raw_tex, mesh.name)
        png = _wants_png(tex8, alpha_mode, fmt)
        key = (id(raw_tex), png)
        if key not in tex_cache:
            t = _limit_texture(tex8, max_side)
            if not png and t.shape[2] == 4:
                t = t[..., :3]
            data = _encode_image(t, "png" if png else "jpeg")
            images.append({"bufferView": b.view(data), "mimeType": "image/png" if png else "image/jpeg"})
            textures.append({"sampler": 0, "source": len(images) - 1})
            tex_cache[key] = len(textures) - 1
        attrs = {
            "POSITION": b.accessor(v, FLOAT, "VEC3", ARRAY_BUFFER, minmax=True),
            "TEXCOORD_0": b.accessor(uv, FLOAT, "VEC2", ARRAY_BUFFER),
        }
        if nrm is not None:
            attrs["NORMAL"] = b.accessor(nrm, FLOAT, "VEC3", ARRAY_BUFFER)
        idx = b.accessor(f.reshape(-1), UNSIGNED_INT, "SCALAR", ELEMENT_ARRAY_BUFFER)
        mat: dict[str, Any] = {
            "name": str(mesh.name),
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": tex_cache[key]},
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
            "extensions": {UNLIT: {}},
            "alphaMode": alpha_mode,
            "doubleSided": bool(mesh.double_sided),
        }
        if alpha_mode == "MASK":
            cutoff = float(mesh.alpha_cutoff)
            if not (math.isfinite(cutoff) and cutoff >= 0):
                raise InputError(f"Mesh '{mesh.name}' has an invalid alpha cutoff {mesh.alpha_cutoff!r}.")
            mat["alphaCutoff"] = cutoff
        materials.append(mat)
        gl_meshes.append(
            {
                "name": str(mesh.name),
                "primitives": [
                    {"attributes": attrs, "indices": idx, "material": len(materials) - 1, "mode": TRIANGLES}
                ],
            }
        )
        nodes.append({"name": str(mesh.name), "mesh": len(gl_meshes) - 1})
    if not gl_meshes:
        raise InputError("Nothing to export: every mesh is empty.")
    doc: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": GENERATOR},
        "extensionsUsed": [UNLIT],
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": gl_meshes,
        "materials": materials,
        "textures": textures,
        "images": images,
        "samplers": [
            {
                "magFilter": LINEAR,
                "minFilter": LINEAR_MIPMAP_LINEAR,
                "wrapS": CLAMP_TO_EDGE,
                "wrapT": CLAMP_TO_EDGE,
            }
        ],
        "accessors": b.accessors,
        "bufferViews": b.views,
        "buffers": [{"byteLength": 0}],
    }
    binary = b.binary()
    doc["buffers"][0]["byteLength"] = len(binary)
    _check(cancel)
    rep(0.95, "Saving GLB")
    _atomic_write(path, _glb_bytes(doc, binary))
    rep(1.0, "GLB saved")
    log.info("Wrote %s (%d meshes, %.1f MB)", path.name, len(gl_meshes), path.stat().st_size / 1e6)
    return path


# --------------------------------------------------------------------------- reader
def _parse_container(path: Path) -> tuple[dict[str, Any], list[bytes]]:
    try:
        data = path.read_bytes()
    except OSError as e:
        raise InputError(f"Cannot read {path.name}: {e}") from e
    if len(data) >= 12 and struct.unpack_from("<I", data, 0)[0] == GLB_MAGIC:
        _, version, total = struct.unpack_from("<III", data, 0)
        if version != 2:
            raise GltfError(f"{path.name} is glTF version {version}; only glTF 2.0 is supported.")
        if total > len(data):
            raise GltfError(f"{path.name} is truncated ({len(data)} of {total} bytes).")
        off, doc, bin_chunk = 12, None, None
        while off + 8 <= total:
            clen, ctype = struct.unpack_from("<II", data, off)
            chunk = data[off + 8 : off + 8 + clen]
            if len(chunk) != clen:
                raise GltfError(f"{path.name} has a truncated chunk.")
            if ctype == CHUNK_JSON and doc is None:
                doc = _parse_json(chunk, path.name)
            elif ctype == CHUNK_BIN and bin_chunk is None:
                bin_chunk = chunk
            off += 8 + clen
        if doc is None:
            raise GltfError(f"{path.name} has no JSON chunk.")
        buffers = []
        for i, buf in enumerate(doc.get("buffers", [])):
            if i == 0 and "uri" not in buf:
                buffers.append(bin_chunk or b"")
            else:
                buffers.append(_load_uri(buf.get("uri", ""), path.parent))
        return doc, buffers
    doc = _parse_json(data, path.name, "is neither a GLB nor a glTF JSON file")
    buffers = [_load_uri(buf.get("uri", ""), path.parent) for buf in doc.get("buffers", [])]
    return doc, buffers


def _parse_json(data: bytes, name: str, what: str = "has a corrupt JSON chunk") -> dict[str, Any]:
    """Decode a glTF JSON document (must be a JSON object)."""
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise GltfError(f"{name} {what}.") from e
    if not isinstance(doc, dict):
        raise GltfError(f"{name} {what} (the top level is not a JSON object).")
    return doc


def _load_uri(uri: str, base: Path) -> bytes:
    if not uri:
        raise GltfError("A glTF buffer/image has no data.")
    if uri.startswith("data:"):
        m = re.match(r"data:[^;,]*(;base64)?,(.*)", uri, re.S)
        if not m:
            raise GltfError("Malformed data URI in glTF file.")
        if m.group(1):
            return base64.b64decode(m.group(2))
        from urllib.parse import unquote_to_bytes

        return unquote_to_bytes(m.group(2))
    from urllib.parse import unquote

    p = (base / unquote(uri)).resolve()
    try:
        return p.read_bytes()
    except OSError as e:
        raise GltfError(f"Cannot read glTF resource {uri}: {e}") from e


def _read_accessor(doc: dict[str, Any], buffers: list[bytes], index: int) -> np.ndarray:
    acc = doc["accessors"][index]
    if "sparse" in acc:
        raise GltfError("Sparse glTF accessors are not supported.")
    ctype = acc["componentType"]
    if ctype not in _COMPONENT_DTYPES:
        raise GltfError(f"Unsupported accessor component type {ctype}.")
    dt = _COMPONENT_DTYPES[ctype]
    ncomp = _TYPE_SIZES[acc["type"]]
    count = int(acc["count"])
    if "bufferView" not in acc:
        out = np.zeros((count, ncomp), dt)
    else:
        bv = doc["bufferViews"][acc["bufferView"]]
        buf = buffers[bv["buffer"]]
        start = int(bv.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
        elem = dt.itemsize * ncomp
        stride = int(bv.get("byteStride", 0)) or elem
        if count and start + stride * (count - 1) + elem > len(buf):
            raise GltfError("A glTF accessor points outside its buffer.")
        if stride == elem:
            out = np.frombuffer(buf, dtype=dt, count=count * ncomp, offset=start).reshape(count, ncomp)
        else:
            raw = np.frombuffer(buf, dtype=np.uint8, count=stride * (count - 1) + elem, offset=start)
            rows = np.lib.stride_tricks.as_strided(raw, shape=(count, elem), strides=(stride, 1))
            out = np.ascontiguousarray(rows).view(dt).reshape(count, ncomp)
    if acc.get("normalized") and ctype != FLOAT:
        info = np.iinfo(dt)
        out = np.maximum(out.astype(np.float32) / info.max, -1.0)
    return out


def _node_matrix(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T  # column-major
    m = np.eye(4)
    if "scale" in node:
        m = np.diag([*node["scale"], 1.0]) @ m
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        r = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        rm = np.eye(4)
        rm[:3, :3] = r
        m = rm @ m
    if "translation" in node:
        t = np.eye(4)
        t[:3, 3] = node["translation"]
        m = t @ m
    return m


def _decode_image(doc: dict[str, Any], buffers: list[bytes], index: int, base: Path) -> np.ndarray:
    img = doc["images"][index]
    if "bufferView" in img:
        bv = doc["bufferViews"][img["bufferView"]]
        off = int(bv.get("byteOffset", 0))
        data = buffers[bv["buffer"]][off : off + int(bv["byteLength"])]
    else:
        data = _load_uri(img.get("uri", ""), base)
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception as e:
        raise GltfError(f"Cannot decode an embedded texture: {e}") from e
    has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
    return np.ascontiguousarray(np.asarray(im.convert("RGBA" if has_alpha else "RGB")))


def read_glb_meshes(path: str | Path) -> list[Mesh]:
    """Read the triangle meshes of a GLB (or ``.gltf``) file as :class:`Mesh` objects.

    Node transforms of the default scene are applied (one Mesh per primitive instance).
    Missing UVs become zeros; a material without texture yields a 1x1 texture of its
    base colour.  Raises :class:`GltfError` for malformed or unsupported files.
    """
    path = Path(path)
    try:
        doc, buffers = _parse_container(path)
        return _meshes_from_doc(doc, buffers, path.parent)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError, struct.error) as e:
        raise GltfError(f"{path.name} is not a valid glTF file: {e}") from e


def _meshes_from_doc(doc: dict[str, Any], buffers: list[bytes], base: Path) -> list[Mesh]:
    instances: list[tuple[int, np.ndarray]] = []
    nodes = doc.get("nodes", [])

    def visit(ni: int, parent: np.ndarray, depth: int) -> None:
        if depth > 64:
            raise GltfError("glTF node hierarchy is too deep or cyclic.")
        node = nodes[ni]
        m = parent @ _node_matrix(node)
        if "mesh" in node:
            instances.append((int(node["mesh"]), m))
        for c in node.get("children", []):
            visit(int(c), m, depth + 1)

    scenes = doc.get("scenes", [])
    if scenes:
        for ni in scenes[int(doc.get("scene", 0))].get("nodes", []):
            visit(int(ni), np.eye(4), 0)
    elif nodes:
        children = {int(c) for n in nodes for c in n.get("children", [])}
        for ni in range(len(nodes)):
            if ni not in children:
                visit(ni, np.eye(4), 0)
    else:
        instances = [(i, np.eye(4)) for i in range(len(doc.get("meshes", [])))]

    image_cache: dict[int, np.ndarray] = {}
    out: list[Mesh] = []
    for mi, mat4 in instances:
        gm = doc["meshes"][mi]
        prims = gm.get("primitives", [])
        for pi, prim in enumerate(prims):
            mode = prim.get("mode", TRIANGLES)
            if mode != TRIANGLES:
                log.warning("Skipping glTF primitive with mode %s (only triangles are supported)", mode)
                continue
            attrs = prim["attributes"]
            v = _read_accessor(doc, buffers, attrs["POSITION"]).astype(np.float64)
            n = len(v)
            uv = (
                _read_accessor(doc, buffers, attrs["TEXCOORD_0"]).astype(np.float32)
                if "TEXCOORD_0" in attrs
                else np.zeros((n, 2), np.float32)
            )
            nrm = (
                _read_accessor(doc, buffers, attrs["NORMAL"]).astype(np.float64)
                if "NORMAL" in attrs
                else None
            )
            if "indices" in prim:
                idx = _read_accessor(doc, buffers, prim["indices"]).reshape(-1).astype(np.int64)
            else:
                idx = np.arange(n, dtype=np.int64)
            idx = idx[: len(idx) - len(idx) % 3]
            if len(idx) and (idx.min() < 0 or idx.max() >= n):
                raise GltfError("glTF indices point outside the vertex array.")
            if not np.allclose(mat4, np.eye(4)):
                v = v @ mat4[:3, :3].T + mat4[:3, 3]
                if nrm is not None:
                    nrm = nrm @ np.linalg.inv(mat4[:3, :3])
                    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
                if np.linalg.det(mat4[:3, :3]) < 0:
                    idx = idx.reshape(-1, 3)[:, [0, 2, 1]].reshape(-1)
            material: dict[str, Any] = {}
            if "material" in prim:
                material = doc["materials"][prim["material"]]
            pbr = material.get("pbrMetallicRoughness", {})
            tex_info = pbr.get("baseColorTexture")
            if tex_info is not None:
                src = doc["textures"][tex_info["index"]].get("source")
                if src is None:
                    texture = np.full((1, 1, 3), 255, np.uint8)
                else:
                    if src not in image_cache:
                        image_cache[src] = _decode_image(doc, buffers, src, base)
                    texture = image_cache[src]
            else:
                factor = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
                texture = (np.clip(np.array(factor[:3]), 0, 1) * 255 + 0.5).astype(np.uint8).reshape(1, 1, 3)
            name = gm.get("name") or f"mesh{mi}"
            if len(prims) > 1:
                name = f"{name}_{pi}"
            out.append(
                Mesh(
                    vertices=v.astype(np.float32),
                    faces=idx.reshape(-1, 3).astype(np.uint32),
                    uvs=np.ascontiguousarray(uv[:, :2], dtype=np.float32),
                    texture=texture,
                    name=name,
                    alpha_mode=material.get("alphaMode", "OPAQUE"),
                    alpha_cutoff=float(material.get("alphaCutoff", 0.5)),
                    double_sided=bool(material.get("doubleSided", False)),
                    normals=None if nrm is None else nrm.astype(np.float32),
                )
            )
    return out


# --------------------------------------------------------------------------- OBJ
def _safe_name(name: str, fallback: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._")
    return s or fallback


def write_obj(meshes: list[Mesh], path: str | Path) -> Path:
    """Write ``meshes`` as Wavefront OBJ + MTL + texture images next to it.

    The OBJ is in the glTF frame (y up, metres, counter-clockwise front faces); texture
    ``v`` is flipped to OBJ's bottom-left origin.  Textures are written as
    ``<stem>_<i>_<name>.jpg`` (``.png`` with alpha / non-opaque materials, referenced by
    both ``map_Kd`` and ``map_d``), capped at 8192 px.  Returns the ``.obj`` path.
    """
    if isinstance(meshes, Mesh):
        meshes = [meshes]
    path = Path(path)
    if path.suffix.lower() != ".obj":
        path = path.with_suffix(".obj")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise GltfError(f"Cannot create the folder for {path.name}: {e}") from e
    mtl_path = path.with_suffix(".mtl")
    stem = path.stem
    obj_tmp = path.with_name(path.name + ".part")
    mtl_lines = [f"# {GENERATOR}\n"]
    written = 0
    tex_files: dict[tuple[int, bool], str] = {}
    keepalive: list[Any] = []
    try:
        with open(obj_tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"# {GENERATOR}\nmtllib {mtl_path.name}\n")
            v_off = vn_off = 0
            for i, mesh in enumerate(meshes):
                prep = _prepare(mesh)
                if prep is None:
                    log.warning("Skipping empty mesh '%s'", mesh.name)
                    continue
                v, f, uv, nrm = prep
                alpha_mode = str(mesh.alpha_mode).upper()
                keepalive.append(mesh.texture)
                tex8 = _texture_u8(mesh.texture, mesh.name)
                png = _wants_png(tex8, alpha_mode, "jpeg")
                key = (id(mesh.texture), png)
                if key not in tex_files:
                    t = _limit_texture(tex8, _OBJ_MAX_TEXTURE)
                    if not png and t.shape[2] == 4:
                        t = t[..., :3]
                    tname = f"{stem}_{i}_{_safe_name(mesh.name, 'mesh')}.{'png' if png else 'jpg'}"
                    _atomic_write(path.parent / tname, _encode_image(t, "png" if png else "jpeg"))
                    tex_files[key] = tname
                tname = tex_files[key]
                mat = f"mat{i}_{_safe_name(mesh.name, 'mesh')}"
                mtl_lines.append(
                    f"\nnewmtl {mat}\nKa 1.000 1.000 1.000\nKd 1.000 1.000 1.000\nKs 0.000 0.000 0.000\n"
                    f"Ns 0\nd 1.0\nillum 1\nmap_Kd {tname}\n"
                )
                if png and alpha_mode != "OPAQUE":
                    mtl_lines.append(f"map_d {tname}\n")
                fh.write(f"o {_safe_name(mesh.name, f'mesh{i}')}\n")
                np.savetxt(fh, v.astype(np.float64), fmt="v %.6f %.6f %.6f")
                vt = np.column_stack([uv[:, 0], 1.0 - uv[:, 1]]).astype(np.float64)
                np.savetxt(fh, vt, fmt="vt %.6f %.6f")
                if nrm is not None:
                    np.savetxt(fh, nrm.astype(np.float64), fmt="vn %.5f %.5f %.5f")
                fh.write(f"usemtl {mat}\ns off\n")
                fi = f.astype(np.int64) + 1 + v_off  # v and vt share numbering (one vt per v)
                if nrm is not None:
                    fn = f.astype(np.int64) + 1 + vn_off
                    rows = np.stack([fi, fi, fn], axis=2).reshape(len(fi), 9)
                    np.savetxt(fh, rows, fmt="f %d/%d/%d %d/%d/%d %d/%d/%d")
                    vn_off += len(v)
                else:
                    rows = np.repeat(fi, 2, axis=1)
                    np.savetxt(fh, rows, fmt="f %d/%d %d/%d %d/%d")
                v_off += len(v)
                written += 1
        if not written:
            raise InputError("Nothing to export: every mesh is empty.")
        with open(mtl_path, "w", encoding="utf-8", newline="\n") as fm:
            fm.write("".join(mtl_lines))
        os.replace(obj_tmp, path)
    except OSError as e:
        raise GltfError(f"Cannot write {path.name}: {e}") from e
    finally:
        if obj_tmp.exists():
            try:
                obj_tmp.unlink()
            except OSError:
                pass
    return path


__all__ = ["write_glb", "read_glb_meshes", "write_obj", "GltfError", "UNLIT"]
