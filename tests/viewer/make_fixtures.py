"""Generate a small scene library with one scene of every viewer mode (for the browser tests).

Usage::

    python tests/viewer/make_fixtures.py OUT_DIR

Scenes are written with :func:`vrphoto.webxr.scene.export_scene`, so this also exercises the
exporter.  Everything is synthetic and tiny (a few hundred KB): a two-layer photo3d GLB, SBS
stereo image (+ a VP8 WebM video when ffmpeg has libvpx), VR180 SBS, 360 mono with a depth map,
a box "object" as GLB and as .splat, a small splat "room" and a capture-like real-size splat
object (person + floor + stray floaters).  Those use an inline GLB writer (KHR_materials_unlit
textured quads) and .splat writer, so they never depend on other modules.

Integration scenes (``fx-real-*``) are additionally built with the app's own geometry code -
``geometry.mesh.layered_meshes``, ``geometry.billboard.billboard``, ``geometry.gltf.write_glb``
and ``geometry.splat.write_splats`` - exactly as the pipeline does, so the browser tests prove
the viewer loads what the exporter really writes.  They are skipped (with a note on stderr)
when those modules are unavailable.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:  # make the repository importable without an editable install
    import vrphoto  # noqa: F401
except ImportError:  # pragma: no cover - depends on the environment
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SCENE_IDS = [
    "fx-photo3d",
    "fx-stereo",
    "fx-stereo-video",
    "fx-sphere180",
    "fx-sphere360",
    "fx-object-glb",
    "fx-object-splat",
    "fx-splat",
    "fx-capture-object",
]
#: Scenes built with the app's geometry modules (see the module docstring).
REAL_SCENE_IDS = ["fx-real-photo3d", "fx-real-object", "fx-real-splat-ply"]


# --------------------------------------------------------------------------- images
def _png(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG")
    return buf.getvalue()


def _jpg(img: np.ndarray, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def gradient(h: int, w: int, c0: tuple[int, int, int], c1: tuple[int, int, int]) -> np.ndarray:
    t = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    a = np.array(c0, np.float32)
    b = np.array(c1, np.float32)
    img = a * (1 - t) + b * t
    img = img * (0.75 + 0.25 * (1 - yy))
    return np.clip(img, 0, 255).astype(np.uint8)


def checker(h: int, w: int, n: int = 8, c0=(220, 220, 220), c1=(60, 90, 160)) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    m = ((xx * n // w) + (yy * n // h)) % 2 == 0
    out = np.empty((h, w, 3), np.uint8)
    out[m] = c0
    out[~m] = c1
    return out


def stereo_pair(h: int, w: int, shift: int) -> tuple[np.ndarray, np.ndarray]:
    """Two eye images of a red disc in front of a gradient (disc shifted between the eyes)."""
    eyes = []
    for s in (shift, -shift):
        img = Image.fromarray(gradient(h, w, (40, 120, 200), (230, 180, 60)))
        d = ImageDraw.Draw(img)
        cx, cy, r = w // 2 + s, h // 2, min(h, w) // 4
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 40, 40))
        d.text((8, 8), "L" if s == shift else "R", fill=(255, 255, 255))
        eyes.append(np.asarray(img))
    return eyes[0], eyes[1]


def equirect(h: int, w: int, seed: int) -> np.ndarray:
    """Colourful equirect with a bright marker straight ahead (image centre)."""
    img = checker(h, w, 16, (200, 200, 210), (70, 110, 70) if seed else (110, 70, 140))
    img = (
        img.astype(np.float32) * gradient(h, w, (255, 200, 160), (160, 200, 255)).astype(np.float32) / 255
    ).astype(np.uint8)
    cy, cx = h // 2, w // 2
    img[cy - h // 20 : cy + h // 20, cx - h // 20 : cx + h // 20] = (255, 30, 30)
    return img


# --------------------------------------------------------------------------- minimal GLB writer
def _pad4(b: bytes, fill: bytes = b"\x00") -> bytes:
    return b + fill * ((4 - len(b) % 4) % 4)


def make_glb(quads: list[tuple[np.ndarray, np.ndarray]]) -> bytes:
    """GLB with one unlit textured quad mesh per (corners 4x3 [bl, br, tr, tl], RGB texture)."""
    bin_parts: list[bytes] = []
    views: list[dict] = []
    accessors: list[dict] = []
    gltf: dict = {
        "asset": {"version": "2.0", "generator": "vrphoto tests/viewer/make_fixtures.py"},
        "extensionsUsed": ["KHR_materials_unlit"],
        "scene": 0,
        "scenes": [{"nodes": list(range(len(quads)))}],
        "nodes": [],
        "meshes": [],
        "materials": [],
        "textures": [],
        "images": [],
        "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
    }
    offset = 0

    def add_view(data: bytes, target: int | None = None) -> int:
        nonlocal offset
        data = _pad4(data)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target:
            view["target"] = target
        views.append(view)
        bin_parts.append(data)
        offset += len(data)
        return len(views) - 1

    for i, (corners, tex) in enumerate(quads):
        pos = np.asarray(corners, np.float32).reshape(4, 3)
        uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], np.float32)  # glTF: (0,0) = top-left
        idx = np.array([0, 1, 2, 0, 2, 3], np.uint16)
        pv = add_view(pos.tobytes(), 34962)
        uvv = add_view(uv.tobytes(), 34962)
        iv = add_view(idx.tobytes(), 34963)
        accessors += [
            {
                "bufferView": pv,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
                "min": pos.min(0).tolist(),
                "max": pos.max(0).tolist(),
            },
            {"bufferView": uvv, "componentType": 5126, "count": 4, "type": "VEC2"},
            {"bufferView": iv, "componentType": 5123, "count": 6, "type": "SCALAR"},
        ]
        a0 = len(accessors) - 3
        imgv = add_view(_png(tex))
        gltf["images"].append({"bufferView": imgv, "mimeType": "image/png"})
        gltf["textures"].append({"sampler": 0, "source": i})
        gltf["materials"].append(
            {
                "name": f"quad{i}",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": i},
                    "metallicFactor": 0,
                    "roughnessFactor": 1,
                },
                "doubleSided": True,
                "extensions": {"KHR_materials_unlit": {}},
            }
        )
        gltf["meshes"].append(
            {
                "primitives": [
                    {"attributes": {"POSITION": a0, "TEXCOORD_0": a0 + 1}, "indices": a0 + 2, "material": i}
                ]
            }
        )
        gltf["nodes"].append({"mesh": i, "name": f"quad{i}"})

    gltf["bufferViews"] = views
    gltf["accessors"] = accessors
    bin_chunk = b"".join(bin_parts)
    gltf["buffers"] = [{"byteLength": len(bin_chunk)}]
    json_chunk = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    out = struct.pack("<III", 0x46546C67, 2, total)
    out += struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk
    out += struct.pack("<II", len(bin_chunk), 0x004E4942) + bin_chunk
    return out


def quad(cx: float, cy: float, z: float, w: float, h: float) -> np.ndarray:
    return np.array(
        [
            [cx - w / 2, cy - h / 2, z],
            [cx + w / 2, cy - h / 2, z],
            [cx + w / 2, cy + h / 2, z],
            [cx - w / 2, cy + h / 2, z],
        ],
        np.float32,
    )


def box_glb(w: float, h: float, d: float) -> bytes:
    """A box standing on y=0 with differently coloured faces (fake 'person')."""
    x0, x1, z0, z1 = -w / 2, w / 2, -d / 2, d / 2
    faces = [
        (np.array([[x0, 0, z1], [x1, 0, z1], [x1, h, z1], [x0, h, z1]]), (230, 90, 70)),  # front +z
        (np.array([[x1, 0, z0], [x0, 0, z0], [x0, h, z0], [x1, h, z0]]), (70, 120, 230)),  # back
        (np.array([[x1, 0, z1], [x1, 0, z0], [x1, h, z0], [x1, h, z1]]), (90, 200, 120)),  # right
        (np.array([[x0, 0, z0], [x0, 0, z1], [x0, h, z1], [x0, h, z0]]), (220, 200, 80)),  # left
        (np.array([[x0, h, z1], [x1, h, z1], [x1, h, z0], [x0, h, z0]]), (240, 240, 240)),  # top
    ]
    return make_glb([(c, checker(64, 64, 4, col, tuple(int(v * 0.6) for v in col))) for c, col in faces])


# --------------------------------------------------------------------------- splats
def splat_bytes(points: np.ndarray, colors: np.ndarray, scale: float, opacity: float = 0.9) -> bytes:
    """antimatter15 .splat: 3xf32 pos, 3xf32 scale, RGBA u8, quat u8 (w first, q*128+128)."""
    n = len(points)
    rec = np.zeros(n, dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("q", "u1", 4)])
    rec["p"] = points
    rec["s"] = scale
    rec["c"][:, :3] = colors
    rec["c"][:, 3] = int(opacity * 255)
    rec["q"] = (255, 128, 128, 128)
    return rec.tobytes()


def splat_object(n: int = 3000) -> bytes:
    rng = np.random.default_rng(1)
    # A capsule-ish column 1 m tall standing on y = 0.
    t = rng.uniform(0, 2 * math.pi, n)
    y = rng.uniform(0.0, 1.0, n)
    r = 0.18 + 0.05 * np.sin(y * 9)
    pts = np.stack([r * np.cos(t), y, r * np.sin(t)], 1)
    cols = np.stack([200 + 50 * y, 120 + 100 * (np.cos(t) * 0.5 + 0.5), 80 + 0 * y], 1).clip(0, 255)
    return splat_bytes(pts, cols.astype(np.uint8), 0.03)


def splat_room(n: int = 6000) -> bytes:
    rng = np.random.default_rng(2)
    pts = []
    # floor 8x8 m at y=0 and four walls 3 m high around the origin
    k = n // 5
    pts.append(np.stack([rng.uniform(-4, 4, k), np.zeros(k), rng.uniform(-4, 4, k)], 1))
    for sx, sz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        u = rng.uniform(-4, 4, k)
        v = rng.uniform(0, 3, k)
        if sx:
            pts.append(np.stack([np.full(k, 4.0 * sx), v, u], 1))
        else:
            pts.append(np.stack([u, v, np.full(k, 4.0 * sz)], 1))
    p = np.concatenate(pts)
    cols = (
        np.stack([128 + 30 * p[:, 0], 100 + 40 * p[:, 1], 128 + 30 * p[:, 2]], 1)
        .clip(0, 255)
        .astype(np.uint8)
    )
    return splat_bytes(p, cols, 0.07, 0.95)


def capture_object() -> bytes:
    """What a capture looks like after the app scaled it to metres: a 1.7 m "person" at the
    origin standing on a 1.5 m floor disc (floor at y = 0), plus a few stray floater splats far
    away / below the floor that must not change its size, centre or floor."""
    rng = np.random.default_rng(3)
    n_person, n_floor = 3500, 4450
    t = rng.uniform(0, 2 * math.pi, n_person)
    y = rng.uniform(0.0, 1.7, n_person)
    r = 0.16 + 0.04 * np.sin(y * 7)
    person = np.stack([r * np.cos(t), y, r * np.sin(t)], 1)
    rf = 1.5 * np.sqrt(rng.uniform(0, 1, n_floor))
    tf = rng.uniform(0, 2 * math.pi, n_floor)
    floor = np.stack([rf * np.cos(tf), rng.normal(0, 0.003, n_floor), rf * np.sin(tf)], 1)
    floaters = np.array(
        [[5.5, -1.0, 0.4], [6.0, 3.5, -0.5], [4.8, -0.9, 5.0], [6.2, -1.1, 4.4], [5.1, 3.9, 5.6]] * 6
    )
    pts = np.concatenate([person, floor, floaters])
    cols = (
        np.concatenate(
            [
                np.tile([[200, 90, 70]], (n_person, 1)) + rng.integers(0, 40, (n_person, 3)),
                np.tile([[90, 100, 110]], (n_floor, 1)) + rng.integers(0, 30, (n_floor, 3)),
                np.tile([[250, 250, 250]], (len(floaters), 1)),
            ]
        )
        .clip(0, 255)
        .astype(np.uint8)
    )
    return splat_bytes(pts.astype(np.float32), cols, 0.025, 0.95)


# --------------------------------------------------------------------------- video
def make_webm(left: np.ndarray, right: np.ndarray, path: Path, seconds: float = 2.0, fps: int = 15) -> bool:
    """SBS VP8 WebM via ffmpeg (imageio-ffmpeg or PATH).  Returns False when unavailable."""
    exe = None
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - optional
        exe = shutil.which("ffmpeg")
    if not exe:
        return False
    h, w = left.shape[:2]
    frames = []
    for i in range(int(seconds * fps)):
        dx = int(10 * math.sin(i / fps * 2 * math.pi))
        frame = np.concatenate([np.roll(left, dx, axis=1), np.roll(right, dx, axis=1)], axis=1)
        frames.append(frame)
    raw = np.stack(frames).tobytes()
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * w}x{h}"]
    cmd += [
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libvpx",
        "-b:v",
        "400k",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, input=raw, capture_output=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and path.is_file() and path.stat().st_size > 0


# --------------------------------------------------------------------------- real pipeline output
def _test_photo(h: int = 180, w: int = 240) -> tuple[np.ndarray, np.ndarray]:
    """A small 'photo' (textured background + a round subject) and the subject's alpha matte."""
    img = Image.fromarray(gradient(h, w, (60, 130, 200), (230, 190, 90)))
    d = ImageDraw.Draw(img)
    for i in range(0, w, 24):
        d.line([(i, 0), (i, h)], fill=(255, 255, 255), width=1)
    cx, cy, r = w // 2, int(h * 0.55), int(h * 0.3)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(210, 50, 60))
    d.ellipse([cx - r // 3, cy - r // 2, cx + r // 3, cy], fill=(250, 220, 120))
    yy, xx = np.mgrid[0:h, 0:w]
    alpha = (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r).astype(np.float32)
    return np.asarray(img).copy(), alpha


def build_real_scenes(export) -> list[str]:  # noqa: ANN001 - export callback of build_library
    """Scenes made by the app's geometry modules (as pipeline/run.py does); [] if unavailable."""
    try:
        from vrphoto.core.types import DepthResult, Intrinsics, Layer, LayeredImage
        from vrphoto.geometry.billboard import billboard
        from vrphoto.geometry.gltf import write_glb
        from vrphoto.geometry.mesh import layered_meshes
        from vrphoto.geometry.splat import splats_from_mesh, write_splats
    except ImportError as e:  # pragma: no cover - depends on which modules exist
        sys.stderr.write(f"make_fixtures: skipping fx-real-* scenes ({e})\n")
        return []

    rgb, alpha = _test_photo()
    h, w = alpha.shape
    K = Intrinsics(fx=0.9 * w, fy=0.9 * w, cx=(w - 1) / 2, cy=(h - 1) / 2, width=w, height=h)
    inv_bg = np.full((h, w), 1 / 4.0, np.float32)
    inv_fg = np.where(alpha > 0.5, 1 / 1.8, 1 / 4.0).astype(np.float32)
    ones = np.ones((h, w), np.float32)
    li = LayeredImage([Layer(rgb, inv_bg, ones), Layer(rgb, inv_fg, ones)], K)
    written: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        meshes = layered_meshes(li, target_faces=6000, anchor_distance_m=2.0, eye_height_m=1.5)
        write_glb(meshes, tmp / "scene.glb")
        export(
            {
                "id": "fx-real-photo3d",
                "title": "Fixture: 3D photo (app geometry)",
                "mode": "photo3d",
                "photo3d": {"meshes": ["scene.glb"], "anchor_distance_m": 2.0, "eye_height_m": 1.5},
            },
            {"scene.glb": tmp / "scene.glb"},
            rgb,
        )
        written.append("fx-real-photo3d")

        disparity = np.where(alpha > 0.5, 0.8, 0.2).astype(np.float32)
        parts = billboard(rgb, alpha, DepthResult(disparity=disparity, model_id="fixture"), K, height_m=1.7)
        write_glb(parts, tmp / "object.glb")
        write_splats(splats_from_mesh(parts[0]), tmp / "object.splat")
        export(
            {
                "id": "fx-real-object",
                "title": "Fixture: object (app billboard)",
                "mode": "object",
                "object": {
                    "asset": "object.glb",
                    "format": "glb",
                    "height_m": 1.7,
                    "pivot": "base",
                    "placement": "hit-test",
                    "alt_assets": ["object.splat"],
                },
                "env": {"prefer": "ar", "background": "passthrough"},
            },
            {"object.glb": tmp / "object.glb", "object.splat": tmp / "object.splat"},
            rgb,
        )
        written.append("fx-real-object")

        write_splats(splats_from_mesh(meshes[0]), tmp / "world.ply")
        export(
            {
                "id": "fx-real-splat-ply",
                "title": "Fixture: splat world (PLY, app writer)",
                "mode": "splat",
                "splat": {"asset": "world.ply", "format": "ply", "up": "y", "scale": 1.0},
            },
            {"world.ply": tmp / "world.ply"},
            rgb,
        )
        written.append("fx-real-splat-ply")
    return written


# --------------------------------------------------------------------------- library
def build_library(out: Path, *, real: bool = True) -> list[str]:
    """Write all fixture scenes into ``out``; returns the scene ids written (newest first).

    ``real=False`` leaves out the ``fx-real-*`` scenes made with the app's geometry modules.
    """
    from vrphoto.webxr.scene import export_scene

    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    t0 = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
    stamp = ((t0 - timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(1000))  # newest first

    def export(manifest: dict, assets: dict, thumb: np.ndarray) -> None:
        manifest.setdefault("created", next(stamp))
        export_scene(manifest, assets, library=out, thumbnail=thumb)
        written.append(manifest["id"])

    # photo3d: background quad at 4 m, foreground card at 1.8 m (eye height 1.5 m)
    bg = make_glb([(quad(0, 1.5, -4.0, 7.0, 4.5), checker(256, 384, 12, (230, 210, 170), (90, 140, 200)))])
    fg = make_glb([(quad(0.2, 1.3, -1.8, 0.9, 1.1), gradient(128, 96, (230, 60, 60), (250, 200, 60)))])
    export(
        {
            "id": "fx-photo3d",
            "title": "Fixture: 3D photo",
            "mode": "photo3d",
            "photo3d": {"meshes": ["bg.glb", "fg.glb"], "anchor_distance_m": 2.0, "eye_height_m": 1.5},
        },
        {"bg.glb": bg, "fg.glb": fg},
        checker(90, 160, 6),
    )

    left, right = stereo_pair(192, 256, 12)
    sbs = np.concatenate([left, right], axis=1)
    export(
        {
            "id": "fx-stereo",
            "title": "Fixture: 3D screen (SBS)",
            "mode": "stereo",
            "stereo": {
                "media": "photo_LR.jpg",
                "kind": "image",
                "layout": "sbs",
                "screen_width_m": 2.5,
                "distance_m": 3.0,
            },
        },
        {"photo_LR.jpg": _jpg(sbs)},
        left,
    )

    with tempfile.TemporaryDirectory() as td:
        webm = Path(td) / "clip_LR.webm"
        if make_webm(left[:, :, :], right, webm):
            export(
                {
                    "id": "fx-stereo-video",
                    "title": "Fixture: 3D video (SBS)",
                    "mode": "stereo",
                    "stereo": {"media": "clip_LR.webm", "kind": "video", "layout": "sbs"},
                },
                {"clip_LR.webm": webm},
                right,
            )

    e_left = equirect(512, 512, 0)
    e_right = np.roll(e_left, -6, axis=1)
    export(
        {
            "id": "fx-sphere180",
            "title": "Fixture: VR180",
            "mode": "sphere",
            "sphere": {
                "media": "photo_180_LR.jpg",
                "kind": "image",
                "coverage": 180,
                "layout": "sbs",
                "radius_m": 20,
            },
        },
        {"photo_180_LR.jpg": _jpg(np.concatenate([e_left, e_right], axis=1))},
        e_left,
    )

    pano = equirect(512, 1024, 1)
    yy = np.linspace(0, 1, 128, dtype=np.float32)[:, None]
    depth = (np.clip(0.2 + 0.6 * np.exp(-((yy - 0.6) ** 2) / 0.02) * np.ones((1, 256)), 0, 1) * 65535).astype(
        np.uint16
    )
    buf = io.BytesIO()
    Image.fromarray(depth).save(buf, "PNG")
    export(
        {
            "id": "fx-sphere360",
            "title": "Fixture: 360 with depth",
            "mode": "sphere",
            "sphere": {
                "media": "pano_360.jpg",
                "kind": "image",
                "coverage": 360,
                "layout": "mono",
                "depth": "depth.png",
                "radius_m": 15,
            },
        },
        {"pano_360.jpg": _jpg(pano), "depth.png": buf.getvalue()},
        pano,
    )

    export(
        {
            "id": "fx-object-glb",
            "title": "Fixture: object (GLB)",
            "mode": "object",
            "object": {
                "asset": "person.glb",
                "format": "glb",
                "height_m": 1.7,
                "pivot": "base",
                "placement": "hit-test",
            },
            "env": {"prefer": "ar"},
        },
        {"person.glb": box_glb(0.5, 1.0, 0.3)},
        checker(96, 64, 4, (230, 90, 70), (120, 50, 40)),
    )

    export(
        {
            "id": "fx-object-splat",
            "title": "Fixture: object (splat)",
            "mode": "object",
            "object": {
                "asset": "thing.splat",
                "format": "splat",
                "height_m": 1.2,
                "placement": "hit-test",
                "distance_m": 1.5,
            },
        },
        {"thing.splat": splat_object()},
        gradient(96, 96, (230, 160, 80), (90, 60, 30)),
    )

    export(
        {
            "id": "fx-splat",
            "title": "Fixture: splat room",
            "mode": "splat",
            "splat": {"asset": "room.splat", "format": "splat", "up": "y", "scale": 1.0},
        },
        {"room.splat": splat_room()},
        gradient(90, 160, (120, 100, 128), (180, 140, 160)),
    )

    export(
        {
            "id": "fx-capture-object",
            "title": "Fixture: captured person (real size)",
            "mode": "object",
            # as the pipeline writes a capture: the splat is already in metres (scale 1) and
            # height_m names the subject's height (here a person, plus floor and floaters)
            "object": {
                "asset": "capture.splat",
                "format": "splat",
                "pivot": "base",
                "placement": "hit-test",
                "scale": 1.0,
                "height_m": 1.7,
            },
            "env": {"prefer": "ar", "background": "passthrough"},
        },
        {"capture.splat": capture_object()},
        gradient(96, 64, (200, 90, 70), (90, 100, 110)),
    )
    if real:
        build_real_scenes(export)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, help="library folder to create")
    ap.add_argument("--no-real", action="store_true", help="skip the fx-real-* scenes (app geometry modules)")
    args = ap.parse_args(argv)
    ids = build_library(args.out, real=not args.no_real)
    sys.stdout.write(json.dumps({"library": str(args.out), "scenes": ids}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
