"""Tests for geometry.gltf: GLB writer/reader and OBJ export.

GLBs are checked structurally, with pygltflib and trimesh, and with the Khronos glTF
validator (npm package ``gltf-validator``) when node/npm are available.
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vrphoto.core.errors import InputError
from vrphoto.core.types import DepthResult, Intrinsics, Layer, LayeredImage, Mesh
from vrphoto.geometry.billboard import billboard
from vrphoto.geometry.gltf import GltfError, read_glb_meshes, write_glb, write_obj
from vrphoto.geometry.mesh import depth_mesh, layered_meshes, vertex_normals

_VALIDATOR_VERSION = "2.0.0-dev.3.10"
_VALIDATE_JS = """
const v = require('gltf-validator');
const fs = require('fs');
v.validateBytes(new Uint8Array(fs.readFileSync(process.argv[2])), {maxIssues: 200})
  .then(r => { process.stdout.write(JSON.stringify(r.issues)); })
  .catch(e => { console.error(String(e)); process.exit(2); });
"""


# --------------------------------------------------------------------------- helpers
def parse_glb(path: Path) -> tuple[dict, bytes, bytes]:
    data = path.read_bytes()
    magic, version, total = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67 and version == 2 and total == len(data)
    jlen, jtype = struct.unpack_from("<II", data, 12)
    assert jtype == 0x4E4F534A and jlen % 4 == 0
    js = data[20 : 20 + jlen]
    off = 20 + jlen
    blen, btype = struct.unpack_from("<II", data, off)
    assert btype == 0x004E4942 and blen % 4 == 0 and off + 8 + blen == total
    return json.loads(js.decode("utf-8")), js, data[off + 8 : off + 8 + blen]


def accessor(doc: dict, binary: bytes, idx: int) -> np.ndarray:
    acc = doc["accessors"][idx]
    bv = doc["bufferViews"][acc["bufferView"]]
    dt = {5126: "<f4", 5125: "<u4"}[acc["componentType"]]
    n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[acc["type"]]
    start = bv["byteOffset"] + acc.get("byteOffset", 0)
    return np.frombuffer(binary, dtype=dt, count=acc["count"] * n, offset=start).reshape(acc["count"], n)


@pytest.fixture
def scene_meshes(two_plane_scene) -> list[Mesh]:
    s = two_plane_scene
    opaque = depth_mesh(s.img, s.inv_z, s.K, target_faces=4000, name="photo")
    alpha = np.ones(s.inv_z.shape, np.float32)
    alpha[:, :60] = 0.0
    masked = depth_mesh(s.img, s.inv_z, s.K, alpha=alpha, target_faces=3000, name="cutout")
    masked.normals = vertex_normals(masked.vertices, masked.faces)
    masked.double_sided = True
    masked.alpha_cutoff = 0.4
    return [opaque, masked]


# --------------------------------------------------------------------------- structure
def test_glb_layout_and_json(scene_meshes, tmp_path):
    p = write_glb(scene_meshes, tmp_path / "scene.glb")
    assert p == tmp_path / "scene.glb" and p.exists()
    doc, js, binary = parse_glb(p)
    assert js.rstrip(b" ").endswith(b"}") and len(js) - len(js.rstrip(b" ")) < 4  # space padding
    assert doc["asset"]["version"] == "2.0"
    assert doc["extensionsUsed"] == ["KHR_materials_unlit"]
    assert "extensionsRequired" not in doc
    assert len(doc["nodes"]) == len(doc["meshes"]) == len(doc["materials"]) == 2
    assert doc["scenes"][doc["scene"]]["nodes"] == [0, 1]
    assert doc["buffers"] == [{"byteLength": len(binary)}]
    for bv in doc["bufferViews"]:
        assert bv["byteOffset"] % 4 == 0 and bv["byteOffset"] + bv["byteLength"] <= len(binary)
    s = doc["samplers"][0]
    assert s == {"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}
    m0, m1 = doc["materials"]
    assert m0["extensions"] == {"KHR_materials_unlit": {}} and m0["alphaMode"] == "OPAQUE"
    assert "alphaCutoff" not in m0 and m0["doubleSided"] is False
    assert m1["alphaMode"] == "MASK" and m1["alphaCutoff"] == pytest.approx(0.4) and m1["doubleSided"] is True
    assert m0["pbrMetallicRoughness"]["metallicFactor"] == 0.0
    assert [im["mimeType"] for im in doc["images"]] == ["image/jpeg", "image/png"]
    for mesh, gm in zip(scene_meshes, doc["meshes"]):
        prim = gm["primitives"][0]
        assert prim["mode"] == 4
        pos = accessor(doc, binary, prim["attributes"]["POSITION"])
        acc = doc["accessors"][prim["attributes"]["POSITION"]]
        assert acc["min"] == [float(x) for x in pos.min(0)] and acc["max"] == [float(x) for x in pos.max(0)]
        np.testing.assert_array_equal(pos, mesh.vertices)
        np.testing.assert_array_equal(accessor(doc, binary, prim["attributes"]["TEXCOORD_0"]), mesh.uvs)
        idx = accessor(doc, binary, prim["indices"])
        assert doc["accessors"][prim["indices"]]["componentType"] == 5125
        np.testing.assert_array_equal(idx.reshape(-1, 3), mesh.faces)
        assert doc["bufferViews"][doc["accessors"][prim["indices"]]["bufferView"]]["target"] == 34963
    prim1 = doc["meshes"][1]["primitives"][0]
    nrm = accessor(doc, binary, prim1["attributes"]["NORMAL"])
    np.testing.assert_allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-6)
    assert "NORMAL" not in doc["meshes"][0]["primitives"][0]["attributes"]


def test_glb_loads_in_pygltflib_and_trimesh(scene_meshes, tmp_path):
    pygltflib = pytest.importorskip("pygltflib")
    trimesh = pytest.importorskip("trimesh")
    p = write_glb(scene_meshes, tmp_path / "scene.glb")
    g = pygltflib.GLTF2().load(str(p))
    assert len(g.meshes) == 2 and g.extensionsUsed == ["KHR_materials_unlit"]
    for mesh, gm in zip(scene_meshes, g.meshes):
        prim = gm.primitives[0]
        assert g.accessors[prim.attributes.POSITION].count == len(mesh.vertices)
        assert g.accessors[prim.indices].count == 3 * len(mesh.faces)
    blob = g.binary_blob()
    img_bv = g.bufferViews[g.images[0].bufferView]
    im = Image.open(io.BytesIO(blob[img_bv.byteOffset : img_bv.byteOffset + img_bv.byteLength]))
    assert im.format == "JPEG" and im.size == (320, 240)
    scene = trimesh.load(p)
    geoms = {k: v for k, v in scene.geometry.items()}
    assert set(geoms) == {"photo", "cutout"}
    for mesh in scene_meshes:
        geo = geoms[mesh.name]
        assert geo.vertices.shape == mesh.vertices.shape and geo.faces.shape == mesh.faces.shape
        tex = geo.visual.material.baseColorTexture
        assert tex is not None and tex.size == (320, 240)
    assert geoms["cutout"].visual.material.baseColorTexture.mode == "RGBA"


def test_textures_are_downscaled_and_png_forced(scene_meshes, tmp_path):
    p = write_glb(scene_meshes, tmp_path / "small.glb", max_texture=64, texture_format="png")
    doc, _, binary = parse_glb(p)
    assert all(im["mimeType"] == "image/png" for im in doc["images"])
    for im in doc["images"]:
        bv = doc["bufferViews"][im["bufferView"]]
        pil = Image.open(io.BytesIO(binary[bv["byteOffset"] : bv["byteOffset"] + bv["byteLength"]]))
        assert max(pil.size) == 64 and pil.size == (64, 48)
    with pytest.raises(InputError):
        write_glb(scene_meshes, tmp_path / "x.glb", texture_format="webp")
    with pytest.raises(InputError):
        write_glb(scene_meshes, tmp_path / "x.glb", max_texture=0)


def test_shared_textures_are_embedded_once(two_plane_scene, tmp_path):
    s = two_plane_scene
    a = depth_mesh(s.img, s.inv_z, s.K, target_faces=1000, name="a")
    b = depth_mesh(s.img, s.inv_z, s.K, target_faces=500, name="b")
    assert a.texture is b.texture
    doc, _, _ = parse_glb(write_glb([a, b], tmp_path / "shared.glb"))
    assert len(doc["images"]) == 1 and len(doc["textures"]) == 1 and len(doc["materials"]) == 2


def test_cleaning_and_validation(tmp_path):
    tex = np.zeros((4, 4, 3), np.uint8)
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [9, 9, 9], [1, 1, 0]], np.float32)
    uv = np.zeros((5, 2), np.float32)
    f = np.array([[0, 1, 2], [1, 1, 2], [1, 4, 2]], np.uint32)  # one degenerate face, vertex 3 unused
    p = write_glb(
        [Mesh(v, f, uv, tex, name="m"), Mesh(v, np.zeros((0, 3), np.uint32), uv, tex, name="empty")],
        tmp_path / "c.glb",
    )
    (m,) = read_glb_meshes(p)
    assert len(m.faces) == 2 and len(m.vertices) == 4 and m.vertices[:, 0].max() == 1.0
    with pytest.raises(InputError):
        write_glb([Mesh(v, np.zeros((0, 3), np.uint32), uv, tex)], tmp_path / "e.glb")
    with pytest.raises(InputError):
        write_glb([Mesh(v, np.array([[0, 1, 7]], np.uint32), uv, tex)], tmp_path / "e.glb")
    with pytest.raises(InputError):
        write_glb([Mesh(v, f, uv[:3], tex)], tmp_path / "e.glb")
    with pytest.raises(InputError):
        write_glb([Mesh(v, f, uv, tex, alpha_mode="FANCY")], tmp_path / "e.glb")  # type: ignore[arg-type]
    assert not list(tmp_path.glob("*.part"))


# --------------------------------------------------------------------------- reader
def test_read_glb_round_trip(scene_meshes, tmp_path, psnr_fn):
    p = write_glb(scene_meshes, tmp_path / "rt.glb")
    back = read_glb_meshes(p)
    assert [m.name for m in back] == ["photo", "cutout"]
    for src, dst in zip(scene_meshes, back):
        np.testing.assert_array_equal(dst.vertices, src.vertices)
        np.testing.assert_array_equal(dst.faces, src.faces)
        np.testing.assert_array_equal(dst.uvs, src.uvs)
        assert dst.alpha_mode == src.alpha_mode and dst.double_sided == src.double_sided
        assert dst.alpha_cutoff == pytest.approx(src.alpha_cutoff)
    assert psnr_fn(back[0].texture, scene_meshes[0].texture) > 30  # JPEG
    np.testing.assert_array_equal(back[1].texture, scene_meshes[1].texture)  # PNG is lossless
    np.testing.assert_allclose(back[1].normals, scene_meshes[1].normals, atol=1e-6)
    assert back[0].normals is None


def test_reader_applies_node_transforms_and_reads_gltf_json(scene_meshes, tmp_path):
    p = write_glb(scene_meshes[:1], tmp_path / "one.glb")
    doc, _, binary = parse_glb(p)
    doc["nodes"][0]["translation"] = [1.0, 2.0, 3.0]
    doc["nodes"][0]["scale"] = [2.0, 2.0, 2.0]
    doc["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(binary).decode(
        "ascii"
    )
    g = tmp_path / "one.gltf"
    g.write_text(json.dumps(doc), encoding="utf-8")
    (m,) = read_glb_meshes(g)
    np.testing.assert_allclose(m.vertices, scene_meshes[0].vertices * 2 + [1, 2, 3], atol=1e-5)
    # external .bin buffer
    (tmp_path / "one.bin").write_bytes(binary)
    doc["buffers"][0]["uri"] = "one.bin"
    doc["nodes"][0].pop("scale")
    g.write_text(json.dumps(doc), encoding="utf-8")
    (m2,) = read_glb_meshes(g)
    np.testing.assert_allclose(m2.vertices, scene_meshes[0].vertices + [1, 2, 3], atol=1e-5)


def test_reader_errors(tmp_path):
    bad = tmp_path / "bad.glb"
    bad.write_bytes(b"glTF" + struct.pack("<II", 1, 20) + b"\x00" * 8)
    with pytest.raises(GltfError):
        read_glb_meshes(bad)
    junk = tmp_path / "junk.glb"
    junk.write_bytes(b"\x00\x01\x02")
    with pytest.raises(GltfError):
        read_glb_meshes(junk)
    with pytest.raises(InputError):
        read_glb_meshes(tmp_path / "missing.glb")


# --------------------------------------------------------------------------- OBJ
def test_write_obj(scene_meshes, tmp_path):
    trimesh = pytest.importorskip("trimesh")
    p = write_obj(scene_meshes, tmp_path / "out" / "scene")
    assert p.suffix == ".obj" and p.exists()
    mtl = p.with_suffix(".mtl")
    assert mtl.exists()
    files = sorted(x.name for x in p.parent.iterdir())
    assert files == ["scene.mtl", "scene.obj", "scene_0_photo.jpg", "scene_1_cutout.png"]
    text = p.read_text(encoding="utf-8")
    assert text.startswith("# VR Photo Studio\nmtllib scene.mtl\n")
    assert text.count("\nv ") == sum(len(m.vertices) for m in scene_meshes)
    assert text.count("\nvn ") == len(scene_meshes[1].vertices)
    assert text.count("\nf ") == sum(len(m.faces) for m in scene_meshes)
    first_vt = next(ln for ln in text.splitlines() if ln.startswith("vt "))
    u, v = map(float, first_vt.split()[1:])
    assert u == pytest.approx(scene_meshes[0].uvs[0, 0], abs=1e-6) and v == pytest.approx(
        1 - scene_meshes[0].uvs[0, 1], abs=1e-6
    )
    m = mtl.read_text(encoding="utf-8")
    assert "map_Kd scene_0_photo.jpg" in m and "map_d scene_1_cutout.png" in m
    loaded = trimesh.load(p)
    geoms = list(loaded.geometry.values())
    assert sorted(len(g.faces) for g in geoms) == sorted(len(x.faces) for x in scene_meshes)
    # normals of mesh 2 index into its own vn block
    last_f = [ln for ln in text.splitlines() if ln.startswith("f ")][-1]
    vn_idx = [int(tok.split("/")[2]) for tok in last_f.split()[1:]]
    assert max(vn_idx) <= len(scene_meshes[1].vertices)


# --------------------------------------------------------------------------- Khronos validator
@pytest.fixture(scope="session")
def gltf_validator(tmp_path_factory):
    """Install the Khronos validator (npm) into a temp dir; skip when node/npm are unavailable."""
    node, npm = shutil.which("node"), shutil.which("npm")
    if not node or not npm:
        pytest.skip("node/npm not available for the Khronos glTF validator")
    d = tmp_path_factory.mktemp("gltf_validator")
    try:
        r = subprocess.run(
            [npm, "install", "--no-audit", "--no-fund", "--silent", f"gltf-validator@{_VALIDATOR_VERSION}"],
            cwd=d,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        pytest.skip(f"npm install gltf-validator failed: {e}")
    if r.returncode != 0 or not (d / "node_modules" / "gltf-validator").is_dir():
        pytest.skip(f"npm install gltf-validator failed: {r.stderr[-300:]}")
    script = d / "validate.js"
    script.write_text(_VALIDATE_JS, encoding="utf-8")

    def run(path: Path) -> dict:
        out = subprocess.run(
            [node, str(script), str(path)], cwd=d, capture_output=True, text=True, timeout=120
        )
        assert out.returncode == 0, out.stderr
        return json.loads(out.stdout)

    return run


def _assert_valid(issues: dict) -> None:
    errors = [m for m in issues["messages"] if m["severity"] == 0]
    warnings = [m for m in issues["messages"] if m["severity"] == 1]
    assert issues["numErrors"] == 0, errors
    assert issues["numWarnings"] == 0, warnings


@pytest.mark.network
def test_khronos_validator_depth_meshes(gltf_validator, scene_meshes, tmp_path):
    _assert_valid(gltf_validator(write_glb(scene_meshes, tmp_path / "scene.glb")))
    _assert_valid(
        gltf_validator(write_glb(scene_meshes, tmp_path / "png.glb", texture_format="png", max_texture=100))
    )


@pytest.mark.network
def test_khronos_validator_layered_and_billboard(gltf_validator, two_plane_scene, person_alpha, tmp_path):
    s = two_plane_scene
    bg_inv = np.where(s.fg_mask, 1.0 / s.z_far, s.inv_z).astype(np.float32)
    ones = np.ones(s.inv_z.shape, np.float32)
    li = LayeredImage([Layer(s.far_rgb, bg_inv, ones), Layer(s.img, s.inv_z, ones)], s.K)
    meshes = layered_meshes(li, target_faces=8000, anchor_distance_m=2.0)
    _assert_valid(gltf_validator(write_glb(meshes, tmp_path / "layered.glb")))
    h, w = person_alpha.shape
    img = np.full((h, w, 3), 150, np.uint8)
    bb = billboard(
        img, person_alpha, DepthResult(np.full((h, w), 0.5, np.float32), "t"), Intrinsics.from_hfov(w, h, 60)
    )
    _assert_valid(gltf_validator(write_glb(bb, tmp_path / "person.glb")))
    mirror = billboard(
        img,
        person_alpha,
        DepthResult(np.full((h, w), 0.5, np.float32), "t"),
        Intrinsics.from_hfov(w, h, 60),
        back="mirror",
    )
    _assert_valid(gltf_validator(write_glb(mirror, tmp_path / "mirror.glb")))


# --------------------------------------------------------------------------- review additions
def _glb_with_json(js: bytes) -> bytes:
    js += b" " * ((-len(js)) % 4)
    return struct.pack("<III", 0x46546C67, 2, 20 + len(js)) + struct.pack("<II", len(js), 0x4E4F534A) + js


@pytest.mark.parametrize(
    "payload",
    [b'{"asset": ', b"\xff\xfe\xfd\xfc", b"[1, 2, 3]", b'{"asset":{"version":"2.0"},"meshes":5}'],
    ids=["truncated-json", "not-utf8", "not-an-object", "wrong-types"],
)
def test_reader_wraps_corrupt_documents(tmp_path, payload):
    p = tmp_path / "bad.glb"
    p.write_bytes(_glb_with_json(payload))
    with pytest.raises(GltfError):
        read_glb_meshes(p)
    g = tmp_path / "bad.gltf"
    g.write_bytes(payload)
    with pytest.raises(GltfError):
        read_glb_meshes(g)


def test_texture_dtypes_are_converted(tmp_path):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    f = np.array([[0, 1, 2]], np.uint32)
    uv = np.array([[0, 0], [1, 0], [0, 1]], np.float32)
    ref = np.arange(48, dtype=np.uint8).reshape(4, 4, 3) * 5
    for tex in (ref.astype(np.uint16) * 257, ref.astype(np.float32) / 255.0):
        (m,) = read_glb_meshes(write_glb([Mesh(v, f, uv, tex)], tmp_path / "t.glb", texture_format="png"))
        np.testing.assert_array_equal(m.texture, ref)
    (g,) = read_glb_meshes(write_glb([Mesh(v, f, uv, ref[..., 0])], tmp_path / "g.glb", texture_format="png"))
    assert g.texture.shape == (4, 4, 3) and np.array_equal(g.texture[..., 2], ref[..., 0])


def test_mirror_billboard_embeds_one_texture(person_alpha, tmp_path):
    h, w = person_alpha.shape
    bb = billboard(
        np.full((h, w, 3), 150, np.uint8),
        person_alpha,
        DepthResult(np.full((h, w), 0.5, np.float32), "t"),
        Intrinsics.from_hfov(w, h, 60),
        back="mirror",
    )
    doc, _, _ = parse_glb(write_glb(bb, tmp_path / "m.glb"))
    assert len(doc["images"]) == 1 and doc["images"][0]["mimeType"] == "image/png"
    assert [m["alphaMode"] for m in doc["materials"]] == ["MASK", "MASK"]
