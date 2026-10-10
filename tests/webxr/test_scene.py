"""Tests for vrphoto.webxr.scene: ids, manifest validation (cross-checked with jsonschema),
export, listing and deletion of viewer scenes."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vrphoto.core.errors import InputError
from vrphoto.webxr import scene as sc

SCHEMA_PATH = Path(sc.__file__).resolve().parent / "viewer" / "scene.schema.json"
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,80}$")


def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def base(mode: str, block: dict, **extra) -> dict:
    m = {"schema": sc.SCHEMA_ID, "id": f"t-{mode}", "title": f"Test {mode}", "mode": mode, mode: block}
    m.update(extra)
    return m


GOOD = [
    base("photo3d", {"meshes": ["bg.glb", "fg.glb"], "anchor_distance_m": 2.0, "eye_height_m": 1.5}),
    base(
        "photo3d", {"meshes": ["only.glb"]}, created="2026-10-09T12:00:00Z", thumbnail="thumb.jpg", units="m"
    ),
    base(
        "stereo",
        {
            "media": "x_LR.jpg",
            "kind": "image",
            "layout": "sbs",
            "aspect": 1.5,
            "screen_width_m": 3,
            "distance_m": 4,
        },
    ),
    base("stereo", {"media": "v_TB.mp4", "kind": "video", "layout": "tb"}),
    base("stereo", {"media": "flat.jpg", "kind": "image", "layout": "mono"}),
    base(
        "sphere",
        {
            "media": "p_180_LR.jpg",
            "kind": "image",
            "coverage": 180,
            "layout": "sbs",
            "depth": "depth.png",
            "radius_m": 20,
        },
    ),
    base("sphere", {"media": "p_360.mp4", "kind": "video", "coverage": 360, "layout": "mono", "depth": None}),
    base(
        "sphere",
        {
            "media": "p.jpg",
            "kind": "image",
            "coverage": 360.0,
            "layout": "tb",
            "depth_near_m": 1.2,
            "depth_far_m": 30,
        },
    ),
    base(
        "object",
        {
            "asset": "person.glb",
            "format": "glb",
            "height_m": 1.7,
            "pivot": "base",
            "placement": "hit-test",
            "distance_m": 1.5,
            "alt_assets": ["person.splat"],
            "up": "y",
        },
    ),
    base(
        "object", {"asset": "thing.spz", "format": "spz"}, env={"prefer": "ar", "background": "passthrough"}
    ),
    # a capture the app already scaled to metres: shown 1:1, height_m names the subject
    base("object", {"asset": "capture.splat", "format": "splat", "scale": 1.0, "height_m": 1.7}),
    base("splat", {"asset": "room.ply", "format": "ply", "up": "-y", "scale": 0.5}),
    base(
        "splat",
        {"asset": "room.splat", "format": "splat"},
        notices=["Contains output of a non-commercial model"],
    ),
    # unknown top-level keys are allowed by the schema (forward compatibility)
    base("splat", {"asset": "a.spz", "format": "spz", "up": "z"}, source="capture", extra={"anything": 1}),
    # a non-matching extra block is fine as long as it is itself valid
    base("photo3d", {"meshes": ["a.glb"]}, stereo={"media": "s.jpg", "kind": "image", "layout": "sbs"}),
]


def bad(desc: str, m: dict) -> tuple[str, dict]:
    return desc, m


def _without(d: dict, key: str) -> dict:
    d = copy.deepcopy(d)
    d.pop(key)
    return d


def _with(d: dict, path: str, value) -> dict:
    d = copy.deepcopy(d)
    cur = d
    parts = path.split(".")
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value
    return d


P3 = GOOD[0]
ST = GOOD[2]
SP = GOOD[5]
OB = GOOD[8]
SL = GOOD[11]

BAD = [
    bad("not an object", ["a"]),
    bad("missing schema", _without(P3, "schema")),
    bad("missing id", _without(P3, "id")),
    bad("missing title", _without(P3, "title")),
    bad("missing mode", _without(P3, "mode")),
    bad("wrong schema", _with(P3, "schema", "vrphoto.scene/2")),
    bad("id uppercase", _with(P3, "id", "Beach")),
    bad("id leading dash", _with(P3, "id", "-beach")),
    bad("id too long", _with(P3, "id", "a" * 82)),
    bad("id with slash", _with(P3, "id", "a/b")),
    bad("id not string", _with(P3, "id", 5)),
    bad("title too long", _with(P3, "title", "x" * 201)),
    bad("title not string", _with(P3, "title", None)),
    bad("created not string", _with(P3, "created", 2026)),
    bad("units feet", _with(P3, "units", "ft")),
    bad("unknown mode", _with(P3, "mode", "hologram")),
    bad("mode block missing", _without(P3, "photo3d")),
    bad("photo3d meshes empty", _with(P3, "photo3d.meshes", [])),
    bad("photo3d meshes not list", _with(P3, "photo3d.meshes", "bg.glb")),
    bad("photo3d meshes non-string", _with(P3, "photo3d.meshes", ["a.glb", 3])),
    bad("photo3d anchor zero", _with(P3, "photo3d.anchor_distance_m", 0)),
    bad("photo3d eye height string", _with(P3, "photo3d.eye_height_m", "1.5")),
    bad("photo3d eye height bool", _with(P3, "photo3d.eye_height_m", True)),
    bad("stereo missing kind", _with(ST, "stereo", {"media": "a.jpg", "layout": "sbs"})),
    bad("stereo bad layout", _with(ST, "stereo.layout", "half_sbs")),
    bad("stereo bad kind", _with(ST, "stereo.kind", "gif")),
    bad("stereo negative distance", _with(ST, "stereo.distance_m", -1)),
    bad("stereo aspect bool", _with(ST, "stereo.aspect", True)),
    bad("stereo block not object", _with(ST, "stereo", "x.jpg")),
    bad("sphere coverage 270", _with(SP, "sphere.coverage", 270)),
    bad("sphere coverage string", _with(SP, "sphere.coverage", "180")),
    bad("sphere coverage bool", _with(SP, "sphere.coverage", True)),
    bad("sphere depth number", _with(SP, "sphere.depth", 3)),
    bad("sphere radius zero", _with(SP, "sphere.radius_m", 0)),
    bad("sphere missing layout", _with(SP, "sphere", {"media": "a.jpg", "kind": "image", "coverage": 180})),
    bad("object bad format", _with(OB, "object.format", "obj")),
    bad("object bad pivot", _with(OB, "object.pivot", "top")),
    bad("object bad placement", _with(OB, "object.placement", "wall")),
    bad("object height negative", _with(OB, "object.height_m", -1.7)),
    bad("object alt not list", _with(OB, "object.alt_assets", "x.splat")),
    bad("object bad up", _with(OB, "object.up", "x")),
    bad("object scale zero", _with(OB, "object.scale", 0)),
    bad("object scale text", _with(OB, "object.scale", "1")),
    bad("splat glb format", _with(SL, "splat.format", "glb")),
    bad("splat bad up", _with(SL, "splat.up", "x")),
    bad("splat scale zero", _with(SL, "splat.scale", 0)),
    bad("splat missing asset", _with(SL, "splat", {"format": "ply"})),
    bad("env not object", _with(SL, "env", "vr")),
    bad("env bad prefer", _with(SL, "env", {"prefer": "mr"})),
    bad("env background number", _with(SL, "env", {"background": 0})),
    bad("notices not list", _with(SL, "notices", "NC")),
    bad("notices non-string", _with(SL, "notices", [1])),
    bad("other block invalid", _with(P3, "stereo", {"media": "s.jpg", "kind": "image", "layout": "x"})),
]

# Python is deliberately stricter than the schema here (jsonschema accepts these).
STRICTER = [
    ("path traversal media", _with(ST, "stereo.media", "../secret.jpg")),
    ("absolute windows path", _with(OB, "object.asset", "C:\\Users\\x\\person.glb")),
    ("subfolder mesh", _with(P3, "photo3d.meshes", ["layers/bg.glb"])),
    ("thumbnail traversal", _with(P3, "thumbnail", "../../etc/passwd")),
    ("id trailing newline", _with(P3, "id", "beach\n")),
]


# --------------------------------------------------------------------------- ids
def test_new_scene_id_format_and_uniqueness():
    ids = {sc.new_scene_id("Beach at Dusk!") for _ in range(300)}
    assert len(ids) == 300
    for sid in ids:
        assert ID_PATTERN.match(sid)
        assert sid.startswith("beach-at-dusk-")
        assert re.fullmatch(r"beach-at-dusk-[0-9a-f]{6}", sid)


@pytest.mark.parametrize(
    "title,prefix",
    [
        ("", "scene-"),
        ("   ", "scene-"),
        ("!!!", "scene-"),
        ("Été à Paris", "ete-a-paris-"),
        ("東京タワー", "scene-"),
        ("My_Photo.v2 (final)", "my-photo-v2-final-"),
        (None, "scene-"),
    ],
)
def test_new_scene_id_slugify(title, prefix):
    sid = sc.new_scene_id(title)
    assert sid.startswith(prefix)
    assert ID_PATTERN.match(sid)


def test_new_scene_id_long_title_is_truncated():
    sid = sc.new_scene_id("word " * 100)
    assert ID_PATTERN.match(sid) and len(sid) <= 81
    assert not sid.startswith("-") and "--" not in sid


# --------------------------------------------------------------------------- validation
@pytest.mark.parametrize("manifest", GOOD, ids=[f"{m['mode']}-{i}" for i, m in enumerate(GOOD)])
def test_validate_good(manifest):
    sc.validate_manifest(manifest)


@pytest.mark.parametrize("desc,manifest", BAD, ids=[d for d, _ in BAD])
def test_validate_bad(desc, manifest):
    with pytest.raises(InputError) as ei:
        sc.validate_manifest(manifest)
    msg = str(ei.value)
    assert msg.startswith("The VR scene description is invalid")
    assert ei.value.problems  # machine-readable list as well


def test_validate_reports_all_problems():
    m = _with(_with(ST, "stereo.layout", "x"), "stereo.distance_m", 0)
    with pytest.raises(InputError) as ei:
        sc.validate_manifest(m)
    assert len(ei.value.problems) == 2
    assert "stereo.layout must be one of: sbs, tb, mono" in str(ei.value)
    assert "stereo.distance_m must be a number greater than 0" in str(ei.value)


@pytest.mark.parametrize("desc,manifest", STRICTER, ids=[d for d, _ in STRICTER])
def test_validate_rejects_unsafe_file_names(desc, manifest):
    with pytest.raises(InputError):
        sc.validate_manifest(manifest)


def test_schema_is_valid_draft_2020_12():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(schema())


def test_cross_check_with_jsonschema():
    """validate_manifest (pure Python) must agree with jsonschema + scene.schema.json."""
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(schema())
    for m in GOOD:
        assert not list(validator.iter_errors(m)), m
        sc.validate_manifest(m)
    for desc, m in BAD:
        assert list(validator.iter_errors(m)), f"jsonschema accepted bad case: {desc}"
        with pytest.raises(InputError):
            sc.validate_manifest(m)
    for desc, m in STRICTER:
        assert not list(validator.iter_errors(m)), f"schema now rejects {desc}; move it to BAD"


def test_python_mirror_covers_schema():
    """Every block/property/required list in the schema is mirrored in scene._BLOCKS."""
    s = schema()
    props = s["properties"]
    assert set(props["mode"]["enum"]) == set(sc.MODES)
    for mode in sc.MODES:
        block = props[mode]
        required, mirror = sc._BLOCKS[mode]
        assert set(block["properties"]) == set(mirror), mode
        assert set(block["required"]) == set(required), mode
    assert set(s["required"]) == {"schema", "id", "title", "mode"}
    assert props["id"]["pattern"] == "^[a-z0-9][a-z0-9._-]{0,80}$"


def test_scene_module_does_not_need_jsonschema():
    code = f"import sys, vrphoto.webxr.scene as s; s.validate_manifest({GOOD[0]!r}); print('jsonschema' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "False"


# --------------------------------------------------------------------------- export
def _rgb(h=300, w=800):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([xx * 255 // w, yy * 255 // h, np.full_like(xx, 90)], -1).astype(np.uint8)


def test_export_scene_writes_everything(tmp_path):
    lib = tmp_path / "lib"
    src = tmp_path / "bg_src.glb"
    src.write_bytes(b"glTF-bg")
    manifest = {"title": "Beach at dusk", "mode": "photo3d", "photo3d": {"meshes": ["bg.glb", "fg.glb"]}}
    original = copy.deepcopy(manifest)
    d = sc.export_scene(manifest, {"bg.glb": src, "fg.glb": b"glTF-fg"}, library=lib, thumbnail=_rgb())
    assert manifest == original  # input not mutated
    assert d.parent == lib and d.is_dir()
    m = json.loads((d / "scene.json").read_text(encoding="utf-8"))
    assert m["schema"] == sc.SCHEMA_ID
    assert m["id"] == d.name and m["id"].startswith("beach-at-dusk-") and ID_PATTERN.match(m["id"])
    assert m["units"] == "m"
    assert m["thumbnail"] == "thumb.jpg"
    assert datetime.strptime(m["created"], "%Y-%m-%dT%H:%M:%SZ")
    assert (d / "bg.glb").read_bytes() == b"glTF-bg"
    assert (d / "fg.glb").read_bytes() == b"glTF-fg"
    with Image.open(d / "thumb.jpg") as im:
        assert im.format == "JPEG" and max(im.size) == 512 and im.size == (512, 192)
    assert not [p for p in d.iterdir() if p.name.startswith(".")], "temporary files left behind"
    sc.validate_manifest(m)


def test_export_scene_default_library_is_settings_library(isolated_home):
    from vrphoto.core.config import get_settings

    d = sc.export_scene(
        {"id": "my-scene", "title": "T", "mode": "splat", "splat": {"asset": "w.splat", "format": "splat"}},
        {"w.splat": b"\0" * 32},
    )
    assert d == get_settings().library_dir / "my-scene"
    assert d.is_relative_to(isolated_home)


def test_export_scene_keeps_given_values_and_small_thumbnail(tmp_path):
    m = {
        "schema": sc.SCHEMA_ID,
        "id": "fixed-id",
        "title": "  Spaced  ",
        "created": "2020-01-01T00:00:00Z",
        "mode": "stereo",
        "stereo": {"media": "a.jpg", "kind": "image", "layout": "sbs"},
    }
    d = sc.export_scene(m, {"a.jpg": b"jpg"}, library=tmp_path, thumbnail=np.zeros((20, 30, 4), np.uint8))
    out = json.loads((d / "scene.json").read_text(encoding="utf-8"))
    assert out["id"] == "fixed-id" and out["created"] == "2020-01-01T00:00:00Z" and out["title"] == "Spaced"
    with Image.open(d / "thumb.jpg") as im:
        assert im.size == (30, 20)  # never upscaled


@pytest.mark.parametrize(
    "thumb",
    [np.zeros((10, 10), np.uint8), np.ones((10, 10, 3), np.float32) * 0.5, np.zeros((10, 10, 1), np.uint8)],
)
def test_export_scene_accepts_thumbnail_variants(tmp_path, thumb):
    d = sc.export_scene(
        {"title": "x", "mode": "splat", "splat": {"asset": "w.spz", "format": "spz"}},
        {"w.spz": b"s"},
        library=tmp_path,
        thumbnail=thumb,
    )
    assert (d / "thumb.jpg").is_file()


def test_export_scene_reexport_replaces_files(tmp_path):
    m = {"id": "again", "title": "A", "mode": "splat", "splat": {"asset": "w.splat", "format": "splat"}}
    sc.export_scene(m, {"w.splat": b"one"}, library=tmp_path)
    d = sc.export_scene(m | {"title": "B"}, {"w.splat": b"two"}, library=tmp_path)
    assert (d / "w.splat").read_bytes() == b"two"
    assert json.loads((d / "scene.json").read_text(encoding="utf-8"))["title"] == "B"
    # Referencing a file that already exists in the scene folder is fine.
    sc.export_scene(m | {"title": "C"}, {}, library=tmp_path)


def test_export_scene_same_source_and_destination(tmp_path):
    m = {"id": "same", "title": "S", "mode": "splat", "splat": {"asset": "w.splat", "format": "splat"}}
    d = sc.export_scene(m, {"w.splat": b"abc"}, library=tmp_path)
    sc.export_scene(m, {"w.splat": d / "w.splat"}, library=tmp_path)
    assert (d / "w.splat").read_bytes() == b"abc"


def test_export_scene_hard_links_assets_on_the_same_volume(tmp_path):
    """E2E-5: the library must not silently keep a second full copy of every video."""
    out = tmp_path / "out" / "clip_LR.mp4"
    out.parent.mkdir()
    out.write_bytes(b"video" * 1000)
    m = {
        "id": "clip",
        "title": "C",
        "mode": "stereo",
        "stereo": {"media": "clip_LR.mp4", "kind": "video", "layout": "sbs"},
    }
    d = sc.export_scene(m, {"clip_LR.mp4": out}, library=tmp_path / "lib")
    assert os.path.samefile(d / "clip_LR.mp4", out)
    assert os.stat(out).st_nlink == 2
    assert not [p for p in d.iterdir() if p.name.startswith(".")]
    # deleting the output keeps the scene playable, and vice versa
    out.unlink()
    assert (d / "clip_LR.mp4").read_bytes() == b"video" * 1000
    sc.delete_scene("clip", library=tmp_path / "lib")
    assert not d.exists()


def test_export_scene_copies_when_hard_links_fail(tmp_path, monkeypatch):
    src = tmp_path / "w_src.splat"
    src.write_bytes(b"splat")

    def no_link(a, b):
        Path(b).write_bytes(b"partial")  # a failing link must not leave a temp file behind
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr(sc.os, "link", no_link)
    m = {"id": "copy", "title": "C", "mode": "splat", "splat": {"asset": "w.splat", "format": "splat"}}
    d = sc.export_scene(m, {"w.splat": src}, library=tmp_path / "lib")
    assert (d / "w.splat").read_bytes() == b"splat"
    assert not os.path.samefile(d / "w.splat", src)
    assert not [p for p in d.iterdir() if p.name.startswith(".")]


@pytest.mark.parametrize("how", ["env", "attribute"])
def test_export_scene_never_hard_links_a_onedrive_output(tmp_path, monkeypatch, how):
    """E2E-5: OneDrive (Pictures is often backed up there) cannot sync a file that has a
    second hard link outside its folder, so outputs in a cloud-synced folder are copied."""
    cloud = tmp_path / "OneDrive"
    src = cloud / "Pictures" / "clip_LR.mp4"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"video")
    monkeypatch.setattr(sc.sys, "platform", "win32")
    for var in sc._ONEDRIVE_VARS:
        monkeypatch.delenv(var, raising=False)
    if how == "env":
        monkeypatch.setenv("OneDrive", str(cloud))
    else:  # e.g. Dropbox: the folders are cloud-files placeholders
        monkeypatch.setattr(sc, "_cloud_marked", lambda p: Path(p) == cloud.resolve())
    links: list = []
    real_link = os.link
    monkeypatch.setattr(sc.os, "link", lambda a, b: links.append((a, b)))
    m = {"id": "od", "title": "C", "mode": "stereo", "stereo": {"media": "clip_LR.mp4", "kind": "video", "layout": "sbs"}}
    d = sc.export_scene(m, {"clip_LR.mp4": src}, library=tmp_path / "lib")
    assert links == []
    assert (d / "clip_LR.mp4").read_bytes() == b"video"
    assert not os.path.samefile(d / "clip_LR.mp4", src)
    # an output outside the synced folder is still linked
    other = tmp_path / "out" / "x_LR.mp4"
    other.parent.mkdir()
    other.write_bytes(b"other")
    monkeypatch.setattr(sc.os, "link", real_link)
    m2 = {**m, "id": "local", "stereo": {**m["stereo"], "media": "x_LR.mp4"}}
    d2 = sc.export_scene(m2, {"x_LR.mp4": other}, library=tmp_path / "lib")
    assert os.path.samefile(d2 / "x_LR.mp4", other)


def test_cloud_marked_reads_windows_placeholder_attributes(monkeypatch):
    def fake_stat(attrs=0, tag=0):
        return lambda p, follow_symlinks=True: types.SimpleNamespace(st_file_attributes=attrs, st_reparse_tag=tag)

    for attrs, tag, want in [
        (0x10, 0, False),  # plain folder
        (0x10 | 0x400, 0xA0000003, False),  # junction (mount point), not a cloud file
        (0x10 | 0x400, 0x9000701A, True),  # IO_REPARSE_TAG_CLOUD_7
        (0x20 | 0x400000, 0, True),  # RECALL_ON_DATA_ACCESS (online-only file)
        (0x20 | 0x80000, 0, True),  # PINNED ("always keep on this device")
    ]:
        monkeypatch.setattr(sc, "os", types.SimpleNamespace(stat=fake_stat(attrs, tag)))
        assert sc._cloud_marked(Path("x")) is want, (hex(attrs), hex(tag))


@pytest.mark.parametrize(
    "name",
    [
        "../evil.glb",
        "sub/dir.glb",
        "sub\\dir.glb",
        "..",
        ".",
        "",
        "C:evil.glb",
        "a\0b.glb",
        " lead.glb",
        "con.glb",
        "dots.",
        "scene.json",
        "Scene.JSON",  # same file as scene.json on Windows
    ],
)
def test_export_scene_rejects_bad_asset_names(tmp_path, name):
    m = {"title": "x", "mode": "splat", "splat": {"asset": "w.splat", "format": "splat"}}
    with pytest.raises(InputError):
        sc.export_scene(m, {"w.splat": b"x", name: b"y"}, library=tmp_path)
    assert not any(tmp_path.rglob("scene.json"))


def test_export_scene_rejects_thumb_asset_with_thumbnail(tmp_path):
    m = {"title": "x", "mode": "splat", "splat": {"asset": "w.splat", "format": "splat"}}
    with pytest.raises(InputError, match="thumb.jpg"):
        sc.export_scene(m, {"w.splat": b"x", "thumb.jpg": b"j"}, library=tmp_path, thumbnail=_rgb(10, 10))


def test_export_scene_rejects_case_insensitive_clashes(tmp_path):
    """Names that differ only in case are one file on Windows: never silently overwrite."""
    m = {"title": "x", "mode": "splat", "splat": {"asset": "w.splat", "format": "splat"}}
    with pytest.raises(InputError, match="upper/lower case"):
        sc.export_scene(m, {"w.splat": b"x", "W.SPLAT": b"y"}, library=tmp_path)
    with pytest.raises(InputError, match="thumb.jpg"):
        sc.export_scene(m, {"w.splat": b"x", "THUMB.jpg": b"j"}, library=tmp_path, thumbnail=_rgb(10, 10))
    assert not any(tmp_path.rglob("scene.json"))


def test_export_scene_rejects_missing_and_bad_assets(tmp_path):
    m = {"title": "x", "mode": "photo3d", "photo3d": {"meshes": ["bg.glb", "fg.glb"]}}
    with pytest.raises(InputError, match=r"photo3d.meshes\[1\] \(fg.glb\)"):
        sc.export_scene(m, {"bg.glb": b"x"}, library=tmp_path)
    with pytest.raises(InputError, match="not found"):
        sc.export_scene(m, {"bg.glb": b"x", "fg.glb": tmp_path / "nope.glb"}, library=tmp_path)
    with pytest.raises(InputError, match="file path or bytes"):
        sc.export_scene(m, {"bg.glb": b"x", "fg.glb": 42}, library=tmp_path)
    assert not any(tmp_path.rglob("scene.json"))


def test_export_scene_rejects_invalid_manifest_and_nan(tmp_path):
    with pytest.raises(InputError):
        sc.export_scene({"title": "x", "mode": "stereo"}, {}, library=tmp_path)
    m = {
        "title": "x",
        "mode": "splat",
        "splat": {"asset": "w.splat", "format": "splat", "scale": float("inf")},
    }
    with pytest.raises(InputError, match="cannot be saved"):
        sc.export_scene(m, {"w.splat": b"x"}, library=tmp_path)
    with pytest.raises(InputError):
        sc.export_scene("not a dict", {}, library=tmp_path)  # type: ignore[arg-type]
    assert not any(tmp_path.rglob("scene.json"))


def test_export_scene_untitled(tmp_path):
    d = sc.export_scene(
        {"mode": "splat", "splat": {"asset": "w.splat", "format": "splat"}},
        {"w.splat": b"x"},
        library=tmp_path,
    )
    m = json.loads((d / "scene.json").read_text(encoding="utf-8"))
    assert m["title"] == "Untitled scene" and m["id"].startswith("untitled-scene-")


# --------------------------------------------------------------------------- list / delete
def _mk(lib: Path, sid: str, created: str | None, *, thumb: bool = True) -> Path:
    m = {"id": sid, "title": sid.upper(), "mode": "splat", "splat": {"asset": "w.splat", "format": "splat"}}
    if created:
        m["created"] = created
    return sc.export_scene(m, {"w.splat": b"x"}, library=lib, thumbnail=_rgb(8, 8) if thumb else None)


def test_list_scenes_sorted_newest_first_with_urls(tmp_path):
    _mk(tmp_path, "old", "2026-01-01T00:00:00Z")
    _mk(tmp_path, "new", "2026-10-09T12:00:00Z", thumb=False)
    _mk(tmp_path, "mid", "2026-05-05T10:00:00+02:00")
    items = sc.list_scenes(tmp_path)
    assert [i["id"] for i in items] == ["new", "mid", "old"]
    assert set(items[0]) == {"id", "title", "mode", "thumbnail", "url", "created"}
    assert items[0]["url"] == "/view.html?scene=new"
    assert items[0]["thumbnail"] is None
    assert items[1]["thumbnail"] == "/scenes/mid/thumb.jpg"
    assert items[2] == {
        "id": "old",
        "title": "OLD",
        "mode": "splat",
        "thumbnail": "/scenes/old/thumb.jpg",
        "url": "/view.html?scene=old",
        "created": "2026-01-01T00:00:00Z",
    }
    json.dumps(items)  # JSON-serialisable for /api/scenes


def test_list_scenes_skips_broken_and_foreign_folders(tmp_path):
    good = _mk(tmp_path, "good", "2026-01-01T00:00:00Z")
    (tmp_path / "empty").mkdir()  # no scene.json (e.g. export in progress)
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "scene.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "invalid").mkdir()
    (tmp_path / "invalid" / "scene.json").write_text(json.dumps({"mode": "nope"}), encoding="utf-8")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "scene.json").write_text((good / "scene.json").read_text(), encoding="utf-8")
    (tmp_path / "Bad Name").mkdir()
    (tmp_path / "Bad Name" / "scene.json").write_text((good / "scene.json").read_text(), encoding="utf-8")
    (tmp_path / "loose-file.txt").write_text("x", encoding="utf-8")
    assert [i["id"] for i in sc.list_scenes(tmp_path)] == ["good"]


def test_list_scenes_uses_folder_name_and_mtime_fallback(tmp_path):
    d = _mk(tmp_path, "orig", None)
    m = json.loads((d / "scene.json").read_text(encoding="utf-8"))
    m.pop("created")
    renamed = tmp_path / "renamed"
    d.rename(renamed)
    (renamed / "scene.json").write_text(json.dumps(m), encoding="utf-8")
    [item] = sc.list_scenes(tmp_path)
    assert item["id"] == "renamed" and item["url"] == "/view.html?scene=renamed"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", item["created"])


def test_list_scenes_empty_and_default_library(isolated_home):
    assert sc.list_scenes() == []


def test_delete_scene(tmp_path):
    d = _mk(tmp_path, "gone", "2026-01-01T00:00:00Z")
    sc.delete_scene("gone", tmp_path)
    assert not d.exists()
    sc.delete_scene("gone", tmp_path)  # idempotent
    assert sc.list_scenes(tmp_path) == []


@pytest.mark.parametrize("sid", ["../x", "a/b", "", "..", "A", None, "x\\y"])
def test_delete_scene_rejects_bad_ids(tmp_path, sid):
    with pytest.raises(InputError):
        sc.delete_scene(sid, tmp_path)


def test_delete_scene_refuses_non_scene_folder(tmp_path):
    (tmp_path / "photos").mkdir()
    (tmp_path / "photos" / "keep.jpg").write_bytes(b"x")
    with pytest.raises(InputError, match="does not look like a VR scene"):
        sc.delete_scene("photos", tmp_path)
    assert (tmp_path / "photos" / "keep.jpg").exists()


def test_read_manifest_errors(tmp_path):
    with pytest.raises(InputError, match="No scene"):
        sc.read_manifest(tmp_path)
    (tmp_path / "scene.json").write_text("[", encoding="utf-8")
    with pytest.raises(InputError, match="damaged"):
        sc.read_manifest(tmp_path)
