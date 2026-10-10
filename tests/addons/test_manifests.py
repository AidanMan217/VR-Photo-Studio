"""Bundled add-on manifests load, validate and are internally consistent."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from _addon_helpers import dummy_manifest, write_manifest

from vrphoto.addons import manager
from vrphoto.addons.manifest import (
    STEP_SCHEMAS,
    STEP_TYPES,
    load_manifest,
    manifests_dir,
    parse_manifest,
    workers_dir,
)
from vrphoto.core.errors import AddonError

BUNDLED = sorted(manifests_dir().glob("*.toml"))
EXPECTED_IDS = {"trellis", "trellis2", "hunyuan", "sharp", "sdxl360", "capture"}
REQUIRED = ("id", "title", "description", "license", "kind", "tasks", "steps")


def test_all_expected_manifests_present():
    ids = {load_manifest(p).id for p in BUNDLED}
    assert EXPECTED_IDS <= ids


@pytest.mark.parametrize("path", BUNDLED, ids=lambda p: p.stem)
def test_manifest_valid(path: Path):
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    for key in REQUIRED:
        assert key in raw, f"{path.name}: missing {key}"
    m = load_manifest(path)
    assert m.id == path.stem, "file name must match the id"
    assert m.sha256 and len(m.sha256) == 64
    assert m.title and m.description and m.license
    assert m.tasks
    assert m.disk_gb > 0 and m.min_vram_gb > 0
    for step in m.steps:
        assert step["type"] in STEP_TYPES
        assert set(step) - {"type", "title", "platforms", "optional", "weight"} <= set(STEP_SCHEMAS[step["type"]])
    if m.kind == "venv":
        assert m.python == "3.11"
        assert m.steps[0]["type"] == "venv"
        wp = m.worker_path()
        assert wp is not None and wp.is_file() and wp.parent == workers_dir()
        assert m.steps[-1]["type"] == "check", "venv add-ons must verify the install last"
    else:
        assert m.entry and m.binaries


@pytest.mark.parametrize("path", BUNDLED, ids=lambda p: p.stem)
def test_manifest_pins(path: Path):
    """Exact pins everywhere: torch from the CUDA 12.8 index, sources by commit, archives by hash."""
    m = load_manifest(path)
    for step in m.steps:
        t = step["type"]
        if t == "pip":
            for pkg in step["packages"]:
                if pkg.startswith("{src}") or " @ " in pkg:
                    continue
                spec = pkg.split(";")[0].strip()
                assert "==" in spec, f"{m.id}: unpinned package {pkg}"
            torch = [p for p in step["packages"] if p.startswith("torch==")]
            if torch:
                assert step.get("index_url") == "https://download.pytorch.org/whl/cu128" or step.get(
                    "torch_backend"
                ) == "cu128"
                major, minor = (int(x) for x in torch[0].split("==")[1].split(".")[:2])
                assert (major, minor) < (2, 12), "torch >= 2.12 has no cu128 wheels"
        elif t == "github_archive":
            assert re.fullmatch(r"[0-9a-f]{40}", step["ref"]), f"{m.id}: github_archive must pin a commit"
        elif t == "archive" and m.kind == "binary":
            assert re.fullmatch(r"[0-9a-f]{64}", step.get("sha256", "")), f"{m.id}: binaries need sha256"
        elif t == "hf_download":
            assert step.get("revision"), f"{m.id}: hf_download needs a revision"


def test_licenses_and_regions():
    by_id = {m.id: m for m in manager.list_addons(include_experimental=True)}
    assert by_id["sharp"].noncommercial
    assert {"GB", "KR", "DE", "FR"} <= set(by_id["hunyuan"].region_restricted)
    assert by_id["hunyuan"].requires_license_ack()
    assert by_id["trellis2"].hf_gated and by_id["trellis2"].experimental
    assert not by_id["trellis"].experimental and not by_id["capture"].experimental
    assert "outpaint_360" in by_id["sdxl360"].tasks
    assert "capture_to_splat" in by_id["capture"].tasks
    notes = " ".join(by_id["hunyuan"].notices())
    assert "European Union" in notes and "South Korea" in notes
    assert by_id["capture"].to_dict()["kind"] == "binary"


def test_list_addons_filters_experimental():
    stable = {m.id for m in manager.list_addons()}
    everything = {m.id for m in manager.list_addons(include_experimental=True)}
    assert "trellis" in stable and "trellis2" not in stable
    assert "trellis2" in everything
    assert manager.addons_for_task("image_to_3d")


def test_get_manifest_unknown():
    with pytest.raises(AddonError, match="Unknown add-on"):
        manager.get_manifest("does-not-exist")


def test_extra_manifest_dir(manifest_dir):
    write_manifest(manifest_dir, "dummy", dummy_manifest())
    manager.reload_manifests()
    m = manager.get_manifest("dummy")
    assert m.path == manifest_dir / "dummy.toml"
    assert m.worker_path() == workers_dir() / "dummy_worker.py"


def test_broken_manifest_is_skipped(manifest_dir, caplog):
    write_manifest(manifest_dir, "broken", 'id = "broken"\n')
    write_manifest(manifest_dir, "dummy", dummy_manifest())
    manager.reload_manifests()
    ids = {m.id for m in manager.list_addons(True)}
    assert "dummy" in ids and "broken" not in ids


# ------------------------------------------------------------------ validation errors
BASE = {
    "id": "x",
    "title": "X",
    "description": "d",
    "license": "MIT",
    "kind": "venv",
    "python": "3.11",
    "tasks": ["t"],
    "worker": "dummy_worker.py",
    "steps": [{"type": "venv"}, {"type": "check"}],
}


def _parse(**changes):
    data = {**BASE, **changes}
    for k in [k for k, v in data.items() if v is None]:
        del data[k]
    return parse_manifest(data, Path("x.toml"))


def test_parse_ok():
    m = _parse()
    assert m.id == "x" and m.kind == "venv" and m.worker == "dummy_worker.py"


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"title": None}, "missing required key 'title'"),
        ({"bogus": 1}, "unknown key"),
        ({"id": "Bad Id"}, "lowercase"),
        ({"kind": "docker"}, "kind must be one of"),
        ({"tasks": []}, "must not be empty"),
        ({"python": "py3"}, "python"),
        ({"worker": None}, "needs a worker"),
        ({"worker": "missing_worker.py"}, "not found"),
        ({"worker": "../evil.py"}, "file name"),
        ({"steps": []}, "non-empty"),
        ({"steps": [{"type": "check"}]}, "first step"),
        ({"steps": [{"type": "venv"}, {"type": "frobnicate"}]}, "unknown step type"),
        ({"steps": [{"type": "venv"}, {"type": "pip"}]}, "missing required key 'packages'"),
        ({"steps": [{"type": "venv"}, {"type": "pip", "packages": []}]}, "must not be empty"),
        ({"steps": [{"type": "venv"}, {"type": "pip", "packages": ["a"], "colour": "red"}]}, "unknown key"),
        ({"steps": [{"type": "venv"}, {"type": "archive", "url": "ftp://x/y.zip", "dest": "d"}]}, "http"),
        ({"steps": [{"type": "venv"}, {"type": "archive", "url": "https://x/y.zip", "dest": "../d"}]}, "relative"),
        ({"steps": [{"type": "venv"}, {"type": "archive", "url": "https://x/y.zip", "dest": "d", "sha256": "abc"}]},
         "64 hex"),
        ({"steps": [{"type": "venv"}, {"type": "github_archive", "repo": "nope", "ref": "1", "dest": "s"}]},
         "owner/name"),
        ({"steps": [{"type": "venv"}, {"type": "pip", "packages": ["a"], "platforms": ["amiga"]}]}, "platform"),
        ({"steps": [{"type": "venv"}, {"type": "venv"}]}, "exactly one venv"),
        ({"region_restricted": ["EU"]}, "ISO 3166"),
        ({"region_restricted": ["UK"]}, "GB not UK"),
        ({"region_restricted": ["deu"]}, "ISO 3166"),
        ({"min_vram_gb": -1}, "negative"),
        ({"steps": [{"type": "venv"}, {"type": "hf_download", "repo": "a/b", "repo_type": "zip"}]}, "repo_type"),
        ({"steps": [{"type": "venv"}, {"type": "worker"}]}, "missing required key 'args'"),
        ({"steps": [{"type": "venv"}, {"type": "archive", "url": "https://x/y.zip", "dest": "{src}/../../x"}]},
         "stay inside"),
        ({"cwd": "{addon_dir}/.."}, "stay inside"),
    ],
)
def test_parse_errors(changes, match):
    with pytest.raises(AddonError, match=match):
        _parse(**changes)


def test_binary_kind_rules():
    with pytest.raises(AddonError, match="binary"):
        _parse(kind="binary", worker=None, steps=[{"type": "pip", "packages": ["a==1"]}])
    with pytest.raises(AddonError, match="entry"):
        _parse(kind="binary", worker=None, entry="nocolon", steps=[{"type": "check"}])
    with pytest.raises(AddonError, match="patterns"):
        _parse(kind="binary", worker=None, steps=[{"type": "check"}], binaries={"tool": {}})
    m = _parse(kind="binary", worker=None, python=None, entry="a.b:c", steps=[{"type": "check"}],
               binaries={"tool": {"path_names": ["tool"]}})
    assert m.kind == "binary" and "tool" in m.binaries


def test_invalid_toml_syntax(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("id = \n", encoding="utf-8")
    with pytest.raises(AddonError, match="Invalid add-on manifest bad.toml"):
        load_manifest(p)


def test_unsupported_platform_notice():
    m = _parse(platforms=["darwin"]) if __import__("sys").platform != "darwin" else _parse(platforms=["win32"])
    assert not m.supported
    assert any("operating system" in n for n in m.notices())


def test_placeholder_paths_inside_are_fine():
    m = _parse(cwd="{src}/trellis", steps=[{"type": "venv"},
                                           {"type": "archive", "url": "https://x/y.zip", "dest": "{addon_dir}/m"}])
    assert m.cwd == "{src}/trellis"


def test_hunyuan_prefetches_the_flashvdm_turbo_vae():
    """enable_flashvdm() swaps in hunyuan3d-vae-v2-mini-turbo; it must come with the install."""
    from vrphoto.addons import manager

    m = manager.get_manifest("hunyuan")
    pats = [p for s in m.steps if s["type"] == "hf_download" for p in s.get("allow_patterns", [])]
    assert "hunyuan3d-dit-v2-mini-turbo/*" in pats and "hunyuan3d-vae-v2-mini-turbo/*" in pats


def test_release_checksums_match_the_published_digests():
    """sha256 of the COLMAP 4.2.1 / Brush 0.3.0 release assets as shown on their GitHub
    release pages (checked 2026-10-10)."""
    from vrphoto.addons import manager

    want = {
        "colmap-x64-windows-cuda.zip": "e9c5cbd84c2ea986d2e970a2473fc2d2e6b34a2cdcf5d3df2765c319a63af881",
        "brush-app-x86_64-pc-windows-msvc.zip": "b68e3e9cf052d51bf3ee30776fa5a364de7f2ba13b58443128ff797bb7bcfcd6",
        "brush-app-x86_64-unknown-linux-gnu.tar.xz": "4f0f9a8785d1951c62df26aae247c02c5bba32b00f40b06df4e1c9b867399e20",
    }
    got = {s["url"].rsplit("/", 1)[-1]: s.get("sha256") for s in manager.get_manifest("capture").steps
           if s["type"] == "archive"}
    assert got == want
