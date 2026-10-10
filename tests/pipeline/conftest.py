"""Fixtures for the pipeline / orchestration tests (no models, no network)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

#: Options that make every job run without AI models.
NO_MODELS = {"depth_model": "heuristic", "inpaint_model": "opencv", "segment_model": "grabcut"}


@pytest.fixture
def out_dir(tmp_path) -> Path:
    d = tmp_path / "out"
    d.mkdir()
    return d


@pytest.fixture
def settings(tmp_path, isolated_home):
    """Process settings with the output folder inside the test's tmp dir."""
    from vrphoto.core import config

    s = config.Settings(output_dir=str(tmp_path / "default_out"))
    config.set_settings(s)
    yield s
    config.set_settings(None)


@pytest.fixture
def common(out_dir, settings) -> dict:
    """CommonOptions dict: no models, outputs in ``out_dir``."""
    return {**NO_MODELS, "output_dir": str(out_dir)}


@pytest.fixture
def make_job(common):
    """``make_job(kind, inputs, **fields)`` -> job spec with model-free common options."""
    from vrphoto.pipeline.jobs import job_from_dict

    def make(kind: str, inputs: list, common_extra: dict | None = None, **fields):
        c = {**common, **(common_extra or {})}
        return job_from_dict({"kind": kind, "inputs": [str(p) for p in inputs], "common": c, **fields})

    return make


@pytest.fixture
def photo_png(tmp_path, photo_rgb) -> Path:
    from PIL import Image

    p = tmp_path / "subject.png"
    Image.fromarray(photo_rgb).save(p)
    return p


@pytest.fixture
def equirect_path(tmp_path) -> Path:
    """A generated 512x256 (2:1) equirect panorama with structure in every direction."""
    from PIL import Image

    h, w = 256, 512
    yy, xx = np.mgrid[0:h, 0:w]
    chk = (((xx // 32) + (yy // 32)) % 2 * 160 + 40).astype(np.uint8)
    img = np.stack([chk, (xx * 255 // w).astype(np.uint8), (yy * 255 // h).astype(np.uint8)], -1)
    img[150:220, 230:290] = (230, 60, 40)  # a near-ish object straight ahead, below the horizon
    p = tmp_path / "pano360.jpg"
    Image.fromarray(img).save(p, quality=95)
    return p


@pytest.fixture
def video_path(tmp_path) -> Path:
    """A 1 s, 10 fps, 128x96 H.264 clip with an AAC tone (moving red square)."""
    from vrphoto.video.encode import VideoWriter, ffmpeg_exe

    silent = tmp_path / "clip_silent.mp4"
    with VideoWriter(silent, 128, 96, 10.0, "h264_compat") as vw:
        for i in range(10):
            f = np.zeros((96, 128, 3), np.uint8)
            f[..., 1] = (np.arange(128) * 2)[None, :].astype(np.uint8)
            f[..., 2] = (np.arange(96) * 2)[:, None].astype(np.uint8)
            f[30:70, 10 + 6 * i : 50 + 6 * i] = (220, 40, 40)
            vw.write(f)
    out = tmp_path / "clip.mp4"
    subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(silent), "-f", "lavfi", "-i",
         "sine=frequency=440:sample_rate=48000:duration=1", "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)],
        check=True, capture_output=True,
    )  # fmt: skip
    return out


@pytest.fixture
def scene_schema() -> dict:
    from vrphoto.core.config import resource_path

    return json.loads(resource_path("webxr", "viewer", "scene.schema.json").read_text(encoding="utf-8"))


@pytest.fixture
def check_scene(scene_schema):
    """``check_scene(result, mode) -> manifest``: validates the exported viewer scene."""

    def check(result, mode: str) -> dict:
        import jsonschema

        from vrphoto.webxr.scene import validate_manifest

        assert result.scene_id, f"no scene exported (warnings: {result.warnings})"
        scene_dirs = [Path(p) for k, p in result.outputs.items() if k.endswith("VR viewer scene")]
        assert scene_dirs, result.outputs
        sdir = scene_dirs[0]
        manifest = json.loads((sdir / "scene.json").read_text(encoding="utf-8"))
        validate_manifest(manifest)
        jsonschema.validate(manifest, scene_schema)
        assert manifest["mode"] == mode
        assert manifest["id"] == sdir.name
        block = manifest[mode]
        refs = []
        for key in ("media", "asset", "depth"):
            if isinstance(block.get(key), str):
                refs.append(block[key])
        refs += list(block.get("meshes", [])) + list(block.get("alt_assets", []))
        if manifest.get("thumbnail"):
            refs.append(manifest["thumbnail"])
        for r in refs:
            assert (sdir / r).is_file(), f"scene file {r} missing in {sdir}"
        return manifest

    return check
