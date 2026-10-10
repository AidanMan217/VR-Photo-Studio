"""Job specifications: one dataclass per output mode, JSON round-trippable.

These are the contract between the GUI / CLI and :func:`vrphoto.pipeline.run.run_job`.
Every field has a sensible default so a job can be built from just its inputs.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from ..core.errors import InputError


@dataclass
class CommonOptions:
    depth_model: str = "auto"  # depth.get_estimator name
    inpaint_model: str = "auto"  # inpaint.get_inpainter name
    segment_model: str = "auto"  # segment.get_segmenter name
    device: str = "auto"
    hfov_deg: float | None = None  # None -> MoGe / EXIF / 65
    near_m: float = 1.2
    far_m: float = 15.0
    depth_refine: float = 1.0  # edge-aware smoothing strength 0..2
    max_process_px: int = 4096
    output_dir: str | None = None  # None -> settings.outputs_dir
    name: str | None = None  # output base name; None -> input stem
    export_scene: bool = True  # also add to the WebXR viewer library
    quest_model: str = "quest3"  # quest3 | quest2 | pcvr


@dataclass
class JobBase:
    kind: ClassVar[str] = "base"
    inputs: list[str] = field(default_factory=list)
    common: CommonOptions = field(default_factory=CommonOptions)


@dataclass
class DepthJob(JobBase):
    """Save depth maps (16-bit PNG + colour preview)."""

    kind: ClassVar[str] = "depth"
    save_color: bool = True
    save_16bit: bool = True


@dataclass
class StereoJob(JobBase):
    """Flat stereoscopic 3D photo (for Quest media players / 3D TVs / anaglyph glasses)."""

    kind: ClassVar[str] = "stereo"
    layout: str = "sbs"  # sbs | half_sbs | tb | half_tb | anaglyph | anaglyph_gray | cross
    ipd_m: float = 0.063
    strength: float = 1.0
    convergence_m: float | None = None  # None = auto
    max_parallax_pct: float = 3.0
    look_behind: str = "band"  # none | band  (inpaint occluded background)
    swap_eyes: bool = False
    image_format: str = "jpg"  # jpg | png
    wiggle_gif: bool = False


@dataclass
class ParallaxJob(JobBase):
    """'Living photo' video: virtual camera motion through the 3D photo."""

    kind: ClassVar[str] = "parallax"
    path: str = "circle"  # dolly | circle | swing | zoom | kenburns | orbit | custom
    keyframes: list[dict] | None = None  # for path == "custom"
    duration_s: float = 6.0
    fps: float = 30.0
    amplitude_m: float = 0.06
    output: str = "flat"  # flat | sbs | vr180
    width: int | None = None  # None -> source width (capped by preset)
    loop: bool = True
    look_behind: str = "band"
    encoder_preset: str = "auto"  # auto -> by output/quest_model
    ipd_m: float = 0.063


@dataclass
class VR180Job(JobBase):
    """VR180 stereo photo (or short looping video) from a normal photo."""

    kind: ClassVar[str] = "vr180"
    eye_px: int | None = None  # None -> preset (quest3 3840, quest2 2880)
    size_mult: float = 1.0  # >1 enlarges the photo in the 180 field (FOV)
    fill: str = "blur"  # blur | black | outpaint (add-on)
    ipd_m: float = 0.063
    look_behind: str = "band"
    as_video_s: float = 0.0  # >0 also writes an N-second still video (for players w/o photo support)
    image_format: str = "jpg"


@dataclass
class PanoJob(JobBase):
    """360 output. Input either an equirect 360 photo (-> stereo ODS) or a normal photo
    (-> 360 sphere with filled surroundings)."""

    kind: ClassVar[str] = "pano"
    input_is_360: bool | None = None  # None -> auto (2:1 aspect)
    stereo: bool = True  # top-bottom stereo via ODS (360 input only)
    width: int | None = None  # equirect width; None -> preset
    fill: str = "blur"  # blur | black | outpaint (add-on) for normal photos
    ipd_m: float = 0.064
    as_video_s: float = 0.0


@dataclass
class SceneJob(JobBase):
    """Interactive 6DoF '3D photo' for the WebXR viewer (layered meshes, look behind)."""

    kind: ClassVar[str] = "scene"
    target_faces: int = 300_000
    look_behind: str = "band"  # none | band | cleanplate
    anchor_distance_m: float = 2.0
    max_baseline_m: float = 0.4  # how far the viewer may move before holes appear
    export_glb: bool = True
    export_obj: bool = False


@dataclass
class ObjectJob(JobBase):
    """Person / object you can walk around (passthrough AR on Quest 3)."""

    kind: ClassVar[str] = "object"
    method: str = "billboard"  # billboard | trellis | trellis2 | hunyuan | sharp
    height_m: float = 1.7
    thickness: float = 0.11  # median body thickness as a fraction of height
    back: str = "diffuse"  # diffuse | mirror (billboard back texture)
    export_splat: bool = True
    placement: str = "hit-test"  # hit-test | front-floor
    seed: int = 0


@dataclass
class CaptureJob(JobBase):
    """Many photos / a phone video -> Gaussian splat (COLMAP + Brush add-on)."""

    kind: ClassVar[str] = "capture"
    video_fps: float = 2.0
    max_images: int = 200
    steps: int = 15000
    max_splats: int = 750_000
    crop_pct: float = 98.0
    as_object: bool = True  # True -> object mode (placed on floor); False -> splat scene


@dataclass
class VideoJob(JobBase):
    """Convert a normal video to 3D (SBS) or VR180 video, frame by frame."""

    kind: ClassVar[str] = "video"
    output: str = "sbs"  # sbs | half_sbs | tb | vr180
    ipd_m: float = 0.063
    strength: float = 1.0
    max_parallax_pct: float = 2.5
    temporal_smooth: float = 0.6
    max_long: int = 1920
    start_s: float = 0.0
    end_s: float | None = None
    keep_audio: bool = True
    encoder_preset: str = "auto"


JOB_TYPES: dict[str, type[JobBase]] = {
    c.kind: c
    for c in (DepthJob, StereoJob, ParallaxJob, VR180Job, PanoJob, SceneJob, ObjectJob, CaptureJob, VideoJob)
}


@dataclass
class JobResult:
    outputs: dict[str, str] = field(default_factory=dict)  # label -> path
    scene_id: str | None = None
    warnings: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def job_to_dict(job: JobBase) -> dict[str, Any]:
    d = dataclasses.asdict(job)
    d["kind"] = job.kind
    return d


def job_from_dict(d: dict[str, Any]) -> JobBase:
    d = dict(d)
    kind = d.pop("kind", None)
    if kind not in JOB_TYPES:
        raise InputError(f"Unknown job kind: {kind!r}. Valid: {', '.join(JOB_TYPES)}")
    cls = JOB_TYPES[kind]
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise InputError(f"Unknown option(s) for {kind}: {', '.join(sorted(unknown))}")
    common = d.pop("common", None) or {}
    cnames = {f.name for f in dataclasses.fields(CommonOptions)}
    bad = set(common) - cnames
    if bad:
        raise InputError(f"Unknown common option(s): {', '.join(sorted(bad))}")
    return cls(common=CommonOptions(**common), **d)


def job_to_json(job: JobBase) -> str:
    return json.dumps(job_to_dict(job), indent=2)


def job_from_json(text_or_path: str | Path) -> JobBase:
    p = Path(text_or_path) if not str(text_or_path).lstrip().startswith("{") else None
    text = p.read_text(encoding="utf-8") if p is not None else str(text_or_path)
    try:
        return job_from_dict(json.loads(text))
    except json.JSONDecodeError as e:
        raise InputError(f"Invalid job JSON: {e}") from e
