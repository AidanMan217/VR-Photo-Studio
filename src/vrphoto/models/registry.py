"""Registry of downloadable ONNX models (depth, inpainting, segmentation).

Every model the app can download is described by a frozen :class:`ModelSpec`.  The
built-in table below carries what is known at release time; exact pins (commit sha,
per-file sha256 and sizes, resolved file names) live in the JSON lock file
``models/models.lock.json`` which is overlaid onto the table at import time, so pins can
be refreshed (``python scripts/pin_models.py``) without code edits.

URL scheme: Hugging Face files resolve to
``https://huggingface.co/<repo>/resolve/<revision>/<path>`` (the host can be overridden
with ``$HF_ENDPOINT`` for mirrors); models with a ``url`` override (GitHub releases) are
fetched from that URL directly.

Checksums in :attr:`ModelSpec.sha256` are lowercase hex sha256 digests.  For upstreams
that only publish md5 (rembg's GitHub release assets) the value may be written as
``"md5:<hex>"``; the downloader verifies with the named algorithm.
"""

from __future__ import annotations

import dataclasses
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from ..core.errors import ModelUnavailable
from ..core.log import get_logger

log = get_logger(__name__)

HF_DEFAULT_ENDPOINT = "https://huggingface.co"
LOCK_FILE_NAME = "models.lock.json"
LOCK_SCHEMA = 1
TASKS = ("depth", "inpaint", "segment", "geometry")


@dataclass(frozen=True)
class ModelSpec:
    """Static description of one downloadable model (see docs/CONTRACTS.md section 2).

    ``files[0]`` is the ONNX entry point; further files (e.g. ``model.onnx_data`` external
    weights) are downloaded next to it, preserving their repo-relative paths.
    """

    id: str
    task: str  # "depth" | "inpaint" | "segment" | "geometry"
    title: str
    repo: str  # HF repo id, or "" when ``url`` is a direct download
    revision: str  # pinned commit sha or "main"
    files: tuple[str, ...]
    url: str | None = None  # direct URL override (single-file models)
    sha256: dict[str, str] = field(default_factory=dict)  # per file; "" / missing -> record on download
    size_mb: float = 0.0
    license: str = ""
    noncommercial: bool = False
    dml_ok: bool = True  # False -> always run on CPU under DirectML (known crashes)
    input_size: tuple[int, int] | None = None  # fixed (H, W) if the graph is static
    notes: str = ""
    # ---- extensions (optional, backwards compatible with the contract) ----
    fp16_file: str | None = None  # half-precision variant of files[0] (same repo/revision)
    sizes: dict[str, int] = field(default_factory=dict)  # exact byte sizes per file (from lock)
    experimental: bool = False
    homepage: str = ""

    def __hash__(self) -> int:  # dict fields are unhashable; ids are unique
        return hash(self.id)

    @property
    def entry(self) -> str:
        """Repo-relative path of the ONNX entry-point file."""
        return self.files[0]

    @property
    def is_hf(self) -> bool:
        """True when files are fetched from a Hugging Face repo (token may apply)."""
        return bool(self.repo) and not self.url

    @property
    def pinned(self) -> bool:
        """True when the revision is an exact commit and every file has a checksum."""
        return self.revision != "main" and all(self.sha256.get(f) for f in self.files)

    @property
    def page_url(self) -> str:
        """Human-facing page (model card / homepage) for licence acceptance etc."""
        if self.homepage:
            return self.homepage
        if self.repo:
            return f"{hf_endpoint()}/{self.repo}"
        return self.url or ""


def hf_endpoint() -> str:
    """Hugging Face host, honouring ``$HF_ENDPOINT`` (mirrors, tests)."""
    return (os.environ.get("HF_ENDPOINT") or HF_DEFAULT_ENDPOINT).rstrip("/")


def hf_url(repo: str, revision: str, path: str, endpoint: str | None = None) -> str:
    """Build a Hugging Face ``resolve`` URL for ``path`` inside ``repo`` at ``revision``."""
    base = (endpoint or hf_endpoint()).rstrip("/")
    return f"{base}/{repo}/resolve/{quote(revision, safe='')}/{quote(path, safe='/')}"


def file_url(spec: ModelSpec, path: str) -> str:
    """Download URL of one of ``spec``'s files.

    A ``url`` override applies to the entry-point file; if it ends with ``/`` it is treated
    as a base URL for every file.  Otherwise HF resolve URLs are used.
    """
    if spec.url:
        if spec.url.endswith("/"):
            return spec.url + quote(path, safe="/")
        if path == spec.files[0]:
            return spec.url
        if spec.repo:
            return hf_url(spec.repo, spec.revision, path)
        raise ModelUnavailable(f"No download location for {path} of model {spec.id}")
    if not spec.repo:
        raise ModelUnavailable(f"Model {spec.id} has neither a repository nor a URL")
    return hf_url(spec.repo, spec.revision, path)


def env_var_for(model_id: str) -> str:
    """Name of the env var that points a model id at a local file (bypasses download)."""
    return "VRPHOTO_MODEL_" + model_id.upper().replace("-", "_").replace(".", "_")


# --------------------------------------------------------------------------- built-in table
# TODO(pin): revisions are "main" until scripts/pin_models.py (CI job with HF access)
# fills exact commit shas, sizes and sha256 into models.lock.json.
_BUILTIN: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="da2-small",
        task="depth",
        title="Depth Anything V2 Small",
        repo="onnx-community/depth-anything-v2-small",
        revision="main",
        files=("onnx/model.onnx",),
        fp16_file="onnx/model_fp16.onnx",
        size_mb=99.0,
        license="Apache-2.0",
        notes=(
            "Input pixel_values f32 [1,3,H,W] (ImageNet-normalised RGB, short side 518, multiples "
            "of 14); output predicted_depth [1,H,W] relative inverse depth."
        ),
        homepage="https://huggingface.co/depth-anything/Depth-Anything-V2-Small",
    ),
    ModelSpec(
        id="da2-base",
        task="depth",
        title="Depth Anything V2 Base",
        repo="onnx-community/depth-anything-v2-base",
        revision="main",
        files=("onnx/model.onnx",),
        fp16_file="onnx/model_fp16.onnx",
        size_mb=389.0,
        license="CC-BY-NC-4.0",
        noncommercial=True,
        notes="Same I/O as da2-small. Weights are licensed for non-commercial use only.",
        homepage="https://huggingface.co/depth-anything/Depth-Anything-V2-Base",
    ),
    ModelSpec(
        id="da2-large",
        task="depth",
        title="Depth Anything V2 Large",
        repo="onnx-community/depth-anything-v2-large",
        revision="main",
        files=("onnx/model.onnx",),
        fp16_file="onnx/model_fp16.onnx",
        size_mb=1340.0,
        license="CC-BY-NC-4.0",
        noncommercial=True,
        notes="Same I/O as da2-small. Weights are licensed for non-commercial use only.",
        homepage="https://huggingface.co/depth-anything/Depth-Anything-V2-Large",
    ),
    ModelSpec(
        id="moge2-vitl",
        task="depth",
        title="MoGe-2 ViT-L (metric depth + field of view)",
        repo="Ruicheng/moge-2-vitl-normal-onnx",
        revision="main",
        # TODO(pin): exact file name unverified (HF unreachable when written);
        # pin_models.py resolves the single .onnx in the repo root if this name is wrong.
        files=("model.onnx",),
        size_mb=1320.0,
        license="MIT",
        notes=(
            "Inputs image f32 [B,3,H,W] RGB in [0,1] (normalised inside the graph), num_tokens "
            "int64 scalar (1200-2500). Outputs points [B,H,W,3] affine-invariant, normal, mask, "
            "metric_scale. Code MIT; DINOv2 backbone code Apache-2.0."
        ),
        homepage="https://github.com/microsoft/MoGe",
    ),
    ModelSpec(
        id="da3-mono-large",
        task="depth",
        title="Depth Anything 3 Mono Large",
        repo="SpatialHub/depth-anything-3-onnx",
        revision="main",
        # TODO(pin): community export; file name taken from the model card listing.
        files=("da3mono_large.onnx",),
        size_mb=1400.0,
        license="Apache-2.0",
        experimental=True,
        notes=(
            "Experimental community ONNX export of DA3Mono-Large (upstream lists Apache-2.0; one "
            "community conversion claims CC-BY-NC-4.0 - verify before commercial use). Predicts "
            "relative DEPTH (not disparity); ImageNet-normalised input, long side 504."
        ),
        homepage="https://github.com/ByteDance-Seed/Depth-Anything-3",
    ),
    ModelSpec(
        id="lama",
        task="inpaint",
        title="LaMa (large-mask inpainting)",
        repo="Carve/LaMa-ONNX",
        revision="main",
        files=("lama_fp32.onnx",),
        size_mb=208.0,
        license="Apache-2.0",
        dml_ok=False,
        input_size=(512, 512),
        notes=(
            "Inputs image f32 [1,3,512,512] in [0,1], mask f32 [1,1,512,512] (1 = hole); output "
            "[1,3,512,512] in 0..255. Crashes reported on DirectML -> forced to CPU."
        ),
        homepage="https://github.com/advimman/lama",
    ),
    ModelSpec(
        id="migan",
        task="inpaint",
        title="MI-GAN (fast inpainting)",
        repo="andraniksargsyan/migan",
        revision="main",
        files=("migan_pipeline_v2.onnx",),
        size_mb=28.0,
        license="MIT",
        notes=(
            "Full pipeline: image uint8 [1,3,H,W] RGB + mask uint8 [1,1,H,W] (255 = known) of any "
            "size -> result uint8 [1,3,H,W]. Code and weights MIT (Picsart-AI-Research/MI-GAN "
            "LICENSE-WEIGHTS)."
        ),
        homepage="https://github.com/Picsart-AI-Research/MI-GAN",
    ),
    ModelSpec(
        id="birefnet-lite",
        task="segment",
        title="BiRefNet Lite (subject cut-out)",
        repo="onnx-community/BiRefNet_lite-ONNX",
        revision="main",
        files=("onnx/model.onnx",),
        size_mb=224.0,
        license="MIT",
        input_size=(1024, 1024),
        notes=(
            "Input input_image f32 [1,3,1024,1024] ImageNet-normalised; output output_image "
            "[1,1,1024,1024] logits (apply a sigmoid)."
        ),
        homepage="https://github.com/ZhengPeng7/BiRefNet",
    ),
    ModelSpec(
        id="birefnet-portrait",
        task="segment",
        title="BiRefNet Portrait (people cut-out)",
        repo="",
        revision="v0.0.0",
        files=("BiRefNet-portrait-epoch_150.onnx",),
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-portrait-epoch_150.onnx",
        sha256={"BiRefNet-portrait-epoch_150.onnx": "md5:c3a64a6abf20250d090cd055f12a3b67"},
        size_mb=973.0,
        license="MIT",
        input_size=(1024, 1024),
        notes="rembg release asset (md5 from rembg sessions/birefnet_portrait.py).",
        homepage="https://github.com/ZhengPeng7/BiRefNet",
    ),
    ModelSpec(
        id="isnet-general",
        task="segment",
        title="IS-Net general use (DIS)",
        repo="",
        revision="v0.0.0",
        files=("isnet-general-use.onnx",),
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx",
        sha256={"isnet-general-use.onnx": "md5:fc16ebd8b0c10d971d3513d564d01e29"},
        size_mb=179.0,
        license="Apache-2.0",
        input_size=(1024, 1024),
        notes="rembg release asset (md5 from rembg sessions/dis_general_use.py).",
        homepage="https://github.com/xuebinqin/DIS",
    ),
    ModelSpec(
        id="u2net",
        task="segment",
        title="U2-Net (salient object)",
        repo="",
        revision="v0.0.0",
        files=("u2net.onnx",),
        url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
        sha256={"u2net.onnx": "md5:60024c5c889badc19c04ad937298a77b"},
        size_mb=176.0,
        license="Apache-2.0",
        input_size=(320, 320),
        notes="rembg release asset (md5 from rembg sessions/u2net.py).",
        homepage="https://github.com/xuebinqin/U-2-Net",
    ),
)

BASE_MODELS: dict[str, ModelSpec] = {s.id: s for s in _BUILTIN}
"""The built-in table before the lock-file overlay (used by scripts/pin_models.py)."""

# Fields a lock entry may override (anything else in the JSON is ignored).
_LOCK_FIELDS = ("revision", "files", "sha256", "sizes", "size_mb", "fp16_file", "url")


def default_lock_path() -> Path:
    """Location of the bundled lock file (works from source and PyInstaller)."""
    env = os.environ.get("VRPHOTO_MODELS_LOCK")
    if env:
        return Path(env)
    from ..core.config import resource_path

    return resource_path("models", LOCK_FILE_NAME)


def apply_lock(base: dict[str, ModelSpec], lock: dict[str, Any]) -> dict[str, ModelSpec]:
    """Return a copy of ``base`` with the pins of a parsed lock document applied.

    Unknown model ids and malformed entries are skipped with a warning; the built-in
    values are kept for anything the lock does not specify.
    """
    out = dict(base)
    if not isinstance(lock, dict):
        log.warning("Model lock file is not a JSON object; ignored")
        return out
    schema = lock.get("schema", LOCK_SCHEMA)
    if schema != LOCK_SCHEMA:
        log.warning("Model lock schema %r unsupported (expected %d); ignored", schema, LOCK_SCHEMA)
        return out
    entries = lock.get("models") or {}
    if not isinstance(entries, dict):
        log.warning("Model lock 'models' must be an object; ignored")
        return out
    for mid, entry in entries.items():
        spec = out.get(mid)
        if spec is None:
            log.debug("Lock entry for unknown model %s ignored", mid)
            continue
        try:
            changes = _lock_changes(entry)
            out[mid] = dataclasses.replace(spec, **changes)
        except (TypeError, ValueError) as e:
            log.warning("Malformed lock entry for %s ignored: %s", mid, e)
    return out


def _lock_changes(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("entry must be an object")
    changes: dict[str, Any] = {}
    for key in _LOCK_FIELDS:
        if key not in entry:
            continue
        val = entry[key]
        if key in ("revision",):
            if not isinstance(val, str) or not val:
                raise ValueError(f"{key} must be a non-empty string")
        elif key in ("url", "fp16_file"):
            if val is not None and not isinstance(val, str):
                raise ValueError(f"{key} must be a string or null")
        elif key == "files":
            if not isinstance(val, list) or not val or not all(isinstance(f, str) and f for f in val):
                raise ValueError("files must be a non-empty list of strings")
            val = tuple(val)
        elif key == "sha256":
            if not isinstance(val, dict) or not all(isinstance(v, str) for v in val.values()):
                raise ValueError("sha256 must map file -> hex string")
            val = {str(k): v.lower() for k, v in val.items()}
        elif key == "sizes":
            if not isinstance(val, dict) or not all(isinstance(v, int) and v >= 0 for v in val.values()):
                raise ValueError("sizes must map file -> non-negative int")
            val = {str(k): int(v) for k, v in val.items()}
        elif key == "size_mb":
            if not isinstance(val, (int, float)) or val < 0:
                raise ValueError("size_mb must be a non-negative number")
            val = float(val)
        changes[key] = val
    return changes


def load_lock(path: Path | None = None) -> dict[str, Any]:
    """Read a lock file; returns ``{}`` if it is missing or unreadable (logged)."""
    path = path or default_lock_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning("Cannot read model lock %s: %s", path, e)
        return {}


_reg_lock = threading.Lock()
MODELS: dict[str, ModelSpec] = {}


def reload_registry(lock_path: Path | None = None) -> dict[str, ModelSpec]:
    """(Re)build :data:`MODELS` from the built-in table plus the lock overlay.

    Mutates :data:`MODELS` in place so ``from ...registry import MODELS`` stays valid.
    """
    merged = apply_lock(BASE_MODELS, load_lock(lock_path))
    with _reg_lock:
        MODELS.clear()
        MODELS.update(merged)
    return MODELS


def get_spec(model_id: str) -> ModelSpec:
    """Look up a model by id.  Raises :class:`ModelUnavailable` for unknown ids."""
    spec = MODELS.get(model_id)
    if spec is None:
        known = ", ".join(sorted(MODELS))
        raise ModelUnavailable(f"Unknown model '{model_id}'", hint=f"Known models: {known}")
    return spec


def list_specs(task: str | None = None) -> list[ModelSpec]:
    """All registered models (optionally only those for ``task``), in registry order."""
    return [s for s in MODELS.values() if task is None or s.task == task]


def needs_license_ack(model_id: str) -> bool:
    """True if the model is non-commercial and the user has not acknowledged its licence."""
    from ..core.config import get_settings

    spec = get_spec(model_id)
    return spec.noncommercial and not get_settings().license_accepted(model_id)


reload_registry()

__all__ = [
    "ModelSpec",
    "MODELS",
    "BASE_MODELS",
    "TASKS",
    "get_spec",
    "list_specs",
    "hf_url",
    "hf_endpoint",
    "file_url",
    "env_var_for",
    "apply_lock",
    "load_lock",
    "reload_registry",
    "default_lock_path",
    "needs_license_ack",
]
