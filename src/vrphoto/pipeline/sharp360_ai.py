"""SHARP 360: the AI photo-editor back sources and the M6 helpers (docs/design/SHARP360.md 3.5.6, 4.4).

Kept out of :mod:`~vrphoto.pipeline.sharp360` (which registers the sources in its ``SOURCES``
table through thin wrappers, so importing this module never runs at app start).

**Back sources** (experimental, never chosen by "auto"):

* ``qwen`` (:func:`_src_qwen`) - the Qwen photo editor add-on (``qwenedit``: Qwen-Image-Edit-2511
  as a GGUF plus a multiple-angles camera LoRA and a Lightning LoRA, run by a pinned headless
  ComfyUI inside the add-on).  Slow on a GTX 10-series card: several minutes per view.
* ``comfyui`` (:func:`_src_comfyui`) - the user's OWN ComfyUI server with their own API-format
  workflow (``comfyui_url`` / ``comfyui_workflow``): the app fills ``{{IMAGE}} {{PROMPT}}
  {{SEED}} {{AZIMUTH}} {{WIDTH}} {{HEIGHT}}`` and queues it once per view.  No add-on; an
  unreachable server is one warning and the next back.

Both make NOMINAL views (only the azimuth is known) and share :func:`_nominal_back`:

1. **Input**: the plain cut-out of the photo (``run._cutout_rgba``, at most 1024 px) on
   #c8c8c8 - light grey keeps hair edges readable (the qwen worker composites it itself).
2. **Phase 1, back candidates**: ``Opts.tries`` pictures from behind (seeds ``seed + k``;
   qwen: one worker call, the model loads once; comfyui: ``{{AZIMUTH}}`` = 180).
3. **Choice** (:func:`_choose_back`): every candidate is cut out (V1), fitted to the
   mirrored front silhouette (V2 / V3, :func:`~vrphoto.geometry.views_lift.lock_back_view`)
   and checked for a face (YuNet, :func:`~vrphoto.segment.faces.frontal_face_score`; above
   :data:`~vrphoto.segment.faces.REJECT_SCORE` in the head region = the editor painted the
   front onto the back of the head: rejected, "a face was found on the back view").  Among
   the usable ones the lowest ``(1 - IoU) + 0.5 chi2 + face`` wins: ``chi2`` of the HSV
   histograms of the rim bands (3 % of the size inside the outline) of the lined-up
   candidate and the levelled photo (the same clothes), ``face`` the face score from 0.5.
   None usable: ``ok=False`` with the reason, so the chain warns and falls back.
4. **Phase 2, side views** (balanced: 90 / 270; best: also 135 / 225; fast: none) made with
   the chosen back as a second reference picture (qwen; the comfyui placeholders carry one
   picture, so its sides are made from the cut-out alone).  A 135 / 225 view with a face is
   dropped.  A failure here keeps the back alone (a note).
5. :func:`~vrphoto.geometry.views_lift.view_based_back` (nominal path: back lock, side
   registration, Reinhard rim colours, left / right check, hull, relief, mould, colours,
   side holes) with the job's relief, colour match, stride and seed; the V11 face term is
   the chosen back's face score.  The contact sheet (``ctx.previews["views"]``) shows the
   lined-up views and the rejected candidates.

Qwen views go to the view cache (:mod:`~vrphoto.pipeline.view_cache`, one entry per picture:
the cut-out's digest, view, seed, steps, prompt, size and - for side views - the chosen
back's digest), so a re-run with another seam, quality or polish skips the minutes of
generation; the views a failed or cancelled worker run finished (``views.partial.jsonl``)
are kept there too.  ComfyUI views are not cached (the user's workflow and models may
change between runs).

**M6 helpers** used by every view-based source:

* face checks (:func:`face_scores`, :func:`with_face_check`): MV-Adapter sets whose back view
  shows a face are rejected, a face on the user's own back photo is only a warning;
* the SHARP relief (:func:`sharp_depth_fn`, 'Back detail' "sharp"): SHARP runs once more on
  the lined-up back view (the add-on's list input) and
  :func:`~vrphoto.geometry.sharp_relief.sharp_relief` turns its cloud into the disparity map
  ``view_based_back`` takes from ``depth_fn``.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import AddonError, AddonNotInstalled, Cancelled, InputError, VRPhotoError
from ..core.log import get_logger
from ..core.types import ProgressReporter
from ..geometry.back_mould import BackPart
from . import run as R
from .sharp360 import (
    QUALITY,
    Ctx,
    _comfy,
    _cutout_input,
    _empty_cloud,
    _levelled_input,
    _relief_fn,
    _relief_notice,
    _view_segmenter,
)

log = get_logger(__name__)

QWEN_ADDON = "qwenedit"
_QWEN_TASK = "views_from_photo"
#: Background the cut-out is composited on (design 3.5.6: #c8c8c8 keeps hair edges readable).
CUTOUT_GREY = (200, 200, 200)
#: Long side of the generated views (the qwen worker's default; the comfyui {{WIDTH}} / {{HEIGHT}}).
VIEW_LONG_SIDE = 768
_SIZE_MULTIPLE = 16  # Qwen-Image's latent grid (8 x VAE, 2 x 2 patches); harmless for other models
#: The qwen worker's step range ('AI steps' outside it are clamped; 0 = its own default: 4 with
#: the 4-step Lightning LoRA, 40 without).
QWEN_STEPS_RANGE = (4, 40)
#: Side views of phase 2 (``(name, phi)``): left / right always (not with 'Fast'), the back
#: quarters with 'Best' (design V10: 135 / 225 for nominal sources).
SIDE_VIEWS: tuple[tuple[str, float], ...] = (("right", 90.0), ("left", 270.0))
QUARTER_VIEWS: tuple[tuple[str, float], ...] = (("back_right", 135.0), ("back_left", 225.0))
#: Views checked for a face (design 3.5.6: the 135 / 180 / 225 views).
FACE_PHIS = (135.0, 180.0, 225.0)
#: Weight of the rim-colour chi-square in the candidate cost.
CHI2_WEIGHT = 0.5
_RIM_FRAC = 0.03  # rim band: within 3 % of the subject's size of the outline (as views_lift V4)
_HSV_BINS = (16, 4, 4)
#: A view slower than this (seconds) makes the qwen source say how slow it is.
SLOW_VIEW_S = 120.0
#: One ComfyUI view may take at most this long (then it is interrupted).
COMFY_VIEW_TIMEOUT_S = 3600.0
_COMFY_POLL_S = 0.5
#: Field of view SHARP is told for the lined-up back view (an orthographic picture: a long lens
#: comes closest; :func:`~vrphoto.geometry.sharp_relief.sharp_relief` normalises the depth, so
#: only the shape matters).  Unverified on real views (M6 tuning).
RELIEF_HFOV_DEG = 30.0
#: ``used`` entry of :func:`sharp_depth_fn` (not a model id: SHARP's notice is always there).
SHARP_RELIEF = "addon:sharp"


def _close(a: float, b: float, tol: float = 1.0) -> bool:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0) < tol


def _clause(text: Any) -> str:
    """An error message as a clause inside a sentence: no final full stop, a plain first
    word in lower case ("The graphics card ..." -> "the graphics card ..."; "ComfyUI ..." stays)."""
    t = str(text or "").strip().rstrip(".")
    word = t.split(" ", 1)[0]
    if len(word) > 1 and word[0].isupper() and word[1:].isalpha() and word[1:].islower():
        t = t[0].lower() + t[1:]
    return t


def _why(e: BaseException) -> str:
    return _clause(e) if isinstance(e, VRPhotoError | OSError) and str(e) else type(e).__name__


def _output_size(width: int, height: int, long_side: int = VIEW_LONG_SIDE) -> tuple[int, int]:
    """``(w, h)``: the aspect of ``width x height``, long side ``long_side``, multiples of 16."""
    w, h = max(1, int(width)), max(1, int(height))
    s = float(long_side) / max(w, h)
    m = _SIZE_MULTIPLE
    return max(m, int(round(w * s / m)) * m), max(m, int(round(h * s / m)) * m)


def _composite(rgba: np.ndarray, grey: tuple[int, int, int] = CUTOUT_GREY) -> np.ndarray:
    """RGBA uint8 -> RGB uint8 over ``grey``."""
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    out = rgba[..., :3].astype(np.float32) * a + np.asarray(grey, np.float32) * (1.0 - a)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _join(*notes: Any) -> str:
    return " ".join(dict.fromkeys(str(n) for n in notes if n))


# --------------------------------------------------------------------------- face checks (M6)
def face_score(view: Any, run: Any = None) -> float:
    """YuNet's best frontal-face score in the head region of a segmented view (0.0 when there
    is none, or the detector is unavailable or fails - the check is optional)."""
    from ..segment import faces

    try:
        return float(faces.frontal_face_score(view.rgb, view.alpha, cancel=getattr(run, "cancel", None)))
    except Cancelled:
        raise
    except Exception as e:  # the face check never costs the back
        log.warning("Face check of the %s view failed (%s); skipped", getattr(view, "name", "?"), e)
        return 0.0


def face_scores(views: Sequence[Any], run: Any = None) -> dict[str, float]:
    """``{view name: score}`` of the views at 135 / 180 / 225 degrees (:data:`FACE_PHIS`)."""
    out: dict[str, float] = {}
    for v in views:
        if any(_close(v.phi_deg, p) for p in FACE_PHIS):
            out[str(v.name)] = round(face_score(v, run), 3)
    return out


def with_face_check(
    part: BackPart, scores: dict[str, float], *, reject: bool, what: str = "the back view"
) -> BackPart:
    """Record a face check in ``part.diag["face"]`` and the V11 ``face`` term (the best score
    from :data:`~vrphoto.segment.faces.MIN_SCORE`, else 0) of a view-based back.  A score above
    :data:`~vrphoto.segment.faces.REJECT_SCORE` makes the part unusable when ``reject`` (an AI
    view: "a face was found on the back view"), else adds a warning (the user's own photo is
    never rejected)."""
    from ..segment import faces

    diag = dict(part.diag or {})
    name, best = max(scores.items(), key=lambda kv: kv[1], default=("", 0.0))
    diag["face"] = {"scores": dict(scores), "max": best, "view": name, "reject_score": faces.REJECT_SCORE}
    qc = diag.get("qc")
    if part.ok and isinstance(qc, dict) and isinstance(qc.get("terms"), dict):
        terms = {**qc["terms"], "face": best if best >= faces.MIN_SCORE else 0.0}
        cost = float(sum(float(t) for t in terms.values() if t is not None))
        diag["qc"] = {**qc, "terms": terms, "cost": cost}
    found = best > faces.REJECT_SCORE
    if found and reject and part.ok:
        reason = f"a face was found on {what} (score {best:.2f})"
        diag.update(ok=False, reason=reason)
        log.warning("View-based back (%s) rejected: %s", part.source, reason)
        return replace(part, ok=False, reason=reason, diag=diag)
    if found and not reject and part.ok:
        diag["warning"] = _join(
            diag.get("warning"),
            f"A face was found on {what}; if it shows the front of the person, the back will too - pick a "
            "photo taken from behind.",
        )
    part.diag = diag
    return part


# --------------------------------------------------------------------------- the SHARP relief (M6)
def _grey_key(rgb: np.ndarray) -> np.ndarray:
    """The subject of a picture composited on views_lift's 0.5 grey (128)."""
    return (np.abs(rgb.astype(np.int16) - 128).max(axis=2) > 3).astype(np.float32)


def _sharp_outputs(outs: dict[str, Any]) -> tuple[str, float, int, int]:
    """``(ply, f_px, width, height)`` of the first picture of a SHARP run (list or single outputs)."""

    def first(many: str, one: str) -> Any:
        v = outs.get(many)
        return v[0] if isinstance(v, list | tuple) and v else outs.get(one)

    ply = first("plys", "ply")
    if not (ply and Path(str(ply)).is_file()):
        raise AddonError("SHARP finished but did not return a 3D model of the back view.")
    return (
        str(ply),
        float(first("f_pxs", "f_px")),
        int(first("widths", "width")),
        int(first("heights", "height")),
    )


def sharp_depth_fn(
    ctx: Ctx,
    rep: ProgressReporter | None = None,
    *,
    ortho: Any = None,
    explicit: bool = False,
    fallback: Callable[[np.ndarray], Any] | None = None,
    used: list[str] | None = None,
) -> Callable[[np.ndarray], Any]:
    """The ``depth_fn`` of 'Back detail' "sharp" for :func:`~vrphoto.geometry.views_lift.
    view_based_back`: SHARP (the add-on that made the front, list input ``{"images": [...]}``,
    field of view :data:`RELIEF_HFOV_DEG`) runs on the lined-up back view it is handed (``S x
    S`` on 0.5 grey - the relief then lines up with the locked back by construction), and
    :func:`~vrphoto.geometry.sharp_relief.sharp_relief` turns SHARP's cloud into the disparity
    map (1 = nearest the back camera) inside the expected back silhouette (the mirrored front
    silhouette of ``ortho``; else the non-grey pixels).

    When SHARP fails (or its cloud has no usable surface) ``fallback`` (the depth model) is
    used instead - with a warning when the user chose "sharp" (``explicit``), quietly for
    'Automatic'.  Every run is recorded in ``ctx.relief["runs"]`` (debug.json); a successful
    one appends :data:`SHARP_RELIEF` to ``used``.  :class:`Cancelled` propagates."""
    run, item = ctx.run, ctx.item
    rep = rep or ProgressReporter.null()
    used = used if used is not None else []
    runs: list[dict[str, Any]] = ctx.relief.setdefault("runs", [])
    alpha = None
    if ortho is not None:
        a0 = np.asarray(getattr(ortho, "alpha0", np.zeros((0, 0))), dtype=bool)
        alpha = a0[:, ::-1].astype(np.float32) if a0.ndim == 2 and a0.any() else None

    def depth_fn(rgb: np.ndarray) -> Any:
        from ..core.imaging import save_image
        from ..geometry.sharp_relief import sharp_relief

        k = len(runs) + 1
        rec: dict[str, Any] = {"run": k, "ok": False}
        runs.append(rec)
        t0 = time.perf_counter()
        try:
            img = np.ascontiguousarray(np.asarray(rgb)[..., :3], dtype=np.uint8)
            a = alpha if alpha is not None and alpha.shape == img.shape[:2] else _grey_key(img)
            path = save_image(run.work_dir(item) / f"sharp_relief_{k}.png", img)
            rep(0.0, "Running SHARP on the back view (surface detail)")

            def worker_progress(frac: float, msg: str = "") -> None:
                rep(frac, f"SHARP on the back view: {msg or 'working'}")

            outs = R._run_addon(
                run,
                item,
                "sharp",
                "image_to_splat",
                {"images": [str(path)]},
                {"hfov_deg": RELIEF_HFOV_DEG, "name": f"relief{k}"},
                ProgressReporter(worker_progress),
            )
            ply, f_px, width, height = _sharp_outputs(outs)
            stats: dict[str, Any] = {}
            d = sharp_relief(
                ply,
                f_px=f_px,
                width=width,
                height=height,
                out_size=img.shape[:2],
                alpha=a,
                diag=stats,
                cancel=run.cancel,
            )
            rec.update(ok=True, seconds=round(time.perf_counter() - t0, 2), stats=stats)
            used.append(SHARP_RELIEF)
            log.info("SHARP relief of the back view: %.1f s", time.perf_counter() - t0)
            return d
        except Cancelled:
            raise
        except Exception as e:  # SHARP or its cloud: the depth model shapes the back instead
            why = _why(e)
            rec.update(error=why, seconds=round(time.perf_counter() - t0, 2))
            log.warning(
                "SHARP relief of the back view failed (%s)", why, exc_info=not isinstance(e, VRPhotoError)
            )
            if explicit:
                instead = "the depth model was used instead" if fallback is not None else "the back is smooth"
                run.warn(f"SHARP could not shape the surface detail of the back ({why}); {instead}.")
            return fallback(rgb) if fallback is not None else None

    return depth_fn


# --------------------------------------------------------------------------- rim colours
def _rim_hist(rgb: np.ndarray, alpha: np.ndarray, width_px: float) -> np.ndarray | None:
    """Normalised HSV histogram of the rim band (inside the outline, within ``width_px``)."""
    import cv2

    m = (np.asarray(alpha) >= 0.5).astype(np.uint8)
    if not m.any():
        return None
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    band = (m > 0) & (d >= 1.5) & (d <= max(2.5, float(width_px)))
    if int(band.sum()) < 16:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], band.astype(np.uint8), list(_HSV_BINS), [0, 180, 0, 256, 0, 256])
    h = hist.ravel().astype(np.float64)
    return h / max(float(h.sum()), 1e-12)


def rim_chi2(rgb: np.ndarray, alpha: np.ndarray, ref_rgb: np.ndarray, ref_alpha: np.ndarray) -> float | None:
    """Chi-square distance (0 = same colours, 1 = none shared) between the rim-band HSV
    histograms of two lined-up pictures (rim: 3 % of the reference subject's size); None when
    a band is too small to say."""
    ref = np.asarray(ref_alpha) >= 0.5
    if not ref.any():
        return None
    rows, cols = np.flatnonzero(ref.any(axis=1)), np.flatnonzero(ref.any(axis=0))
    width = _RIM_FRAC * max(rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1)
    p, q = _rim_hist(rgb, alpha, width), _rim_hist(ref_rgb, ref, width)
    if p is None or q is None:
        return None
    s = p + q
    nz = s > 0
    return float(0.5 * np.sum((p[nz] - q[nz]) ** 2 / s[nz]))


# --------------------------------------------------------------------------- back candidates
@dataclass
class _Cand:
    """One AI picture from behind: where it is, how it fits, why it was not used."""

    index: int  # 1-based
    seed: int
    file: Path
    cached: bool = False
    view: Any = None  # segmented GenView (None: unreadable / no subject)
    locked: Any = None  # lined up in the levelled back frame (contact sheet)
    lock: dict[str, Any] = field(default_factory=dict)
    face: float = 0.0
    chi2: float | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.view is not None and not self.reason

    def cost(self) -> float:
        from ..segment import faces

        iou = float(self.lock.get("iou") or 0.0)
        face = self.face if self.face >= faces.MIN_SCORE else 0.0
        return (1.0 - iou) + CHI2_WEIGHT * float(self.chi2 or 0.0) + face

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "candidate": self.index,
            "seed": self.seed,
            "file": self.file.name,
            "cached": self.cached,
            "ok": self.ok,
            "face": round(self.face, 3),
        }
        for k in ("iou", "iou_locked", "pose_frac"):
            if k in self.lock:
                out[k] = self.lock[k]
        if self.chi2 is not None:
            out["chi2"] = round(self.chi2, 4)
        if self.ok:
            out["cost"] = round(self.cost(), 4)
        if self.reason:
            out["reason"] = self.reason
        return out


def _load_view(
    path: Path, phi: float, name: str, seg: Any, run: Any, meta: dict[str, Any]
) -> tuple[Any, str]:
    """``(segmented GenView | None, why not)`` of one generated picture."""
    from ..geometry import views_lift as vl

    dropped: list[str] = []
    try:
        views = vl.load_views([(phi, path)], dropped=dropped)
    except InputError as e:
        return None, (dropped[0] if dropped else str(e))
    views = [replace(v, name=name, meta={**v.meta, **meta}) for v in views]
    views = vl.segment_views(views, seg, cancel=run.cancel, dropped=dropped)
    if not views:
        return None, (dropped[0] if dropped else "no subject was found in it")
    return views[0], ""


def _choose_back(ctx: Ctx, cands: list[_Cand], ortho: Any, seg: Any, rep: ProgressReporter) -> _Cand | None:
    """Design 3.5.6 phase 2: cut out, line up and face-check every candidate; the usable one
    with the lowest :meth:`_Cand.cost` (the first on a tie), None when none is usable."""
    from ..geometry import views_lift as vl
    from ..segment import faces

    run = ctx.run
    ref_rgb = np.asarray(ortho.rgba)[..., :3]
    ref_alpha = np.asarray(ortho.alpha0, dtype=bool)
    for k, c in enumerate(cands):
        run.check()
        rep(k / max(1, len(cands)), "Choosing the back view")
        c.view, why = _load_view(c.file, 180.0, "back", seg, run, {"candidate": c.index, "seed": c.seed})
        if c.view is None:
            c.reason = _clause(why)
            continue
        c.face = face_score(c.view, run)
        c.locked, c.lock = vl.lock_back_view(ortho, ctx.canon, c.view, cancel=run.cancel)
        if c.face > faces.REJECT_SCORE:
            c.reason = "a face was found on the back view"
        elif not c.lock.get("ok"):
            c.reason = str(c.lock.get("reason") or "it does not fit the photo's outline")
        else:
            c.chi2 = rim_chi2(c.locked.rgb, c.locked.alpha, ref_rgb, ref_alpha)
    usable = [c for c in cands if c.ok]
    log.info("Back candidates: %s", [c.summary() for c in cands])
    return min(usable, key=lambda c: (c.cost(), c.index)) if usable else None


def _candidates_reason(cands: list[_Cand], title: str) -> str:
    """Why no back candidate could be used, in a few plain words."""
    if not cands:
        return f"{title} made no back view"
    reasons = list(dict.fromkeys(c.reason or "it could not be used" for c in cands))
    if len(cands) == 1:
        return reasons[0]
    if reasons == ["a face was found on the back view"]:
        return f"a face was found on all {len(cands)} back views"
    return f"none of its {len(cands)} back views could be used: {'; '.join(reasons)}"


def _rejected_views(cands: list[_Cand], chosen: _Cand | None) -> list[Any]:
    """The candidates that were not used, as views for the contact sheet ("not used")."""
    out = []
    for c in cands:
        if c is chosen:
            continue
        v = c.locked if c.locked is not None else c.view
        if v is None:
            continue
        why = c.reason or "another back view fitted better"
        out.append(replace(v, name=f"back {c.index}", meta={**v.meta, "used": False, "reason": why}))
    return out


# --------------------------------------------------------------------------- generators
class _Generator:
    """What :func:`_nominal_back` needs from a source."""

    source = ""
    title = ""  # in sentences ("The Qwen photo editor could not ...")
    label = ""  # in progress messages ("Qwen photo editor: ...")
    references = False  # side views get the chosen back as a reference picture
    made = 0  # pictures generated in this job (not from the cache)
    seconds = 0.0  # time spent generating them
    hits = 0  # pictures found in the view cache
    from_cache: frozenset[int] = frozenset()  # back seeds found in the view cache

    def start(self, rep: ProgressReporter) -> None:  # pragma: no cover - interface
        pass

    def backs(self, seeds: list[int], rep: ProgressReporter) -> tuple[dict[int, Path], str]:
        raise NotImplementedError  # pragma: no cover - interface

    def sides(
        self, views: list[tuple[str, float]], back: Path, rep: ProgressReporter
    ) -> tuple[dict[str, Path], str]:
        raise NotImplementedError  # pragma: no cover - interface

    def info(self) -> dict[str, Any]:
        return {"made": self.made, "seconds": round(self.seconds, 1), "cache_hits": self.hits}


class _QwenGen(_Generator):
    """The Qwen photo editor add-on (``views_from_photo``) with the view cache."""

    source = "qwen"
    references = True

    def __init__(self, ctx: Ctx, manifest: Any, image: Path, digest: str):
        from . import view_cache

        self.ctx = ctx
        self.manifest = manifest
        self.title = "The Qwen photo editor"
        self.label = "Qwen photo editor"
        self.addon_title = str(getattr(manifest, "title", "") or self.label)  # names the worker's own notes
        self.image = image
        self.digest = digest
        o = ctx.opts
        self.steps = int(min(max(o.steps, QWEN_STEPS_RANGE[0]), QWEN_STEPS_RANGE[1])) if o.steps else 0
        self.prompt = o.back_prompt
        self.cache = view_cache.ViewCache(view_cache.default_root(getattr(ctx.run, "settings", None)))
        self.worker = view_cache.worker_version(manifest)
        self.outs: dict[str, Any] = {}
        self.folder = ctx.run.work_dir(ctx.item) / QWEN_ADDON

    # ---- cache
    def _key(self, name: str, phi: float, seed: int, back: str = "") -> str:
        from . import view_cache

        params: dict[str, Any] = {
            "view": name,
            "phi": float(phi),
            "seed": int(seed),
            "steps": self.steps,
            "prompt": self.prompt,
            "long_side": VIEW_LONG_SIDE,
        }
        if back:
            params["back"] = back
        return view_cache.make_key(self.digest, source=self.source, params=params, worker=self.worker)

    def _cached(self, key: str) -> Path | None:
        hit = self.cache.get(key)
        if hit is None:
            return None
        pngs = sorted(p for p in hit.iterdir() if p.suffix.lower() == ".png")
        if not pngs:
            self.cache.discard(key)
            return None
        self.hits += 1
        return pngs[0]

    def _store(self, key: str, path: Path, meta: dict[str, Any]) -> Path:
        try:
            stored = self.cache.put(key, [path], meta={"source": self.source, **meta})
        except Exception:  # the cache is a convenience
            log.warning("Could not keep a Qwen view in the view cache", exc_info=True)
            stored = None
        return stored / path.name if stored is not None else path

    def _partial(self) -> list[dict[str, Any]]:
        """The views a failed / cancelled worker run finished (its ``views.partial.jsonl``)."""
        out = []
        try:
            lines = (self.folder / "views.partial.jsonl").read_text(encoding="utf-8").splitlines()
        except OSError:
            return out
        for line in lines:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and Path(str(rec.get("file", ""))).is_file():
                out.append(rec)
        return out

    def _keep(self, entries: list[dict[str, Any]], keys: dict[Any, str], by: str) -> dict[Any, Path]:
        """Store the worker's finished pictures under ``keys`` (``by`` = "seed" or "name")."""
        got: dict[Any, Path] = {}
        for e in entries:
            try:
                ident = int(e["seed"]) if by == "seed" else str(e["name"])
                path = Path(str(e["file"]))
            except (KeyError, TypeError, ValueError):
                continue
            if ident in keys and ident not in got and path.is_file():
                got[ident] = self._store(keys[ident], path, {"view": e.get("name"), "seed": e.get("seed")})
        return got

    # ---- the worker
    def _call(
        self,
        views: list[dict[str, Any]],
        *,
        candidates: int,
        seed: int,
        references: Sequence[Path],
        name: str,
        rep: ProgressReporter,
    ) -> list[dict[str, Any]]:
        run, item = self.ctx.run, self.ctx.item

        def worker_progress(frac: float, msg: str = "") -> None:
            rep(frac, f"Qwen photo editor: {msg or 'working'}")

        params: dict[str, Any] = {
            "views": views,
            "candidates": int(candidates),
            "seed": int(seed),
            "prompt_extra": self.prompt,
            "long_side": VIEW_LONG_SIDE,
            "name": name,
        }
        if self.steps:
            params["steps"] = self.steps
        inputs: dict[str, Any] = {"image": str(self.image)}
        if references:
            inputs["references"] = [str(p) for p in references]
        t0 = time.perf_counter()
        try:
            outs = R._run_addon(
                run, item, QWEN_ADDON, _QWEN_TASK, inputs, params, ProgressReporter(worker_progress)
            )
        finally:
            self.seconds += time.perf_counter() - t0
        R._addon_warnings(run, self.addon_title, outs)
        self.outs = outs
        entries = [e for e in outs.get("views") or [] if isinstance(e, dict)]
        self.made += len(entries)
        return entries

    def backs(self, seeds: list[int], rep: ProgressReporter) -> tuple[dict[int, Path], str]:
        keys = {s: self._key("back", 180.0, s) for s in seeds}
        found: dict[int, Path] = {}
        for s in seeds:
            p = self._cached(keys[s])
            if p is not None:
                found[s] = p
        missing = [s for s in seeds if s not in found]
        self.from_cache = frozenset(found)
        note = ""
        if not missing:
            rep(1.0, "Qwen photo editor: using the back views made before (cache)")
            return found, note
        first, last = min(missing), max(missing)
        n = last - first + 1
        rep(
            0.0,
            f"Qwen photo editor: making {n} back view{'s' if n > 1 else ''} (several minutes each on a "
            "GTX 10-series card)",
        )
        try:
            entries = self._call(
                [{"name": "back", "phi_deg": 180.0}],
                candidates=n,
                seed=first,
                references=(),
                name="back",
                rep=rep,
            )
        except Cancelled:
            self._keep(self._partial(), keys, "seed")  # the next run does not make them again
            raise
        except AddonNotInstalled:
            raise
        except (AddonError, VRPhotoError, MemoryError, OSError) as e:
            found.update(self._keep(self._partial(), keys, "seed"))
            self.made += len([s for s in missing if s in found])
            if not found:
                raise
            note = f"{self.title} could not make every back view ({_why(e)}); the ones made before were used."
            return found, note
        found.update(self._keep(entries, keys, "seed"))
        return found, note

    def sides(
        self, views: list[tuple[str, float]], back: Path, rep: ProgressReporter
    ) -> tuple[dict[str, Path], str]:
        seed = int(self.ctx.opts.seed)
        back_digest = _file_digest(back)[:32]
        keys = {name: self._key(name, phi, seed, back=back_digest) for name, phi in views}
        found: dict[str, Path] = {}
        for name, _ in views:
            p = self._cached(keys[name])
            if p is not None:
                found[name] = p
        todo = [{"name": name, "phi_deg": phi} for name, phi in views if name not in found]
        if not todo:
            return found, ""
        rep(0.0, f"Qwen photo editor: making {len(todo)} side views")
        try:
            entries = self._call(todo, candidates=1, seed=seed, references=[back], name="sides", rep=rep)
        except Cancelled:
            self._keep(self._partial(), keys, "name")
            raise
        except (AddonError, VRPhotoError, MemoryError, OSError) as e:
            got = self._keep(self._partial(), keys, "name")
            self.made += len(got)
            found.update(got)
            return found, f"{self.title} could not make the side views ({_why(e)})"
        found.update(self._keep(entries, keys, "name"))
        return found, ""

    def info(self) -> dict[str, Any]:
        out = super().info()
        out["worker"] = {
            k: self.outs[k]
            for k in (
                "comfyui_version",
                "steps",
                "cfg",
                "lightning",
                "ram_tier",
                "vram_mode",
                "precision",
                "worker_version",
            )
            if k in self.outs
        }
        out["steps_asked"] = self.steps
        return out


_VIEW_WORDS = {
    180.0: "seen from directly behind (back view)",
    90.0: "seen from their left side (side view)",
    270.0: "seen from their right side (side view)",
    135.0: "seen from behind and to their left (back-left three-quarter view)",
    225.0: "seen from behind and to their right (back-right three-quarter view)",
}


def view_prompt(phi_deg: float, extra: str = "") -> str:
    """``{{PROMPT}}`` of the comfyui source for a view at the app's azimuth (90 = "you walked
    to your right": the subject's LEFT side is seen), followed by 'Describe the hidden side'."""
    words = next(
        (w for p, w in _VIEW_WORDS.items() if _close(p, phi_deg, 1.0)), f"seen from {phi_deg:.0f} degrees"
    )
    base = f"the same person {words}, full body, same pose, same clothes, plain light grey background"
    tidy = " ".join(str(extra or "").split())
    return f"{base}. {tidy}" if tidy else base


class _ComfyGen(_Generator):
    """The user's own ComfyUI server and API-format workflow."""

    source = "comfyui"
    title = "Your ComfyUI server"
    label = "Your ComfyUI"

    def __init__(self, ctx: Ctx, image_rgb: np.ndarray):
        self.ctx = ctx
        o = ctx.opts
        if not o.comfyui_workflow:  # resolve_options requires it; a hand-made Opts may not
            raise InputError("No ComfyUI workflow was given ('ComfyUI workflow').")
        self.c = _comfy()
        self.url = str(o.comfyui_url or self.c.DEFAULT_URL)
        self.client = self.c.ComfyClient(self.url, timeout=30)
        self.graph = self.c.load_workflow(o.comfyui_workflow)
        self.folder = ctx.run.work_dir(ctx.item) / "comfyui"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.size = (int(image_rgb.shape[1]), int(image_rgb.shape[0]))
        self.image_rgb = image_rgb
        self.uploaded = ""
        self.version = ""
        self.prompts: list[dict[str, Any]] = []

    def _translate(self, e: Exception) -> Exception:
        if isinstance(e, self.c.ComfyCancelled):
            return Cancelled("Operation cancelled")
        return AddonError(_clause(e), hint=getattr(e, "hint", None))  # said inside the chain's sentence

    def start(self, rep: ProgressReporter) -> None:
        from ..core.imaging import save_image

        rep(0.0, f"Connecting to your ComfyUI ({self.url})")
        try:
            stats = self.client.system_stats()
            self.version = str((stats.get("system") or {}).get("comfyui_version") or "")
            R._free_app_gpu()  # the server may share the graphics card with the app
            path = save_image(self.folder / "comfyui_input.png", self.image_rgb)
            self.uploaded = self.client.upload_image(path)
        except self.c.ComfyError as e:
            raise self._translate(e) from e
        log.info("ComfyUI %s at %s: input uploaded as %s", self.version or "?", self.url, self.uploaded)

    def _one(self, name: str, phi: float, seed: int, label: str, rep: ProgressReporter) -> Path:
        run = self.ctx.run
        mapping = {
            "IMAGE": self.uploaded,
            "PROMPT": view_prompt(phi, self.ctx.opts.back_prompt),
            "SEED": int(seed),
            "AZIMUTH": int(round(phi)) % 360,
            "WIDTH": self.size[0],
            "HEIGHT": self.size[1],
        }

        def on_progress(p: dict[str, Any]) -> None:
            el = int(p.get("elapsed") or 0)
            if p.get("state") == "pending" and p.get("ahead"):
                rep(0.1, f"Your ComfyUI: {label} (waiting in its queue, {p['ahead']} ahead)")
            else:
                rep(0.5, f"Your ComfyUI: {label} ({el // 60}:{el % 60:02d})")

        t0 = time.perf_counter()
        try:
            graph = self.c.substitute(self.graph, mapping, strict=True)
            rep(0.0, f"Your ComfyUI: {label}")
            pid = self.client.queue(graph)
            self.prompts.append({"view": name, "phi": phi, "seed": int(seed), "prompt_id": pid})
            entry = self.client.wait(
                pid,
                cancel=run.cancel,
                poll_s=_COMFY_POLL_S,
                timeout_s=COMFY_VIEW_TIMEOUT_S,
                on_progress=on_progress,
            )
            refs = self.client.outputs(entry)
            if not refs:
                raise self.c.ComfyError(
                    "The ComfyUI workflow returned no picture.",
                    hint="It needs a 'Save Image' (or 'Preview Image') node at the end.",
                    kind="protocol",
                )
            path = self.client.download(refs[-1], self.folder / f"{name}_{int(seed)}.png")
        except self.c.ComfyError as e:
            raise self._translate(e) from e
        finally:
            self.seconds += time.perf_counter() - t0
        self.made += 1
        run.check()
        return path

    def backs(self, seeds: list[int], rep: ProgressReporter) -> tuple[dict[int, Path], str]:
        found: dict[int, Path] = {}
        for k, s in enumerate(seeds):
            label = "making the back view" + (f" ({k + 1}/{len(seeds)})" if len(seeds) > 1 else "")
            try:
                found[s] = self._one("back", 180.0, s, label, rep.sub(k / len(seeds), (k + 1) / len(seeds)))
            except Cancelled:
                raise
            except AddonError as e:
                if not found:
                    raise
                note = f"Your ComfyUI could not make back view {k + 1} ({_why(e)}); the ones made before were used."
                return found, note
        return found, ""

    def sides(
        self, views: list[tuple[str, float]], back: Path, rep: ProgressReporter
    ) -> tuple[dict[str, Path], str]:
        found: dict[str, Path] = {}
        seed = int(self.ctx.opts.seed)
        for k, (name, phi) in enumerate(views):
            try:
                found[name] = self._one(
                    name,
                    phi,
                    seed,
                    f"making the {name.replace('_', ' ')} view",
                    rep.sub(k / len(views), (k + 1) / len(views)),
                )
            except Cancelled:
                raise
            except AddonError as e:
                return found, f"Your ComfyUI could not make the side views ({_why(e)})"
        return found, ""

    def info(self) -> dict[str, Any]:
        out = super().info()
        out.update(
            url=self.url,
            comfyui_version=self.version,
            workflow=Path(str(self.ctx.opts.comfyui_workflow)).name,
        )
        return out


# --------------------------------------------------------------------------- the shared nominal pipeline
def _side_plan(quality: str) -> list[tuple[str, float]]:
    """Phase 2 views by back quality: none with 'Fast' (every view costs minutes), left /
    right with 'Balanced', plus the back quarters with 'Best'."""
    if quality == "fast":
        return []
    return list(SIDE_VIEWS) + (list(QUARTER_VIEWS) if quality == "best" else [])


def _nominal_back(ctx: Ctx, rep: ProgressReporter, gen: _Generator) -> BackPart:
    """Design 3.5.6 for a generator: back candidates -> choice -> side views -> view lifting
    (see the module docstring).  Returns the part (``ok=False`` with a plain reason when no
    back candidate is usable or the views do not make a back); raises the generator's error
    when it made no back view at all."""
    from ..geometry import views_lift as vl
    from ..segment import faces

    run, item, o = ctx.run, ctx.item, ctx.opts
    q = QUALITY.get(o.quality, QUALITY["balanced"])
    source = gen.source
    n = max(1, int(o.tries))
    seeds = [int(o.seed) + i for i in range(n)]
    notes: list[str] = []

    gen.start(rep.sub(0.0, 0.01))  # comfyui: is the server there? (before any slow work)
    rep(0.01, f"{gen.label}: levelling the photo")
    ortho = _levelled_input(ctx, rep.sub(0.01, 0.03))
    run.check()

    # ---- phase 1: back candidates (generation dominates the time)
    files, note = gen.backs(seeds, rep.sub(0.03, 0.45))
    notes.append(note)
    run.check()
    cands = [
        _Cand(index=i + 1, seed=s, file=files[s], cached=s in gen.from_cache)
        for i, s in enumerate(seeds)
        if s in files
    ]
    seg = _view_segmenter(run) if cands else None
    chosen = _choose_back(ctx, cands, ortho, seg, rep.sub(0.45, 0.5))
    base_diag: dict[str, Any] = {
        "candidates": [c.summary() for c in cands],
        "tries": n,
        "input": "cutout",
        "generator": gen.info(),
        "prompt": o.back_prompt,
    }
    if chosen is None:
        ctx.previews["views"] = (ortho, _rejected_views(cands, None))
        ctx.previews["views_source"] = source
        return BackPart(
            sc=_empty_cloud(),
            kind="mould",
            source=source,
            ok=False,
            reason=_candidates_reason(cands, gen.title),
            views=_rejected_views(cands, None),
            diag=base_diag,
        )
    for c in cands:
        if c is not chosen and c.reason:
            notes.append(f"{gen.title}'s back view {c.index} of {len(cands)} was not used ({c.reason}).")
    run.check()

    # ---- phase 2: side views with the chosen back as a reference
    sides: list[Any] = []
    side_files: dict[str, Path] = {}
    plan = _side_plan(o.quality)
    if plan:
        side_files, note = gen.sides(plan, chosen.file, rep.sub(0.5, 0.85))
        if note:
            notes.append(
                f"{note}; the back was built from the back view alone." if not side_files else f"{note}."
            )
        for name, phi in plan:
            path = side_files.get(name)
            if path is None:
                continue
            v, why = _load_view(path, phi, name, seg, run, {"seed": int(o.seed)})
            if v is None:
                notes.append(f"The {name.replace('_', ' ')} view was not used ({_clause(why)}).")
                continue
            if any(_close(phi, p) for p in FACE_PHIS):
                score = face_score(v, run)
                if score > faces.REJECT_SCORE:
                    notes.append(f"The {name.replace('_', ' ')} view was not used (it shows a face).")
                    continue
            sides.append(v)
    run.check()

    # ---- lift (V2-V11, nominal)
    vb = rep.sub(0.85, 1.0)
    relief, depth_fn, used = _relief_fn(ctx, vb.sub(0.40, 0.45), ortho=ortho)
    part = vl.view_based_back(
        ctx.front,
        ctx.canon,
        ortho,
        [chosen.view, *sides],
        depth_fn=depth_fn,
        relief=relief,
        color_match=o.color_match,
        stride=o.stride,
        seed=o.seed,
        fill_holes=q.fill_holes,
        source=source,
        progress=vb,
        cancel=run.cancel,
    )
    part = with_face_check(part, {"back": round(chosen.face, 3)}, reject=False)
    rejected = _rejected_views(cands, chosen)
    ctx.previews["views"] = (ortho, [*list(part.views or [chosen.view, *sides]), *rejected])
    ctx.previews["views_source"] = source
    if gen.made and gen.seconds / gen.made > SLOW_VIEW_S and source == "qwen":
        notes.append(
            f"{gen.title} took {gen.seconds / 60.0:.0f} minutes for {gen.made} views; on a GTX 10-series card "
            "every view takes several minutes (fewer 'AI steps' are faster; the views are reused when you "
            "re-run this photo)."
        )
    diag = {
        **(part.diag or {}),
        **base_diag,
        "candidates": [c.summary() for c in cands],
        "chosen": chosen.index,
        "seed": chosen.seed,
        "sides": sorted(side_files),
        "generator": gen.info(),
    }
    note_text = _join(*notes)
    if note_text:
        diag["warning"] = _join(note_text, diag.get("warning"))
    part.diag = diag
    if part.ok:
        if seg is not None and any(
            getattr(v, "meta", {}).get("matte") == "segmenter" for v in [chosen.view, *sides]
        ):
            run.model_notice(item, str(getattr(seg, "resolved", None) or getattr(seg, "id", "") or ""))
        _relief_notice(run, item, part.diag, used)
    log.info(
        "%s back: candidate %d/%d (seed %d), %d side view(s), ok %s, %d Gaussians",
        gen.title,
        chosen.index,
        len(cands),
        chosen.seed,
        len(sides),
        part.ok,
        len(part.sc),
    )
    rep(1.0, f"{gen.label}: back ready")
    return part


def _src_qwen(ctx: Ctx, rep: ProgressReporter) -> BackPart:
    """The Qwen photo editor back (design 3.5.6, add-on ``qwenedit``; experimental, never
    "auto"): see the module docstring.  Raises :class:`AddonNotInstalled` when the add-on is
    missing and the add-on's error when it made no back view (the chain warns and falls back);
    returns ``ok=False`` with a plain reason when no back view is usable."""
    from ..core.imaging import save_image
    from .cache import content_hash

    run, item = ctx.run, ctx.item
    manifest = R._require_addon(QWEN_ADDON)  # not installed -> the chain warns and falls back
    rgba = np.ascontiguousarray(_cutout_input(ctx), dtype=np.uint8)
    image = save_image(run.work_dir(item) / "qwen_input.png", rgba)  # the worker composites it on #c8c8c8
    gen = _QwenGen(ctx, manifest, image, content_hash(rgba))
    part = _nominal_back(ctx, rep, gen)
    if part.ok:
        run.addon_notice(item, manifest)  # its output is in the result from here on
    return part


def _src_comfyui(ctx: Ctx, rep: ProgressReporter) -> BackPart:
    """Your own ComfyUI server (design 3.5.6; expert, never "auto"): the API-format workflow
    is queued once per view with the placeholders filled in (``{{IMAGE}}`` = the uploaded
    cut-out on #c8c8c8, ``{{PROMPT}}`` = :func:`view_prompt`, ``{{SEED}}``, ``{{AZIMUTH}}`` =
    the view's azimuth in degrees - 180 = behind, 90 = "you walked to your right" -,
    ``{{WIDTH}}`` / ``{{HEIGHT}}`` = the cut-out's size, long side 768).  A server that does
    not answer, a workflow ComfyUI rejects or a failed run raises :class:`AddonError` with
    ComfyUI's own message (the chain warns and falls back); a cancel interrupts the queued
    prompt."""
    import cv2

    rgb = _composite(np.ascontiguousarray(_cutout_input(ctx), dtype=np.uint8))
    w, h = _output_size(rgb.shape[1], rgb.shape[0])
    interp = cv2.INTER_AREA if w < rgb.shape[1] else cv2.INTER_CUBIC
    rgb = cv2.resize(rgb, (w, h), interpolation=interp)
    c = _comfy()
    try:
        gen = _ComfyGen(ctx, rgb)
    except c.ComfyError as e:  # the workflow file went missing or the address is odd
        raise InputError(str(e), hint=getattr(e, "hint", None)) from e
    return _nominal_back(ctx, rep, gen)


__all__ = [
    "CHI2_WEIGHT",
    "COMFY_VIEW_TIMEOUT_S",
    "CUTOUT_GREY",
    "FACE_PHIS",
    "QUARTER_VIEWS",
    "QWEN_ADDON",
    "QWEN_STEPS_RANGE",
    "RELIEF_HFOV_DEG",
    "SHARP_RELIEF",
    "SIDE_VIEWS",
    "SLOW_VIEW_S",
    "VIEW_LONG_SIDE",
    "face_score",
    "face_scores",
    "rim_chi2",
    "sharp_depth_fn",
    "view_prompt",
    "with_face_check",
]
