"""SHARP 360: SHARP's photo-real front plus a predicted back, life-size (object method ``sharp360``).

Design: docs/design/SHARP360.md (sections 1.3-1.5, 3.1, 3.12 and 4.2); conventions and the
front guarantee: docs/CONTRACTS.md section 13.  :func:`vrphoto.pipeline.run._mode_object`
validates the options (:func:`resolve_options`), checks that SHARP is installed and finds
the subject, then hands over to :func:`mode_sharp360`:

1. SHARP reconstructs the whole photo (the same add-on call as the plain ``sharp`` method).
2. :func:`~vrphoto.geometry.splat_fusion.sharp_front` cleans it down to the subject's
   photographed front (hidden wall layer, edge streaks and background removed) and
   :func:`~vrphoto.geometry.splat_fusion.canonical_frame` finds the subject's upright frame.
3. A **back source** makes the hidden side.  Sources are tried in order (:data:`SOURCES`,
   ``Opts.sources``): one that is missing, fails or returns an unusable back only adds a
   warning and the next one runs; the built-in shell is always last, and when even it fails
   the cleaned front is exported alone.  A cancel is never turned into a fallback
   (``Cancelled`` is a ``VRPhotoError``, so it is re-raised first).
4. Front and back are stitched (:mod:`~vrphoto.geometry.splat_merge`: dithered seam,
   colour feathering, floater removal).  Any unexpected failure there exports the cleaned
   front with a warning - once SHARP has succeeded the job only fails when cancelled.
5. Export (design 3.12): glTF frame, rotated upright about the found "up", life size from
   the entered height or from SHARP's own metric size, box crop, decimation to the viewer
   budget; the scene block carries ``scale: 1.0`` so the viewer shows it 1:1.

This version (0.2.0-beta) ships only the built-in ``shell`` back; TRELLIS, back photos,
MV-Adapter and Qwen join :data:`SOURCES` (and ``run.BACK_SOURCES``) in later milestones,
as do the Brush polish and the turntable preview.

Outputs: ``<base>_cutout.png``, ``<base>_sharp360.splat`` and, with ``save_extras``,
``_sharp360.ply``, ``_sharp360_front.splat`` (the cleaned SHARP front alone, same frame),
``_sharp360_back.splat`` and ``_sharp360_debug.json``.  The debug data is always written
to the job's work folder and summarised in one log line.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import (
    UI_ADDONS,
    UI_LOG,
    AddonError,
    AddonNotInstalled,
    Cancelled,
    InputError,
    VRPhotoError,
)
from ..core.fsutil import atomic_write_bytes
from ..core.log import get_logger
from ..core.types import FloatMap, ProgressReporter, SplatCloud
from ..geometry.back_mould import BackPart, shell_back
from ..geometry.camera import cv_to_gltf
from ..geometry.splat import (
    crop_splats,
    decimate_splats,
    normalize_orientation,
    read_splats,
    subset_splats,
    write_splats,
)
from ..geometry.splat_fusion import MOULD_LONG, Canon, FrontModel, SharpCam, canonical_frame, sharp_front
from ..geometry.splat_merge import add_underlayer, feather_colors, merge_parts, remove_floaters, to_life_size
from . import run as R

log = get_logger(__name__)

#: The source that always works and ends every chain (no AI, no download).
FINAL_SOURCE = "shell"
#: Seam angle limits (deg) of the "Keep the photo up to" option.
SEAM_RANGE = (35.0, 80.0)
#: Thickness (a fraction of the height) the life-size back accepts.  The option is shared
#: with Billboard, whose presets go up to 0.24 for a stronger 3D look; a back that deep
#: makes a person about twice as deep as a real body, so it is clamped (with a warning).
THICKNESS_RANGE = (0.06, 0.18)


@dataclass(frozen=True)
class Quality:
    """What a ``back_quality`` setting means (design 1.3; more knobs arrive with the AI sources)."""

    mould_long: int  # long side of the isotropic mould grid (cells)
    stride: int  # one back Gaussian every ``stride`` mould cells
    auto: tuple[str, ...]  # sources "auto" tries, best first (only installed ones are used)


QUALITY: dict[str, Quality] = {
    "fast": Quality(MOULD_LONG["fast"], 2, (FINAL_SOURCE,)),
    "balanced": Quality(MOULD_LONG["balanced"], 1, (FINAL_SOURCE,)),
    "best": Quality(MOULD_LONG["best"], 1, (FINAL_SOURCE,)),
}

#: How each source is named in messages (mid-sentence) and in the output's label.
SOURCE_NAMES: dict[str, str] = {"shell": "the built-in back"}
SOURCE_LABELS: dict[str, str] = {"shell": "built-in back"}
#: The add-on a source needs; "auto" skips sources whose add-on is not installed.  The
#: built-in back needs none (TRELLIS and MV-Adapter are added here in later versions).
SOURCE_ADDONS: dict[str, str] = {}


@dataclass(frozen=True)
class Opts:
    """Validated SHARP 360 options of one job (see :func:`resolve_options`)."""

    back_source: str  # as chosen: "auto" or a source id
    quality: str  # fast | balanced | best
    sources: tuple[str, ...]  # tried in this order; always ends with FINAL_SOURCE
    mould_long: int
    stride: int
    seam: float  # degrees
    size_from: str  # auto | height | photo
    save_extras: bool
    back: str  # built-in back colours: diffuse | mirror
    thickness: float  # median body thickness as a fraction of the height (clamped)
    seed: int  # seam dither (and, later, the AI sources' seed)
    thickness_asked: float | None = None  # the job's value when it was clamped, else None


def _seam(value: Any) -> float:
    lo, hi = SEAM_RANGE
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float("nan")
    if not (math.isfinite(v) and lo <= v <= hi):
        raise InputError(
            f"'Keep the photo up to' must be between {lo:g} and {hi:g} degrees (got {value!r}).",
            hint="60 degrees suits most photos.",
        )
    return v


def resolve_options(spec: Any, *, installed: Callable[[str], bool]) -> Opts:
    """Check the SHARP 360 options of an object job and pick the back sources to try.

    Raises :class:`InputError` for an unknown back source, quality or size setting, a
    seam angle outside 35-80 degrees, or a bad thickness / back colour, so a bad job fails
    before any slow work.  A valid thickness outside :data:`THICKNESS_RANGE` is clamped
    into it (``Opts.thickness_asked`` keeps the job's value; the job warns).  ``"auto"``
    tries the quality's :attr:`Quality.auto` sources whose add-on is ``installed``; an
    explicit source is tried even when its add-on is missing (the job then warns and falls
    back).  The built-in back always ends the chain.
    ``seed`` is 0: the 'Variation' option (the AI sources' seed and the seam dither) arrives
    with the AI back sources; until then the same photo always gives the same result."""
    source = R._choice(spec.back_source, R.BACK_SOURCES, "back source")
    quality = R._choice(spec.back_quality, R.BACK_QUALITIES, "back quality")
    size_from = R._choice(spec.size_from, R.SIZE_FROM, "'Size from' setting")
    seam = _seam(spec.seam_deg)
    back = R._choice(spec.back, R.BACKS, "back texture")
    asked = R._positive(spec.thickness, "Thickness", allow_zero=True)
    thickness = min(max(asked, THICKNESS_RANGE[0]), THICKNESS_RANGE[1])
    q = QUALITY[quality]
    if source == "auto":
        chain = [
            s for s in q.auto if s in SOURCES and (s not in SOURCE_ADDONS or installed(SOURCE_ADDONS[s]))
        ]
    else:
        chain = [source]
    if FINAL_SOURCE not in chain:
        chain.append(FINAL_SOURCE)
    return Opts(
        back_source=source,
        quality=quality,
        sources=tuple(chain),
        mould_long=q.mould_long,
        stride=q.stride,
        seam=seam,
        size_from=size_from,
        save_extras=bool(spec.save_extras),
        back=back,
        thickness=thickness,
        seed=0,
        thickness_asked=None if thickness == asked else asked,
    )


# --------------------------------------------------------------------------- back sources
@dataclass
class Ctx:
    """What a back source gets: the job, the photo and SHARP's cleaned front."""

    run: Any  # run._Run
    item: Any  # run._Item
    photo: Any  # run._Photo
    alpha: FloatMap
    sc_full: SplatCloud  # SHARP's whole reconstruction (frame C)
    front: FrontModel
    canon: Canon
    opts: Opts


def _src_shell(ctx: Ctx, rep: ProgressReporter) -> BackPart:
    """The built-in back: the silhouette inflated behind the front (no AI, no download)."""
    o = ctx.opts
    return shell_back(
        ctx.front,
        ctx.canon,
        thickness=o.thickness,
        back=o.back,
        stride=o.stride,
        progress=rep,
        cancel=ctx.run.cancel,
    )


#: Back sources by id: ``fn(ctx, rep) -> BackPart`` in frame C.  A source raises (or returns
#: ``ok=False`` with a plain-language ``reason``) when it cannot make a back.
SOURCES: dict[str, Callable[[Ctx, ProgressReporter], BackPart]] = {"shell": _src_shell}


def _name(src: str) -> str:
    return SOURCE_NAMES.get(src, src)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _seed_given(spec: Any) -> bool:
    try:
        return int(spec.seed or 0) != 0
    except (TypeError, ValueError):
        return False


def run_sources(ctx: Ctx, rep: ProgressReporter, attempts: list[dict[str, Any]]) -> BackPart | None:
    """Try ``ctx.opts.sources`` in order and return the first usable back (None: none worked).

    Every failure is one plain warning naming what is used instead; ``attempts`` gets one
    JSON-safe record per source tried (for debug.json)."""
    run = ctx.run
    srcs = ctx.opts.sources
    for i, src in enumerate(srcs):
        nxt = _name(srcs[i + 1]) if i + 1 < len(srcs) else "the photo front alone"
        who = _cap(_name(src))
        t0 = time.perf_counter()
        rec: dict[str, Any] = {"source": src, "ok": False}
        attempts.append(rec)
        fn = SOURCES.get(src)
        if fn is None:  # named by a job file from a newer version
            rec["reason"] = "not available in this version"
            run.warn(f"{who} is not available in this version; {nxt} was used instead.")
            continue
        try:
            part = fn(ctx, rep)
            rec["seconds"] = round(time.perf_counter() - t0, 2)
            if part is not None and part.ok and len(part.sc):
                rec["ok"] = True
                return part
            reason = (part.reason if part is not None else "") or "it came out empty"
            rec["reason"] = reason
            run.warn(f"{who} could not make a usable back ({reason}); {nxt} was used instead.")
        except Cancelled:
            raise  # Cancelled is a VRPhotoError: never a fallback
        except AddonNotInstalled as e:
            rec["reason"] = f"not installed: {e}"
            run.warn(f"{who} is not installed; {nxt} was used instead.")
        except (AddonError, VRPhotoError, MemoryError, OSError) as e:
            rec["reason"] = str(e) or type(e).__name__
            run.warn(f"{who} could not make the back ({rec['reason']}); {nxt} was used instead.")
        except Exception as e:  # a bug in the fusion code: degrade, keep the traceback
            log.exception("SHARP 360 back source %s failed", src)
            rec["reason"] = f"{type(e).__name__}: {e}"
            run.warn(f"{who} failed unexpectedly ({type(e).__name__}); {nxt} was used instead.")
    return None


# --------------------------------------------------------------------------- stitching and export
def _stitch(
    ctx: Ctx, part: BackPart | None, rep: ProgressReporter, dbg: dict[str, Any]
) -> tuple[SplatCloud, np.ndarray]:
    """Merge the back into the front, close cloud backs, feather colours, drop floaters."""
    run, front, canon, o = ctx.run, ctx.front, ctx.canon, ctx.opts
    rep(0.0, "Stitching front and back")
    merged, is_front, dbg["merge"] = merge_parts(
        front,
        part,
        seam_deg=o.seam,
        seed=o.seed,
        budget=R.MAX_OBJECT_SPLATS,
        cancel=run.cancel,
        progress=rep.sub(0.0, 0.45),
    )
    if part is not None and part.kind == "cloud":
        rep(0.45, "Closing holes in the back")
        merged, is_front = add_underlayer(merged, is_front, front, canon, part, cancel=run.cancel)
    rep(0.5, "Matching colours")
    merged = feather_colors(merged, is_front, cancel=run.cancel)
    run.check()
    rep(0.6, "Removing floaters")
    merged, is_front, dbg["floaters"] = remove_floaters(
        merged,
        is_front,
        front,
        canon,
        hull=None if part is None else part.hull,
        seam_deg=o.seam,
        spacing_m=None if part is None else part.spacing_m,
        cancel=run.cancel,
        progress=rep.sub(0.6, 1.0),
    )
    return merged, is_front


@dataclass
class _Export:
    sc: SplatCloud  # final cloud (glTF frame, feet at y = 0, life-size, cropped, decimated)
    front: SplatCloud  # its front rows before decimation (save_extras)
    back: SplatCloud  # its generated rows before decimation (save_extras)
    height: float  # final height (m)
    size_mode: str  # height | photo
    info: dict[str, Any] = field(default_factory=dict)


def _export_cloud(
    sc: SplatCloud,
    is_front: np.ndarray,
    canon: Canon,
    o: Opts,
    height: float,
    run: Any,
    rep: ProgressReporter,
) -> _Export:
    """Design 3.12: glTF frame -> upright about ``canon.up_C`` (floor at y = 0) -> life size ->
    box crop -> decimate to the viewer budget.  Every step before the decimation keeps the
    row order, so the front / back split of ``is_front`` still applies.

    The box crop removes generated rows only: the front rows are SHARP's cleaned subject
    (matte- and depth-filtered already), and the percentile box would cut thin parts held
    out from the body (a cane, a pointing hand, fingertips above a raised arm) out of the
    protected photo front."""
    P = np.asarray(sc.means)
    ok = np.isfinite(P).all(axis=1) & np.isfinite(np.asarray(sc.scales)).all(axis=1)
    if not ok.all():  # normalize_orientation would drop them and break the row order
        sc, is_front = subset_splats(sc, ok), np.asarray(is_front)[ok]
    rep(0.0, "Making it life-size")
    g = R._splats_cv_to_gltf(sc)
    g = normalize_orientation(g, up=cv_to_gltf(np.asarray(canon.up_C, dtype=np.float64)))
    cut = bool(canon.cut_top or canon.cut_bottom)
    g, h, box, used = to_life_size(g, mode=o.size_from, height_m=height, cut=cut)
    isf = np.asarray(is_front, dtype=bool)
    P = np.asarray(g.means, dtype=np.float64)
    in_box = np.all(P >= box[0], axis=1) & np.all(P <= box[1], axis=1)
    keep = isf | in_box
    main = subset_splats(g, keep)
    front_part = subset_splats(g, isf)
    back_part = subset_splats(g, keep & ~isf)
    run.check()
    n_crop = len(main)
    main = decimate_splats(main, max_n=R.MAX_OBJECT_SPLATS, progress=rep.sub(0.3, 1.0), cancel=run.cancel)
    info = {
        "size_from": o.size_from,
        "size_used": used,
        "height_m": h,
        "box": np.asarray(box).tolist(),
        "front_outside_box": int((isf & ~in_box).sum()),
        "after_crop": n_crop,
        "final": len(main),
    }
    return _Export(main, front_part, back_part, h, used, info)


def _cut_where(canon: Canon) -> str:
    if canon.cut_top and canon.cut_bottom:
        return "at the top and bottom"
    return "at the top" if canon.cut_top else "at the bottom"


# --------------------------------------------------------------------------- debug data
def _jsonable(x: Any) -> Any:
    """``x`` with numpy scalars / small arrays, paths and non-finite floats made JSON-safe."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist()) if x.size <= 64 else f"<array {x.shape}>"
    if isinstance(x, np.generic):
        return _jsonable(x.item())
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, bool | int | str) or x is None:
        return x
    if isinstance(x, Path):
        return str(x)
    return str(x)


def _canon_info(canon: Canon) -> dict[str, Any]:
    return {
        "up_source": canon.up_source,
        "up_C": canon.up_C,
        "pitch_deg": canon.pitch_deg,
        "height_s": canon.height_s,
        "width_s": canon.width_s,
        "T_prior": canon.T_prior,
        "z_axis": canon.z_axis,
        "cut_top": canon.cut_top,
        "cut_bottom": canon.cut_bottom,
        "diag": canon.diag,
    }


def _write_debug(path: Path, dbg: dict[str, Any]) -> Path:
    data = json.dumps(_jsonable(dbg), indent=2, ensure_ascii=False).encode("utf-8")
    atomic_write_bytes(path, data)
    return path


# --------------------------------------------------------------------------- the mode
def mode_sharp360(
    run: Any,
    item: Any,
    photo: Any,
    alpha: FloatMap,
    spec: Any,
    *,
    placement: str,
    height: float,
    rep: ProgressReporter,
    opts: Opts | None = None,
) -> None:
    """Run SHARP 360 for one input once the subject's matte ``alpha`` (photo resolution) is
    known; writes the outputs and the viewer scene (see the module docstring).

    ``opts`` (an optional addition) are options already checked by :func:`resolve_options`
    (``run._mode_object`` validates them before any work); None resolves them here.
    Raises :class:`AddonError` when SHARP fails or returns nothing (as the plain ``sharp``
    method) and :class:`Cancelled`; every later failure becomes a warning."""
    o = opts if opts is not None else resolve_options(spec, installed=R._addon_installed)
    t_start = time.perf_counter()
    times: dict[str, float] = {}
    dbg: dict[str, Any] = {"version": 1, "options": asdict(o)}

    def lap(name: str, t0: float) -> float:
        t = time.perf_counter()
        times[name] = round(t - t0, 3)
        return t

    # ---- 1. SHARP (fails the job like the plain sharp method)
    manifest = R._addon_manifest("sharp")
    run.addon_notice(item, manifest)
    if _seed_given(spec):  # as plain SHARP says
        run.warn(
            "SHARP 360° always gives the same result for the same photo; the 'Variation' number is not used."
        )
    if o.thickness_asked is not None:
        lo, hi = THICKNESS_RANGE
        run.warn(
            f"SHARP 360° used a Thickness of {o.thickness:.2f} instead of {o.thickness_asked:.2f}: its "
            f"life-size back allows {lo:.2f}-{hi:.2f} (0.08-0.15 suits most people; the presets set "
            "Thickness for Billboard)."
        )
    R._save_cutout(run, item, photo, alpha)
    image_in, params = R._sharp_inputs(run, item, photo)
    rep(0.10, "Running SHARP (front)")
    t = time.perf_counter()
    outs = R._run_addon(
        run, item, "sharp", "image_to_splat", {"image": str(image_in)}, params, rep.sub(0.10, 0.25)
    )
    R._addon_warnings(run, manifest.title, outs)
    ply = outs.get("ply") or outs.get("splat")
    if not (ply and Path(ply).is_file()):
        raise AddonError(
            f"{manifest.title} finished but did not return a 3D model.",
            hint=f"Details are in the log ({UI_LOG}). Try again, or reinstall the add-on in the {UI_ADDONS}.",
        )
    frame = str(outs.get("frame", "opencv") or "opencv").lower()
    if frame != "opencv":
        log.warning("SHARP reported frame %r; SHARP 360 assumes OpenCV camera coordinates", frame)
    sc_full = read_splats(ply)
    if not len(sc_full):
        raise AddonError(
            f"{manifest.title} returned an empty 3D result.",
            hint=f"Try again; if it keeps happening, reinstall the add-on in the {UI_ADDONS}.",
        )
    t = lap("sharp", t)
    run.check()

    # ---- 2. clean the front, find the upright frame
    rep(0.25, "Cleaning the SHARP front (removing background layers and edge streaks)")
    try:
        cam = SharpCam.from_outputs(outs, params, n_raw=len(sc_full), shape=alpha.shape)
        dbg["camera"] = {k: getattr(cam, k) for k in ("f", "width", "height", "n_raw", "grid", "layers")}
        front = sharp_front(
            sc_full,
            alpha,
            cam,
            photo.rgb,
            mould_long=o.mould_long,
            progress=rep.sub(0.25, 0.29),
            cancel=run.cancel,
        )
        dbg["front"] = {"kept": len(front.sc), "stats": front.stats}
        rep(0.29, "Finding the subject's upright direction")
        canon = canonical_frame(front, sc_full, alpha, thickness=o.thickness)
        dbg["canon"] = _canon_info(canon)
    except Cancelled:
        raise
    except Exception as e:
        log.exception("SHARP 360: cleaning SHARP's result failed")
        why = str(e) if isinstance(e, VRPhotoError) else type(e).__name__
        run.warn(
            f"SHARP 360 could not clean SHARP's result ({why}); plain SHARP's subject was saved instead, "
            "without a back."
        )
        dbg["front_failed"] = f"{type(e).__name__}: {e}"
        _plain_fallback(run, item, photo, alpha, sc_full, outs, params, o, placement, height, rep, dbg)
        return
    for w in list(front.stats.get("warnings") or []) + list(canon.diag.get("warnings") or []):
        run.warn(str(w))
    t = lap("front", t)
    run.check()

    # ---- 3. the back
    ctx = Ctx(run, item, photo, alpha, sc_full, front, canon, o)
    attempts: list[dict[str, Any]] = []
    part = run_sources(ctx, rep.sub(0.30, 0.72), attempts)
    dbg["sources"] = attempts
    if part is not None:
        dbg["back"] = {"source": part.source, "kind": part.kind, "n": len(part.sc), "diag": part.diag or {}}
        warning = (part.diag or {}).get("warning")
        if warning:
            run.warn(str(warning))
    t = lap("back", t)
    run.check()

    # ---- 4. stitch (never fails the job: falls back to the cleaned front)
    stitched = True
    try:
        merged, is_front = _stitch(ctx, part, rep.sub(0.72, 0.90), dbg)
    except Cancelled:
        raise
    except Exception as e:
        log.exception("SHARP 360: stitching failed")
        run.warn("Stitching the front and back failed; the cleaned SHARP front was saved instead.")
        dbg["stitch_failed"] = f"{type(e).__name__}: {e}"
        merged, is_front, stitched = front.sc, np.ones(len(front.sc), bool), False
    t = lap("stitch", t)

    # ---- 5. export (an export problem with the back retries with the front alone)
    try:
        ex = _export_cloud(merged, is_front, canon, o, height, run, rep.sub(0.90, 0.95))
    except Cancelled:
        raise
    except Exception as e:
        if not stitched or part is None:
            raise
        log.exception("SHARP 360: export with the back failed")
        run.warn("The back could not be added to the final splat; the cleaned SHARP front was saved instead.")
        dbg["export_failed"] = f"{type(e).__name__}: {e}"
        stitched = False
        ex = _export_cloud(front.sc, np.ones(len(front.sc), bool), canon, o, height, run, rep.sub(0.90, 0.95))
    if o.size_from == "auto" and ex.size_mode == "photo":
        run.warn(
            f"The subject is cut off {_cut_where(canon)} of the photo, so SHARP's own size estimate was used "
            f"({ex.height:.2f} m). Set 'Size from' to 'Real-world height' to force your height."
        )
    dbg["export"] = ex.info
    used_back = part is not None and stitched and len(ex.back) > 0
    what = "front only"
    if used_back and part is not None:
        what = SOURCE_LABELS.get(part.source, f"back by {_name(part.source)}")
    dst = run.out_path(item, "_sharp360", "splat")
    write_splats(ex.sc, dst)
    run.add_output(item, f"3D object (Gaussian splat, SHARP 360°, {what})", dst)
    t = lap("export", t)

    # ---- 6. extras, debug data, viewer scene
    times["total"] = round(time.perf_counter() - t_start, 3)
    dbg["seconds"] = times
    dbg["warnings"] = list(run.result.warnings)
    _write_extras(run, item, ex, o, used_back)
    _save_debug(run, item, dbg, o.save_extras)
    log.info(
        "SHARP 360 %s: %d SHARP Gaussians -> front %d, back %d (%s), final %d, height %.2f m (%s), %.1f s",
        item.path.name,
        len(sc_full),
        len(front.sc),
        0 if part is None else len(part.sc),
        what,
        len(ex.sc),
        ex.height,
        ex.size_mode,
        times["total"],
    )
    # the viewer shows the splat 1:1 (it is already life-size); height_m only names the height
    block = {"asset": "object.splat", "format": "splat", "height_m": round(float(ex.height), 4), "scale": 1.0}
    _scene(run, item, photo, dst, {**block, "pivot": "base", "placement": placement}, rep)


def _scene(
    run: Any, item: Any, photo: Any, splat: Path, block: dict[str, Any], rep: ProgressReporter
) -> None:
    R._export_scene(
        run,
        item,
        mode="object",
        block=block,
        assets={"object.splat": splat},
        thumbnail=photo.rgb,
        env=dict(R.OBJECT_ENV),
        rep=rep.sub(0.95, 1.0),
    )
    rep(1.0, "Done")


def _save_debug(run: Any, item: Any, dbg: dict[str, Any], as_output: bool) -> None:
    """debug.json in the work folder (always, for the log) and, with ``save_extras``, as an
    output.  Never fails the job."""
    try:
        _write_debug(run.work_dir(item) / "sharp360_debug.json", dbg)
        if as_output:
            dj = run.out_path(item, "_sharp360_debug", "json")
            _write_debug(dj, dbg)
            run.add_output(item, "SHARP 360° debug data (JSON)", dj)
    except OSError as e:
        log.warning("Could not write the SHARP 360 debug data: %s", e)


def _write_extras(run: Any, item: Any, ex: _Export, o: Opts, used_back: bool) -> None:
    """With ``save_extras``: the PLY and the front / back parts (same frame and size as the
    main splat).  Extras never fail the job."""
    if not o.save_extras:
        return
    try:
        ply = run.out_path(item, "_sharp360", "ply")
        write_splats(ex.sc, ply)
        run.add_output(item, "3D object (PLY, SHARP 360°)", ply)
        parts = [("_sharp360_front", ex.front, "SHARP 360° photo front only (Gaussian splat)")]
        if used_back:
            parts.append(("_sharp360_back", ex.back, "SHARP 360° back only (Gaussian splat)"))
        for kind, sc, label in parts:
            if len(sc) > R.MAX_OBJECT_SPLATS:
                sc = decimate_splats(sc, max_n=R.MAX_OBJECT_SPLATS, cancel=run.cancel)
            p = run.out_path(item, kind, "splat")
            write_splats(sc, p)
            run.add_output(item, label, p)
    except Cancelled:
        raise
    except (VRPhotoError, OSError) as e:
        run.warn(f"The extra SHARP 360 files could not all be saved: {e}")


def _plain_fallback(
    run: Any,
    item: Any,
    photo: Any,
    alpha: FloatMap,
    sc_full: SplatCloud,
    outs: dict[str, Any],
    params: dict[str, Any],
    o: Opts,
    placement: str,
    height: float,
    rep: ProgressReporter,
    dbg: dict[str, Any],
) -> None:
    """Last resort when the front cannot be cleaned: the plain ``sharp`` method's result
    (matte filter, fitted to the height by the viewer), saved under the SHARP 360 name."""
    rep(0.9, "Converting the splat")
    sc = R._subject_splats(run, sc_full, alpha, outs, params)
    sc = R._splats_cv_to_gltf(sc)
    sc = crop_splats(sc, radius_pct=98.0)
    sc = normalize_orientation(sc)
    sc = decimate_splats(sc, max_n=R.MAX_OBJECT_SPLATS, cancel=run.cancel)
    dst = run.out_path(item, "_sharp360", "splat")
    write_splats(sc, dst)
    run.add_output(item, "3D object (Gaussian splat, SHARP 360°, plain SHARP front)", dst)
    dbg["warnings"] = list(run.result.warnings)
    _save_debug(run, item, dbg, o.save_extras)
    block = {"asset": "object.splat", "format": "splat", "height_m": height, "pivot": "base"}
    _scene(run, item, photo, dst, {**block, "placement": placement}, rep)


__all__ = [
    "FINAL_SOURCE",
    "QUALITY",
    "SEAM_RANGE",
    "SOURCES",
    "SOURCE_ADDONS",
    "SOURCE_LABELS",
    "SOURCE_NAMES",
    "THICKNESS_RANGE",
    "Ctx",
    "Opts",
    "Quality",
    "mode_sharp360",
    "resolve_options",
    "run_sources",
]
