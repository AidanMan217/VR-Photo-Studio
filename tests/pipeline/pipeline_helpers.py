"""Helpers for the pipeline tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def decode_frames(path: Path) -> list[np.ndarray]:
    from vrphoto.video.decode import iter_frames

    return list(iter_frames(path))


_CLOUD_FIELDS = ("means", "scales", "quats", "colors", "opacity", "sh_rest")


def _snapshot(sc) -> dict:
    """Deep copies of every array of a SplatCloud (None stays None)."""
    return {k: None if getattr(sc, k) is None else np.array(getattr(sc, k), copy=True) for k in _CLOUD_FIELDS}


def capture_stitch(monkeypatch) -> dict:
    """Record what SHARP 360 stitches: wraps ``sharp360._stitch`` and fills the returned dict
    with ``front`` (the cleaned SHARP front model), ``alpha`` (the photo's matte), ``merged``
    and ``is_front`` (the fused cloud in SHARP's camera frame, before the export).

    It also wraps ``sharp360.sharp_front`` and keeps ``front_sc0``: deep copies of the
    cleaned front's arrays taken the moment SHARP's front is made - before any back source,
    colour match or merge runs - so a change made to ``ctx.front.sc`` in place cannot move
    the reference along with it."""
    from vrphoto.pipeline import sharp360

    seen: dict = {}
    real_front = sharp360.sharp_front
    real = sharp360._stitch

    def front_spy(*args, **kwargs):
        front = real_front(*args, **kwargs)
        seen["front_sc0"] = _snapshot(front.sc)
        return front

    def spy(ctx, part, rep, dbg):
        merged, is_front = real(ctx, part, rep, dbg)
        seen.update(front=ctx.front, alpha=ctx.alpha, merged=merged, is_front=is_front, part=part)
        return merged, is_front

    monkeypatch.setattr(sharp360, "sharp_front", front_spy)
    monkeypatch.setattr(sharp360, "_stitch", spy)
    return seen


def front_changes(seen: dict) -> list[str]:
    """Fields of the cleaned front's cloud that differ from the snapshot taken when it was
    made (empty: untouched, bit for bit)."""
    snap, sc = seen["front_sc0"], seen["front"].sc
    out = []
    for k in _CLOUD_FIELDS:
        a, b = snap[k], getattr(sc, k)
        if (a is None) != (b is None) or (a is not None and not np.array_equal(a, b)):
            out.append(k)
    return out


def front_render_difference(seen: dict, erode_px: int = 2) -> tuple[float, int]:
    """Design 5.2 "front untouched", seen from the photo's camera with the CPU splat renderer:
    the mean absolute difference (0-255 units, mean over RGB) inside the photo matte eroded
    by ``erode_px`` pixels between the fused cloud and the cleaned SHARP front alone with the
    generated part showing only *through* it (``front + (1 - front alpha) * generated``, both
    rendered over black), and the number of pixels compared.

    The front alone is rendered from the snapshot :func:`capture_stitch` took when SHARP's
    front was made, and the helper first asserts that the front model's own cloud still
    equals it (:func:`front_changes`): a reference read after the code under test ran would
    follow any change made to it in place.

    SHARP's front is not fully opaque (its accumulated alpha is ~0.97 on the synthetic
    person), so a little of whatever is behind it always shows; comparing with the front over
    black would measure the back's colour, not a change to the front.  A front Gaussian
    removed or recoloured, or a generated one in front of the photographed surface, shows up
    here as in the design's plain comparison."""
    import cv2

    from vrphoto.core.types import SplatCloud
    from vrphoto.geometry.splat import subset_splats
    from vrphoto.geometry.splat_render import SplatCamera, render_splats

    changed = front_changes(seen)
    assert not changed, f"SHARP's cleaned front was changed in place: {changed}"
    front = seen["front"]
    cam = SplatCamera.pinhole(front.cam.f, front.cam.width, front.cam.height)
    alone = render_splats(SplatCloud(**seen["front_sc0"]), cam)
    merged, is_front = seen["merged"], np.asarray(seen["is_front"], dtype=bool)
    behind = render_splats(subset_splats(merged, ~is_front), cam).rgb.astype(np.float64)
    expected = alone.rgb.astype(np.float64) + (1.0 - alone.alpha[..., None]) * behind
    fused = render_splats(merged, cam).rgb.astype(np.float64)
    matte = np.asarray(seen["alpha"]) >= 0.5
    if matte.shape != fused.shape[:2]:
        matte = cv2.resize(matte.astype(np.uint8), fused.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
    k = 2 * erode_px + 1
    inner = cv2.erode(matte.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    diff = np.abs(fused - expected).mean(axis=2)
    return float(diff[inner].mean()), int(inner.sum())


def generated_on_the_front(seen: dict, *, margin_m: float = 0.01, min_dt: float = 3.0) -> tuple[float, int]:
    """Design 5.2 "front untouched", geometrically: the share of the opaque (> 0.5) generated
    Gaussians of the fused cloud that lie ON the photographed surface - projecting inside
    the matte well away from its outline (more than ``min_dt`` mould cells) and nearer the
    camera than the front depth plus ``margin_m`` - and how many generated opaque Gaussians
    there are.  A back source must leave the photographed side to SHARP: e.g. TRELLIS's own
    front half, sparse and coloured like the photo, hardly shows in a render but is counted
    here."""
    front = seen["front"]
    merged, is_front = seen["merged"], np.asarray(seen["is_front"], dtype=bool)
    gen = ~is_front & (np.asarray(merged.opacity) > 0.5)
    P = np.asarray(merged.means, dtype=np.float64)[gen]
    if not len(P):
        return 0.0, 0
    u, v, z = front.cam.project(P)
    cell = front.grid.cell_of(u, v)
    ok = cell >= 0
    c = np.where(ok, cell, 0)
    D_f = np.asarray(front.D_f, dtype=np.float64).ravel()[c]
    inner = ok & front.matte.ravel()[c] & (np.asarray(front.dt).ravel()[c] > min_dt)
    with np.errstate(invalid="ignore"):
        on = inner & np.isfinite(D_f) & (z < D_f + margin_m)
    return float(on.mean()), int(len(P))
