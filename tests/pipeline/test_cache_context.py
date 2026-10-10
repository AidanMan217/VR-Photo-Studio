"""Depth cache (npz, LRU size cap) and RunContext (cached models, cache integration)."""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

from vrphoto.core.types import DepthResult, Intrinsics
from vrphoto.pipeline.cache import DepthCache, content_hash, hash_array, make_key
from vrphoto.pipeline.context import RunContext


def _res(h=48, w=64, seed=0, metric=False, K=False, model="da2-small") -> DepthResult:
    rng = np.random.default_rng(seed)
    d = rng.random((h, w), dtype=np.float32)
    return DepthResult(
        disparity=d,
        model_id=model,
        metric_m=(1.0 + 5.0 * d).astype(np.float32) if metric else None,
        intrinsics=Intrinsics.from_hfov(w, h, 60.0) if K else None,
    )


# --------------------------------------------------------------------------- keys
def test_make_key_depends_on_content_model_and_size(tmp_path):
    img = np.zeros((10, 12, 3), np.uint8)
    k = make_key(img, "da2-small", (12, 10))
    assert len(k) == 40 and k.isalnum()
    assert make_key(img.copy(), "da2-small", (12, 10)) == k
    img2 = img.copy()
    img2[0, 0, 0] = 1
    assert make_key(img2, "da2-small", (12, 10)) != k
    assert make_key(img, "da2-base", (12, 10)) != k
    assert make_key(img, "da2-small", (24, 20)) != k
    assert make_key(img, "da2-small", (12, 10), extra="pano:518") != k
    # files are hashed by content, not by name
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"same bytes")
    b.write_bytes(b"same bytes")
    assert make_key(a, "m", 5) == make_key(b, "m", 5)
    b.write_bytes(b"other")
    assert make_key(a, "m", 5) != make_key(b, "m", 5)
    # a precomputed digest is accepted as-is
    assert make_key(content_hash(a), "m", 5) == make_key(a, "m", 5)


def test_hash_array_includes_shape_and_dtype():
    a = np.zeros((4, 4), np.uint8)
    assert hash_array(a) != hash_array(a.reshape(2, 8))
    assert hash_array(a) != hash_array(a.astype(np.uint16))


# --------------------------------------------------------------------------- cache
def test_put_get_roundtrip(tmp_path):
    c = DepthCache(tmp_path / "depth")
    r = _res(metric=True, K=True)
    assert c.get("k1") is None
    path = c.put("k1", r)
    assert path is not None and path.suffix == ".npz"
    got = c.get("k1")
    assert got is not None
    np.testing.assert_array_equal(got.disparity, r.disparity)
    np.testing.assert_array_equal(got.metric_m, r.metric_m)
    assert got.intrinsics == r.intrinsics
    assert got.model_id == "da2-small"
    assert "k1" in c and "nope" not in c and len(c) == 1
    assert c.hits == 1 and c.misses == 1


def test_get_without_optional_fields(tmp_path):
    c = DepthCache(tmp_path)
    c.put("plain", _res())
    got = c.get("plain")
    assert got.metric_m is None and got.intrinsics is None


def test_corrupt_entry_is_discarded(tmp_path):
    c = DepthCache(tmp_path)
    c.put("bad", _res())
    c.path_for("bad").write_bytes(b"not an npz file")
    assert c.get("bad") is None
    assert not c.path_for("bad").exists()


def test_invalid_keys_are_rejected_safely(tmp_path):
    c = DepthCache(tmp_path)
    assert c.get("../evil") is None
    assert c.put("../evil", _res()) is None
    with pytest.raises(ValueError):
        c.path_for("a/b")


def test_lru_prune_keeps_recently_used(tmp_path):
    one = len(DepthCache(tmp_path / "probe").put("x", _res(seed=9)).read_bytes())
    c = DepthCache(tmp_path / "lru", max_bytes=int(one * 3.5))
    for i, key in enumerate(["a", "b", "c"]):
        c.put(key, _res(seed=i))
        t = time.time() - 100 + i
        os.utime(c.path_for(key), (t, t))
    assert c.get("a") is not None  # refreshes "a" -> "b" is now the oldest
    c.put("d", _res(seed=3))
    assert "b" not in c
    assert all(k in c for k in ("a", "c", "d"))
    assert c.size_bytes() <= c.max_bytes


def test_zero_cap_disables_storing(tmp_path):
    c = DepthCache(tmp_path, max_bytes=0)
    assert c.put("k", _res()) is None
    assert c.get("k") is None


def test_prune_and_clear(tmp_path):
    c = DepthCache(tmp_path)
    for i in range(3):
        c.put(f"k{i}", _res(seed=i))
    assert c.prune(0) > 0
    assert len(c) == 0
    c.put("z", _res())
    (tmp_path / ".stray.tmp").write_bytes(b"x")
    c.clear()
    assert len(c) == 0 and not (tmp_path / ".stray.tmp").exists()


def test_concurrent_puts_and_gets(tmp_path):
    c = DepthCache(tmp_path)
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            for j in range(5):
                key = f"k{(i + j) % 4}"
                c.put(key, _res(seed=j))
                got = c.get(key)
                assert got is None or got.disparity.shape == (48, 64)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors


# --------------------------------------------------------------------------- context
class _FakeEstimator:
    def __init__(self, ident="fake-model", result_id=None):
        self.id = ident
        self.result_id = result_id or ident
        self.calls = 0

    def estimate(self, img, *, progress=None, cancel=None):
        self.calls += 1
        h, w = img.shape[:2]
        d = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))
        return DepthResult(d, self.result_id)


def test_context_models_are_cached_per_name_and_device(settings):
    ctx = RunContext(settings)
    e1 = ctx.estimator("heuristic", "cpu")
    assert ctx.estimator("heuristic", "cpu") is e1
    assert ctx.estimator("HEURISTIC", "CPU") is e1
    assert ctx.estimator("heuristic", "0") is not e1
    i1 = ctx.inpainter("opencv")
    assert ctx.inpainter("opencv") is i1
    s1 = ctx.segmenter("grabcut")
    assert ctx.segmenter("grabcut") is s1
    ctx.clear()
    assert ctx.estimator("heuristic", "cpu") is not e1
    assert "heuristic" in repr(ctx)


def test_auto_names_resolve_to_settings(settings):
    settings.depth_model = "heuristic"
    settings.inpaint_model = "opencv"
    settings.device = "cpu"
    ctx = RunContext(settings)
    assert ctx.resolve_name("depth", "auto") == "heuristic"
    assert ctx.resolve_name("depth", None) == "heuristic"
    assert ctx.resolve_name("depth", "da2-base") == "da2-base"
    assert ctx.resolve_device("auto") == "cpu"
    assert ctx.estimator("auto").id == "heuristic"
    assert ctx.inpainter(None).id == "opencv"


def test_creation_is_thread_safe(settings):
    ctx = RunContext(settings)
    got = []
    ts = [threading.Thread(target=lambda: got.append(ctx.estimator("heuristic"))) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len({id(g) for g in got}) == 1


def test_default_cache_location_and_disable(settings):
    ctx = RunContext(settings)
    assert ctx.cache is not None
    assert ctx.cache.root == settings.cache_dir / "depth"
    assert RunContext(settings, use_cache=False).cache is None


def test_depth_uses_cache_for_model_estimators(settings, photo_png, photo_rgb):
    ctx = RunContext(settings)
    est = _FakeEstimator()
    r1 = ctx.depth(photo_rgb, estimator=est, source=photo_png)
    r2 = ctx.depth(photo_rgb, estimator=est, source=photo_png)
    assert est.calls == 1
    np.testing.assert_array_equal(r1.disparity, r2.disparity)
    # a new context (new process) still hits the disk cache
    est2 = _FakeEstimator()
    RunContext(settings).depth(photo_rgb, estimator=est2, source=photo_png)
    assert est2.calls == 0
    # different processing size -> recomputed
    ctx.depth(photo_rgb[:100], estimator=est, source=photo_png)
    assert est.calls == 2


def test_depth_hashes_array_without_source(settings, photo_rgb):
    ctx = RunContext(settings)
    est = _FakeEstimator()
    ctx.depth(photo_rgb, estimator=est)
    ctx.depth(photo_rgb.copy(), estimator=est)
    assert est.calls == 1


def test_heuristic_results_are_not_cached(settings, photo_rgb):
    ctx = RunContext(settings)
    est = _FakeEstimator("heuristic")
    ctx.depth(photo_rgb, estimator=est)
    ctx.depth(photo_rgb, estimator=est)
    assert est.calls == 2
    assert len(ctx.cache) == 0


def test_fallback_result_does_not_poison_cache(settings, photo_rgb):
    """An 'auto' estimator that fell back to the heuristic must not be cached as the model."""
    ctx = RunContext(settings)
    est = _FakeEstimator("da2-small", result_id="heuristic")
    ctx.depth(photo_rgb, estimator=est)
    assert len(ctx.cache) == 0
    est.result_id = "da2-small"
    ctx.depth(photo_rgb, estimator=est)
    assert est.calls == 2 and len(ctx.cache) == 1


def test_depth_compute_override_and_pano_ids(settings, photo_rgb):
    ctx = RunContext(settings)
    est = _FakeEstimator("da2-small")
    calls = []

    def compute(e):
        calls.append(e)
        return DepthResult(np.zeros(photo_rgb.shape[:2], np.float32), f"pano:{e.id}")

    ctx.depth(photo_rgb, estimator=est, extra="pano:518", compute=compute)
    ctx.depth(photo_rgb, estimator=est, extra="pano:518", compute=compute)
    assert len(calls) == 1 and est.calls == 0
    # the same image without the pano marker is a different computation
    ctx.depth(photo_rgb, estimator=est)
    assert est.calls == 1


def test_depth_cache_hit_reports_progress(settings, photo_rgb):
    ctx = RunContext(settings)
    est = _FakeEstimator()
    ctx.depth(photo_rgb, estimator=est)
    seen = []
    ctx.depth(photo_rgb, estimator=est, progress=lambda f, m: seen.append((f, m)))
    assert seen and seen[-1][0] == 1.0 and "cache" in seen[-1][1].lower()
