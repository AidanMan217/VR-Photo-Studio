"""Review fixes: failure memory of the auto chains, download-failure memo, error texts,
robust post-processing of model outputs."""

from __future__ import annotations

import numpy as np
import pytest

from vrphoto.core.errors import DownloadError, InferenceError, ModelUnavailable
from vrphoto.segment import AutoSegmenter
from vrphoto.segment import _runtime as rt
from vrphoto.segment.birefnet import BiRefNetSegmenter
from vrphoto.segment.isnet import ISNetSegmenter


def _bright_sess(fake_session_cls):
    return fake_session_cls(
        ["input_image"], [[1, 3, 1024, 1024]], lambda f: 8.0 * f["input_image"].mean(axis=1, keepdims=True)
    )


# ----------------------------------------------------------------------------- FailureMemo
def test_failure_memo_ttl_expiry(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(rt, "_clock", lambda: clock[0])
    monkeypatch.setattr(rt, "is_available", lambda mid: False)
    memo = rt.FailureMemo(ttl=60.0)
    memo.add("birefnet-lite", "birefnet-lite")
    assert memo.blocked("birefnet-lite") and "birefnet-lite" in memo
    clock[0] += 59.0
    assert memo.blocked("birefnet-lite")
    clock[0] += 2.0
    assert not memo.blocked("birefnet-lite")
    assert "birefnet-lite" not in memo


def test_failure_memo_retries_once_model_is_downloaded(monkeypatch):
    avail = {"birefnet-lite": False}
    monkeypatch.setattr(rt, "is_available", lambda mid: avail.get(mid, False))
    memo = rt.FailureMemo()
    memo.add("birefnet-lite", "birefnet-lite")
    assert memo.blocked("birefnet-lite")
    avail["birefnet-lite"] = True  # user downloaded it from Settings
    assert not memo.blocked("birefnet-lite")


def test_failure_memo_corrupt_local_model_stays_blocked(monkeypatch):
    monkeypatch.setattr(rt, "is_available", lambda mid: True)
    memo = rt.FailureMemo()
    memo.add("birefnet-lite", "birefnet-lite")  # was available and still failed (corrupt)
    assert memo.blocked("birefnet-lite")
    memo.add("grabcut", None)
    assert memo.blocked("grabcut")


# ----------------------------------------------------------------------------- download memo
def test_open_model_records_download_failures(monkeypatch):
    pytest.importorskip("vrphoto.models.ort")
    import vrphoto.models.ort as ort_mod

    def offline(model_id, **kw):
        raise DownloadError("Cannot reach github.com. You appear to be offline.")

    monkeypatch.setattr(ort_mod, "load_model", offline)
    assert not rt.recent_download_failure("u2net")
    with pytest.raises(DownloadError):
        rt.open_model("u2net")
    assert rt.recent_download_failure("u2net")

    sentinel = object()
    monkeypatch.setattr(ort_mod, "load_model", lambda model_id, **kw: sentinel)
    assert rt.open_model("u2net") is sentinel
    assert not rt.recent_download_failure("u2net")  # a success clears it


def test_open_model_other_errors_are_not_download_failures(monkeypatch):
    pytest.importorskip("vrphoto.models.ort")
    import vrphoto.models.ort as ort_mod

    def corrupt(model_id, **kw):
        raise ModelUnavailable("Cannot load model u2net.onnx: INVALID_PROTOBUF")

    monkeypatch.setattr(ort_mod, "load_model", corrupt)
    with pytest.raises(ModelUnavailable):
        rt.open_model("u2net")
    assert not rt.recent_download_failure("u2net")


def test_download_memo_expires(monkeypatch):
    clock = [50.0]
    monkeypatch.setattr(rt, "_clock", lambda: clock[0])
    rt.note_download_failure("u2net")
    assert rt.recent_download_failure("u2net")
    clock[0] += rt.DOWNLOAD_RETRY_S + 1
    assert not rt.recent_download_failure("u2net")


def test_auto_candidate_ok(monkeypatch):
    monkeypatch.setattr(rt, "is_available", lambda mid: False)
    monkeypatch.setattr(rt, "is_registered", lambda mid: True)
    assert rt.auto_candidate_ok(None)
    assert rt.auto_candidate_ok("birefnet-lite")
    rt.note_download_failure("birefnet-lite")
    assert not rt.auto_candidate_ok("birefnet-lite")
    monkeypatch.setattr(rt, "is_available", lambda mid: True)  # present locally after all
    assert rt.auto_candidate_ok("birefnet-lite")


def test_new_auto_instance_skips_recently_failed_download(monkeypatch, photo_rgb):
    """Offline: the second job must not wait for the same failing download again."""
    monkeypatch.setattr(rt, "is_available", lambda mid: False)
    monkeypatch.setattr(rt, "is_registered", lambda mid: True)
    attempts: list[str] = []

    def offline(model_id, **kw):
        attempts.append(model_id)
        rt.note_download_failure(model_id)  # what the real open_model does
        raise DownloadError("offline")

    monkeypatch.setattr(rt, "open_model", offline)
    AutoSegmenter().segment(photo_rgb)
    assert attempts == ["birefnet-lite"]
    seg2 = AutoSegmenter()
    assert seg2.preferred == "grabcut"
    seg2.segment(photo_rgb)
    assert attempts == ["birefnet-lite"]
    # an explicitly chosen model is still attempted (the user asked for it)
    with pytest.raises(DownloadError):
        BiRefNetSegmenter().segment(photo_rgb)
    assert attempts == ["birefnet-lite", "birefnet-lite"]


# ----------------------------------------------------------------------------- auto order
def test_auto_returns_to_model_once_downloaded(monkeypatch, fake_session_cls, subject_img):
    """A fallback to GrabCut must not stick once the model has been downloaded."""
    avail = {"birefnet-lite": False}
    monkeypatch.setattr(rt, "is_available", lambda mid: avail.get(mid, False))
    monkeypatch.setattr(rt, "is_registered", lambda mid: True)
    sess = _bright_sess(fake_session_cls)

    def opener(model_id, **kw):
        if not avail[model_id]:
            raise DownloadError("offline")
        return sess

    monkeypatch.setattr(rt, "open_model", opener)
    seg = AutoSegmenter()
    seg.segment(subject_img)
    assert seg.resolved == "grabcut"
    seg.segment(subject_img)
    assert seg.resolved == "grabcut"  # still not downloaded: not retried
    avail["birefnet-lite"] = True
    rt.reset_download_failures()
    assert seg.preferred == "birefnet-lite"
    alpha = seg.segment(subject_img)
    assert seg.resolved == "birefnet-lite" and seg.id == "birefnet-lite"
    assert alpha[80, 100] > 0.5


def test_auto_dedupes_candidates():
    seg = AutoSegmenter(["isnet", "isnet-general", "auto", "grabcut", "u2net"])
    assert seg.candidates == ["isnet", "grabcut", "u2net"]


# ----------------------------------------------------------------------------- error texts
@pytest.mark.parametrize(
    "exc",
    [
        MemoryError(),
        RuntimeError("[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : bad allocation"),
        InferenceError("Model x.onnx failed: Failed to allocate memory for requested buffer"),
    ],
)
def test_out_of_memory_is_explained(fake_session_cls, subject_img, exc):
    def boom(feeds):
        raise exc

    seg = BiRefNetSegmenter(model=fake_session_cls(["input_image"], [[1, 3, 1024, 1024]], boom))
    with pytest.raises(InferenceError) as ei:
        seg.segment(subject_img)
    assert "ran out of memory" in str(ei.value)
    assert ei.value.hint and "smaller photo" in ei.value.hint


def test_inference_error_from_models_layer_gets_hint(fake_session_cls, subject_img):
    def boom(feeds):
        raise InferenceError("Model model.onnx failed on GPU and CPU: shape mismatch")

    seg = BiRefNetSegmenter(model=fake_session_cls(["input_image"], [[1, 3, 1024, 1024]], boom))
    with pytest.raises(InferenceError) as ei:
        seg.segment(subject_img)
    assert "shape mismatch" in str(ei.value)
    assert ei.value.hint


# ----------------------------------------------------------------------------- post-processing
def test_minmax_ignores_non_finite_values(fake_session_cls, subject_img):
    def fn(feeds):
        x = feeds["input.1"].mean(axis=1, keepdims=True)
        x[..., :4, :4] = np.inf
        x[..., -4:, -4:] = np.nan
        return x

    seg = ISNetSegmenter(model=fake_session_cls(["input.1"], [[1, 3, 1024, 1024]], fn))
    alpha = seg.segment(subject_img)
    assert np.isfinite(alpha).all()
    assert alpha[80, 100] > 0.9 and alpha[150, 10] < 0.1


def test_sigmoid_nan_is_background(fake_session_cls, subject_img):
    def fn(feeds):
        x = 8.0 * feeds["input_image"].mean(axis=1, keepdims=True)
        x[..., :100, :100] = np.nan
        return x

    seg = BiRefNetSegmenter(model=fake_session_cls(["input_image"], [[1, 3, 1024, 1024]], fn))
    alpha = seg.segment(subject_img)
    assert np.isfinite(alpha).all()
    assert alpha[2, 2] == pytest.approx(0.0, abs=1e-6)


def test_small_image_alpha_is_area_downsampled(fake_session_cls):
    """Output shrunk from 1024 to a small photo keeps a clean, non-aliased edge."""
    img = np.zeros((50, 50, 3), np.uint8)
    img[:, 25:] = 255

    def fn(feeds):
        # a 1-px checker pattern on the right half: area filtering averages it to ~0.5
        out = np.full((1, 1, 1024, 1024), -60.0, np.float32)
        out[..., 512:] = np.where((np.indices((1024, 512)).sum(0) % 2) == 0, 60.0, -60.0)
        return out

    seg = BiRefNetSegmenter(model=fake_session_cls(["input_image"], [[1, 3, 1024, 1024]], fn))
    alpha = seg.segment(img)
    assert alpha.shape == (50, 50)
    np.testing.assert_allclose(alpha[:, 30:], 0.5, atol=0.05)
    assert alpha[:, :20].max() < 0.01


def test_waiting_for_another_threads_download_is_cancellable(monkeypatch, subject_img):
    """A second job waiting while the first downloads the model can still be cancelled."""
    import threading

    from vrphoto.core.errors import Cancelled
    from vrphoto.core.types import CancelToken

    started, release = threading.Event(), threading.Event()

    def slow_download(model_id, **kw):
        started.set()
        release.wait(10)
        raise DownloadError("offline")

    monkeypatch.setattr(rt, "open_model", slow_download)
    seg = BiRefNetSegmenter()
    errors: list[BaseException] = []

    def first():
        try:
            seg.segment(subject_img)
        except BaseException as e:  # collected for the assertion below
            errors.append(e)

    t = threading.Thread(target=first)
    t.start()
    assert started.wait(5)
    tok = CancelToken()
    timer = threading.Timer(0.3, tok.cancel)
    timer.start()
    try:
        with pytest.raises(Cancelled):
            seg.segment(subject_img, cancel=tok)
    finally:
        release.set()
        t.join(5)
        timer.cancel()
    assert len(errors) == 1 and isinstance(errors[0], DownloadError)
