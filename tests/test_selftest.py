"""Installation self-test."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vrphoto import selftest as st

EXPECTED = {
    "imports",
    "data_folder",
    "onnxruntime",
    "ffmpeg",
    "spatial_metadata",
    "https_server",
    "viewer_assets",
    "addons",
    "uv",
    "geometry",
    "devices",
}


@pytest.mark.timeout(240)
def test_selftest_passes_here(tmp_path):
    out = tmp_path / "report" / "selftest.json"
    seen: list[str] = []
    report = st.run_selftest(out, progress=seen.append)
    failed = {k: v.get("error") for k, v in report["checks"].items() if not v["ok"]}
    assert report["ok"] is True, failed
    assert set(report["checks"]) == EXPECTED
    assert seen == list(report["checks"]) == st.check_names()
    assert st.check_names(models=True) == [*st.check_names(), "models"]
    for c in report["checks"].values():
        assert c["seconds"] >= 0 and "detail" in c
    assert report["failed"] == [] and report["version"] and report["python"]
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["ok"] is True and set(saved["checks"]) == EXPECTED
    json.dumps(report)  # JSON-able
    det = report["checks"]
    assert det["onnxruntime"]["detail"]["runs"]["CPUExecutionProvider"] == "ok"
    assert det["ffmpeg"]["detail"]["encoder"]
    assert det["spatial_metadata"]["detail"]["metadata"]["coverage"] == 180
    assert det["https_server"]["detail"]["index_bytes"] > 100
    assert "trellis" in det["addons"]["detail"]["addons"]
    assert det["geometry"]["detail"]["glb_bytes"] > 1000


def test_failing_check_is_reported_and_others_still_run(monkeypatch):
    def broken():
        raise st.CheckFailed("Missing viewer files: three.core.js")

    def crash():
        raise RuntimeError("unexpected")

    monkeypatch.setattr(st, "check_viewer_assets", broken)
    monkeypatch.setattr(st, "check_uv", crash)
    monkeypatch.setattr(st, "check_ffmpeg", lambda tmp: {"skipped": True})
    monkeypatch.setattr(st, "check_spatial", lambda tmp: {"skipped": True})
    report = st.run_selftest()
    assert report["ok"] is False
    assert set(report["failed"]) == {"viewer_assets", "uv"}
    assert report["checks"]["viewer_assets"]["error"] == "Missing viewer files: three.core.js"
    assert "traceback" not in report["checks"]["viewer_assets"]
    assert "traceback" in report["checks"]["uv"]  # unexpected errors keep a traceback for support
    assert report["checks"]["geometry"]["ok"] is True


def test_informational_checks_do_not_fail(monkeypatch):
    def no_devices():
        raise RuntimeError("no DXGI")

    for name in (
        "check_imports",
        "check_onnxruntime",
        "check_viewer_assets",
        "check_addons",
        "check_uv",
        "check_data_folder",
    ):
        monkeypatch.setattr(st, name, lambda: {})
    for name in ("check_ffmpeg", "check_spatial", "check_https_server", "check_geometry"):
        monkeypatch.setattr(st, name, lambda tmp: {})
    monkeypatch.setattr(st, "_devices", no_devices)
    report = st.run_selftest()
    assert report["ok"] is True
    assert report["checks"]["devices"]["ok"] is False and report["checks"]["devices"]["required"] is False


def test_models_check_only_when_requested(monkeypatch):
    called = []
    for name in (
        "check_imports",
        "check_onnxruntime",
        "check_viewer_assets",
        "check_addons",
        "check_uv",
        "check_data_folder",
        "_devices",
    ):
        monkeypatch.setattr(st, name, lambda: {})
    for name in ("check_ffmpeg", "check_spatial", "check_https_server", "check_geometry"):
        monkeypatch.setattr(st, name, lambda tmp: {})
    monkeypatch.setattr(st, "check_models", lambda tmp: called.append(1) or {"model": "x"})
    assert "models" not in st.run_selftest()["checks"]
    report = st.run_selftest(models=True)
    assert report["checks"]["models"]["ok"] and called == [1]


def test_models_check_fails_offline(monkeypatch):
    """Without network (or with a broken download) the models check fails cleanly."""
    from vrphoto.core.errors import DownloadError
    from vrphoto.models import download

    def offline(*a, **k):
        raise DownloadError("No internet connection", hint="Check your network")

    monkeypatch.setattr(download, "ensure_model", offline)
    entry: dict = {}
    st._run_check(entry, "models", lambda: st.check_models(None))
    assert entry["models"]["ok"] is False
    assert (
        "No internet connection" in entry["models"]["error"]
        and "Check your network" in entry["models"]["error"]
    )


def test_embedded_onnx_model_matches_reference():
    import onnxruntime as ort

    sess = ort.InferenceSession(st.tiny_onnx_model(), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(5).random((1, 3, 8, 8), dtype=np.float32)
    y = sess.run(None, {"x": x})[0]
    assert y.shape == (1, 4, 8, 8)
    np.testing.assert_allclose(y, st.tiny_onnx_reference(x), atol=1e-5)


def test_embedded_model_is_valid_onnx():
    onnx = pytest.importorskip("onnx")
    m = onnx.load_from_string(st.tiny_onnx_model())
    onnx.checker.check_model(m)
    assert m.opset_import[0].version == 13


def test_wrong_provider_results_fail(monkeypatch):
    monkeypatch.setattr(st, "tiny_onnx_reference", lambda x: np.zeros((1, 4, 8, 8), np.float32) - 5)
    with pytest.raises(st.CheckFailed, match="CPU provider|Wrong results"):
        st.check_onnxruntime()


def test_viewer_asset_list_is_complete():
    from vrphoto.core.config import resource_path

    root = resource_path("webxr", "viewer")
    for f in st.VIEWER_FILES:
        assert (root / f).is_file(), f
