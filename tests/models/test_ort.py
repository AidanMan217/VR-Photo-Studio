"""OrtModel: provider selection, GPU->CPU fallback, fp16 handling, caching, threads."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import onnxruntime as real_ort
import pytest

from vrphoto.core.config import get_settings
from vrphoto.core.errors import Cancelled, InferenceError, ModelUnavailable
from vrphoto.core.types import CancelToken
from vrphoto.models import ort
from vrphoto.models.gpu import Device
from vrphoto.models.ort import CPU, CUDA, DML, OrtModel, load_model, parse_device, select_providers


class FakeOrt:
    """Stand-in for the onnxruntime module: GPU providers are simulated, computation is
    delegated to a real CPU session so outputs are genuine."""

    def __init__(
        self,
        available,
        *,
        fail_create=(),
        fail_run=(),
        nan_on=(),
        internal_fallback=False,
        hang_run=False,
        run_error="DXGI_ERROR_DEVICE_REMOVED",
    ):
        self.available = list(available)
        self.hang_run = hang_run
        self.run_error = run_error
        self.fail_create = set(fail_create)
        self.fail_run = set(fail_run)
        self.nan_on = set(nan_on)
        self.internal_fallback = internal_fallback
        self.calls: list[tuple] = []
        outer = self

        class SessionOptions:
            def __init__(self):
                self.enable_mem_pattern = True
                self.execution_mode = "parallel"
                self.graph_optimization_level = None
                self.log_severity_level = 2

        class InferenceSession:
            def __init__(self, path, sess_options=None, providers=None, provider_options=None):
                prov = providers[0]
                outer.calls.append(("create", prov, dict(provider_options[0]), sess_options))
                if prov in outer.fail_create:
                    raise RuntimeError(f"{prov}: D3D12CreateDevice failed")
                self._p = [CPU] if outer.internal_fallback else list(providers)
                self._real = real_ort.InferenceSession(path, providers=["CPUExecutionProvider"])

            def get_providers(self):
                return self._p

            def get_inputs(self):
                return self._real.get_inputs()

            def get_outputs(self):
                return self._real.get_outputs()

            def disable_fallback(self):
                outer.calls.append(("disable_fallback",))

            def run(self, names, feeds, run_options=None):
                outer.calls.append(("run", self._p[0]))
                if outer.hang_run:  # a long inference: only RunOptions.terminate stops it
                    while not (run_options is not None and run_options.terminate):
                        time.sleep(0.01)
                    raise RuntimeError("Exiting due to terminate flag being set to true.")
                if self._p[0] in outer.fail_run:
                    raise RuntimeError(outer.run_error)
                res = self._real.run(names, feeds)
                if self._p[0] in outer.nan_on:
                    res = [np.full_like(r, np.nan) for r in res]
                return res

        class RunOptions:
            def __init__(self):
                self.terminate = False

        self.SessionOptions = SessionOptions
        self.InferenceSession = InferenceSession
        self.RunOptions = RunOptions
        self.GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL="all")
        self.ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL="sequential")

    def get_available_providers(self):
        return list(self.available)

    def creates(self) -> list[str]:
        return [c[1] for c in self.calls if c[0] == "create"]


@pytest.fixture
def fake(monkeypatch):
    """Install a FakeOrt; call it with options to (re)configure."""

    def install(available=(DML, CPU), **kw):
        f = FakeOrt(available, **kw)
        monkeypatch.setattr(ort, "_import_ort", lambda: f)
        return f

    return install


@pytest.fixture
def no_adapters(monkeypatch):
    monkeypatch.setattr("vrphoto.models.gpu.list_devices", lambda **k: [])


def _x(h=28, w=42):
    rng = np.random.default_rng(0)
    return rng.random((1, 3, h, w), dtype=np.float32)


# --------------------------------------------------------------------------- real CPU path
def test_real_cpu_session(stub_models):
    m = OrtModel(stub_models["da2-small"], "cpu")
    assert m.provider == CPU
    assert m.input_names() == ["pixel_values"]
    assert m.input_shapes() == [[1, 3, "height", "width"]]
    assert m.input_types() == ["tensor(float)"]
    assert m.output_names() == ["predicted_depth"]
    x = _x()
    out = m.run({"pixel_values": x})
    assert set(out) == {"predicted_depth"}
    np.testing.assert_allclose(out["predicted_depth"][0], np.maximum(x[0].mean(0), 0), rtol=1e-5, atol=1e-6)
    # float64 / named outputs work too
    out2 = m.run({"pixel_values": x.astype(np.float64)}, ["predicted_depth"])
    assert out2["predicted_depth"].dtype == np.float32
    assert "da2-small.onnx" in repr(m)


def test_moge_stub_scalar_int_input(stub_models):
    m = OrtModel(stub_models["moge2-vitl"], "cpu")
    assert m.input_names() == ["image", "num_tokens"]
    assert m.input_shapes()[1] == []
    out = m.run({"image": np.zeros((1, 3, 256, 256), np.float32), "num_tokens": np.array(1800)})
    assert out["points"].shape == (1, 256, 256, 3)
    assert out["metric_scale"].shape == (1,)


def test_fp16_inputs_are_cast(stub_models):
    m = OrtModel(stub_models["da2-fp16"], "cpu", fp16=True)
    assert m.input_types() == ["tensor(float16)"]
    assert m.has_fp16_io
    x = _x()
    out = m.run({"pixel_values": x})
    assert out["predicted_depth"].dtype == np.float32
    np.testing.assert_allclose(out["predicted_depth"][0], np.maximum(x[0].mean(0), 0), atol=2e-3)


def test_bad_feeds(stub_models):
    m = OrtModel(stub_models["da2-small"], "cpu")
    with pytest.raises(ValueError, match="Missing input"):
        m.run({})
    with pytest.raises(ValueError, match="Unknown input"):
        m.run({"pixel_values": _x(), "extra": _x()})


def test_cpu_failure_raises_inference_error(stub_models):
    m = OrtModel(stub_models["lama"], "cpu")
    with pytest.raises(InferenceError):
        m.run({"image": np.zeros((1, 3, 10, 10), np.float32), "mask": np.zeros((1, 1, 10, 10), np.float32)})


def test_missing_and_corrupt_files(tmp_path):
    with pytest.raises(ModelUnavailable, match="not found"):
        OrtModel(tmp_path / "nope.onnx", "cpu")
    bad = tmp_path / "bad.onnx"
    bad.write_bytes(b"this is not a protobuf")
    with pytest.raises(ModelUnavailable) as ei:
        OrtModel(bad, "cpu")
    assert "delete" in (ei.value.hint or "").lower()


def test_onnxruntime_missing(monkeypatch, stub_models):
    def boom():
        raise ModelUnavailable("ONNX Runtime could not be loaded: DLL load failed")

    monkeypatch.setattr(ort, "_import_ort", boom)
    assert ort.available_providers() == []
    with pytest.raises(ModelUnavailable):
        OrtModel(stub_models["da2-small"], "cpu")


# --------------------------------------------------------------------------- device parsing / selection
@pytest.mark.parametrize(
    "dev,expect",
    [
        ("cpu", ("cpu", None)),
        ("CPU", ("cpu", None)),
        (0, ("gpu", 0)),
        (2, ("gpu", 2)),
        ("1", ("gpu", 1)),
        ("dml:3", ("gpu", 3)),
        ("cuda:0", ("gpu", 0)),
        ("gpu", ("gpu", None)),
        ("auto", ("auto", None)),
        (None, ("auto", None)),
        ("weird", ("auto", None)),
        (-1, ("cpu", None)),
    ],
)
def test_parse_device(dev, expect):
    assert parse_device(dev) == expect


def test_parse_device_auto_uses_settings():
    get_settings().device = "cpu"
    assert parse_device("auto") == ("cpu", None)
    get_settings().device = "1"
    assert parse_device("auto") == ("gpu", 1)


def test_select_providers(monkeypatch):
    devs = [Device(0, "Intel UHD", 128, "intel"), Device(1, "NVIDIA RTX 4070", 12282, "nvidia")]
    monkeypatch.setattr("vrphoto.models.gpu.list_devices", lambda **k: devs)
    assert select_providers("cpu", available=[DML, CPU]) == [(CPU, {})]
    # auto picks the adapter with the most VRAM (its DXGI index = DirectML device_id)
    assert select_providers("auto", available=[DML, CPU]) == [(DML, {"device_id": "1"}), (CPU, {})]
    assert select_providers(0, available=[DML, CPU])[0] == (DML, {"device_id": "0"})
    assert select_providers("dml:2", available=[DML, CPU])[0] == (DML, {"device_id": "2"})
    assert select_providers("auto", dml_ok=False, available=[DML, CPU]) == [(CPU, {})]
    assert select_providers("auto", available=[CPU]) == [(CPU, {})]
    assert select_providers("auto", available=["AzureExecutionProvider", CPU]) == [(CPU, {})]
    assert select_providers("auto", available=[CUDA, CPU]) == [(CUDA, {"device_id": "0"}), (CPU, {})]
    # Settings hold the DXGI index; CUDA numbers only the NVIDIA cards (iGPU listed first)
    assert select_providers(1, dml_ok=False, available=[CUDA, CPU])[0] == (CUDA, {"device_id": "0"})
    assert select_providers(0, available=[CUDA, CPU])[0] == (CUDA, {"device_id": "0"})


def test_cuda_ordinal_mapping(monkeypatch):
    devs = [
        Device(0, "Intel UHD", 128, "intel"),
        Device(1, "RTX 3060", 12288, "nvidia"),
        Device(3, "RTX 4090", 24564, "nvidia"),  # index 2 was a skipped software adapter
    ]
    monkeypatch.setattr("vrphoto.models.gpu.list_devices", lambda **k: devs)
    assert select_providers(3, available=[CUDA, CPU])[0] == (CUDA, {"device_id": "1"})
    assert select_providers("cuda:1", available=[CUDA, CPU])[0] == (CUDA, {"device_id": "0"})
    monkeypatch.setattr("vrphoto.models.gpu.list_devices", lambda **k: [])  # Linux: no DXGI list
    assert select_providers(2, available=[CUDA, CPU])[0] == (CUDA, {"device_id": "2"})


def test_select_providers_without_adapter_info(no_adapters):
    chain = select_providers("auto", available=[DML, CPU])
    assert chain[0] == (DML, {"performance_preference": "high_performance", "device_filter": "gpu"})


# --------------------------------------------------------------------------- fallback behaviour
def test_dml_session_options(fake, stub_models, no_adapters):
    f = fake()
    m = OrtModel(stub_models["da2-small"])
    assert m.provider == DML
    _, prov, _opts, so = f.calls[0]
    assert prov == DML
    assert so.enable_mem_pattern is False
    assert so.execution_mode == "sequential"
    assert ("disable_fallback",) in f.calls
    assert m.run({"pixel_values": _x()})["predicted_depth"].shape == (1, 28, 42)


def test_dml_creation_failure_falls_back_to_cpu_and_sticks(fake, stub_models, no_adapters):
    f = fake(fail_create=[DML])
    m = OrtModel(stub_models["da2-small"])
    assert m.provider == CPU
    assert f.creates() == [DML, CPU]
    m.run({"pixel_values": _x()})
    # sticky: a new model for the same file does not try DirectML again
    m2 = OrtModel(stub_models["da2-small"])
    assert m2.provider == CPU
    assert f.creates() == [DML, CPU, CPU]
    # other files still try the GPU
    m3 = OrtModel(stub_models["nan"])
    assert f.creates()[-2:] == [DML, CPU]
    assert m3.provider == CPU
    ort.reset_cpu_fallbacks()
    f.fail_create.clear()
    assert OrtModel(stub_models["da2-small"]).provider == DML


def test_first_run_failure_retries_on_cpu(fake, stub_models, no_adapters):
    f = fake(fail_run=[DML])
    m = OrtModel(stub_models["da2-small"])
    assert m.provider == DML
    x = _x()
    out = m.run({"pixel_values": x})
    np.testing.assert_allclose(out["predicted_depth"][0], np.maximum(x[0].mean(0), 0), rtol=1e-5, atol=1e-6)
    assert m.provider == CPU
    runs = [c[1] for c in f.calls if c[0] == "run"]
    assert runs == [DML, CPU]
    assert OrtModel(stub_models["da2-small"]).provider == CPU  # sticky


def test_ort_internal_fallback_detected(fake, stub_models, no_adapters):
    f = fake(internal_fallback=True)
    m = OrtModel(stub_models["da2-small"])
    assert m.provider == CPU
    assert f.creates() == [DML]
    OrtModel(stub_models["da2-small"])
    assert f.creates() == [DML, CPU]  # remembered


def test_dml_ok_false_never_uses_dml(fake, stub_models, no_adapters):
    f = fake()
    m = OrtModel(stub_models["lama"], dml_ok=False)
    assert m.provider == CPU
    assert f.creates() == [CPU]


def test_cuda_preferred_when_available(fake, stub_models):
    f = fake(available=[CUDA, CPU])
    m = OrtModel(stub_models["da2-small"], 0)
    assert m.provider == CUDA
    assert f.calls[0][2] == {"device_id": "0"}
    assert f.calls[0][3].enable_mem_pattern is True  # DML-only options not applied


def test_device_index_passed_to_dml(fake, stub_models):
    f = fake()
    OrtModel(stub_models["da2-small"], 1)
    assert f.calls[0][2] == {"device_id": "1"}


def test_nan_output_on_gpu_retries_on_cpu(fake, stub_models, no_adapters):
    fake(nan_on=[DML])
    m = OrtModel(stub_models["da2-small"])
    out = m.run({"pixel_values": _x()})
    assert np.isfinite(out["predicted_depth"]).all()
    assert m.provider == CPU


def test_nan_fp16_switches_to_fp32_variant(stub_models):
    calls = []

    def fb():
        calls.append(1)
        return stub_models["da2-small"]

    m = OrtModel(stub_models["nan"], "cpu", fp16=True, fallback_path=fb)
    out = m.run({"pixel_values": _x()})
    assert calls == [1]
    assert np.isfinite(out["predicted_depth"]).all()
    assert m.path == stub_models["da2-small"]
    assert m.fp16 is False


def test_nan_on_cpu_without_fallback_is_returned(stub_models, caplog):
    m = OrtModel(stub_models["nan"], "cpu", fp16=True)
    with caplog.at_level("WARNING"):
        out = m.run({"pixel_values": _x()})
    assert np.isnan(out["predicted_depth"]).all()
    assert any("NaN" in r.message for r in caplog.records)


def test_concurrent_runs_are_safe(stub_models):
    m = OrtModel(stub_models["da2-small"], "cpu")
    errors: list[BaseException] = []

    def worker(seed):
        rng = np.random.default_rng(seed)
        for _ in range(10):
            x = rng.random((1, 3, 14, 28), dtype=np.float32)
            try:
                y = m.run({"pixel_values": x})["predicted_depth"][0]
                np.testing.assert_allclose(y, np.maximum(x[0].mean(0), 0), rtol=1e-5, atol=1e-6)
            except BaseException as e:  # pragma: no cover
                errors.append(e)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    assert not errors


# --------------------------------------------------------------------------- load_model cache
def test_load_model_caches(monkeypatch, stub_models):
    monkeypatch.setenv("VRPHOTO_MODEL_DA2_SMALL", str(stub_models["da2-small"]))
    a = load_model("da2-small", device="cpu")
    b = load_model("da2-small", device="cpu")
    assert a is b
    assert ("da2-small", "cpu", CPU) in ort.cached_models()
    ort.unload_model("da2-small")
    assert load_model("da2-small", device="cpu") is not a
    # changing the file behind the id gives a new session
    monkeypatch.setenv("VRPHOTO_MODEL_DA2_SMALL", str(stub_models["da2-fp16"]))
    c = load_model("da2-small", device="cpu")
    assert c.input_types() == ["tensor(float16)"]
    ort.unload_model()
    assert ort.cached_models() == []


def test_load_model_uses_registry_dml_flag(monkeypatch, fake, stub_models, no_adapters):
    f = fake()
    monkeypatch.setenv("VRPHOTO_MODEL_LAMA", str(stub_models["lama"]))
    m = load_model("lama")
    assert m.provider == CPU
    assert f.creates() == [CPU]


def test_load_model_fp16_variant(monkeypatch, fake, stub_models, no_adapters):
    fake()
    seen = []

    def fake_ensure(model_id, *, progress=None, cancel=None, token=None, variant="default"):
        seen.append(variant)
        return stub_models["da2-fp16"] if variant == "fp16" else stub_models["da2-small"]

    monkeypatch.setattr("vrphoto.models.download.ensure_model", fake_ensure)
    get_settings().use_fp16 = True
    m = load_model("da2-small")
    assert seen == ["fp16"]
    assert m.fp16 and m.has_fp16_io
    # fp16 is never used on the CPU
    m2 = load_model("da2-small", device="cpu")
    assert seen[-1] == "default"
    assert not m2.fp16


def test_load_model_unknown_id():
    with pytest.raises(ModelUnavailable):
        load_model("nope")


def test_inherent_nan_does_not_pin_model_to_cpu(fake, stub_models, no_adapters):
    f = fake()
    m = OrtModel(stub_models["nan"])  # NaN on every provider (it is the model, not the GPU)
    out = m.run({"pixel_values": _x()})
    assert np.isnan(out["predicted_depth"]).all()
    assert m.provider == DML  # restored after the CPU comparison run
    assert f.creates() == [DML, CPU, DML]
    m.run({"pixel_values": _x()})
    assert f.creates() == [DML, CPU, DML]  # no repeated recovery
    assert OrtModel(stub_models["nan"]).provider == DML  # not sticky


# --------------------------------------------------------------------------- review additions
def test_run_cancel_terminates_real_inference(stub_models):
    m = OrtModel(stub_models["slow"], "cpu")
    x = np.ones((1, 384, 384), np.float32)
    tok = CancelToken()
    threading.Timer(0.05, tok.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        m.run({"pixel_values": x}, cancel=tok)
    cancelled_after = time.monotonic() - t0
    # the session is still usable afterwards and stays on its provider
    t0 = time.monotonic()
    out = m.run({"pixel_values": x}, cancel=CancelToken())
    full = time.monotonic() - t0
    assert np.allclose(out["predicted_depth"], x)
    assert cancelled_after < 0.6 * full  # stopped mid-graph, not after the whole run
    pre = CancelToken()
    pre.cancel()
    with pytest.raises(Cancelled):
        m.run({"pixel_values": x}, cancel=pre)


def test_cancel_on_gpu_is_not_treated_as_a_gpu_failure(fake, stub_models, no_adapters):
    f = fake(hang_run=True)
    m = OrtModel(stub_models["da2-small"])
    assert m.provider == DML
    tok = CancelToken()
    threading.Timer(0.1, tok.cancel).start()
    with pytest.raises(Cancelled):
        m.run({"pixel_values": _x()}, cancel=tok)
    assert m.provider == DML and not ort._is_cpu_sticky(m.path)
    assert f.creates() == [DML]  # no CPU re-creation


def test_closed_model(stub_models):
    m = OrtModel(stub_models["da2-small"], "cpu", title="Depth Anything V2 Small")
    assert m.closed is False
    m.close()
    assert m.closed is True
    with pytest.raises(ModelUnavailable, match="Depth Anything V2 Small"):
        m.run({"pixel_values": _x()})


def test_inference_errors_are_friendly(fake, stub_models, no_adapters):
    fake(
        available=[CPU],
        fail_run=[CPU],
        run_error="[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : bad allocation",
    )
    m = OrtModel(stub_models["da2-small"], title="Depth Anything V2 Small")
    with pytest.raises(InferenceError) as ei:
        m.run({"pixel_values": _x()})
    assert "Depth Anything V2 Small" in str(ei.value) and "bad allocation" in str(ei.value)
    assert "memory" in (ei.value.hint or "").lower()
    fake(fail_run=[DML, CPU], run_error="shape mismatch")
    m = OrtModel(stub_models["da2-small"], title="X")
    with pytest.raises(InferenceError, match="GPU and on the CPU") as ei:
        m.run({"pixel_values": _x()})
    assert ei.value.hint


def test_load_model_sets_title(monkeypatch, stub_models):
    monkeypatch.setenv("VRPHOTO_MODEL_DA2_SMALL", str(stub_models["da2-small"]))
    m = load_model("da2-small", device="cpu")
    assert m.title == "Depth Anything V2 Small"


def test_unload_waits_for_running_inference_without_blocking_cache(monkeypatch, stub_models):
    monkeypatch.setenv("VRPHOTO_MODEL_DA2_SMALL", str(stub_models["slow"]))
    m = load_model("da2-small", device="cpu")
    x = np.ones((1, 384, 384), np.float32)
    t = threading.Thread(target=lambda: m.run({"pixel_values": x}))
    t.start()
    time.sleep(0.05)
    ort.unload_model("da2-small")  # waits for the run, then closes
    t.join(10)
    assert m.closed and ort.cached_models() == []


def test_load_model_fp16_missing_upstream_uses_fp32(monkeypatch, fake, stub_models, no_adapters):
    from vrphoto.core.errors import DownloadError

    fake()
    seen = []

    def fake_ensure(model_id, *, progress=None, cancel=None, token=None, variant="default"):
        seen.append(variant)
        if variant == "fp16":
            raise DownloadError("model_fp16.onnx was not found on huggingface.co (HTTP 404).")
        return stub_models["da2-small"]

    monkeypatch.setattr("vrphoto.models.download.ensure_model", fake_ensure)
    get_settings().use_fp16 = True
    m = load_model("da2-small")
    assert seen == ["fp16", "default"]
    assert not m.fp16 and m.provider == DML


def test_gpu_failure_of_fp16_model_switches_to_fp32_file_on_cpu(monkeypatch, fake, stub_models, no_adapters):
    fake(fail_run=[DML])
    m = OrtModel(stub_models["da2-fp16"], fp16=True, fallback_path=stub_models["da2-small"])
    assert m.provider == DML and m.has_fp16_io
    out = m.run({"pixel_values": _x()})
    assert m.provider == CPU and not m.fp16 and m.path == stub_models["da2-small"]
    assert out["predicted_depth"].dtype == np.float32 and not m.has_fp16_io
