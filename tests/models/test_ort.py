"""OrtModel: provider selection, GPU->CPU fallback, fp16 handling, caching, threads."""

from __future__ import annotations

import gc
import threading
import time
import weakref
from pathlib import Path
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


class _NodeArg:
    """Like onnxruntime's ``NodeArg``: it keeps the session that returned it alive (pybind11
    ``reference_internal``), so holding one prevents the session (and its GPU memory) from
    being freed."""

    def __init__(self, session, arg):
        self._session = session
        self.name, self.shape, self.type = arg.name, arg.shape, arg.type


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
        run_hook=None,
        create_hook=None,
    ):
        self.available = list(available)
        self.run_hook = run_hook  # (provider, file name) -> None, may raise
        self.create_hook = create_hook  # (provider, file name) -> None, may raise
        self.runs: list[tuple[str, str]] = []  # (provider, file name) per session.run
        self.freed: list[tuple[str, str]] = []  # (provider, file name) of garbage-collected sessions
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
                self._name = Path(path).name
                if outer.create_hook is not None:
                    outer.create_hook(prov, self._name)
                self._p = [CPU] if outer.internal_fallback else list(providers)
                self._real = real_ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                weakref.finalize(self, outer.freed.append, (self._p[0], self._name))

            def get_providers(self):
                return self._p

            def get_inputs(self):
                return [_NodeArg(self, a) for a in self._real.get_inputs()]

            def get_outputs(self):
                return [_NodeArg(self, a) for a in self._real.get_outputs()]

            def disable_fallback(self):
                outer.calls.append(("disable_fallback",))

            def run(self, names, feeds, run_options=None):
                outer.calls.append(("run", self._p[0]))
                outer.runs.append((self._p[0], self._name))
                if outer.run_hook is not None:
                    outer.run_hook(self._p[0], self._name)
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


# --------------------------------------------------------------------------- out of GPU memory (v0.1.1)
OOM = "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : bad allocation"


def _oom_on(fail: dict[tuple[str, str], int]):
    """run/create hook: raise an out-of-memory error ``fail[(provider, file)]`` times."""

    def hook(prov, name):
        left = fail.get((prov, name), 0)
        if left:
            fail[(prov, name)] = left - 1
            raise RuntimeError(OOM)

    return hook


def _moge_feeds():
    return {"image": np.zeros((1, 3, 256, 256), np.float32), "num_tokens": np.array(1800)}


@pytest.fixture
def stub_env(monkeypatch, stub_models):
    from vrphoto.models.registry import env_var_for

    for mid in ("da2-small", "moge2-vitl", "birefnet-lite"):
        monkeypatch.setenv(env_var_for(mid), str(stub_models[mid]))
    return stub_models


def test_gpu_oom_releases_other_models_and_retries_on_the_gpu(fake, stub_env, no_adapters):
    """A GPU out-of-memory error frees the other cached sessions and retries on the GPU:
    no slow CPU run and nothing sticky; the released model reloads transparently."""
    fail = {(DML, "da2-small.onnx"): 1}
    f = fake(run_hook=_oom_on(fail))
    other = load_model("moge2-vitl")
    m = load_model("da2-small")
    assert other.provider == DML and m.provider == DML
    x = _x()
    out = m.run({"pixel_values": x})
    np.testing.assert_allclose(out["predicted_depth"][0], np.maximum(x[0].mean(0), 0), rtol=1e-5, atol=1e-6)
    assert m.provider == DML and not ort._is_cpu_sticky(m.path)
    assert [r for r in f.runs if r[1] == "da2-small.onnx"] == [(DML, "da2-small.onnx")] * 2
    # the other model's GPU session was really dropped, but the object stays cached and usable
    assert (DML, "moge2-vitl.onnx") in f.freed
    assert other.released and not other.closed
    assert ("moge2-vitl", f"auto|{get_settings().device}", DML) in ort.cached_models()
    assert load_model("moge2-vitl") is other
    creates = len(f.creates())
    assert other.run(_moge_feeds())["points"].shape == (1, 256, 256, 3)
    assert f.creates()[creates:] == [DML]  # re-created on the GPU on next use
    assert not other.released and other.provider == DML


def test_gpu_oom_twice_runs_only_this_call_on_the_cpu(fake, stub_env, no_adapters):
    fail = {(DML, "da2-small.onnx"): 99}
    f = fake(run_hook=_oom_on(fail))
    load_model("moge2-vitl")
    m = load_model("da2-small")
    x = _x()
    out = m.run({"pixel_values": x})
    np.testing.assert_allclose(out["predicted_depth"][0], np.maximum(x[0].mean(0), 0), rtol=1e-5, atol=1e-6)
    da2 = [r[0] for r in f.runs if r[1] == "da2-small.onnx"]
    assert da2 == [DML, DML, CPU]  # GPU, GPU after freeing the other model, then CPU
    # not sticky: the model keeps its GPU session and the next call tries the GPU first
    assert m.provider == DML and not m.released and not ort._is_cpu_sticky(m.path)
    m.run({"pixel_values": x})
    da2 = [r[0] for r in f.runs if r[1] == "da2-small.onnx"]
    assert da2 == [DML, DML, CPU, DML, CPU]  # nothing left to free: no pointless GPU retry
    assert OrtModel(stub_env["da2-small"]).provider == DML
    fail.clear()  # memory available again -> back on the GPU
    m.run({"pixel_values": x})
    assert [r[0] for r in f.runs if r[1] == "da2-small.onnx"][-1] == DML


def test_gpu_oom_on_the_cpu_retry_raises_a_memory_hint(fake, stub_env, no_adapters):
    fake(run_hook=_oom_on({(DML, "da2-small.onnx"): 99, (CPU, "da2-small.onnx"): 99}))
    m = load_model("da2-small")
    with pytest.raises(InferenceError, match="GPU and on the CPU") as ei:
        m.run({"pixel_values": _x()})
    assert "memory" in (ei.value.hint or "").lower()
    assert m.provider == DML and not ort._is_cpu_sticky(m.path)


def test_gpu_oom_of_fp16_model_uses_the_fp32_file_on_the_cpu_once(fake, stub_models, no_adapters):
    f = fake(run_hook=_oom_on({(DML, "da2-fp16.onnx"): 99}))
    m = OrtModel(stub_models["da2-fp16"], fp16=True, fallback_path=lambda: stub_models["da2-small"])
    out = m.run({"pixel_values": _x()})
    assert out["predicted_depth"].dtype == np.float32
    assert f.runs[-1] == (CPU, "da2-small.onnx")
    # the model itself is unchanged: still the fp16 file on the GPU
    assert m.provider == DML and m.fp16 and m.path == stub_models["da2-fp16"]


def test_non_oom_failure_after_oom_retry_is_sticky(fake, stub_env, no_adapters):
    calls = {"n": 0}

    def hook(prov, name):
        if prov == DML and name == "da2-small.onnx":
            calls["n"] += 1
            raise RuntimeError(OOM if calls["n"] == 1 else "DXGI_ERROR_DEVICE_REMOVED")

    fake(run_hook=hook)
    load_model("moge2-vitl")
    m = load_model("da2-small")
    m.run({"pixel_values": _x()})
    assert m.provider == CPU and ort._is_cpu_sticky(m.path)


def test_gpu_oom_while_loading_releases_others_and_is_not_sticky(fake, stub_env, no_adapters):
    fail = {(DML, "birefnet-lite.onnx"): 1}
    f = fake(create_hook=_oom_on(fail))
    other = load_model("moge2-vitl")
    m = load_model("birefnet-lite")
    assert m.provider == DML and other.released
    assert f.creates() == [DML, DML, DML]  # moge, birefnet (out of memory), birefnet again
    # nothing left to free: this session falls back to the CPU, but only for the next run
    ort.unload_model()
    fail[(DML, "birefnet-lite.onnx")] = 1
    m2 = load_model("birefnet-lite")
    assert m2.provider == CPU and not ort._is_cpu_sticky(m2.path)
    feeds = {"input_image": np.zeros((1, 3, 1024, 1024), np.float32)}
    m2.run(feeds)
    assert f.runs[-1] == (CPU, "birefnet-lite.onnx")
    assert m2.released  # the CPU stand-in is dropped after its run
    m2.run(feeds)
    assert m2.provider == DML and f.runs[-1] == (DML, "birefnet-lite.onnx")  # GPU tried again


def test_gpu_oom_while_recreating_runs_one_call_on_the_cpu_then_retries_the_gpu(
    fake, stub_models, no_adapters
):
    fail: dict[tuple[str, str], int] = {}
    f = fake(create_hook=_oom_on(fail))
    m = OrtModel(stub_models["da2-fp16"], fp16=True, fallback_path=lambda: stub_models["da2-small"])
    assert m.provider == DML and m.release()
    fail[(DML, "da2-fp16.onnx")] = 1  # no other model to free: the GPU stays full
    out = m.run({"pixel_values": _x()})
    assert out["predicted_depth"].dtype == np.float32
    # the stand-in CPU session used the fp32 file; the model itself is unchanged
    assert f.runs[-1] == (CPU, "da2-small.onnx")
    assert m.fp16 and m.path == stub_models["da2-fp16"] and not ort._is_cpu_sticky(m.path)
    assert m.released
    m.run({"pixel_values": _x()})
    assert m.provider == DML and f.runs[-1] == (DML, "da2-fp16.onnx") and m.has_fp16_io


def test_release_and_close_free_the_native_session(stub_models):
    """Real onnxruntime: the model must not keep anything (e.g. a ``NodeArg``) that holds
    the native session, or releasing a model would not give its memory back."""
    m = OrtModel(stub_models["moge2-vitl"], "cpu")
    native = getattr(m._sess, "_sess", None)
    if native is None:  # pragma: no cover - onnxruntime internals changed
        pytest.skip("onnxruntime.InferenceSession has no native _sess")
    ref = weakref.ref(native)
    del native
    assert m.release()
    gc.collect()
    assert ref() is None
    # the metadata survives the release and the model reloads on use
    assert m.input_names() == ["image", "num_tokens"] and m.input_shapes()[1] == []
    assert m.input_types() == ["tensor(float)", "tensor(int64)"]
    assert m.run(_moge_feeds())["points"].shape == (1, 256, 256, 3)
    ref = weakref.ref(m._sess._sess)
    m.close()
    gc.collect()
    assert ref() is None


def test_release_keeps_model_usable_close_does_not(fake, stub_models, no_adapters):
    f = fake()
    m = OrtModel(stub_models["da2-small"], title="Depth Anything V2 Small")
    assert m.release() is True and m.release() is False
    assert m.released and not m.closed
    assert m.input_names() == ["pixel_values"] and m.output_names() == ["predicted_depth"]
    assert m.run({"pixel_values": _x()})["predicted_depth"].shape == (1, 28, 42)
    assert f.creates() == [DML, DML] and not m.released
    m.close()
    assert m.closed and not m.released and m.release() is False
    with pytest.raises(ModelUnavailable, match="Depth Anything V2 Small"):
        m.run({"pixel_values": _x()})


def test_release_does_not_wait_for_a_busy_model(fake, stub_models, no_adapters):
    gate, started = threading.Event(), threading.Event()

    def hook(prov, name):
        started.set()
        gate.wait(10)

    fake(run_hook=hook)
    m = OrtModel(stub_models["da2-small"])
    t = threading.Thread(target=lambda: m.run({"pixel_values": _x()}))
    t.start()
    try:
        assert started.wait(10)
        assert m.release(wait=False) is False and m.close(wait=False) is False
        assert not m.released and not m.closed
    finally:
        gate.set()
        t.join(10)
    assert m.release(wait=False) is True


def test_estimator_holding_a_released_model_keeps_working(fake, stub_env, no_adapters):
    """Depth / cut-out wrappers keep their OrtModel: a session released by another model's
    out-of-memory recovery must be re-created on use, not fail as "unloaded"."""
    from vrphoto.depth.moge import MoGeEstimator

    fake(run_hook=_oom_on({(DML, "da2-small.onnx"): 1}))
    est = MoGeEstimator("moge2-vitl")
    held = est._load(None, None)
    load_model("da2-small").run({"pixel_values": _x()})  # out of memory -> releases MoGe
    assert held.released
    img = np.full((120, 160, 3), 90, np.uint8)
    res = est.estimate(img)
    assert res is not None and est._model is held and not held.released


def test_big_model_releases_other_gpu_sessions_only_when_needed(fake, stub_env, monkeypatch):
    devs = [Device(0, "NVIDIA GeForce GTX 1080", 8192, "nvidia")]
    monkeypatch.setattr("vrphoto.models.gpu.list_devices", lambda **k: devs)
    f = fake()
    monkeypatch.setattr(ort, "BIG_MODEL_BYTES", 100)  # the stubs count as "big"
    cpu_model = load_model("birefnet-lite", device="cpu")
    other = load_model("moge2-vitl")
    assert other.provider == DML and f.creates()[-1] == DML
    monkeypatch.setattr(ort, "VRAM_FOOTPRINT", 1.0)  # fits next to the other model
    load_model("da2-small")
    assert not other.released
    ort.unload_model("da2-small")
    monkeypatch.setattr(ort, "VRAM_FOOTPRINT", 1e9)  # does not fit: free the GPU first
    m = load_model("da2-small")
    assert m.provider == DML and other.released
    assert not cpu_model.released  # CPU sessions do not use GPU memory


def test_big_model_with_unknown_vram_does_not_release(fake, stub_env, no_adapters, monkeypatch):
    fake()
    monkeypatch.setattr(ort, "BIG_MODEL_BYTES", 100)
    monkeypatch.setattr(ort, "VRAM_FOOTPRINT", 1e9)
    other = load_model("moge2-vitl")
    load_model("da2-small")
    assert not other.released  # no adapter list: only the out-of-memory retry applies


def test_two_models_out_of_memory_at_once_do_not_deadlock(fake, stub_env, no_adapters):
    barrier = threading.Barrier(2, timeout=10)
    first = {(DML, "da2-small.onnx"): 1, (DML, "moge2-vitl.onnx"): 1}

    def hook(prov, name):
        if first.get((prov, name)):
            first[(prov, name)] = 0
            barrier.wait()  # both models are inside run() holding their own lock
            raise RuntimeError(OOM)

    fake(run_hook=hook)
    a, b = load_model("da2-small"), load_model("moge2-vitl")
    errors: list[BaseException] = []

    def go(fn):
        try:
            fn()
        except BaseException as e:  # pragma: no cover
            errors.append(e)

    ts = [
        threading.Thread(target=go, args=(lambda: a.run({"pixel_values": _x()}),)),
        threading.Thread(target=go, args=(lambda: b.run(_moge_feeds()),)),
    ]
    for t in ts:
        t.start()
    for t in ts:
        t.join(20)
    assert not any(t.is_alive() for t in ts) and not errors
    assert not ort._is_cpu_sticky(a.path) and not ort._is_cpu_sticky(b.path)


def test_unload_without_waiting_closes_busy_models_after_their_run(fake, stub_env, no_adapters):
    gate, started = threading.Event(), threading.Event()

    def hook(prov, name):
        if name == "da2-small.onnx":
            started.set()
            gate.wait(10)

    fake(run_hook=hook)
    busy, idle = load_model("da2-small"), load_model("moge2-vitl")
    t = threading.Thread(target=lambda: busy.run({"pixel_values": _x()}))
    t.start()
    try:
        assert started.wait(10)
        t0 = time.monotonic()
        ort.unload_model(wait=False)
        assert time.monotonic() - t0 < 1.0  # never blocks the GUI thread
        assert ort.cached_models() == [] and idle.closed and not busy.closed
    finally:
        gate.set()
        t.join(10)
    deadline = time.monotonic() + 10
    while not busy.closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert busy.closed


def test_hidden_duplicate_adapter_index_maps_to_the_real_gpu(monkeypatch):
    """A device saved as "1" before the duplicate GTX 1080 entry was hidden still uses it."""
    devs = [Device(0, "NVIDIA GeForce GTX 1080", 8192, "nvidia", aliases=(1,))]
    monkeypatch.setattr("vrphoto.models.gpu.list_devices", lambda **k: devs)
    assert select_providers(1, available=[DML, CPU])[0] == (DML, {"device_id": "0"})
    assert select_providers("dml:3", available=[DML, CPU])[0] == (DML, {"device_id": "3"})  # unknown: as is
    assert select_providers(1, available=[CUDA, CPU])[0] == (CUDA, {"device_id": "0"})
