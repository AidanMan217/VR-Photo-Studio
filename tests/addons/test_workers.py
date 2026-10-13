"""Worker scripts: syntax, stdlib-only plumbing, graceful errors without their heavy
dependencies, and the pure-numpy helpers of the SDXL 360 outpainting worker."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from vrphoto.addons.manifest import load_manifest, manifests_dir, workers_dir
from vrphoto.addons.runner import run_worker
from vrphoto.core.errors import AddonError

WORKERS = sorted(p for p in workers_dir().glob("*_worker.py"))
HEAVY = [p for p in WORKERS if p.name != "dummy_worker.py"]
STDLIB = set(sys.stdlib_module_names)


def test_every_manifest_worker_exists():
    names = {p.name for p in WORKERS}
    for m in map(load_manifest, manifests_dir().glob("*.toml")):
        if m.worker:
            assert m.worker in names


@pytest.mark.parametrize("path", WORKERS + [workers_dir() / "_common.py"], ids=lambda p: p.name)
def test_worker_top_level_imports_are_light(path: Path):
    """Workers must not import vrphoto, and heavy libraries only inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    allowed = STDLIB | {"_common", "numpy", "__future__"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "").split(".")[0]]
        else:
            continue
        for mod in mods:
            assert mod in allowed, f"{path.name}: top-level import of {mod}"
            assert mod != "vrphoto"


def test_common_is_stdlib_only():
    tree = ast.parse((workers_dir() / "_common.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            assert node.module.split(".")[0] in STDLIB | {"__future__"} or node.module == "torch"
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] in STDLIB | {"torch"}


@pytest.mark.parametrize("path", HEAVY, ids=lambda p: p.name)
def test_heavy_worker_check_fails_cleanly_without_deps(path: Path, tmp_path):
    """Without torch etc. the worker reports a protocol error (not a crash) with a hint."""
    pytest.importorskip("numpy")
    try:
        importlib.import_module("torch")
        pytest.skip("torch is installed here")
    except ImportError:
        pass
    with pytest.raises(AddonError) as ei:
        run_worker(sys.executable, path, "", {}, {}, tmp_path, args=["--check", "--no-gpu"], send_request=False,
                   timeout=60)
    assert "torch" in str(ei.value) or "No module" in str(ei.value) or "missing" in str(ei.value)
    assert ei.value.hint


@pytest.mark.parametrize("path", HEAVY, ids=lambda p: p.name)
def test_heavy_worker_rejects_unknown_task(path: Path, tmp_path):
    with pytest.raises(AddonError, match="Unknown task"):
        run_worker(sys.executable, path, "bogus_task", {}, {}, tmp_path, timeout=60)


@pytest.mark.parametrize("path", HEAVY, ids=lambda p: p.name)
def test_heavy_worker_missing_input(path: Path, tmp_path):
    m = {p.worker: p for p in map(load_manifest, manifests_dir().glob("*.toml")) if p.worker}[path.name]
    with pytest.raises(AddonError, match="Missing input 'image'|not found"):
        run_worker(sys.executable, path, m.tasks[0], {}, {}, tmp_path, timeout=60)


# ------------------------------------------------------------------ sdxl360 helpers
@pytest.fixture
def sdx(monkeypatch):
    monkeypatch.syspath_prepend(str(workers_dir()))
    sys.modules.pop("sdxl360_worker", None)
    return importlib.import_module("sdxl360_worker")


def test_sdxl_geometry_helpers(sdx):
    assert sdx.is_full_360(4096, 2048) and sdx.is_full_360(4100, 2048)
    assert not sdx.is_full_360(2048, 2048)
    assert sdx.work_size(4096, 2048, 2048) == (2048, 1024)
    assert sdx.work_size(1000, 1000, 2048) == (1000, 1000)
    ww, wh = sdx.work_size(3000, 1499, 1024)
    assert ww % 8 == 0 and wh % 8 == 0
    a = np.arange(24).reshape(2, 12)
    assert np.array_equal(sdx.roll_half(sdx.roll_half(a), inverse=True), a)
    assert sdx.roll_half(a)[0, 6] == a[0, 0]  # left edge moved to the centre
    m = sdx.seam_band_mask(10, 200, 0.1)
    cols = np.where(m[0])[0]
    assert m.all(axis=0)[cols].all() and len(cols) == 20 and cols.min() == 90


def test_sdxl_seam_after_roll_covers_original_edges(sdx):
    w = 400
    img_edges = np.zeros((4, w), bool)
    img_edges[:, :5] = True
    img_edges[:, -5:] = True
    rolled = sdx.roll_half(img_edges)
    band = sdx.seam_band_mask(4, w, 0.1)
    assert (rolled & band).sum() == img_edges.sum()


def test_sdxl_feather_and_composite_keep_known_pixels(sdx):
    rng = np.random.default_rng(0)
    orig = rng.integers(0, 256, (64, 128, 3), dtype=np.uint8)
    gen = rng.integers(0, 256, (64, 128, 3), dtype=np.uint8)
    mask = np.zeros((64, 128), bool)
    mask[:, 40:100] = True
    w = sdx.inward_feather(mask, 6)
    assert w.dtype == np.float32 and w.min() >= 0 and w.max() <= 1
    assert (w[~mask] == 0).all()
    assert w[32, 70] == pytest.approx(1.0)
    out = sdx.composite(orig, gen, w)
    assert np.array_equal(out[~mask], orig[~mask])
    assert np.array_equal(out[32, 70], gen[32, 70])


def test_sdxl_prefill(sdx):
    img = np.zeros((32, 64, 3), np.uint8)
    img[:, :32] = (200, 100, 50)
    mask = np.zeros((32, 64), bool)
    mask[:, 32:] = True
    filled = sdx.prefill(img, mask)
    assert np.array_equal(filled[~mask], img[~mask])
    assert np.abs(filled[mask].astype(int) - [200, 100, 50]).max() <= 2  # smooth guess from known colour


def test_sdxl_dilate_erode(sdx):
    m = np.zeros((20, 20), bool)
    m[10, 10] = True
    d = sdx.dilate(m, 2)
    assert d.sum() == 25 and d[8:13, 8:13].all()
    assert np.array_equal(sdx.erode(d, 2), m)


def test_sdxl_load_inputs(sdx, tmp_path):
    from PIL import Image

    rgba = np.zeros((10, 20, 4), np.uint8)
    rgba[..., 3] = 255
    rgba[:, 15:, 3] = 0
    Image.fromarray(rgba).save(tmp_path / "a.png")
    rgb, fill = sdx.load_inputs(tmp_path / "a.png", None)
    assert rgb.shape == (10, 20, 3) and fill[:, 15:].all() and not fill[:, :15].any()
    Image.fromarray(np.full((10, 20), 255, np.uint8)).save(tmp_path / "m.png")
    _, fill2 = sdx.load_inputs(tmp_path / "a.png", tmp_path / "m.png")
    assert fill2.all()
    Image.fromarray(rgba[..., :3]).save(tmp_path / "rgb.png")
    with pytest.raises(sdx.WorkerError, match="nothing to fill"):
        sdx.load_inputs(tmp_path / "rgb.png", None)


def test_common_param_and_progress(monkeypatch):
    monkeypatch.syspath_prepend(str(workers_dir()))
    common = importlib.import_module("_common")
    assert common.param({"a": "5"}, "a", 1, int) == 5
    assert common.param({"a": 500}, "a", 1, int, hi=10) == 10
    assert common.param({}, "a", 2.5, float) == 2.5
    with pytest.raises(common.WorkerError, match="Invalid value"):
        common.param({"a": "x"}, "a", 1, int)
    sent = []
    monkeypatch.setattr(common, "emit", lambda t, **kw: sent.append((t, kw)))
    p = common.Progress()
    sub = p.sub(0.5, 1.0)
    sub(0.5, "half")
    sub(0.5, "half")  # duplicate suppressed
    assert sent == [("progress", {"frac": 0.75, "msg": "half"})]
    tq = common.tqdm_factory(p, "loop")
    assert list(tq(range(4))) == [0, 1, 2, 3]
    assert sent[-1][1]["frac"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "stdin, expect",
    [
        ("", "No request received"),
        ("{not json\n", "Malformed request"),
        ('{"type": "hello"}\n', "not a request"),
        ('{"type": "request", "protocol": 99, "task": "echo"}\n', "Protocol mismatch"),
    ],
)
def test_common_rejects_bad_requests(stdin, expect):
    from vrphoto.addons import protocol
    from vrphoto.addons.runner import run_process

    lines: list[str] = []
    dummy = workers_dir() / "dummy_worker.py"
    res = run_process([sys.executable, "-u", str(dummy)], stdin_data=stdin, on_line=lines.append)
    msgs = [m for m in map(protocol.decode, lines) if m]
    assert res.returncode == 1
    assert msgs and msgs[-1]["type"] == "error" and expect in msgs[-1]["message"]


# ------------------------------------------------------------------ trellis2: remote-code import check
@pytest.fixture
def t2(monkeypatch):
    """trellis2_worker with a stand-in for transformers.dynamic_module_utils.check_imports
    (same contract: raise ImportError for missing packages, return relative imports)."""
    import types

    monkeypatch.syspath_prepend(str(workers_dir()))
    sys.modules.pop("trellis2_worker", None)

    def check_imports(filename):
        tree = ast.parse(Path(filename).read_text(encoding="utf-8"))
        absolute, relative = [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                absolute += [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                (relative if node.level else absolute).append((node.module or "").split(".")[0])
        missing = []
        for mod in absolute:
            try:
                importlib.import_module(mod)
            except ImportError:
                missing.append(mod)
        if missing:
            raise ImportError(
                "This modeling file requires the following packages that were not found in your "
                f"environment: {', '.join(missing)}. Run `pip install {' '.join(missing)}`"
            )
        return relative

    tf = types.ModuleType("transformers")
    dmu = types.ModuleType("transformers.dynamic_module_utils")
    dmu.check_imports = check_imports
    tf.dynamic_module_utils = dmu
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "transformers.dynamic_module_utils", dmu)
    return importlib.import_module("trellis2_worker")


def _birefnet_folder(root: Path, config_import: str = "json") -> Path:
    import json

    d = root / "MODELS" / "RMBG-2.0"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"auto_map": {
        "AutoConfig": "BiRefNet_config.BiRefNetConfig",
        "AutoModelForImageSegmentation": "birefnet.BiRefNet",
    }}), encoding="utf-8")
    (d / "birefnet.py").write_text("import os\nfrom .BiRefNet_config import BiRefNetConfig\n", encoding="utf-8")
    (d / "BiRefNet_config.py").write_text(f"import {config_import}\nclass BiRefNetConfig: pass\n", encoding="utf-8")
    (d / "handler.py").write_text("import not_needed_for_loading_xyz\n", encoding="utf-8")  # not in auto_map
    return d


def test_trellis2_remote_code_check_ok(t2, tmp_path):
    assert t2.check_remote_code(_birefnet_folder(tmp_path)) is None
    assert t2.check_remote_code(tmp_path / "nothing_downloaded") is None


def test_trellis2_remote_code_check_reports_missing_package(t2, tmp_path):
    """A package missing for the (relatively imported) model code fails the install check."""
    msg = t2.check_remote_code(_birefnet_folder(tmp_path, config_import="einops_missing_for_test_xyz"))
    assert msg and "einops_missing_for_test_xyz" in msg and "BiRefNet_config.py" in msg


def test_trellis2_check_includes_remote_code(t2, tmp_path, monkeypatch):
    assert "einops" in t2._IMPORTS
    _birefnet_folder(tmp_path, config_import="einops_missing_for_test_xyz")
    monkeypatch.setenv("VRPHOTO_ADDON_DIR", str(tmp_path))
    monkeypatch.setattr(t2, "cuda_info", lambda require: {})
    monkeypatch.setattr(t2, "_IMPORTS", ())
    monkeypatch.setattr(t2, "_setup_path", lambda: tmp_path)
    with pytest.raises(t2.WorkerError, match="BiRefNet.*einops_missing_for_test_xyz"):
        t2.check(require_gpu=False)


# ------------------------------------------------------------------ hunyuan: texture painting
@pytest.fixture
def hy(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(workers_dir()))
    monkeypatch.setenv("HY3DGEN_MODELS", str(tmp_path / "models"))
    sys.modules.pop("hunyuan_worker", None)
    return importlib.import_module("hunyuan_worker")


class _Rec:
    def __init__(self):
        self.events = []

    def __call__(self, frac, msg=""):
        self.events.append((frac, msg))


def test_hunyuan_texture_flag(hy):
    assert hy.flag({}, "texture") is False
    assert hy.flag({"texture": True}, "texture") is True
    assert hy.flag({"texture": "false"}, "texture") is False  # bool("false") would be True
    assert hy.flag({"texture": "1"}, "texture") is True
    with pytest.raises(hy.WorkerError, match="Invalid value"):
        hy.flag({"texture": "maybe"}, "texture")


def test_hunyuan_paint_models_download_once_with_progress(hy, tmp_path):
    calls = []

    def fake_download(*, repo_id, allow_patterns, local_dir, max_workers):
        calls.append((repo_id, tuple(allow_patterns)))
        for sub in (hy.DELIGHT_SUBFOLDER, hy.PAINT_SUBFOLDER):
            (Path(local_dir) / sub).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / sub / "model.safetensors").write_bytes(b"x" * 10)

    rec = _Rec()
    local = hy.ensure_paint_models(rec, download=fake_download)
    assert local == tmp_path / "models" / "tencent" / "Hunyuan3D-2"  # where from_pretrained looks first
    assert calls == [("tencent/Hunyuan3D-2", ("hunyuan3d-delight-v2-0/*", "hunyuan3d-paint-v2-0-turbo/*"))]
    assert rec.events and rec.events[-1][0] == 1.0
    assert (local / hy.PAINT_COMPLETE).is_file()
    assert hy.ensure_paint_models(rec, download=fake_download) == local
    assert len(calls) == 1  # second textured run: no network


def test_hunyuan_paint_models_download_failure(hy):
    def offline(**kw):
        raise OSError("Temporary failure in name resolution")

    with pytest.raises(hy.WorkerError, match="could not be downloaded") as ei:
        hy.ensure_paint_models(_Rec(), download=offline)
    assert "internet" in ei.value.hint

    with pytest.raises(hy.WorkerError, match="incomplete"):
        hy.ensure_paint_models(_Rec(), download=lambda **kw: None)
    assert not (hy.paint_models_dir() / hy.PAINT_COMPLETE).exists()


def test_hunyuan_manifest_explains_texture_option():
    m = load_manifest(manifests_dir() / "hunyuan.toml")
    assert "Paint texture" in m.notes and "first textured run" in m.notes
    assert any(s.get("optional") and "custom_rasterizer" in " ".join(s.get("packages", [])) for s in m.steps)


# ------------------------------------------------------------------ _common: tqdm stand-in with tensors
class _TensorLike:
    """Behaves like a multi-element torch tensor: iterable and sized, but bool() raises."""

    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        return iter(self.values)

    def __bool__(self):
        raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")


class _CountLike:
    """A tensor-like 0-d count (e.g. total=torch.tensor(4)): int() works, bool() raises."""

    def __init__(self, n):
        self.n = n

    def __int__(self):
        return self.n

    def __index__(self):
        return self.n

    def __bool__(self):
        raise RuntimeError("Boolean value of Tensor with more than one value is ambiguous")


@pytest.fixture
def common(monkeypatch):
    monkeypatch.syspath_prepend(str(workers_dir()))
    mod = importlib.import_module("_common")
    sent: list[tuple[str, dict]] = []
    monkeypatch.setattr(mod, "emit", lambda t, **kw: sent.append((t, kw)))
    mod.sent = sent  # type: ignore[attr-defined]
    return mod


def _fracs(common):
    return [kw["frac"] for t, kw in common.sent if t == "progress"]


def test_tqdm_factory_iterates_tensor_like_timesteps(common):
    """Hunyuan3D passes a torch tensor of timesteps to tqdm: iterating must never call
    bool() on it ("Boolean value of Tensor ... is ambiguous" crashed every shape job)."""
    tq = common.tqdm_factory(common.Progress(), "Generating the shape")
    steps = _TensorLike([999, 749, 499, 249, 0])
    bar = tq(steps, desc="Diffusion Sampling:", disable=False)
    assert len(bar) == 5
    assert list(bar) == [999, 749, 499, 249, 0]
    assert _fracs(common)[-1] == pytest.approx(1.0)


def test_tqdm_factory_iterates_numpy_arrays(common):
    tq = common.tqdm_factory(common.Progress(), "loop")
    arr = np.array([3.0, 2.0, 1.0])
    assert [float(x) for x in tq(arr)] == [3.0, 2.0, 1.0]
    assert _fracs(common)[-1] == pytest.approx(1.0)
    assert list(tq(np.array([], dtype=np.float32))) == []


def test_tqdm_factory_counts_and_manual_updates(common):
    tq = common.tqdm_factory(common.Progress(), "loop")
    assert list(tq()) == []  # no iterable: empty, no crash
    with tq(total=_CountLike(4)) as bar:  # tensor-like total
        assert len(bar) == 4
        bar.update(2)
        bar.update(2)
    assert _fracs(common)[-1] == pytest.approx(1.0)
    assert list(tq((i for i in range(3)), total=np.int64(3))) == [0, 1, 2]
    assert len(tq(np.array(5.0))) == 0  # 0-d array: no len, no crash


def test_tqdm_factory_and_track_accept_infinite_total(common):
    """tqdm documents total=float("inf") for unbounded loops; int() of it raises OverflowError."""
    tq = common.tqdm_factory(common.Progress(), "loop")
    bar = tq(iter([1, 2]), total=float("inf"))
    assert bar.total is None and len(bar) == 0
    assert list(bar) == [1, 2]
    assert list(common.track(iter([1, 2]), common.Progress(), total=float("inf"))) == [1, 2]


def test_track_with_tensor_like_inputs(common):
    p = common.Progress()
    assert list(common.track(_TensorLike([1, 2]), p)) == [1, 2]
    assert list(common.track(iter([1, 2, 3]), p, total=_CountLike(3))) == [1, 2, 3]
    assert _fracs(common)[-1] == pytest.approx(1.0)


def test_trellis_stage_tqdm_with_tensor_like(common, monkeypatch):
    sys.modules.pop("trellis_worker", None)
    tw = importlib.import_module("trellis_worker")
    p = common.Progress()
    stages = tw._StageTqdm([(p.sub(0.0, 0.5), "shape"), (p.sub(0.5, 1.0), "details")])
    make = stages.factory()
    assert list(make(_TensorLike([1, 2]), desc="Sampling")) == [1, 2]
    assert list(make(np.arange(3))) == [0, 1, 2]
    assert _fracs(common)[-1] == pytest.approx(1.0)


# ------------------------------------------------------------------ _common: GPU architecture guard
WIN_TORCH_270_CU128 = ["sm_50", "sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]
WIN_TORCH_280_CU128 = ["sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]
LINUX_TORCH_280_CU128 = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


@pytest.mark.parametrize(
    "cap, archs, expect",
    [
        ((6, 1), WIN_TORCH_270_CU128, True),  # GTX 1080 with the TRELLIS torch build
        ((6, 1), WIN_TORCH_280_CU128, True),
        ((6, 1), LINUX_TORCH_280_CU128, False),  # Pascal dropped from the Linux cu128 builds
        ((6, 0), WIN_TORCH_280_CU128, False),  # sm_61 SASS does not run on 6.0
        ((5, 2), WIN_TORCH_280_CU128, False),
        ((6, 2), ["sm_61"], True),  # SASS: same major, higher minor
        ((7, 0), ["sm_75", "sm_80"], False),  # same major is not enough
        ((8, 9), ["sm_80", "sm_86", "sm_90"], True),
        ((9, 0), ["sm_80", "sm_86"], False),  # SASS never runs on another major
        ((12, 0), ["sm_80", "compute_90"], True),  # PTX: any newer GPU
        ((13, 0), ["sm_120"], False),
        ((13, 0), ["sm_120", "compute_120"], True),
        ((6, 1), ["compute_70"], False),  # PTX does not run on older GPUs
        ((6, 1), ["compute_60"], True),
        ((9, 0), ["sm_90a"], True),
        ((10, 0), ["sm_90a", "compute_90a"], False),  # arch-specific: exactly 9.0 only
        ((10, 3), ["sm_100f"], True),  # family-specific
        ((10, 0), ["sm_103f"], False),
        ((12, 0), ["SM_120 "], True),
    ],
)
def test_gpu_arch_supported(common, cap, archs, expect):
    assert common.gpu_arch_supported(cap, archs) is expect


def test_gpu_arch_supported_unknown(common):
    assert common.gpu_arch_supported((6, 1), []) is None
    assert common.gpu_arch_supported((9, 0), ["gfx90a", "gfx1100"]) is None  # ROCm: do not block


def _fake_torch(cap=(6, 1), archs=LINUX_TORCH_280_CU128, name="NVIDIA GeForce GTX 1080", cuda=True):
    import types

    torch = types.ModuleType("torch")
    torch.__version__ = "2.8.0+cu128"
    torch.version = types.SimpleNamespace(cuda="12.8")

    def props(i=0):
        return types.SimpleNamespace(name=name, total_memory=8 * 1024**3, major=cap[0], minor=cap[1])

    torch.cuda = types.SimpleNamespace(
        is_available=lambda: cuda,
        get_device_properties=props,
        get_device_capability=lambda device=0: cap,
        get_arch_list=lambda: list(archs),
        get_device_name=lambda device=0: name,
        empty_cache=lambda: None,
    )
    return torch


def test_cuda_info_rejects_unsupported_gpu(common, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    with pytest.raises(common.WorkerError) as ei:
        common.cuda_info(require=True)
    msg = str(ei.value)
    assert "NVIDIA GeForce GTX 1080" in msg and "sm_61" in msg
    assert "not supported by the installed PyTorch build" in msg and "sm_70" in msg
    assert "7.0 or newer" in ei.value.hint
    with pytest.raises(common.WorkerError, match="sm_61"):
        common.require_supported_gpu()
    info = common.cuda_info(require=False)  # --no-gpu checks report instead of failing
    assert info["gpu_supported"] is False and "sm_61" in info["gpu_problem"]


def test_cuda_info_accepts_supported_gpu(common, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(archs=WIN_TORCH_270_CU128))
    info = common.cuda_info(require=True)
    assert info["gpu"] == "NVIDIA GeForce GTX 1080" and info["capability"] == "6.1"
    assert "gpu_supported" not in info
    common.require_supported_gpu()
    assert common.gpu_arch_problem() is None


def test_gpu_arch_problem_newer_gpu_and_unknown(common, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cap=(13, 0), archs=["sm_90", "sm_120"], name="Future GPU"))
    msg, hint = common.gpu_arch_problem()
    assert "Future GPU, sm_130" in msg and "newer" in hint
    # between the built architectures (sm_61 SASS does not run on 7.x): neither "too old" nor "newer"
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cap=(7, 5), archs=["sm_61", "sm_80", "sm_90"],
                                                          name="NVIDIA GeForce RTX 2080"))
    msg, hint = common.gpu_arch_problem()
    assert "RTX 2080, sm_75" in msg and "too old" not in hint and "newer" not in hint and "generation" in hint
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(archs=[]))  # unknown build: do not block
    assert common.gpu_arch_problem() is None
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    assert common.gpu_arch_problem() is None


def _import_worker(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


@pytest.mark.parametrize(
    "name", ["trellis_worker", "trellis2_worker", "hunyuan_worker", "sdxl360_worker", "mvadapter_worker",
             "qwenedit_worker"]
)
def test_worker_check_rejects_unsupported_gpu(common, monkeypatch, tmp_path, name):
    """The install check fails with a readable message instead of a mid-job "no kernel image"."""
    monkeypatch.setenv("VRPHOTO_ADDON_DIR", str(tmp_path))
    monkeypatch.setenv("HY3DGEN_MODELS", str(tmp_path / "models"))
    worker = _import_worker(name)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    with pytest.raises(worker.WorkerError) as ei:
        worker.check(require_gpu=True)
    assert "This GPU (NVIDIA GeForce GTX 1080, sm_61) is not supported" in str(ei.value)
    assert ei.value.hint
    # a supported GPU passes the guard (the check then fails later on the missing packages)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(archs=WIN_TORCH_280_CU128))
    with pytest.raises(worker.WorkerError) as ei:
        worker.check(require_gpu=True)
    assert "not supported" not in str(ei.value)


def test_sharp_falls_back_to_cpu_on_unsupported_gpu(common, monkeypatch, tmp_path):
    """SHARP also runs on the CPU: an unsupported GPU is a warning, not an install failure."""
    import types

    ckpt = tmp_path / "sharp.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setenv("SHARP_CHECKPOINT", str(ckpt))
    for mod in ("sharp.models", "sharp.utils.gaussians", "sharp.utils.io", "timm", "plyfile"):
        monkeypatch.setitem(sys.modules, mod, types.ModuleType(mod))
    sharp = _import_worker("sharp_worker")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    info = sharp.check(require_gpu=True)
    assert info["gpu_supported"] is False and "sm_61" in info["warnings"][0] and "CPU" in info["warnings"][0]
    assert sharp._device("auto") == "cpu"
    assert any(t == "log" and "sm_61" in kw["msg"] for t, kw in common.sent)
    with pytest.raises(sharp.WorkerError, match="sm_61"):
        sharp._device("cuda")
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(archs=WIN_TORCH_280_CU128))
    assert sharp._device("auto") == "cuda"
    assert "warnings" not in sharp.check(require_gpu=True)


# ------------------------------------------------------------------ sharp_worker: list input (SHARP 360 M6)
class _FakeOOM(RuntimeError):
    pass


_FakeOOM.__name__ = "OutOfMemoryError"  # what torch.cuda.OutOfMemoryError looks like to is_oom


class _FakePredictor:
    def __init__(self, rec):
        self.rec = rec

    def load_state_dict(self, state):
        self.rec["loaded"] += 1

    def eval(self):
        return self

    def to(self, device):
        self.rec["to"].append(device)
        fail = self.rec["to_fail"]
        if fail and device == "cuda" and len(self.rec["to"]) in fail:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return self


def _sharp_env(common, monkeypatch, tmp_path, *, sizes=None, oom=(), to_fail=(), cuda=True):
    """The SHARP worker with fake torch / sharp modules.  ``sizes``: picture name -> (w, h);
    ``oom``: (picture name, device) pairs whose prediction runs out of memory; ``to_fail``:
    1-based indices of ``predictor.to`` calls (cuda only) that run out of memory."""
    import types

    ckpt = tmp_path / "sharp.pt"
    ckpt.write_bytes(b"x")
    monkeypatch.setenv("SHARP_CHECKPOINT", str(ckpt))
    sizes = sizes or {"a.png": (64, 48), "b.png": (40, 40), "c.png": (32, 24)}
    pics = {}
    for name in sizes:
        (tmp_path / name).write_bytes(b"png")
        pics[name] = tmp_path / name
    rec = {"loaded": 0, "loads": 0, "created": 0, "to": [], "to_fail": set(to_fail), "predict": [], "saved": []}
    torch = _fake_torch(archs=WIN_TORCH_280_CU128, cuda=cuda)

    def load(path, map_location=None, weights_only=None):
        rec["loads"] += 1
        return {}

    torch.load = load
    monkeypatch.setitem(sys.modules, "torch", torch)
    models = types.ModuleType("sharp.models")
    models.PredictorParams = lambda: None

    def create_predictor(_params):
        rec["created"] += 1
        return _FakePredictor(rec)

    models.create_predictor = create_predictor
    sio = types.ModuleType("sharp.utils.io")

    def load_rgb(path):
        w, h = sizes[Path(path).name]
        return np.zeros((h, w, 3), np.uint8), None, 100.0 + w  # EXIF focal length per picture

    sio.load_rgb = load_rgb
    utils = types.ModuleType("sharp.utils")
    utils.io = sio
    gauss = types.ModuleType("sharp.utils.gaussians")

    def save_ply(g, f_px, hw, path):
        rec["saved"].append((Path(path).name, g, round(float(f_px), 4), tuple(hw)))
        Path(path).write_bytes(b"ply")

    gauss.save_ply = save_ply
    for name, mod in (("sharp", types.ModuleType("sharp")), ("sharp.models", models), ("sharp.utils", utils),
                      ("sharp.utils.io", sio), ("sharp.utils.gaussians", gauss)):
        monkeypatch.setitem(sys.modules, name, mod)
    sharp = _import_worker("sharp_worker")

    def fake_predict(predictor, image, f_px, device):
        key = (image.shape[1], device)
        rec["predict"].append(key)
        hit = [n for n, d in oom if sizes[n][0] == image.shape[1] and d == device]
        if hit:
            raise _FakeOOM("CUDA out of memory. Tried to allocate 1.00 GiB\nmore details")
        return f"g{image.shape[1]}@{device}"

    monkeypatch.setattr(sharp, "_predict", fake_predict)
    freed = []
    monkeypatch.setattr(sharp, "free_cuda", lambda: freed.append(len(rec["predict"])))
    rec["freed"] = freed
    return sharp, rec, pics


def test_sharp_list_input_loads_the_model_once(common, monkeypatch, tmp_path):
    import math

    sharp, rec, pics = _sharp_env(common, monkeypatch, tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    images = [str(pics["a.png"]), str(pics["b.png"]), str(pics["c.png"])]
    outs = sharp.handle("image_to_splat", {"images": images}, {"hfov_deg": [50.0, None, 70.0], "name": "back"},
                        out_dir, common.Progress())
    assert rec["loads"] == 1 and rec["created"] == 1 and rec["loaded"] == 1 and rec["to"] == ["cuda"]
    assert outs["plys"] == [str(out_dir / f"back_{i}.ply") for i in range(3)]
    assert all(Path(p).is_file() for p in outs["plys"])
    f0 = 32.0 / math.tan(math.radians(25.0))
    f2 = 16.0 / math.tan(math.radians(35.0))
    assert outs["f_pxs"] == pytest.approx([f0, 140.0, f2])  # None -> the picture's own (EXIF) value
    assert outs["widths"] == [64, 40, 32] and outs["heights"] == [48, 40, 24]
    assert outs["devices"] == ["cuda"] * 3 and outs["warnings"] == []
    # the single-picture keys describe the first picture (SharpCam.from_outputs keeps working)
    assert outs["ply"] == outs["plys"][0] and outs["f_px"] == pytest.approx(f0)
    assert (outs["width"], outs["height"], outs["frame"]) == (64, 48, "opencv")
    assert [s[0] for s in rec["saved"]] == ["back_0.ply", "back_1.ply", "back_2.ply"]
    assert rec["saved"][1][3] == (40, 40)
    msgs = [kw["msg"] for t, kw in common.sent if t == "progress"]
    assert "Predicting the 3D scene (picture 2/3)" in msgs and "Saving the splat (picture 3/3)" in msgs
    fracs = _fracs(common)
    assert fracs == sorted(fracs) and fracs[-1] == 1.0


def test_sharp_list_input_one_hfov_for_all_and_image_first(common, monkeypatch, tmp_path):
    import math

    sharp, rec, pics = _sharp_env(common, monkeypatch, tmp_path)
    inputs = {"image": str(pics["c.png"]), "images": [str(pics["a.png"]), str(pics["b.png"])]}
    outs = sharp.handle("image_to_splat", inputs, {"hfov_deg": 60}, tmp_path, common.Progress())
    assert outs["widths"] == [32, 64, 40]  # 'image' comes first ...
    assert outs["f_pxs"] == pytest.approx([w / 2 / math.tan(math.radians(30.0)) for w in (32, 64, 40)])
    inputs = {"image": str(pics["b.png"]), "images": [str(pics["a.png"]), str(pics["b.png"])]}
    outs = sharp.handle("image_to_splat", inputs, {}, tmp_path, common.Progress())
    assert outs["widths"] == [64, 40]  # ... unless the list already holds it
    assert outs["f_pxs"] == pytest.approx([164.0, 140.0])
    outs = sharp.handle("image_to_splat", {"images": str(pics["a.png"])}, {}, tmp_path, common.Progress())
    assert outs["plys"] == [str(tmp_path / "sharp_0.ply")]  # one path instead of a list


def test_sharp_single_image_outputs_unchanged(common, monkeypatch, tmp_path):
    import math

    sharp, rec, pics = _sharp_env(common, monkeypatch, tmp_path)
    for inputs in ({"image": str(pics["a.png"])}, {"image": str(pics["a.png"]), "images": []}):
        outs = sharp.handle("image_to_splat", inputs, {"hfov_deg": 40.0}, tmp_path, common.Progress())
        assert set(outs) == {"ply", "frame", "f_px", "width", "height"}
        assert outs["ply"] == str(tmp_path / "sharp.ply") and Path(outs["ply"]).is_file()
        assert outs["f_px"] == pytest.approx(32.0 / math.tan(math.radians(20.0)))
    msgs = [kw["msg"] for t, kw in common.sent if t == "progress"]
    assert "Predicting the 3D scene" in msgs and "Saving the splat" in msgs and "Model loaded (cuda)" in msgs
    outs = sharp.handle("image_to_splat", {"image": str(pics["b.png"])}, {"hfov_deg": [90.0]}, tmp_path,
                        common.Progress())
    assert outs["f_px"] == pytest.approx(20.0)  # a one-value list is fine too


@pytest.mark.parametrize(
    "inputs, params, match",
    [
        ({}, {}, "Missing input 'image'"),
        ({"images": []}, {}, "Missing input 'image'"),
        ({"images": None, "image": ""}, {}, "Missing input 'image'"),
        ({"images": 3}, {}, "must be a list of picture files"),
        ({"images": ["A", None]}, {}, "item 2 is not a picture file"),
        ({"images": ["A", "missing.png"]}, {}, "Input file not found"),
        ({"images": ["A", "B"]}, {"hfov_deg": [50.0]}, "'hfov_deg' has 1 values for 2 pictures"),
        ({"image": "A"}, {"hfov_deg": [50.0, 60.0]}, "'hfov_deg' has 2 values for 1 picture;"),
        ({"images": ["A"]}, {"hfov_deg": ["wide"]}, "Invalid value for 'hfov_deg'"),
    ],
)
def test_sharp_list_input_validation(common, monkeypatch, tmp_path, inputs, params, match):
    sharp, rec, pics = _sharp_env(common, monkeypatch, tmp_path)
    names = {"A": str(pics["a.png"]), "B": str(pics["b.png"]), "missing.png": str(tmp_path / "missing.png")}

    def fix(v):
        if isinstance(v, list):
            return [names.get(x, x) if isinstance(x, str) else x for x in v]
        return names.get(v, v) if isinstance(v, str) else v

    with pytest.raises(sharp.WorkerError, match=match):
        sharp.handle("image_to_splat", {k: fix(v) for k, v in inputs.items()}, params, tmp_path, common.Progress())
    assert rec["loads"] == 0  # checked before the model is loaded


def test_sharp_hfov_list_helper(common, monkeypatch, tmp_path):
    sharp, _rec, _pics = _sharp_env(common, monkeypatch, tmp_path)
    assert sharp.hfov_list(None, 2) == [None, None]
    assert sharp.hfov_list(55, 3) == [55.0, 55.0, 55.0]
    assert sharp.hfov_list([5, None, 200, "65"], 4) == [10.0, None, 170.0, 65.0]  # clamped like before
    assert sharp.hfov_list((30.0,), 1) == [30.0]


def test_sharp_out_of_memory_runs_that_picture_on_the_cpu(common, monkeypatch, tmp_path):
    sharp, rec, pics = _sharp_env(common, monkeypatch, tmp_path, oom=[("b.png", "cuda")])
    images = [str(pics["a.png"]), str(pics["b.png"]), str(pics["c.png"])]
    outs = sharp.handle("image_to_splat", {"images": images}, {}, tmp_path, common.Progress())
    assert outs["devices"] == ["cuda", "cpu", "cuda"]
    assert rec["predict"] == [(64, "cuda"), (40, "cuda"), (40, "cpu"), (32, "cuda")]
    assert rec["to"] == ["cuda", "cpu", "cuda"]  # off the GPU for picture 2, back for picture 3
    assert 2 in rec["freed"]  # the cache was freed after the failed attempt, before the CPU run
    assert [s[1] for s in rec["saved"]] == ["g64@cuda", "g40@cpu", "g32@cuda"]
    assert len(outs["warnings"]) == 1 and "picture 2 (b.png)" in outs["warnings"][0]
    assert "CPU" in outs["warnings"][0]
    assert any(t == "log" and kw["level"] == "warning" and "out of memory" in kw["msg"] for t, kw in common.sent)


def test_sharp_out_of_memory_single_image_falls_back_with_a_warning(common, monkeypatch, tmp_path):
    sharp, rec, pics = _sharp_env(common, monkeypatch, tmp_path, oom=[("a.png", "cuda")])
    outs = sharp.handle("image_to_splat", {"image": str(pics["a.png"])}, {}, tmp_path, common.Progress())
    assert set(outs) == {"ply", "frame", "f_px", "width", "height", "warnings"}
    assert "SHARP made it on the CPU" in outs["warnings"][0]
    assert rec["saved"][0][1] == "g64@cpu"


def test_sharp_stays_on_the_cpu_when_the_model_cannot_go_back(common, monkeypatch, tmp_path):
    # to() calls: 1 = initial cuda, 2 = cpu (fallback), 3 = back to cuda -> out of memory
    sharp, rec, pics = _sharp_env(common, monkeypatch, tmp_path, oom=[("a.png", "cuda")], to_fail={3})
    images = [str(pics["a.png"]), str(pics["b.png"]), str(pics["c.png"])]
    outs = sharp.handle("image_to_splat", {"images": images}, {}, tmp_path, common.Progress())
    assert outs["devices"] == ["cpu", "cpu", "cpu"]
    assert rec["to"] == ["cuda", "cpu", "cuda", "cpu"]
    assert rec["predict"] == [(64, "cuda"), (64, "cpu"), (40, "cpu"), (32, "cpu")]


def test_sharp_model_too_big_for_the_gpu_uses_the_cpu(common, monkeypatch, tmp_path):
    sharp, rec, pics = _sharp_env(common, monkeypatch, tmp_path, to_fail={1})
    outs = sharp.handle("image_to_splat", {"images": [str(pics["a.png"])]}, {}, tmp_path, common.Progress())
    assert outs["devices"] == ["cpu"] and rec["to"] == ["cuda", "cpu"]
    assert "did not fit on the graphics card" in outs["warnings"][0]
    assert any(t == "progress" and kw["msg"] == "Model loaded (cpu)" for t, kw in common.sent)


def test_sharp_other_errors_and_cpu_out_of_memory_propagate(common, monkeypatch, tmp_path):
    sharp, rec, pics = _sharp_env(common, monkeypatch, tmp_path, oom=[("a.png", "cpu")])
    with pytest.raises(_FakeOOM):  # already on the CPU: nothing left to fall back to
        sharp.handle("image_to_splat", {"image": str(pics["a.png"])}, {"device": "cpu"}, tmp_path,
                     common.Progress())
    assert rec["predict"] == [(64, "cpu")]

    def broken(*a, **k):
        raise ValueError("bad tensor shape")

    monkeypatch.setattr(sharp, "_predict", broken)
    with pytest.raises(ValueError, match="bad tensor shape"):
        sharp.handle("image_to_splat", {"images": [str(pics["b.png"])]}, {}, tmp_path, common.Progress())


def test_sharp_is_oom(common, monkeypatch, tmp_path):
    sharp, _rec, _pics = _sharp_env(common, monkeypatch, tmp_path)
    assert sharp.is_oom(_FakeOOM("x"))
    assert sharp.is_oom(RuntimeError("CUDA error: out of memory"))
    assert sharp.is_oom(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"))
    assert not sharp.is_oom(ValueError("shape mismatch"))


def test_sharp_worker_list_validation_without_torch(tmp_path):
    """The real worker process (no torch in CI) checks the list input before importing torch."""
    path = workers_dir() / "sharp_worker.py"
    pic = tmp_path / "a.png"
    pic.write_bytes(b"png")
    with pytest.raises(AddonError, match="Input file not found"):
        run_worker(sys.executable, path, "image_to_splat", {"images": [str(pic), str(tmp_path / "b.png")]}, {},
                   tmp_path, timeout=60)
    with pytest.raises(AddonError, match="'hfov_deg' has 3 values for 2 pictures"):
        run_worker(sys.executable, path, "image_to_splat", {"images": [str(pic), str(pic)]},
                   {"hfov_deg": [40, 50, 60]}, tmp_path, timeout=60)
