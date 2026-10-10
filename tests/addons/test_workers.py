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
