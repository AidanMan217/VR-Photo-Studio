"""capture.py: frame extraction, blur filter, COLMAP model io, tool commands and the full
pipeline driven by fake COLMAP / Brush executables (no real tools needed)."""

from __future__ import annotations

import json
import struct
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from _addon_helpers import write_manifest

from vrphoto.addons import capture, manager
from vrphoto.addons.capture import ColmapInfo
from vrphoto.core.errors import AddonError, AddonNotInstalled, Cancelled, InputError
from vrphoto.core.types import CancelToken, ProgressReporter, SplatCloud

FAKES = Path(__file__).parent / "fakes"


def _texture(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(h // 4 + 1, w // 4 + 1, 3), dtype=np.uint8)
    return np.kron(base, np.ones((4, 4, 1), dtype=np.uint8))[:h, :w].copy()


def _write_jpg(path: Path, img: np.ndarray) -> Path:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    path.write_bytes(buf.tobytes())  # unicode-safe on Windows (cv2.imwrite is not)
    return path


@pytest.fixture
def photos(tmp_path) -> list[Path]:
    d = tmp_path / "photos"
    d.mkdir()
    return [_write_jpg(d / f"IMG_{i:03d}.jpg", _texture(96, 128, i)) for i in range(8)]


@pytest.fixture
def video(tmp_path) -> Path:
    import imageio_ffmpeg

    path = tmp_path / "walk.mp4"
    w = imageio_ffmpeg.write_frames(str(path), (160, 120), fps=10, codec="libx264", pix_fmt_in="rgb24",
                                    output_params=["-crf", "18"], macro_block_size=8)
    w.send(None)
    for i in range(30):  # 3 seconds
        frame = _texture(120, 160, i // 3)
        if (i // 5) % 2 == 1:
            frame = cv2.GaussianBlur(frame, (0, 0), 4)
        w.send(np.ascontiguousarray(frame))
    w.close()
    return path


class FakeSplat:
    """Stand-in for vrphoto.geometry.splat recording the calls."""

    def __init__(self, transform: np.ndarray | None = None):
        self.calls: list[tuple[str, dict]] = []
        self.transform = transform if transform is not None else np.ones(3)

    def read_splats(self, path):
        xyz = capture.ply_xyz(path).astype(np.float32) * self.transform.astype(np.float32)
        n = len(xyz)
        self.calls.append(("read", {"path": str(path)}))
        return SplatCloud(xyz, np.full((n, 3), 0.01, np.float32), np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
                          np.full((n, 3), 0.5, np.float32), np.full(n, 0.9, np.float32))

    def crop_splats(self, sc, *, radius_pct=98.0, aabb=None):
        self.calls.append(("crop", {"radius_pct": radius_pct}))
        return sc

    def normalize_orientation(self, sc, up=None):
        self.calls.append(("orient", {"up": None if up is None else np.asarray(up).tolist()}))
        return sc

    def decimate_splats(self, sc, max_n=750_000):
        self.calls.append(("decimate", {"max_n": max_n}))
        return sc

    def write_splats(self, sc, path):
        path = Path(path)
        path.write_bytes(b"splat" * len(sc))
        self.calls.append(("write", {"path": str(path)}))
        return path

    def kinds(self):
        return [c[0] for c in self.calls]


@pytest.fixture
def fake_tools(monkeypatch, tmp_path):
    log = tmp_path / "tools.jsonl"
    monkeypatch.setenv("VRPHOTO_COLMAP", str(FAKES / "fake_colmap.py"))
    monkeypatch.setenv("VRPHOTO_BRUSH", str(FAKES / "fake_brush.py"))
    monkeypatch.setenv("FAKE_TOOL_LOG", str(log))
    for k in ("FAKE_REG", "FAKE_GLOBAL_REG", "FAKE_NO_GLOBAL", "FAKE_GLOBAL_FAIL", "FAKE_GPU_FAIL", "FAKE_LEGACY",
              "FAKE_BRUSH_SLEEP", "FAKE_BRUSH_FAIL", "FAKE_CAMS", "FAKE_BRUSH_PLY"):
        monkeypatch.delenv(k, raising=False)
    capture._info_cache.clear()
    splat = FakeSplat()
    monkeypatch.setattr(capture, "_splat_api", lambda: splat)
    monkeypatch.setattr(capture, "_has_nvidia_gpu", lambda: True)

    def calls():
        if not log.exists():
            return []
        return [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()]

    return SimpleNamespace(log=log, calls=calls, splat=splat)


# ------------------------------------------------------------------ frames / sharpness
def test_sharpness_orders_blur():
    img = _texture(120, 160, 1)
    assert capture.sharpness(img) > 5 * capture.sharpness(cv2.GaussianBlur(img, (0, 0), 3))


def test_select_sharp_groups_and_threshold(tmp_path):
    frames = [tmp_path / f"f{i}.jpg" for i in range(10)]
    scores = [100, 10, 5, 90, 80, 85, 1, 2, 95, 99]
    keep = capture.select_sharp(frames, group=2, scores=scores)
    # pairs -> best of each: 0, 3, 5(85>80), 7(2>1), 9 ; frame 7 is far below the median -> dropped
    assert keep == [frames[0], frames[3], frames[5], frames[9]]
    capped = capture.select_sharp(frames, group=1, scores=[50] * 10, max_images=4)
    assert len(capped) == 4 and capped[0] == frames[0] and capped[-1] == frames[-1]
    assert capture.select_sharp([], scores=[]) == []


def test_extract_frames_and_blur_filter(video, tmp_path):
    rec = []
    frames = capture.extract_frames(video, tmp_path / "frames", fps=4, max_long=80,
                                    progress=ProgressReporter(lambda f, m: rec.append(f)))
    assert 10 <= len(frames) <= 14
    img = cv2.imdecode(np.fromfile(str(frames[0]), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[:2] == (60, 80)  # long side limited, aspect kept
    scores = [capture.sharpness(f) for f in frames]
    keep = capture.select_sharp(frames, group=2, scores=scores)
    kept_scores = [scores[frames.index(k)] for k in keep]
    assert len(keep) < len(frames)
    assert min(kept_scores) >= 0.35 * float(np.median(kept_scores))
    assert rec and rec[-1] == pytest.approx(1.0)


def test_extract_frames_bad_video(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(InputError, match="Could not read frames"):
        capture.extract_frames(bad, tmp_path / "f")
    with pytest.raises(InputError, match="not found"):
        capture.extract_frames(tmp_path / "missing.mp4", tmp_path / "f")


# ------------------------------------------------------------------ COLMAP model io
def _images_bin(path: Path, quats: list[tuple[float, float, float, float]]) -> Path:
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(quats)))
        for i, q in enumerate(quats):
            fh.write(struct.pack("<I7dI", i + 1, *q, 0.0, 0.0, 0.0, 1))
            fh.write(f"img{i}.jpg".encode() + b"\0")
            fh.write(struct.pack("<Q", 2))
            fh.write(struct.pack("<ddqddq", 1, 2, -1, 3, 4, 7))
    return path


def test_read_images_bin_and_up_vector(tmp_path):
    p = _images_bin(tmp_path / "images.bin", [(1, 0, 0, 0)] * 3)
    imgs = capture.read_images_bin(p)
    assert [n for n, _, _ in imgs] == ["img0.jpg", "img1.jpg", "img2.jpg"]
    assert capture.count_registered(tmp_path) == 3
    np.testing.assert_allclose(capture.camera_up_vector(p), [0, -1, 0], atol=1e-9)
    # cameras rolled 180 degrees about z: their "up" is world +y
    p2 = _images_bin(tmp_path / "rolled.bin", [(0, 0, 0, 1)] * 2)
    np.testing.assert_allclose(capture.camera_up_vector(p2), [0, 1, 0], atol=1e-9)
    assert capture.count_registered(tmp_path / "nope") == 0
    p3 = _images_bin(tmp_path / "empty.bin", [])
    assert capture.camera_up_vector(p3) is None


def test_map_vector_to_cloud():
    raw = np.random.default_rng(0).normal(size=(50, 3))
    flipped = raw * [1, -1, -1] + [5, 0, 2]
    v = capture._map_vector_to_cloud(raw, flipped, np.array([0.0, -1.0, 0.0]))
    np.testing.assert_allclose(v, [0, 1, 0], atol=1e-6)


# ------------------------------------------------------------------ command construction
def test_colmap_args_new_and_legacy(tmp_path):
    new = ColmapInfo("4.2.1", True, True, True)
    old = ColmapInfo("3.9.1", False, False, False)
    db, im = tmp_path / "db", tmp_path / "im"
    a = capture.colmap_feature_args(db, im, use_gpu=True, info=new)
    assert a[0] == "feature_extractor" and "--ImageReader.camera_model" in a and "OPENCV" in a
    assert a[a.index("--ImageReader.single_camera_per_folder") + 1] == "1"
    assert a[a.index("--FeatureExtraction.use_gpu") + 1] == "1" and a[-2:] == ["--log_target", "stderr"]
    b = capture.colmap_feature_args(db, im, use_gpu=False, info=old)
    assert b[b.index("--SiftExtraction.use_gpu") + 1] == "0" and "--log_target" not in b
    m = capture.colmap_matcher_args(db, sequential=True, use_gpu=True, info=new, overlap=12)
    assert m[0] == "sequential_matcher" and m[m.index("--SequentialMatching.overlap") + 1] == "12"
    assert "--FeatureMatching.use_gpu" in m
    e = capture.colmap_matcher_args(db, sequential=False, use_gpu=False, info=old)
    assert e[0] == "exhaustive_matcher" and "--SiftMatching.use_gpu" in e
    assert capture.colmap_mapper_args(db, im, tmp_path, global_mapper=True, info=new)[0] == "global_mapper"
    assert capture.colmap_mapper_args(db, im, tmp_path, global_mapper=False, info=old)[0] == "mapper"
    u = capture.colmap_undistort_args(im, tmp_path / "0", tmp_path / "u", info=new)
    assert u[u.index("--output_type") + 1] == "COLMAP" and u[u.index("--max_image_size") + 1] == "1600"


def test_brush_args(tmp_path):
    a = capture.brush_args(tmp_path / "ds", tmp_path / "out", steps=5000, max_splats=2_000_000)
    assert a[0] == str(tmp_path / "ds")
    assert a[a.index("--total-steps") + 1] == "5000"
    assert a[a.index("--export-every") + 1] == "5000"  # export once, at the end
    assert a[a.index("--export-name") + 1] == "splat.ply"
    assert a[a.index("--max-splats") + 1] == "2000000"
    assert "--with-viewer" not in a  # a source path implies headless


def test_colmap_info_probe(fake_tools):
    cmd = capture.find_colmap()
    info = capture.colmap_info(cmd)
    assert info.version == "4.2.1" and info.has_global_mapper and info.new_option_names and info.has_log_target


def test_tools_missing(monkeypatch):
    monkeypatch.delenv("VRPHOTO_COLMAP", raising=False)
    monkeypatch.setattr(manager.shutil, "which", lambda name: None)
    with pytest.raises(AddonNotInstalled, match="colmap"):
        capture.find_colmap()


# ------------------------------------------------------------------ full pipeline (fake tools)
def test_capture_photos_end_to_end(fake_tools, photos, tmp_path):
    rec: list[tuple[float, str]] = []
    out = capture.capture_to_splat(photos, tmp_path / "out", steps=1000, max_splats=5000,
                                   progress=ProgressReporter(lambda f, m: rec.append((f, m))), name="statue")
    assert out == tmp_path / "out" / "statue.splat" and out.is_file()
    assert (tmp_path / "out" / "statue.ply").is_file()
    assert not (tmp_path / "out" / ".capture_work_statue").exists()  # work dir removed
    assert not list((tmp_path / "out").glob("*_capture.log"))  # log only kept on failure
    calls = fake_tools.calls()
    names = [c[1] for c in calls if c[0] == "colmap"]
    assert names[:2] == ["help", "feature_extractor"]  # capability probe
    steps = [n for n in names if n not in ("help",) and "-h" not in [c for c in calls if c[1] == n][0]]
    assert "exhaustive_matcher" in steps and "global_mapper" in steps and "image_undistorter" in steps
    fe = next(c for c in calls if c[:2] == ["colmap", "feature_extractor"] and "-h" not in c)
    assert fe[fe.index("--FeatureExtraction.use_gpu") + 1] == "1"
    brush = next(c for c in calls if c[0] == "brush" and "--total-steps" in c)
    assert brush[brush.index("--total-steps") + 1] == "1000"
    assert brush[brush.index("--max-splats") + 1] == "1000000"  # training cap >= 1M
    assert fake_tools.splat.kinds() == ["read", "crop", "orient", "decimate", "write", "write"]
    orient = dict(fake_tools.splat.calls)["orient"]
    np.testing.assert_allclose(orient["up"], [0, -1, 0], atol=1e-6)
    assert dict(fake_tools.splat.calls)["decimate"]["max_n"] == 5000
    fr = [f for f, _ in rec]
    assert fr[-1] == pytest.approx(1.0) and all(b >= a - 1e-9 for a, b in zip(fr, fr[1:]))
    assert any("Training the splat (step" in m for _, m in rec)


def test_capture_video_uses_sequential_matcher(fake_tools, video, tmp_path):
    out = capture.capture_to_splat([video], tmp_path / "out", video_fps=2, max_images=50, steps=400)
    assert out.name == "walk.splat"
    names = [c[1] for c in fake_tools.calls() if c[0] == "colmap"]
    assert "sequential_matcher" in names and "exhaustive_matcher" not in names


def test_capture_mixed_inputs_budget_and_camera_folders(fake_tools, video, photos, tmp_path):
    capture.capture_to_splat([video, *photos], tmp_path / "out", video_fps=2, max_images=8, steps=200,
                             keep_workdir=True, name="mixed")
    files = [f for f in (tmp_path / "out" / ".capture_work_mixed" / "images").rglob("*") if f.is_file()]
    by_dir: dict[str, int] = {}
    for f in files:
        by_dir[f.parent.name] = by_dir.get(f.parent.name, 0) + 1
    assert set(by_dir) == {"video0", "photos0"}
    assert by_dir["photos0"] == 4 and by_dir["video0"] <= 4  # max_images shared between sources


def test_capture_up_vector_follows_reader_frame(fake_tools, photos, tmp_path, monkeypatch):
    splat = FakeSplat(transform=np.array([1.0, -1.0, -1.0]))  # a reader that converts to y-up
    monkeypatch.setattr(capture, "_splat_api", lambda: splat)
    capture.capture_to_splat(photos, tmp_path / "out", steps=200)
    np.testing.assert_allclose(dict(splat.calls)["orient"]["up"], [0, 1, 0], atol=1e-6)


def test_capture_gpu_failure_falls_back_to_cpu(fake_tools, photos, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_GPU_FAIL", "1")
    capture.capture_to_splat(photos, tmp_path / "out", steps=200)
    fes = [c for c in fake_tools.calls() if c[:2] == ["colmap", "feature_extractor"] and "-h" not in c]
    assert [c[c.index("--FeatureExtraction.use_gpu") + 1] for c in fes] == ["1", "0"]
    m = next(c for c in fake_tools.calls() if c[:2] == ["colmap", "exhaustive_matcher"])
    assert m[m.index("--FeatureMatching.use_gpu") + 1] == "0"


def test_capture_legacy_colmap_incremental_mapper(fake_tools, photos, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_LEGACY", "1")
    capture.capture_to_splat(photos, tmp_path / "out", steps=200, use_gpu=False)
    calls = fake_tools.calls()
    names = [c[1] for c in calls if c[0] == "colmap"]
    assert "mapper" in names and "global_mapper" not in names
    fe = next(c for c in calls if c[:2] == ["colmap", "feature_extractor"] and "-h" not in c)
    assert "--SiftExtraction.use_gpu" in fe and "--log_target" not in fe


def test_capture_global_mapper_failure_falls_back(fake_tools, photos, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_GLOBAL_FAIL", "1")
    capture.capture_to_splat(photos, tmp_path / "out", steps=200)
    names = [c[1] for c in fake_tools.calls() if c[0] == "colmap"]
    assert names.index("global_mapper") < names.index("mapper")


def test_capture_too_few_registered(fake_tools, photos, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_REG", "1")
    with pytest.raises(AddonError, match="could not register enough images") as ei:
        capture.capture_to_splat(photos, tmp_path / "out", steps=200, name="cup")
    assert "overlap" in ei.value.hint
    assert (tmp_path / "out" / "cup_capture.log").is_file()  # diagnostics kept on failure


def test_capture_brush_failure(fake_tools, photos, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_BRUSH_FAIL", "1")
    with pytest.raises(AddonError, match="Brush training failed") as ei:
        capture.capture_to_splat(photos, tmp_path / "out", steps=200)
    assert "memory" in ei.value.hint


def test_capture_cancel_kills_brush(fake_tools, photos, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_BRUSH_SLEEP", "60")
    token = CancelToken()
    seen = threading.Event()

    def prog(f, m):
        if "Training the splat (step" in m and not seen.is_set():
            seen.set()
            threading.Timer(0.3, token.cancel).start()

    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        capture.capture_to_splat(photos, tmp_path / "out", steps=400, progress=ProgressReporter(prog), cancel=token)
    assert seen.is_set() and time.monotonic() - t0 < 15


def test_capture_input_validation(fake_tools, tmp_path):
    with pytest.raises(InputError, match="No photos"):
        capture.capture_to_splat([], tmp_path / "out")
    with pytest.raises(InputError, match="not found"):
        capture.capture_to_splat([tmp_path / "missing.jpg"], tmp_path / "out")
    txt = tmp_path / "notes.txt"
    txt.write_text("x")
    with pytest.raises(InputError, match="Unsupported"):
        capture.capture_to_splat([txt], tmp_path / "out")
    one = _write_jpg(tmp_path / "one.jpg", _texture(64, 64, 1))
    with pytest.raises(InputError, match="at least 3"):
        capture.capture_to_splat([one], tmp_path / "out")


def test_capture_folder_input_and_max_images(fake_tools, photos, tmp_path):
    capture.capture_to_splat([photos[0].parent], tmp_path / "out", max_images=5, steps=200, keep_workdir=True,
                             name="folder")
    work = tmp_path / "out" / ".capture_work_folder"
    files = [f for f in (work / "images").rglob("*") if f.is_file()]
    assert work.is_dir() and len(files) == 5
    assert {f.parent.name for f in files} == {"photos0"}  # same camera -> one COLMAP camera folder


def test_run_task_via_manager(fake_tools, photos, tmp_path, manifest_dir):
    """A binary add-on (entry = capture.run_task) through manager.install / run."""
    write_manifest(manifest_dir, "captest", """
        id = "captest"
        title = "Capture test"
        description = "fake tools"
        license = "MIT"
        kind = "binary"
        tasks = ["capture_to_splat"]
        entry = "vrphoto.addons.capture:run_task"

        [binaries.colmap]
        path_names = ["colmap-not-on-path"]
        check_args = ["help"]
        env_var = "VRPHOTO_COLMAP"

        [binaries.brush]
        path_names = ["brush-not-on-path"]
        check_args = ["--help"]
        env_var = "VRPHOTO_BRUSH"

        [[steps]]
        type = "check"
    """)
    manager.reload_manifests()
    manager.install("captest")
    assert manager.status("captest")["installed"]
    assert set(manager.check("captest")) == {"colmap", "brush"}
    out = manager.run("captest", "capture_to_splat", {"paths": [str(p) for p in photos]},
                      {"steps": 200, "max_splats": 1000, "name": "via"}, tmp_path / "out")
    assert out == {"splat": str(tmp_path / "out" / "via.splat"), "ply": str(tmp_path / "out" / "via.ply")}
    with pytest.raises(InputError, match="Unknown capture option"):
        manager.run("captest", "capture_to_splat", {"paths": []}, {"bogus": 1}, tmp_path / "out")


def test_capture_manifest_check_reports_missing_tool(tmp_path, manifest_dir, monkeypatch):
    monkeypatch.delenv("VRPHOTO_COLMAP", raising=False)
    write_manifest(manifest_dir, "captest2", """
        id = "captest2"
        title = "Capture test 2"
        description = "fake tools"
        license = "MIT"
        kind = "binary"
        tasks = ["capture_to_splat"]
        entry = "vrphoto.addons.capture:run_task"
        notes = "Install COLMAP with your package manager."

        [binaries.colmap]
        path_names = ["colmap-not-on-path-xyz"]

        [[steps]]
        type = "check"
    """)
    manager.reload_manifests()
    with pytest.raises(AddonError, match="was not found after installing") as ei:
        manager.install("captest2")
    assert "package manager" in ei.value.hint


def test_capture_with_real_splat_module(fake_tools, photos, tmp_path, monkeypatch):
    """End-to-end with the real vrphoto.geometry.splat (when it is available)."""
    splat = pytest.importorskip("vrphoto.geometry.splat")
    monkeypatch.setattr(capture, "_splat_api", lambda: splat)
    out = capture.capture_to_splat(photos, tmp_path / "out", steps=200, max_splats=1000, name="real")
    sc = splat.read_splats(out)
    assert len(sc) > 0 and out.suffix == ".splat"
    sc_ply = splat.read_splats(out.with_suffix(".ply"))
    assert len(sc_ply) == len(sc)
    y = np.asarray(sc.means[:, 1], dtype=np.float64)
    assert y.min() >= -0.05 * (y.max() - y.min() + 1e-6)  # floor at ~y = 0 after orientation


# ------------------------------------------------------------------ review regressions
def test_colmap_probe_reads_the_whole_long_help(fake_tools, monkeypatch):
    """COLMAP's help texts are longer than the runner's 60-line tail; the version header
    and the option names come first, so a tail-only probe misdetected COLMAP 3.x."""
    monkeypatch.setenv("FAKE_LEGACY", "1")
    info = capture.colmap_info(capture.find_colmap())
    assert info.version == "3.9.1"
    assert info.new_option_names is False and not info.has_log_target and not info.has_global_mapper


def test_colmap_probe_unknown_help_falls_back_to_version(monkeypatch):
    lines = {"help": ["COLMAP 3.11.1 -- Structure-from-Motion", "  mapper"], "feature_extractor": ["usage only"]}

    def fake_run(cmd, *, on_line=None, **kw):
        for ln in lines["help" if cmd[-1] == "help" else "feature_extractor"]:
            on_line(ln)
        return capture.ProcResult(0, [])

    monkeypatch.setattr(capture, "run_process", fake_run)
    capture._info_cache.clear()
    info = capture.colmap_info(["colmap-x"])
    assert info.version == "3.11.1" and info.new_option_names is False
    capture._info_cache.clear()


def test_capture_keeps_the_better_mapper_result(fake_tools, photos, tmp_path, monkeypatch):
    """Global mapper registers 3/8 (below the 50% bar), incremental only 2/8: the global
    model must be used, not overwritten by the worse fallback."""
    monkeypatch.setenv("FAKE_GLOBAL_REG", "3")
    monkeypatch.setenv("FAKE_REG", "2")
    capture.capture_to_splat(photos, tmp_path / "out", steps=200, name="best")
    calls = fake_tools.calls()
    names = [c[1] for c in calls if c[0] == "colmap"]
    assert names.index("global_mapper") < names.index("mapper")
    und = next(c for c in calls if c[:2] == ["colmap", "image_undistorter"])
    model = Path(und[und.index("--input_path") + 1])
    assert model.parent.name == "global"


def test_ascii_names_for_colmap():
    assert capture._ascii_name("Café Señor 東京") == "Caf_Se_or"
    assert capture._ascii_name("東京") == "img"
    assert capture._ascii_name("IMG_0001") == "IMG_0001"


def test_work_dir_is_ascii_on_windows(tmp_path, monkeypatch):
    out = tmp_path / "Jösé" / "out"
    out.mkdir(parents=True)
    monkeypatch.setattr(capture.sys, "platform", "win32")
    monkeypatch.setattr(capture, "_short_path", lambda p: p)  # 8.3 names disabled
    monkeypatch.setenv("PUBLIC", str(tmp_path / "Public"))
    w = capture._work_dir(out, "Statue_Ü")
    assert str(w).isascii() and w.is_relative_to(tmp_path / "Public")
    monkeypatch.setattr(capture, "_short_path", lambda p: tmp_path / "JOSE~1" / "out")
    assert capture._work_dir(out, "x") == tmp_path / "JOSE~1" / "out" / ".capture_work_x"
    monkeypatch.setattr(capture.sys, "platform", "linux")
    assert capture._work_dir(out, "x") == out / ".capture_work_x"


def test_capture_unicode_photo_names_reach_colmap_as_ascii(fake_tools, tmp_path):
    d = tmp_path / "fotos"
    d.mkdir()
    photos = [_write_jpg(d / f"Bild_ü_{i}.jpg", _texture(96, 128, i)) for i in range(4)]
    capture.capture_to_splat(photos, tmp_path / "out", steps=200, keep_workdir=True, name="ü")
    work = next((tmp_path / "out").glob(".capture_work_*"))
    names = [f.name for f in (work / "images").rglob("*.jpg")]
    assert len(names) == 4 and all(n.isascii() for n in names)


# ------------------------------------------------------------------ F2: metric scale
def _rot_to_quat(R: np.ndarray) -> list[float]:
    """Rotation matrix -> (w, x, y, z) (Shepperd), inverse of capture._quat_to_R."""
    t = np.trace(R)
    if t > 0:
        w = np.sqrt(1 + t) / 2
        q = [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        r = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        v = np.zeros(3)
        v[i] = r / 2
        v[j] = (R[j, i] + R[i, j]) / (2 * r)
        v[k] = (R[k, i] + R[i, k]) / (2 * r)
        q = [(R[k, j] - R[j, k]) / (2 * r), *v]
    return [float(x) for x in q]


def _yaw(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


UNITS_PER_M = 2.2  # COLMAP's arbitrary scale in the synthetic capture
PERSON_M = 1.7
# metric "gravity" frame (y up, floor y=0) -> COLMAP world (up = -y, yawed, shifted, scaled)
R_WG = np.diag([1.0, -1.0, -1.0]) @ _yaw(30)
T_WG = np.array([1.0, 0.5, -2.0])


def _to_world(p: np.ndarray) -> np.ndarray:
    return UNITS_PER_M * (np.asarray(p, float) @ R_WG.T) + T_WG


def _synthetic_capture(tmp_path: Path, monkeypatch) -> None:
    """A person (1.7 m) on a floor with a wall behind, seen by 24 phones orbiting at 2 m and
    1.2 / 1.7 m height (median 1.45 m), in COLMAP's arbitrary frame (2.2 units per metre)."""
    rng = np.random.default_rng(3)
    a, r = rng.uniform(0, 2 * np.pi, 6000), 3.0 * np.sqrt(rng.uniform(0, 1, 6000))
    floor = np.stack([r * np.cos(a), np.zeros(6000), r * np.sin(a)], 1)
    a2 = rng.uniform(0, 2 * np.pi, 4000)
    person = np.stack([0.2 * np.cos(a2), rng.uniform(0, PERSON_M, 4000), 0.2 * np.sin(a2)], 1)
    wall = np.stack([rng.uniform(-1.5, 1.5, 1500), rng.uniform(0, 2.5, 1500), np.full(1500, -2.6)], 1)
    pts = _to_world(np.vstack([floor, person, wall])).astype(np.float32)
    n = len(pts)
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]
    rec = np.zeros(n, dtype=[(p, "<f4") for p in props])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["opacity"], rec["rot_0"] = 3.0, 1.0
    for k in ("scale_0", "scale_1", "scale_2"):
        rec[k] = np.log(0.01 * UNITS_PER_M)
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
    header += "".join(f"property float {p}\n" for p in props) + "end_header\n"
    ply = tmp_path / "scene.ply"
    ply.write_bytes(header.encode("ascii") + rec.tobytes())
    poses = []
    for i in range(24):
        th = 2 * np.pi * i / 24
        h = 1.2 if i % 2 else 1.7
        c_g = np.array([2.0 * np.cos(th), h, 2.0 * np.sin(th)])
        f = np.array([-np.cos(th), 0.0, -np.sin(th)])
        d = np.array([0.0, -1.0, 0.0])
        R_cg = np.stack([np.cross(d, f), d, f])  # rows: camera x (right), y (down), z (forward)
        R_cw = R_cg @ R_WG.T
        c_w = _to_world(c_g)
        poses.append([*_rot_to_quat(R_cw), *(-R_cw @ c_w)])
    cams = tmp_path / "cams.json"
    cams.write_text(json.dumps(poses), encoding="utf-8")
    monkeypatch.setenv("FAKE_CAMS", str(cams))
    monkeypatch.setenv("FAKE_BRUSH_PLY", str(ply))


def test_rot_to_quat_roundtrip():
    R = _yaw(40) @ np.diag([1.0, -1.0, -1.0])
    np.testing.assert_allclose(capture._quat_to_R(np.array(_rot_to_quat(R))), R, atol=1e-12)


def test_camera_centers(tmp_path):
    p = tmp_path / "images.bin"
    with open(p, "wb") as fh:
        fh.write(struct.pack("<Q", 2))
        for i, (q, t) in enumerate([((1, 0, 0, 0), (1.0, 2.0, 3.0)), ((0, 0, 0, 1), (1.0, 0.0, 0.0))]):
            fh.write(struct.pack("<I7dI", i + 1, *q, *t, 1) + b"x\0" + struct.pack("<Q", 0))
    np.testing.assert_allclose(capture.camera_centers(p), [[-1, -2, -3], [1, 0, 0]], atol=1e-12)


def test_capture_gets_metric_scale_from_camera_height(fake_tools, photos, tmp_path, monkeypatch):
    splat = pytest.importorskip("vrphoto.geometry.splat")
    monkeypatch.setattr(capture, "_splat_api", lambda: splat)
    _synthetic_capture(tmp_path, monkeypatch)
    out = capture.capture_to_splat(photos, tmp_path / "out", steps=200, name="person")
    info = json.loads((tmp_path / "out" / "person.capture.json").read_text(encoding="utf-8"))
    assert info["method"] == "camera" and "subject_height_m" not in info
    assert info["scale"] == pytest.approx(1 / UNITS_PER_M, rel=0.03)
    assert info["camera_height_m"] == pytest.approx(1.45, rel=0.03)
    sc = splat.read_splats(out)
    m = np.asarray(sc.means, dtype=np.float64)
    near = np.linalg.norm(m[:, [0, 2]], axis=1) < 0.3
    assert m[near, 1].max() == pytest.approx(PERSON_M, rel=0.03)  # the person is 1.7 m tall again
    assert np.median(np.asarray(sc.scales)) == pytest.approx(0.01, rel=0.05)  # splat sizes scaled too
    assert info["height_m"] == pytest.approx(m[:, 1].max() - m[:, 1].min(), abs=1e-3)
    assert PERSON_M + 0.3 < info["height_m"] <= 2.5 + 1e-3  # bounding box incl. the wall behind
    ply = splat.read_splats(out.with_suffix(".ply"))
    np.testing.assert_allclose(np.asarray(ply.means), m, atol=1e-4)


def test_capture_subject_height_override(fake_tools, photos, tmp_path, monkeypatch):
    splat = pytest.importorskip("vrphoto.geometry.splat")
    monkeypatch.setattr(capture, "_splat_api", lambda: splat)
    _synthetic_capture(tmp_path, monkeypatch)
    res = capture.run_task("capture_to_splat", {"paths": [str(p) for p in photos]},
                           {"steps": 200, "name": "tall", "subject_height_m": 1.8}, tmp_path / "out")
    info = json.loads((tmp_path / "out" / "tall.capture.json").read_text(encoding="utf-8"))
    assert info["method"] == "subject"
    # height_m: the whole box; subject_height_m: the given height (the viewer's size label)
    assert set(info) == {"scale", "height_m", "method", "camera_height_m", "subject_height_m"}
    assert info["subject_height_m"] == pytest.approx(1.8)
    assert info["height_m"] > 1.8
    m = np.asarray(splat.read_splats(res["splat"]).means, dtype=np.float64)
    near = np.linalg.norm(m[:, [0, 2]], axis=1) <= 0.6
    lo, hi = np.percentile(m[near, 1], [2, 98])
    assert hi - lo == pytest.approx(1.8, rel=0.02)


def test_capture_scale_options_validated(fake_tools, photos, tmp_path):
    with pytest.raises(InputError, match="subject height"):
        capture.capture_to_splat(photos, tmp_path / "out", subject_height_m=-1)
    with pytest.raises(InputError, match="camera height"):
        capture.capture_to_splat(photos, tmp_path / "out", camera_height_m=0)


def test_capture_without_usable_cameras_still_writes_scale_info(fake_tools, photos, tmp_path):
    out = capture.capture_to_splat(photos, tmp_path / "out", steps=200, name="flat")
    info = json.loads((tmp_path / "out" / "flat.capture.json").read_text(encoding="utf-8"))
    assert set(info) >= {"scale", "height_m", "method"}
    assert info["method"] in ("camera", "radius", "none") and info["scale"] > 0
    assert out.is_file()


def test_metric_scale_methods():
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.normal(0, 0.1, 500), rng.uniform(0, 3.0, 500), rng.normal(0, 0.1, 500)])
    th = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    cams = np.column_stack([4 * np.cos(th), np.full(12, 2.9), 4 * np.sin(th)])
    s, how = capture.metric_scale(pts, cams)
    assert how == "camera" and s == pytest.approx(1.45 / 2.9)
    s, how = capture.metric_scale(pts, cams, camera_height_m=1.6)
    assert s == pytest.approx(1.6 / 2.9)
    # cameras at floor level: camera-height method unusable -> orbit radius
    flat = cams.copy()
    flat[:, 1] = 0.0
    s, how = capture.metric_scale(pts, flat)
    assert how == "radius" and s == pytest.approx(capture.ORBIT_RADIUS_M / 4.0)
    # no cameras at all
    assert capture.metric_scale(pts, None) == (1.0, "none")
    assert capture.metric_scale(pts, cams[:1]) == (1.0, "none")
    # subject override: robust 2..98 percentile height of the central points
    s, how = capture.metric_scale(pts, cams, subject_height_m=1.5)
    lo, hi = np.percentile(pts[:, 1], [2, 98])
    assert how == "subject" and s == pytest.approx(1.5 / (hi - lo), rel=1e-6)
    # unusable subject (all points at one height) -> falls back to the cameras
    s, how = capture.metric_scale(np.zeros((50, 3)), cams, subject_height_m=1.5)
    assert how == "camera"


def test_fit_affine():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(300, 3))
    R = _yaw(25)
    dst = src @ R.T * 1.0 + [0.5, -1.0, 2.0]
    M = capture._fit_affine(src, dst)
    assert M is not None
    np.testing.assert_allclose(capture._apply_affine(M, [[1.0, 2.0, 3.0]]), [R @ [1, 2, 3] + [0.5, -1, 2]], atol=1e-9)
    np.testing.assert_allclose(capture._fit_affine(src, src), np.vstack([np.eye(3), np.zeros((1, 3))]))
    assert capture._fit_affine(src, rng.normal(size=(300, 3))) is None  # rows do not correspond
    line = np.column_stack([np.arange(10.0), np.zeros(10), np.zeros(10)])
    assert capture._fit_affine(line, line * [1, -1, -1] + [0, 1, 0]) is None  # ambiguous
    assert capture._fit_affine(src[:3], dst[:3]) is None
