"""Add-on manager: find_uv, install / status / run / uninstall lifecycle with a dummy add-on."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest
from _addon_helpers import dummy_manifest, write_manifest

from vrphoto.addons import installer, manager
from vrphoto.addons.env import venv_python
from vrphoto.core.config import get_settings
from vrphoto.core.errors import AddonError, AddonNotInstalled, Cancelled
from vrphoto.core.types import CancelToken, ProgressReporter


class Recorder:
    def __init__(self):
        self.events: list[tuple[float, str]] = []

    def __call__(self, frac, msg):
        self.events.append((frac, msg))


# ------------------------------------------------------------------ find_uv
def test_find_uv_env_override(monkeypatch, tmp_path):
    fake = tmp_path / "uv"
    fake.write_text("")
    monkeypatch.setenv(manager.UV_ENV, str(fake))
    assert manager.find_uv() == str(fake)


def test_find_uv_prefers_bundled_binary(monkeypatch, tmp_path):
    monkeypatch.delenv(manager.UV_ENV, raising=False)
    (tmp_path / "bin").mkdir()
    exe = tmp_path / "bin" / ("uv.exe" if sys.platform == "win32" else "uv")
    exe.write_text("")
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert manager.find_uv() == str(exe)


def test_find_uv_pip_package_then_path(monkeypatch, tmp_path):
    monkeypatch.delenv(manager.UV_ENV, raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    import uv as uv_pkg

    monkeypatch.setattr(uv_pkg, "find_uv_bin", lambda: str(tmp_path / "missing-uv"))
    monkeypatch.setattr(manager.shutil, "which", lambda name: "/opt/bin/uv" if name == "uv" else None)
    assert manager.find_uv() == "/opt/bin/uv"
    monkeypatch.setattr(manager.shutil, "which", lambda name: None)
    with pytest.raises(AddonError, match="uv") as ei:
        manager.find_uv()
    assert ei.value.hint


def test_find_uv_real():
    assert Path(manager.find_uv()).exists()


# ------------------------------------------------------------------ status / errors before install
def test_status_not_installed():
    st = manager.status("sdxl360")
    assert st["installed"] is False and st["version"] is None
    assert st["path"] == str(get_settings().addons_dir / "sdxl360")
    assert st["size_mb"] == 0 and st["partial"] is False and st["busy"] is False


def test_run_requires_install(tmp_path):
    with pytest.raises(AddonNotInstalled) as ei:
        manager.run("sdxl360", "outpaint_360", {}, {}, tmp_path)
    assert "Add-ons window (toolbar > Add-ons)" in ei.value.hint and "Settings" not in ei.value.hint


def test_run_rejects_unknown_task(tmp_path):
    with pytest.raises(AddonError, match="cannot do"):
        manager.run("sdxl360", "image_to_3d", {}, {}, tmp_path)


def test_install_unsupported_platform(manifest_dir):
    other = "darwin" if sys.platform != "darwin" else "win32"
    write_manifest(manifest_dir, "dummy", dummy_manifest(header_extra=f'platforms = ["{other}"]\n'))
    manager.reload_manifests()
    with pytest.raises(AddonError, match="not available on this operating system"):
        manager.install("dummy")


def test_install_gated_needs_token(manifest_dir, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    write_manifest(manifest_dir, "dummy", dummy_manifest(header_extra="hf_gated = true\n"))
    manager.reload_manifests()
    with pytest.raises(AddonError, match="access token"):
        manager.install("dummy")


def test_install_disk_check(manifest_dir, monkeypatch):
    write_manifest(manifest_dir, "dummy", dummy_manifest(header_extra="disk_gb = 1000000\n"))
    manager.reload_manifests()
    with pytest.raises(AddonError, match="Not enough disk space"):
        manager.install("dummy")


# ------------------------------------------------------------------ lifecycle (offline)
def test_full_lifecycle_offline(installed_dummy, tmp_path):
    d = manager.addon_dir("dummy")
    st = manager.status("dummy")
    assert st["installed"] and st["version"] == "1.0" and st["size_mb"] >= 0 and not st["update_available"]
    state = json.loads((d / "state.json").read_text(encoding="utf-8"))
    assert state["installed"] and len(state["manifest_sha256"]) == 64 and state["date"].endswith("Z")
    assert state["size_bytes"] > 0
    assert len(state["steps_done"]) == 3
    assert venv_python(d / "venv").is_file()
    assert (d / "prefetch.txt").read_text(encoding="utf-8") == "ok"  # worker step ran with VRPHOTO_ADDON_DIR
    log = (d / "install.log").read_text(encoding="utf-8")
    assert "[step 1/3]" in log and "check result" in log
    assert manager.is_installed("dummy")

    rec = Recorder()
    out = manager.run("dummy", "echo", {"image": tmp_path / "in.png"}, {"steps": 3}, tmp_path / "out",
                      progress=ProgressReporter(rec))
    assert out["echo"]["inputs"]["image"] == str(tmp_path / "in.png")
    assert [round(f, 3) for f, _ in rec.events] == [0.333, 0.667, 1.0]
    assert (d / "run.log").is_file()

    info = manager.check("dummy", check_gpu=False)
    assert info["gpu_required"] is False

    manager.uninstall("dummy")
    assert not d.exists()
    assert manager.status("dummy")["installed"] is False
    manager.uninstall("dummy")  # idempotent


def test_worker_env_inside_venv(installed_dummy, tmp_path, monkeypatch):
    """The worker runs with the venv interpreter and a clean environment."""
    monkeypatch.setenv("PYTHONPATH", "/should/not/leak")
    info = manager.check("dummy", check_gpu=True)
    d = manager.addon_dir("dummy")
    assert Path(info["executable"]).parent == venv_python(d / "venv").parent


def test_run_failure_and_hint(installed_dummy, tmp_path):
    with pytest.raises(AddonError) as ei:
        manager.run("dummy", "echo", {}, {"oom": True, "steps": 0}, tmp_path)
    assert "GPU" in ei.value.hint
    with pytest.raises(AddonError, match="kaputt"):
        manager.run("dummy", "echo", {}, {"fail": "kaputt", "steps": 0}, tmp_path)


def test_run_cancel_kills_worker(installed_dummy, tmp_path):
    token = CancelToken()
    threading.Timer(0.7, token.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        manager.run("dummy", "sleep", {}, {"seconds": 60}, tmp_path, cancel=token)
    assert time.monotonic() - t0 < 0.7 + 2.0
    assert manager.status("dummy")["busy"] is False


def test_uninstall_refused_while_running(installed_dummy, tmp_path):
    token = CancelToken()
    errors: list[BaseException] = []

    def bg():
        try:
            manager.run("dummy", "sleep", {}, {"seconds": 30}, tmp_path, cancel=token)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=bg)
    t.start()
    deadline = time.monotonic() + 5
    while not manager.status("dummy")["busy"] and time.monotonic() < deadline:
        time.sleep(0.02)
    try:
        with pytest.raises(AddonError, match="while it is running"):
            manager.uninstall("dummy")
    finally:
        token.cancel()
        t.join(5)
    assert errors and isinstance(errors[0], Cancelled)


def test_reinstall_skips_done_steps(installed_dummy):
    d = manager.addon_dir("dummy")
    mtime = venv_python(d / "venv").stat().st_mtime
    manager.install("dummy")
    log = (d / "install.log").read_text(encoding="utf-8")
    assert "All steps already done" in log
    assert venv_python(d / "venv").stat().st_mtime == mtime


def test_manifest_change_resumes_from_changed_step(installed_dummy, manifest_dir):
    d = manager.addon_dir("dummy")
    (d / "prefetch.txt").unlink()
    write_manifest(manifest_dir, "dummy", dummy_manifest(version="1.1").replace(
        'args = ["--prefetch"]', 'args = ["--prefetch"]\ntitle = "Prefetch again"'))
    manager.reload_manifests()
    assert manager.status("dummy")["update_available"]
    manager.install("dummy")
    assert (d / "prefetch.txt").exists()
    log = (d / "install.log").read_text(encoding="utf-8")
    assert "Resuming at step 2 of 3" in log
    st = manager.status("dummy")
    assert st["version"] == "1.1" and not st["update_available"]


def test_failed_install_records_error_and_resumes(manifest_dir, addon_python, uv_exe, monkeypatch):
    write_manifest(manifest_dir, "dummy", dummy_manifest())
    manager.reload_manifests()
    monkeypatch.setenv("DUMMY_CHECK_FAIL", "1")
    with pytest.raises(AddonError, match="dummy check failed"):
        manager.install("dummy")
    st = manager.status("dummy")
    assert not st["installed"] and st["partial"] and "dummy check failed" in st["error"]
    monkeypatch.delenv("DUMMY_CHECK_FAIL")
    manager.install("dummy")
    assert manager.status("dummy")["installed"]
    assert "Resuming at step 3 of 3" in (manager.addon_dir("dummy") / "install.log").read_text(encoding="utf-8")


def test_install_cancel(manifest_dir, addon_python, uv_exe):
    write_manifest(manifest_dir, "dummy", dummy_manifest())
    manager.reload_manifests()
    token = CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        manager.install("dummy", cancel=token)
    assert manager.status("dummy")["error"] == "Cancelled"


def test_install_progress_monotonic(manifest_dir, addon_python, uv_exe):
    write_manifest(manifest_dir, "dummy", dummy_manifest())
    manager.reload_manifests()
    rec = Recorder()
    manager.install("dummy", progress=ProgressReporter(rec))
    fracs = [f for f, _ in rec.events]
    assert fracs[-1] == pytest.approx(1.0)
    assert all(b >= a - 1e-6 for a, b in zip(fracs, fracs[1:]))
    assert any("environment" in m.lower() for _, m in rec.events)


def test_force_reinstall(installed_dummy):
    d = manager.addon_dir("dummy")
    (d / "junk.bin").write_bytes(b"x")
    manager.install("dummy", force=True)
    assert not (d / "junk.bin").exists() and manager.status("dummy")["installed"]


def test_concurrent_install_is_refused(manifest_dir, addon_python, uv_exe, monkeypatch):
    write_manifest(manifest_dir, "dummy", dummy_manifest())
    manager.reload_manifests()
    gate = threading.Event()
    release = threading.Event()
    real = installer.STEP_RUNNERS["venv"]

    def slow_venv(ctx, step, progress):
        gate.set()
        release.wait(10)
        real(ctx, step, progress)

    monkeypatch.setitem(installer.STEP_RUNNERS, "venv", slow_venv)
    t = threading.Thread(target=lambda: manager.install("dummy"))
    t.start()
    assert gate.wait(10)
    try:
        assert manager.status("dummy")["busy"]
        with pytest.raises(AddonError, match="busy"):
            manager.install("dummy")
        with pytest.raises(AddonError, match="busy"):
            manager.uninstall("dummy")
    finally:
        release.set()
        t.join(30)
    assert manager.status("dummy")["installed"]


def test_venv_deleted_means_not_installed(installed_dummy):
    d = manager.addon_dir("dummy")
    import shutil

    shutil.rmtree(d / "venv")
    assert manager.status("dummy")["installed"] is False
    with pytest.raises(AddonNotInstalled):
        manager.run("dummy", "echo", {}, {}, d / "out")
    manager.install("dummy")  # starts over from the venv step
    assert manager.status("dummy")["installed"]


def test_platform_specific_step_skipped(manifest_dir, addon_python, uv_exe):
    other = "darwin" if sys.platform != "darwin" else "win32"
    extra = (f'\n[[steps]]\ntype = "pip"\nplatforms = ["{other}"]\n'
             'packages = ["this-package-does-not-exist-anywhere==9.9"]\n')
    write_manifest(manifest_dir, "dummy", dummy_manifest(extra_steps=extra))
    manager.reload_manifests()
    manager.install("dummy")
    assert "skipped on" in (manager.addon_dir("dummy") / "install.log").read_text(encoding="utf-8")


@pytest.mark.network
def test_optional_step_failure_is_a_warning(manifest_dir, addon_python, uv_exe):
    extra = ('\n[[steps]]\ntype = "pip"\noptional = true\ntitle = "Optional thing"\n'
             'packages = ["this-package-does-not-exist-anywhere-vrphoto==9.9"]\n')
    write_manifest(manifest_dir, "dummy", dummy_manifest(extra_steps=extra))
    manager.reload_manifests()
    manager.install("dummy")
    st = manager.status("dummy")
    assert st["installed"] and any("Optional thing" in w for w in st["warnings"])


@pytest.mark.network
def test_pip_step_installs_from_pypi(manifest_dir, addon_python, uv_exe):
    write_manifest(manifest_dir, "dummy", dummy_manifest(pip=True))
    manager.reload_manifests()
    manager.install("dummy")
    d = manager.addon_dir("dummy")
    py = venv_python(d / "venv")
    import subprocess

    out = subprocess.run([str(py), "-c", "import six; print(six.__version__)"], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "1.17.0"
    cache = get_settings().addons_dir / ".uv-cache"
    assert cache.is_dir() or os.environ.get("UV_CACHE_DIR")
    log = (d / "install.log").read_text(encoding="utf-8")
    assert "uv" in log and "six" in log


def test_pip_failure_message(manifest_dir, addon_python, uv_exe):
    extra = '\n[[steps]]\ntype = "pip"\npackages = ["{addon_dir}/no-such-wheel-1.0-py3-none-any.whl"]\n'
    write_manifest(manifest_dir, "dummy", dummy_manifest(extra_steps=extra))
    manager.reload_manifests()
    with pytest.raises(AddonError) as ei:
        manager.install("dummy")
    assert "failed" in str(ei.value) and ei.value.hint


def test_github_archive_and_archive_steps(manifest_dir, addon_python, uv_exe, monkeypatch, tmp_path):
    """Download steps use fetch.download/extract (network mocked with local archives)."""
    import io
    import zipfile

    from vrphoto.addons import fetch

    def make_zip(entries):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for n, data in entries.items():
                zf.writestr(n, data)
        return buf.getvalue()

    blobs = {
        "https://github.com/acme/pkg/archive/" + "a" * 40 + ".zip": make_zip(
            {"pkg-aaaa/setup.py": "from setuptools import setup\nsetup(name='acmepkg', version='0.1', py_modules=['acmepkg'])\n",
             "pkg-aaaa/acmepkg.py": "VALUE = 42\n"}),
        "https://example.com/tool.zip": make_zip({"tool/bin/tool.txt": "hi"}),
        "https://example.com/model.pt": b"weights",
    }
    urls: list[str] = []

    def fake_download(url, dest, *, sha256=None, on_bytes=None, cancel=None, title=None, **kw):
        urls.append(url)
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blobs[url])
        if on_bytes:
            on_bytes(len(blobs[url]), len(blobs[url]))
        return dest

    monkeypatch.setattr(fetch, "download", fake_download)
    extra = (
        '\n[[steps]]\ntype = "github_archive"\nrepo = "acme/pkg"\nref = "' + "a" * 40 + '"\ndest = "src/pkg"\n'
        '\n[[steps]]\ntype = "archive"\nurl = "https://example.com/tool.zip"\ndest = "tools/tool"\n'
        '\n[[steps]]\ntype = "archive"\nurl = "https://example.com/model.pt"\ndest = "models/model.pt"\nextract = false\n'
    )
    write_manifest(manifest_dir, "dummy", dummy_manifest(extra_steps=extra))
    manager.reload_manifests()
    manager.install("dummy")
    d = manager.addon_dir("dummy")
    assert (d / "src" / "pkg" / "acmepkg.py").is_file()
    assert (d / "tools" / "tool" / "bin" / "tool.txt").read_text() == "hi"
    assert (d / "models" / "model.pt").read_bytes() == b"weights"
    assert not any((d / "downloads").glob("*.zip"))
    assert len(urls) == 3


def test_hf_download_step_runs_snippet_in_venv(manifest_dir, addon_python, uv_exe, monkeypatch, tmp_path):
    """The hf snippet gets its arguments (incl. token) on stdin; errors map to a token hint."""
    calls = []
    real = installer.run_process

    def fake_run_process(cmd, **kw):
        if len(cmd) >= 3 and cmd[1] == "-c" and "snapshot_download" in cmd[2]:
            calls.append((cmd, kw))
            args = json.loads(kw["stdin_data"])
            assert args["token"] == "hf_secret" and args["repo"] == "org/gated"
            assert kw["env"]["HF_HOME"].endswith("hf")
            kw["on_line"]('@@error {"type": "GatedRepoError", "message": "401 Client Error"}')
            from vrphoto.addons.runner import ProcResult

            return ProcResult(1, ["@@error ..."])
        return real(cmd, **kw)

    monkeypatch.setattr(installer, "run_process", fake_run_process)
    extra = '\n[[steps]]\ntype = "hf_download"\nrepo = "org/gated"\nrevision = "main"\n'
    write_manifest(manifest_dir, "dummy", dummy_manifest(extra_steps=extra))
    manager.reload_manifests()
    with pytest.raises(AddonError, match="denied") as ei:
        manager.install("dummy", hf_token="hf_secret")
    assert "token" in ei.value.hint
    assert calls
    assert "hf_secret" not in (manager.addon_dir("dummy") / "install.log").read_text(encoding="utf-8")


def test_uv_env_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("UV_INDEX_URL", "https://evil.example/simple")
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    m = manager.get_manifest("sdxl360")
    ctx = installer.InstallContext(m, tmp_path / "addons" / "sdxl360", "uv", None, None, None)  # type: ignore[arg-type]
    env = ctx.uv_env()
    assert "UV_INDEX_URL" not in env
    assert env["UV_CACHE_DIR"] == str(tmp_path / "addons" / ".uv-cache")
    assert env["UV_PYTHON_INSTALL_DIR"] == str(tmp_path / "addons" / ".uv-python")
    assert env["UV_NO_CONFIG"] == "1"
    wenv = installer.worker_env_extra(m, tmp_path / "a", "tok")
    assert wenv["HF_HOME"] == str(tmp_path / "a" / "hf") and wenv["HF_TOKEN"] == "tok"
    assert wenv["SDXL_INPAINT_MODEL"].startswith("diffusers/")
    sharp = installer.worker_env_extra(manager.get_manifest("sharp"), tmp_path / "s", None)
    assert sharp["SHARP_CHECKPOINT"] == f"{tmp_path / 's'}/models/sharp_2572gikvuh.pt"
    assert "HF_TOKEN" not in sharp


def test_frozen_paths_resolve_into_meipass(monkeypatch, tmp_path):
    """When frozen, manifests and workers are data files under sys._MEIPASS/vrphoto."""
    from vrphoto.addons.manifest import manifests_dir, workers_dir

    bundle = tmp_path / "_internal"
    (bundle / "vrphoto" / "addons" / "workers").mkdir(parents=True)
    (bundle / "vrphoto" / "addons" / "manifests").mkdir(parents=True)
    worker = bundle / "vrphoto" / "addons" / "workers" / "dummy_worker.py"
    worker.write_text("# bundled copy\n", encoding="utf-8")
    write_manifest(bundle / "vrphoto" / "addons" / "manifests", "dummy", dummy_manifest())
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    manager.reload_manifests()
    assert workers_dir() == bundle / "vrphoto" / "addons" / "workers"
    assert manifests_dir() == bundle / "vrphoto" / "addons" / "manifests"
    m = manager.get_manifest("dummy")
    assert m.worker_path() == worker
    with pytest.raises(AddonError, match="packaged app"):
        manager.tool_command("fake_tool.py")
    assert manager.tool_command("colmap.exe") == ["colmap.exe"]


# ------------------------------------------------------------------ review regressions
def test_stale_step_records_are_dropped_on_resume(manifest_dir, addon_python, uv_exe, monkeypatch):
    """A step that failed after an earlier step changed must run again next time, even
    though an older record of it (from the previous install) is still in state.json."""
    early = '\n[[steps]]\ntype = "check"\ntitle = "{t}"\n'
    write_manifest(manifest_dir, "dummy", dummy_manifest(extra_steps=early.format(t="Early check")))
    manager.reload_manifests()
    manager.install("dummy")
    d = manager.addon_dir("dummy")
    (d / "prefetch.txt").unlink()
    # manifest update changes step 2; step 3 (prefetch) then fails
    write_manifest(manifest_dir, "dummy", dummy_manifest(version="1.1", extra_steps=early.format(t="Early check v2")))
    manager.reload_manifests()
    monkeypatch.setenv("DUMMY_PREFETCH_FAIL", "1")
    with pytest.raises(AddonError, match="dummy prefetch failed"):
        manager.install("dummy")
    state = json.loads((d / "state.json").read_text(encoding="utf-8"))
    assert len(state["steps_done"]) == 2  # venv + changed step; the stale prefetch record is gone
    monkeypatch.delenv("DUMMY_PREFETCH_FAIL")
    manager.install("dummy")
    assert (d / "prefetch.txt").exists(), "the failed step was skipped on the next install"
    assert "Resuming at step 3 of 4" in (d / "install.log").read_text(encoding="utf-8")
    assert manager.status("dummy")["installed"]


def test_resume_point_only_counts_unbroken_prefix(manifest_dir):
    write_manifest(manifest_dir, "dummy", dummy_manifest())
    manager.reload_manifests()
    m = manager.get_manifest("dummy")
    keys = [installer.step_key(i, s) for i, s in enumerate(m.steps)]
    assert installer.resume_point(m, keys) == (3, keys)
    assert installer.resume_point(m, [keys[0], keys[2]]) == (1, keys[:1])
    assert installer.resume_point(m, ["0:venv:deadbeef", *keys[1:]]) == (0, [])


def test_install_refuses_too_old_uv(manifest_dir, addon_python, tmp_path, monkeypatch):
    if sys.platform == "win32":
        pytest.skip("POSIX shell script fake")
    fake = tmp_path / "uv"
    fake.write_text("#!/bin/sh\necho 'uv 0.5.31 (abc 2024-12-01)'\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setenv(manager.UV_ENV, str(fake))
    installer._uv_checked.clear()
    write_manifest(manifest_dir, "dummy", dummy_manifest())
    manager.reload_manifests()
    with pytest.raises(AddonError, match="too old") as ei:
        manager.install("dummy")
    assert "pip install -U uv" in (ei.value.hint or "")
    installer._uv_checked.clear()


def test_uv_version_of_real_uv(uv_exe):
    ver = installer.uv_version(uv_exe)
    assert ver is not None and ver >= installer.MIN_UV_VERSION


def test_resolve_binary_with_glob_chars_in_data_path(monkeypatch, tmp_path, manifest_dir):
    from vrphoto.core import config

    home = tmp_path / "data [beta] {x}"
    monkeypatch.setenv("VRPHOTO_HOME", str(home))
    config.set_settings(None)
    write_manifest(manifest_dir, "tools", """
        id = "tools"
        title = "Tools"
        description = "d"
        license = "MIT"
        kind = "binary"
        tasks = ["t"]
        [binaries.colmap]
        patterns = ["colmap/**/bin/colmap*"]
        [[steps]]
        type = "check"
    """)
    manager.reload_manifests()
    exe = manager.addon_dir("tools") / "colmap" / "COLMAP-4.2" / "bin" / "colmap.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZ")
    monkeypatch.setattr(manager.shutil, "which", lambda name: None)
    assert manager.resolve_binary(manager.get_manifest("tools"), "colmap") == str(exe)
    config.set_settings(None)


def test_last_uninstall_prunes_shared_uv_dirs(installed_dummy):
    root = get_settings().addons_dir
    for name in manager.SHARED_DIRS:
        (root / name / "sub").mkdir(parents=True)
        (root / name / "sub" / "blob").write_bytes(b"x" * 10)
    (root / "other").mkdir()  # another add-on (even partial) still needs the cache
    manager.uninstall("dummy")
    assert all((root / n).exists() for n in manager.SHARED_DIRS)
    (root / "other").rmdir()
    write_manifest(Path(os.environ[manager.MANIFESTS_ENV]), "dummy", dummy_manifest())
    manager.uninstall("dummy")  # nothing left to remove, but the shared dirs go now
    assert not any((root / n).exists() for n in manager.SHARED_DIRS)
