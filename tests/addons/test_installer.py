"""Installer steps with a fake uv: exact-patch managed Python for the venv (Windows error
448 with uv's minor-version junction) and the error hints of failed steps."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from vrphoto.addons import installer, manager
from vrphoto.addons.installer import INSTALL_LOG_HINT, InstallContext, find_managed_python
from vrphoto.addons.runner import NETWORK_HINT, REDIRECTION_GUARD_HINT, LogSink, ProcResult, error_hint
from vrphoto.core.errors import AddonError
from vrphoto.core.types import ProgressReporter

PLAT = "windows-x86_64-none" if sys.platform == "win32" else "linux-x86_64-gnu"
ERR_448 = ("error: Failed to inspect Python interpreter from managed installations at `C:\\Users\\A\\AppData\\Local"
           "\\VRPhotoStudio\\addons\\.uv-python\\cpython-3.11-windows-x86_64-none\\python.exe`\n"
           "  Caused by: The path cannot be traversed because it contains an untrusted mount point. (os error 448)")


def _link_dir(target: Path, link: Path) -> None:
    """A directory symlink, or a junction on Windows without symlink rights (what uv makes)."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return
    except (OSError, NotImplementedError):
        pass
    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
        return
    pytest.skip("cannot create directory links here")


def _fake_install_dir(root: Path) -> Path:
    """Layout of uv's UV_PYTHON_INSTALL_DIR after `uv python install 3.11` (plus distractors)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / ".lock").write_text("")
    (root / ".temp").mkdir()
    for name in (f"cpython-3.11.9-{PLAT}", f"cpython-3.11.13-{PLAT}", f"cpython-3.12.10-{PLAT}",
                 f"cpython-3.11.14+freethreaded-{PLAT}", f"cpython-3.11.0rc1-{PLAT}", "pypy-3.11.99-x"):
        (root / name / "bin").mkdir(parents=True)
    (root / f"cpython-3.11.15-{PLAT}").write_text("a file, not an install")
    _link_dir(root / f"cpython-3.11.13-{PLAT}", root / f"cpython-3.11-{PLAT}")  # uv's minor-version link
    _link_dir(root / f"cpython-3.11.13-{PLAT}", root / f"cpython-3.11.20-{PLAT}")  # a link is never chosen
    return root


# ------------------------------------------------------------------ find_managed_python
def test_find_managed_python_picks_newest_real_patch_folder(tmp_path):
    root = _fake_install_dir(tmp_path / ".uv-python")
    assert installer._is_link(root / f"cpython-3.11-{PLAT}")
    assert installer._is_link(root / f"cpython-3.11.20-{PLAT}")
    assert not installer._is_link(root / f"cpython-3.11.13-{PLAT}")
    assert find_managed_python(root, "3.11") == "3.11.13"
    assert find_managed_python(root, "3.12") == "3.12.10"
    assert find_managed_python(root, "3.10") is None
    assert find_managed_python(tmp_path / "missing", "3.11") is None


def test_is_link_is_failure_safe(tmp_path):
    assert installer._is_link(tmp_path / "does-not-exist")  # unknown: treated as a link (skipped)
    (tmp_path / "real").mkdir()
    assert not installer._is_link(tmp_path / "real")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junctions")
def test_is_link_detects_windows_junctions(tmp_path):
    import _winapi

    (tmp_path / "cpython-3.11.13-x").mkdir()
    _winapi.CreateJunction(str(tmp_path / "cpython-3.11.13-x"), str(tmp_path / "cpython-3.11.14-x"))
    assert installer._is_link(tmp_path / "cpython-3.11.14-x")
    assert find_managed_python(tmp_path, "3.11") == "3.11.13"


# ------------------------------------------------------------------ step_venv with a fake uv
class FakeUv:
    """Stands in for ``installer.run_process``: answers the uv / venv-python commands."""

    def __init__(self, *, install_ok: bool = True, make_install: bool = True, venv_tail: list[str] | None = None):
        self.calls: list[list[str]] = []
        self.envs: list[dict] = []
        self.install_ok = install_ok
        self.make_install = make_install
        self.venv_tail = venv_tail

    def __call__(self, cmd, **kw):
        cmd = [os.fspath(c) for c in cmd]
        self.calls.append(cmd)
        self.envs.append(dict(kw.get("env") or {}))
        if cmd[1:] == ["--version"]:
            return ProcResult(0, ["uv 0.11.32 (x86_64-unknown-linux-gnu)"])
        if cmd[1:3] == ["python", "install"]:
            if self.make_install:
                _fake_install_dir(Path(kw["env"]["UV_PYTHON_INSTALL_DIR"]))
            if self.install_ok:
                for line in ("Downloading cpython-3.11.13 (download) (21.1MiB)", "Installed Python 3.11.13 in 2.1s"):
                    kw["on_line"](line)
                return ProcResult(0, ["Installed Python 3.11.13 in 2.1s"])
            return ProcResult(2, ["error: Failed to install cpython-3.11.13", "  Caused by: something else"])
        if cmd[1] == "venv":
            if self.venv_tail is not None:
                return ProcResult(2, list(self.venv_tail))
            venv = Path(cmd[-1])
            py = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("")
            return ProcResult(0, [f"Creating virtual environment at: {venv}"])
        if len(cmd) >= 3 and cmd[1] == "-c":  # the venv's "which Python version" check
            return ProcResult(0, ["3.11"])
        raise AssertionError(f"unexpected command {cmd}")

    def venv_args(self) -> list[str]:
        return next(c for c in self.calls if len(c) > 1 and c[1] == "venv")


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    for var in (installer.PYTHON_ENV, "UV_PYTHON_INSTALL_DIR", "UV_CACHE_DIR"):
        monkeypatch.delenv(var, raising=False)
    installer._uv_checked.clear()
    sink = LogSink(tmp_path / "install.log")
    c = InstallContext(manager.get_manifest("sdxl360"), tmp_path / "addons" / "sdxl360", "uv", sink, None, None)
    c.addon_dir.mkdir(parents=True)
    yield c
    sink.close()
    installer._uv_checked.clear()


def _log(ctx) -> str:
    return ctx.sink.path.read_text(encoding="utf-8")


def test_step_venv_uses_exact_patch_version(ctx, monkeypatch):
    """`uv python install` first, then the venv from the real cpython-3.11.N folder, never
    from uv's cpython-3.11 junction (blocked with os error 448 under RedirectionGuard)."""
    uv = FakeUv()
    monkeypatch.setattr(installer, "run_process", uv)
    events: list[tuple[float, str]] = []
    installer.step_venv(ctx, {"type": "venv"}, ProgressReporter(lambda f, m: events.append((f, m))))
    install = next(c for c in uv.calls if c[1:3] == ["python", "install"])
    assert install == ["uv", "python", "install", "--no-bin", "--no-registry", "3.11"]
    venv = uv.venv_args()
    assert venv[venv.index("--python") + 1] == "3.11.13"
    assert "--managed-python" in venv and venv[-1] == str(ctx.venv)
    assert uv.calls.index(install) < uv.calls.index(venv)
    env = uv.envs[uv.calls.index(install)]  # same isolated uv environment as the venv command
    assert env["UV_PYTHON_INSTALL_DIR"] == str(ctx.addon_dir.parent / ".uv-python")
    assert env == uv.envs[uv.calls.index(venv)]
    assert "Using the managed Python 3.11.13" in _log(ctx)
    fracs = [f for f, _ in events]
    assert fracs == sorted(fracs) and fracs[-1] == pytest.approx(1.0)


def test_step_venv_falls_back_to_minor_request(ctx, monkeypatch):
    """No real cpython-3.11.N folder found (install failed): the old `--python 3.11`."""
    uv = FakeUv(install_ok=False, make_install=False)
    monkeypatch.setattr(installer, "run_process", uv)
    installer.step_venv(ctx, {"type": "venv"}, ProgressReporter())
    venv = uv.venv_args()
    assert venv[venv.index("--python") + 1] == "3.11"
    log = _log(ctx)
    assert "WARNING: uv python install failed" in log and "using --python 3.11" in log


def test_step_venv_uses_existing_patch_folder_when_install_fails(ctx, monkeypatch):
    uv = FakeUv(install_ok=False)  # e.g. offline, but an earlier add-on already installed 3.11.13
    monkeypatch.setattr(installer, "run_process", uv)
    installer.step_venv(ctx, {"type": "venv"}, ProgressReporter())
    venv = uv.venv_args()
    assert venv[venv.index("--python") + 1] == "3.11.13"


def test_step_venv_override_path_unchanged(ctx, monkeypatch):
    monkeypatch.setenv(installer.PYTHON_ENV, sys.executable)
    uv = FakeUv()
    monkeypatch.setattr(installer, "run_process", uv)
    installer.step_venv(ctx, {"type": "venv", "seed": True}, ProgressReporter())
    assert not any(c[1:3] == ["python", "install"] for c in uv.calls)
    assert uv.venv_args() == ["uv", "venv", "--no-project", "--python", sys.executable, "--no-python-downloads",
                              "--seed", str(ctx.venv)]


def test_step_venv_448_error_gets_restart_hint(ctx, monkeypatch):
    uv = FakeUv(venv_tail=ERR_448.splitlines())
    monkeypatch.setattr(installer, "run_process", uv)
    with pytest.raises(AddonError, match="Creating the Python environment failed") as ei:
        installer.step_venv(ctx, {"type": "venv"}, ProgressReporter())
    assert ei.value.hint == REDIRECTION_GUARD_HINT
    assert "Start menu" in ei.value.hint and "internet" not in ei.value.hint


# ------------------------------------------------------------------ hints
@pytest.mark.parametrize(
    "tail",
    [
        "error: Failed to fetch: `https://download.pytorch.org/whl/cu128/torch/`",
        "  Caused by: error sending request for url (https://pypi.org/simple/numpy/)",
        "  Caused by: dns error: failed to lookup address information: No such host is known. (os error 11001)",
        "  Caused by: operation timed out",
        "  Caused by: Connection reset by peer (os error 10054)",
        "  Caused by: received fatal alert: HandshakeFailure (tls handshake failed)",
        "requests.exceptions.ConnectionError: Max retries exceeded with url",
        # Python-side download errors (huggingface_hub / requests / urllib3), which the
        # hf_download fallback no longer answers with an internet hint by default
        "ChunkedEncodingError: ('Connection broken: IncompleteRead(0 bytes read, 512 more expected)'",
        "urllib3.exceptions.ProtocolError: ('Connection aborted.', RemoteDisconnected('Remote end closed'))",
        "LocalEntryNotFoundError: An error happened while trying to locate the file on the Hub. "
        "Please check your internet connection and try again.",
    ],
)
def test_pip_hint_network_errors(tail):
    assert installer._pip_hint(tail) == NETWORK_HINT
    assert error_hint(tail) == NETWORK_HINT


@pytest.mark.parametrize("text", [ERR_448, "OSError: [WinError 448] The path cannot be traversed because it "
                                           "contains an untrusted mount point: 'C:\\x'"])
def test_448_hint(text):
    assert installer._pip_hint(text) == REDIRECTION_GUARD_HINT
    assert error_hint(text) == REDIRECTION_GUARD_HINT


def test_pip_hint_default_does_not_blame_the_internet():
    hint = installer._pip_hint("error: Because foo==1 depends on bar>2 and you require bar<2, no solution found")
    assert hint == INSTALL_LOG_HINT
    assert "install.log" in hint and "Open folder" in hint and "internet" not in hint.lower()
    assert "build tools" in installer._pip_hint("error: Failed to build `xyz==1.0`")


def test_network_pattern_does_not_match_tls_inside_words():
    assert error_hint("utils loaded; settings saved; rustls initialised; outlets ok") is None


def test_hf_download_failure_hints(ctx, monkeypatch):
    ctx.python.parent.mkdir(parents=True)
    ctx.python.write_text("")
    payload: dict[str, str] = {}

    def fake_run_process(cmd, **kw):
        kw["on_line"]("@@error " + json.dumps(payload))
        return ProcResult(1, ["@@error ..."])

    monkeypatch.setattr(installer, "run_process", fake_run_process)
    step = {"type": "hf_download", "repo": "org/model", "revision": "main"}
    cases = [
        ({"type": "ValueError", "message": "something unexpected"}, INSTALL_LOG_HINT),
        ({"type": "ConnectionError", "message": "error sending request for url"}, NETWORK_HINT),
        ({"type": "OSError", "message": "[WinError 448] untrusted mount point"}, REDIRECTION_GUARD_HINT),
    ]
    for err, want in cases:
        payload.clear()
        payload.update(err)
        with pytest.raises(AddonError, match="Downloading org/model failed") as ei:
            installer.step_hf_download(ctx, step, ProgressReporter())
        assert ei.value.hint == want


_FAKE_HF = {
    "__init__.py": (
        "import json, os\n"
        "class _S:\n"
        "    def __init__(self, name):\n"
        "        self.rfilename, self.size = name, 10\n"
        "class _Info:\n"
        "    siblings = [_S(f) for f in json.loads(os.environ['FAKE_HF_FILES'])]\n"
        "class HfApi:\n"
        "    def repo_info(self, *a, **kw):\n"
        "        return _Info()\n"
        "def snapshot_download(**kw):\n"
        "    with open(os.environ['FAKE_HF_CALLS'], 'a', encoding='utf-8') as fh:\n"
        "        fh.write(json.dumps(kw['allow_patterns']) + '\\n')\n"
        "    return kw.get('local_dir') or '.'\n"
    ),
    "utils.py": (
        "import fnmatch\n"
        "def filter_repo_objects(items, allow_patterns=None, ignore_patterns=None):\n"
        "    for i in items:\n"
        "        if allow_patterns and not any(fnmatch.fnmatch(i, p) for p in allow_patterns):\n"
        "            continue\n"
        "        if ignore_patterns and any(fnmatch.fnmatch(i, p) for p in ignore_patterns):\n"
        "            continue\n"
        "        yield i\n"
    ),
    "constants.py": "HF_HUB_CACHE = '.'\n",
}


@pytest.mark.parametrize("files, allow, missing", [
    (["Qwen-Image-Edit-2511-Q4_K_M.gguf"], ["qwen-image-edit-2511-Q4_K_M.gguf"],
     "org/model@main has no file qwen-image-edit-2511-Q4_K_M.gguf"),  # a wrong exact name
    (["a.safetensors"], ["*.gguf"], "org/model@main has no file *.gguf"),  # nothing matches
    (["a.gguf", "mmproj-a.gguf"], ["a.gguf", "mmproj-b.gguf"], "org/model@main has no file mmproj-b.gguf"),
    (["split_files/vae/v.safetensors", "x.bin"], ["split_files/vae/v.safetensors"], ""),
    (["a.gguf", "b.gguf"], ["*.gguf"], ""),
])
def test_hf_snippet_refuses_patterns_that_match_no_file(tmp_path, files, allow, missing):
    """snapshot_download downloads nothing and succeeds when no file matches allow_patterns; the
    snippet must fail instead (naming the file and repo@revision), so the step is not recorded as done."""
    import subprocess

    pkg = tmp_path / "fakehf" / "huggingface_hub"
    pkg.mkdir(parents=True)
    for name, text in _FAKE_HF.items():
        (pkg / name).write_text(text, encoding="utf-8")
    calls = tmp_path / "calls.jsonl"
    env = {**os.environ, "PYTHONPATH": str(pkg.parent), "FAKE_HF_FILES": json.dumps(files),
           "FAKE_HF_CALLS": str(calls)}
    args = {"repo": "org/model", "revision": "main", "allow_patterns": allow, "local_dir": str(tmp_path / "dl")}
    res = subprocess.run([sys.executable, "-c", installer.HF_SNIPPET], input=json.dumps(args) + "\n", env=env,
                         capture_output=True, text=True, timeout=60)
    tags = {ln[2:].partition(" ")[0]: json.loads(ln[2:].partition(" ")[2]) for ln in res.stdout.splitlines()
            if ln.startswith("@@")}
    if missing:
        assert res.returncode == 1 and tags["error"] == {"type": "MissingFiles", "message": missing}, res
        assert not calls.exists()  # nothing was downloaded
    else:
        assert res.returncode == 0 and "done" in tags and "error" not in tags, res
        assert json.loads(calls.read_text(encoding="utf-8")) == allow


def test_hf_download_missing_files_error(ctx, monkeypatch):
    ctx.python.parent.mkdir(parents=True)
    ctx.python.write_text("")

    def fake_run_process(cmd, **kw):
        kw["on_line"]("@@error " + json.dumps({"type": "MissingFiles", "message": "org/model@main has no file x.gguf"}))
        return ProcResult(1, ["@@error ..."])

    monkeypatch.setattr(installer, "run_process", fake_run_process)
    step = {"type": "hf_download", "repo": "org/model", "revision": "main", "allow_patterns": ["x.gguf"]}
    with pytest.raises(AddonError, match=r"Model files of org/model were not found .*has no file x\.gguf") as ei:
        installer.step_hf_download(ctx, step, ProgressReporter())
    assert "renamed or moved" in ei.value.hint
