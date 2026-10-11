"""Executes the ``[[steps]]`` of an add-on manifest (used by :mod:`vrphoto.addons.manager`).

Each step runs with logging to ``<addons_dir>/<id>/install.log``, progress and
cooperative cancel.  Completed steps are recorded in ``state.json`` by a key that
includes a hash of the step definition; a re-install resumes at the first step that is
not recorded (or whose definition changed) and re-runs everything after it, because
later steps depend on earlier ones (a new venv needs its packages again).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..core.errors import AddonError, Cancelled
from ..core.log import get_logger
from ..core.types import CancelToken, ProgressReporter
from . import fetch
from .env import clean_env, native_tool_env, prepend_path, venv_python
from .manifest import AddonManifest
from .runner import LogSink, error_hint, run_process, run_worker

log = get_logger(__name__)

PYTHON_ENV = "VRPHOTO_ADDON_PYTHON"
_IS_WINDOWS = sys.platform == "win32"

_DEFAULT_WEIGHTS = {
    "venv": 1.0,
    "pip": 6.0,
    "pip_requirements": 8.0,
    "github_archive": 3.0,
    "archive": 6.0,
    "hf_download": 20.0,
    "check": 1.0,
    "worker": 4.0,
}

# Executed in the add-on venv for hf_download steps.  Reads its arguments (including the
# token) from stdin so the token never appears in a command line or log.
HF_SNIPPET = r"""
import json, os, sys, threading, time
args = json.loads(sys.stdin.readline())
def out(tag, obj):
    sys.stdout.write("@@" + tag + " " + json.dumps(obj) + "\n"); sys.stdout.flush()
try:
    from huggingface_hub import snapshot_download, HfApi
    token = args.get("token") or None
    total = 0
    watch = args.get("local_dir")
    try:
        api = HfApi()
        info = api.repo_info(args["repo"], repo_type=args.get("repo_type") or "model",
                             revision=args.get("revision") or None, files_metadata=True, token=token)
        files = [s.rfilename for s in (info.siblings or [])]
        try:
            from huggingface_hub.utils import filter_repo_objects
            keep = set(filter_repo_objects(files, allow_patterns=args.get("allow_patterns"),
                                           ignore_patterns=args.get("ignore_patterns")))
        except Exception:
            keep = set(files)
        total = sum((s.size or 0) for s in (info.siblings or []) if s.rfilename in keep)
        out("total", {"bytes": total, "files": len(keep)})
        if not watch:
            from huggingface_hub.constants import HF_HUB_CACHE
            kind = (args.get("repo_type") or "model") + "s"
            watch = os.path.join(HF_HUB_CACHE, kind + "--" + args["repo"].replace("/", "--"))
    except Exception as e:
        out("warn", {"message": "no size information: %s" % (e,)})
    stop = threading.Event()
    def poll():
        while not stop.wait(1.0):
            size = 0
            for root, _d, fs in os.walk(watch or "."):
                for f in fs:
                    try:
                        size += os.lstat(os.path.join(root, f)).st_size
                    except OSError:
                        pass
            if total:
                out("progress", {"frac": min(1.0, size / total), "bytes": size})
    if total and watch:
        threading.Thread(target=poll, daemon=True).start()
    kw = dict(repo_id=args["repo"], revision=args.get("revision") or None,
              allow_patterns=args.get("allow_patterns"), ignore_patterns=args.get("ignore_patterns"),
              repo_type=args.get("repo_type") or "model", token=token, max_workers=4)
    if args.get("local_dir"):
        kw["local_dir"] = args["local_dir"]
    path = snapshot_download(**kw)
    stop.set()
    out("done", {"path": str(path)})
except Exception as e:
    out("error", {"type": type(e).__name__, "message": str(e)[:2000]})
    sys.exit(1)
"""


def step_key(index: int, step: dict) -> str:
    """Stable identity of a step (index, type and a hash of its definition)."""
    blob = json.dumps(step, sort_keys=True, separators=(",", ":"))
    return f"{index}:{step.get('type')}:{hashlib.sha1(blob.encode('utf-8')).hexdigest()[:12]}"


def step_applies(step: dict) -> bool:
    plats = step.get("platforms")
    return not plats or sys.platform in plats


def _expand(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        out = value
        for k, v in mapping.items():
            out = out.replace("{" + k + "}", v)
        return out
    if isinstance(value, list):
        return [_expand(v, mapping) for v in value]
    return value


@dataclass
class InstallContext:
    """Everything a step needs."""

    manifest: AddonManifest
    addon_dir: Path
    uv: str
    sink: LogSink
    cancel: CancelToken | None
    hf_token: str | None
    check_gpu: bool = True

    # ----------------------------------------------------------------- paths
    @property
    def venv(self) -> Path:
        return self.addon_dir / "venv"

    @property
    def python(self) -> Path:
        return venv_python(self.venv)

    @property
    def downloads(self) -> Path:
        return self.addon_dir / "downloads"

    def placeholders(self) -> dict[str, str]:
        return {
            "addon_dir": str(self.addon_dir),
            "src": str(self.addon_dir / "src"),
            "venv": str(self.venv),
            "python": str(self.python),
            "manifests": str(self.manifest.manifest_dir),
        }

    def expand(self, value: Any) -> Any:
        return _expand(value, self.placeholders())

    def resolve(self, rel: str) -> Path:
        """Path relative to the add-on folder (placeholders expanded)."""
        p = Path(self.expand(rel))
        return p if p.is_absolute() else self.addon_dir / p

    # ----------------------------------------------------------------- envs
    def uv_env(self) -> dict[str, str]:
        root = self.addon_dir.parent
        extra: dict[str, Any] = {
            "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR") or str(root / ".uv-cache"),
            "UV_PYTHON_INSTALL_DIR": os.environ.get("UV_PYTHON_INSTALL_DIR") or str(root / ".uv-python"),
            "UV_NO_CONFIG": "1",
            "UV_HTTP_TIMEOUT": os.environ.get("UV_HTTP_TIMEOUT") or "300",
            "UV_PYTHON_DOWNLOADS": None,
            "UV_PROJECT_ENVIRONMENT": None,
            "UV_INDEX_URL": None,
            "UV_EXTRA_INDEX_URL": None,
            "UV_DEFAULT_INDEX": None,
            "UV_INDEX": None,
            "UV_TORCH_BACKEND": None,
            "UV_PYTHON": None,
            "UV_SYSTEM_PYTHON": None,
            "UV_MANAGED_PYTHON": None,
            "UV_NO_MANAGED_PYTHON": None,
            "UV_PYTHON_PREFERENCE": None,
            "UV_OFFLINE": None,
            "PIP_REQUIRE_VIRTUALENV": None,
        }
        if _IS_WINDOWS:
            extra["UV_LINK_MODE"] = "copy"
        return clean_env(extra)

    def venv_env(self) -> dict[str, str]:
        """Environment for commands running inside the add-on venv (hf, checks)."""
        env = clean_env(worker_env_extra(self.manifest, self.addon_dir, self.hf_token))
        return prepend_path(env, self.python.parent)


def worker_env_extra(manifest: AddonManifest, addon_dir: Path, hf_token: str | None) -> dict[str, Any]:
    """Extra environment variables for worker processes of ``manifest``."""
    cache = addon_dir / "cache"
    extra: dict[str, Any] = {
        "VRPHOTO_ADDON_ID": manifest.id,
        "VRPHOTO_ADDON_DIR": str(addon_dir),
        "VRPHOTO_ADDON_SRC": str(addon_dir / "src"),
        "HF_HOME": str(addon_dir / "hf"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TORCH_HOME": str(addon_dir / "torch"),
        "U2NET_HOME": str(addon_dir / "u2net"),
        "HY3DGEN_MODELS": str(addon_dir / "models"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
        "XDG_CACHE_HOME": str(cache),
        "MPLCONFIGDIR": str(cache / "matplotlib"),
        "TQDM_DISABLE": None,
        "HF_HUB_OFFLINE": None,
        "TRANSFORMERS_OFFLINE": None,
    }
    if hf_token:
        extra["HF_TOKEN"] = hf_token
    mapping = {
        "addon_dir": str(addon_dir),
        "src": str(addon_dir / "src"),
        "venv": str(addon_dir / "venv"),
        "python": str(venv_python(addon_dir / "venv")),
        "manifests": str(manifest.manifest_dir),
    }
    for k, v in manifest.env.items():
        extra[k] = _expand(v, mapping)
    return extra


# --------------------------------------------------------------------------- uv output parsing
class _UvProgress:
    """Turns uv's plain (non-tty) output into progress callbacks."""

    _re_dl = re.compile(r"^\s*Downloading (\S+)")
    _re_done = re.compile(r"^\s*Downloaded (\S+)")
    _re_phase = re.compile(r"^\s*(Resolved|Prepared|Installed|Uninstalled|Audited|Built|Building)\b")

    def __init__(self, progress: ProgressReporter, title: str):
        self.progress = progress
        self.title = title
        self.started: set[str] = set()
        self.finished: set[str] = set()
        self.frac = 0.02

    def __call__(self, line: str) -> None:
        m = self._re_dl.match(line)
        if m:
            self.started.add(m.group(1))
            self.progress(self._frac(), f"{self.title}: downloading {m.group(1)}")
            return
        m = self._re_done.match(line)
        if m:
            self.finished.add(m.group(1))
            self.progress(self._frac(), f"{self.title}: downloaded {m.group(1)}")
            return
        m = self._re_phase.match(line)
        if m:
            phase = m.group(1)
            if phase == "Resolved":
                self.frac = max(self.frac, 0.05)
            elif phase in ("Prepared", "Built"):
                self.frac = max(self.frac, 0.85)
            elif phase == "Installed":
                self.frac = 1.0
            self.progress(self._frac(), f"{self.title}: {line.strip()}")

    def _frac(self) -> float:
        if self.started:
            dl = 0.05 + 0.8 * len(self.finished) / max(len(self.started), 1)
            self.frac = max(self.frac, min(dl, 0.85))
        return self.frac


def _uv_error_line(tail: list[str]) -> str:
    for ln in tail:
        s = ln.strip()
        if s.startswith(("error:", "×", "x ")) or "No solution found" in s:
            return s.lstrip("×x ").strip()
    for ln in reversed(tail):
        if ln.strip():
            return ln.strip()
    return ""


_PIP_HINTS = (
    (re.compile(r"not a supported wheel|no matching distribution|has no wheels|platform", re.I),
     "This add-on's packages are not available for your system (Windows 64-bit with an NVIDIA GPU "
     "is required for most add-ons)."),
    (re.compile(r"Failed to build|Microsoft Visual C\+\+|CUDA_HOME|nvcc", re.I),
     "A package needed compiling and the build tools are missing. Retry later or report the problem."),
)


#: Fallback hint when no known error pattern matched (not every failure is a network problem).
INSTALL_LOG_HINT = "Installation failed - details are in install.log (Add-ons window > Open folder)."


def _pip_hint(text: str) -> str | None:
    h = error_hint(text)
    if h:
        return h
    for pat, hint in _PIP_HINTS:
        if pat.search(text):
            return hint
    return INSTALL_LOG_HINT


# --------------------------------------------------------------------------- step runners
StepFn = Callable[[InstallContext, dict, ProgressReporter], None]


#: Oldest uv that understands every flag used here (``uv venv --managed-python --clear``).
MIN_UV_VERSION = (0, 8, 0)
_uv_checked: dict[str, tuple[int, ...]] = {}


def uv_version(uv: str) -> tuple[int, ...] | None:
    """``(major, minor, patch)`` of a uv executable (cached), or None if it cannot be read."""
    if uv in _uv_checked:
        return _uv_checked[uv]
    try:
        res = run_process([uv, "--version"], env=clean_env(), timeout=60, title="uv")
    except AddonError:
        return None
    m = re.search(r"\buv (\d+)\.(\d+)\.(\d+)", " ".join(res.tail))
    if not res.ok or m is None:
        return None
    ver = tuple(int(g) for g in m.groups())
    _uv_checked[uv] = ver
    return ver


def require_uv(uv: str) -> None:
    """Raise :class:`AddonError` when ``uv`` is older than :data:`MIN_UV_VERSION`."""
    ver = uv_version(uv)
    if ver is not None and ver < MIN_UV_VERSION:
        need = ".".join(map(str, MIN_UV_VERSION))
        raise AddonError(
            f"The uv package manager is too old ({'.'.join(map(str, ver))}; {need} or newer is needed).",
            hint="Update uv (pip install -U uv) or reinstall VR Photo Studio.",
        )


def _run_uv(ctx: InstallContext, args: list[str], title: str, progress: ProgressReporter) -> None:
    require_uv(ctx.uv)
    res = run_process(
        [ctx.uv, *args],
        env=ctx.uv_env(),
        cwd=ctx.addon_dir,
        log_sink=ctx.sink,
        on_line=_UvProgress(progress, title),
        cancel=ctx.cancel,
        title=title,
    )
    if not res.ok:
        detail = _uv_error_line(res.tail)
        raise AddonError(
            f"{title} failed" + (f": {detail}" if detail else "."),
            hint=_pip_hint("\n".join(res.tail)),
        )


def _index_args(ctx: InstallContext, step: dict) -> list[str]:
    args: list[str] = []
    if step.get("index_url"):
        args += ["--index-url", ctx.expand(step["index_url"])]
    for url in step.get("extra_index_url", []):
        args += ["--extra-index-url", ctx.expand(url)]
    if step.get("torch_backend"):
        args += ["--torch-backend", step["torch_backend"]]
    if step.get("index_strategy"):
        args += ["--index-strategy", step["index_strategy"]]
    if step.get("no_deps"):
        args.append("--no-deps")
    if step.get("no_build_isolation"):
        args.append("--no-build-isolation")
    if step.get("reinstall"):
        args.append("--reinstall")
    if step.get("pre"):
        args.append("--prerelease=allow")
    return args


def _require_venv(ctx: InstallContext) -> None:
    if not ctx.python.is_file():
        raise AddonError(
            f"The Python environment of {ctx.manifest.title} is missing.",
            hint="Reinstall the add-on.",
        )


def _is_link(path: Path) -> bool:
    """True for a symlink, a Windows junction or any other reparse point - and, to be safe,
    whenever it cannot be determined.  Never follows the link itself."""
    try:
        if path.is_symlink():
            return True
        isjunction = getattr(os.path, "isjunction", None)  # Python 3.12+
        if isjunction is not None and isjunction(path):
            return True
        st = os.lstat(path)
    except (OSError, ValueError):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if getattr(st, "st_file_attributes", 0) & reparse:  # Windows only
        return True
    try:
        os.readlink(path)  # also reads junctions on Windows
    except (OSError, ValueError, NotImplementedError):
        return False
    return True


def find_managed_python(install_dir: Path, minor: str) -> str | None:
    """Exact version (``"3.11.13"``) of the newest uv-managed CPython ``minor`` (``"3.11"``)
    installed in ``install_dir`` as a *real* folder, or None.

    uv also keeps a ``cpython-3.11-<platform>`` link (a junction on Windows) to the newest
    patch release; such links are skipped.
    """
    pat = re.compile(r"cpython-" + re.escape(minor) + r"\.(\d+)-", re.IGNORECASE)
    best: int | None = None
    try:
        entries = list(os.scandir(install_dir))
    except OSError:
        return None
    for entry in entries:
        m = pat.match(entry.name)
        if m is None:
            continue
        p = Path(entry.path)
        if _is_link(p):
            continue
        try:
            if not stat.S_ISDIR(os.lstat(p).st_mode):
                continue
        except OSError:
            continue
        patch = int(m.group(1))
        if best is None or patch > best:
            best = patch
    return f"{minor}.{best}" if best is not None else None


def _managed_python_request(ctx: InstallContext, progress: ProgressReporter) -> str:
    """Install the manifest's managed Python and return the exact ``x.y.N`` to create the
    venv from (falls back to ``x.y`` when the installed version cannot be determined).

    ``uv venv --python 3.11`` points the venv at uv's ``cpython-3.11-*`` minor-version link,
    a directory junction on Windows.  Under Windows' RedirectionGuard (which some processes
    run with) Windows refuses to follow such a junction ("untrusted mount point", os error
    448); an exact ``--python 3.11.N`` request uses the real folder instead.
    """
    want = ctx.manifest.python
    env = ctx.uv_env()
    title = f"Installing Python {want}"
    progress(0.02, title)
    require_uv(ctx.uv)
    res = run_process(
        [ctx.uv, "python", "install", "--no-bin", "--no-registry", want],
        env=env,
        cwd=ctx.addon_dir,
        log_sink=ctx.sink,
        on_line=_UvProgress(progress, title),
        cancel=ctx.cancel,
        title=title,
    )
    if not res.ok:
        ctx.sink.write(f"WARNING: uv python install failed ({_uv_error_line(res.tail) or res.returncode}); "
                       "creating the environment directly")
    install_dir = env.get("UV_PYTHON_INSTALL_DIR")
    exact = find_managed_python(Path(install_dir), want) if install_dir else None
    if exact is None:
        ctx.sink.write(f"No installed Python {want}.N folder found in {install_dir}; using --python {want}")
        return want
    ctx.sink.write(f"Using the managed Python {exact} (exact version, not the {want} folder link)")
    return exact


def step_venv(ctx: InstallContext, step: dict, progress: ProgressReporter) -> None:
    """Create ``<addon_dir>/venv`` with uv (managed Python, or ``VRPHOTO_ADDON_PYTHON``)."""
    override = os.environ.get(PYTHON_ENV, "").strip()
    args = ["venv", "--no-project"]
    if override:
        if not Path(override).exists() and not shutil.which(override):
            raise AddonError(f"{PYTHON_ENV} points to a missing interpreter: {override}")
        args += ["--python", override, "--no-python-downloads"]
        ctx.sink.write(f"Using interpreter from {PYTHON_ENV}: {override}")
        venv_progress = progress
    else:
        request = _managed_python_request(ctx, progress.sub(0.0, 0.6))
        args += ["--managed-python", "--python", request]
        venv_progress = progress.sub(0.6, 1.0)
    if step.get("seed"):
        args.append("--seed")
    if ctx.venv.exists():
        args.append("--clear")
    args.append(str(ctx.venv))
    venv_progress(0.1, f"Creating Python {ctx.manifest.python} environment")
    _run_uv(ctx, args, "Creating the Python environment", venv_progress)
    _require_venv(ctx)
    res = run_process(
        [str(ctx.python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        env=ctx.venv_env(),
        log_sink=ctx.sink,
        cancel=ctx.cancel,
        title="Python check",
    )
    got = res.tail[-1].strip() if res.tail else "?"
    if not res.ok:
        raise AddonError("The new Python environment does not start.", hint=error_hint(res.tail_text()))
    if got != ctx.manifest.python:
        msg = f"Add-on {ctx.manifest.id} expects Python {ctx.manifest.python}, venv has {got}"
        if override:
            ctx.sink.write("WARNING: " + msg)
            log.warning(msg)
        else:
            raise AddonError(msg)
    progress(1.0, "Python environment ready")


def step_pip(ctx: InstallContext, step: dict, progress: ProgressReporter) -> None:
    """``uv pip install`` the listed packages into the add-on venv."""
    _require_venv(ctx)
    packages = [str(p) for p in ctx.expand(step["packages"])]
    title = step.get("title") or "Installing Python packages"
    _run_uv(ctx, ["pip", "install", "--python", str(ctx.python), *_index_args(ctx, step), *packages], title, progress)


def step_pip_requirements(ctx: InstallContext, step: dict, progress: ProgressReporter) -> None:
    """``uv pip install -r <file>`` (file bundled next to the manifest or inside the add-on folder)."""
    _require_venv(ctx)
    rel = ctx.expand(step["file"])
    cand = [Path(rel)] if Path(rel).is_absolute() else [ctx.manifest.manifest_dir / rel, ctx.addon_dir / rel]
    req = next((c for c in cand if c.is_file()), None)
    if req is None:
        raise AddonError(f"Requirements file not found: {rel}")
    title = step.get("title") or "Installing Python packages"
    _run_uv(ctx, ["pip", "install", "--python", str(ctx.python), *_index_args(ctx, step), "-r", str(req)], title, progress)


def _bytes_progress(progress: ProgressReporter, lo: float, hi: float, title: str) -> Callable[[int, int | None], None]:
    last = [0.0]

    def cb(done: int, total: int | None) -> None:
        now = time.monotonic()
        if now - last[0] < 0.25 and (total is None or done < total):
            return
        last[0] = now
        mb = done / 1e6
        if total:
            progress(lo + (hi - lo) * done / total, f"{title}: {mb:,.0f} / {total / 1e6:,.0f} MB")
        else:
            progress(lo, f"{title}: {mb:,.0f} MB")

    return cb


def _download_name(url: str, fallback: str) -> str:
    name = Path(urlparse(url).path).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or fallback


def step_github_archive(ctx: InstallContext, step: dict, progress: ProgressReporter) -> None:
    """Download ``github.com/<repo>/archive/<ref>.zip`` (no git needed) and extract it."""
    repo, ref = step["repo"], step["ref"]
    url = ctx.expand(step.get("url") or f"https://github.com/{repo}/archive/{ref}.zip")
    title = step.get("title") or f"Downloading {repo}"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", f"{repo.split('/')[-1]}-{ref[:16]}.zip")
    zpath = ctx.downloads / name
    fetch.download(url, zpath, sha256=step.get("sha256"), cancel=ctx.cancel, title=repo,
                   on_bytes=_bytes_progress(progress, 0.0, 0.7, title))
    dest = ctx.resolve(step["dest"])
    progress(0.72, f"Extracting {repo}")
    fetch.extract(zpath, dest, strip=True, cancel=ctx.cancel)
    with contextlib.suppress(OSError):
        zpath.unlink()
    if step.get("pip_install"):
        _require_venv(ctx)
        target = ["-e", str(dest)] if step.get("editable") else [str(dest)]
        _run_uv(ctx, ["pip", "install", "--python", str(ctx.python), *_index_args(ctx, step), *target],
                f"Installing {repo}", progress.sub(0.75, 1.0))
    progress(1.0, f"{repo} ready")


_ARCHIVE_EXTS = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2", ".tbz2")


def step_archive(ctx: InstallContext, step: dict, progress: ProgressReporter) -> None:
    """Download a file (e.g. a COLMAP / Brush release zip or a checkpoint) and extract it."""
    url = ctx.expand(step["url"])
    name = _download_name(url, "download.bin")
    title = step.get("title") or f"Downloading {name}"
    extract = step.get("extract", name.lower().endswith(_ARCHIVE_EXTS))
    dest = ctx.resolve(step["dest"])
    if extract:
        tmp = ctx.downloads / name
        fetch.download(url, tmp, sha256=step.get("sha256"), cancel=ctx.cancel, title=name,
                       on_bytes=_bytes_progress(progress, 0.0, 0.85, title))
        progress(0.86, f"Extracting {name}")
        fetch.extract(tmp, dest, strip=step.get("strip"), cancel=ctx.cancel,
                      on_progress=lambda f: progress(0.86 + 0.14 * f, f"Extracting {name}"))
        with contextlib.suppress(OSError):
            tmp.unlink()
    else:
        fetch.download(url, dest, sha256=step.get("sha256"), cancel=ctx.cancel, title=name,
                       on_bytes=_bytes_progress(progress, 0.0, 1.0, title))
    if not step.get("sha256"):
        ctx.sink.write(f"WARNING: no sha256 pinned for {url}; recorded {fetch.sha256_file(dest) if dest.is_file() else '(folder)'}")
    progress(1.0, f"{name} ready")


def step_hf_download(ctx: InstallContext, step: dict, progress: ProgressReporter) -> None:
    """Download Hugging Face model files inside the venv (``huggingface_hub.snapshot_download``)."""
    _require_venv(ctx)
    repo = step["repo"]
    gated = bool(step.get("gated", ctx.manifest.hf_gated))
    if gated and not ctx.hf_token:
        raise AddonError(
            f"{repo} is a gated model and needs a Hugging Face access token.",
            hint=f"Accept the license at https://huggingface.co/{repo} and enter your token in Settings.",
        )
    args = {
        "repo": repo,
        "revision": step.get("revision") or None,
        "allow_patterns": step.get("allow_patterns") or None,
        "ignore_patterns": step.get("ignore_patterns") or None,
        "repo_type": step.get("repo_type", "model"),
        "local_dir": str(ctx.resolve(step["local_dir"])) if step.get("local_dir") else None,
        "token": ctx.hf_token or None,
    }
    title = step.get("title") or f"Downloading {repo}"
    state: dict[str, Any] = {"error": None}

    def on_line(line: str) -> None:
        if not line.startswith("@@"):
            return
        tag, _, payload = line[2:].partition(" ")
        try:
            data = json.loads(payload) if payload else {}
        except ValueError:
            return
        if tag == "progress":
            gb = float(data.get("bytes", 0)) / 1e9
            progress(float(data.get("frac", 0.0)), f"{title}: {gb:.2f} GB")
        elif tag == "total":
            progress(0.0, f"{title}: {float(data.get('bytes', 0)) / 1e9:.1f} GB in {data.get('files')} files")
        elif tag == "error":
            state["error"] = data

    res = run_process(
        [str(ctx.python), "-c", HF_SNIPPET],
        env=ctx.venv_env(),
        cwd=ctx.addon_dir,
        log_sink=ctx.sink,
        on_line=on_line,
        cancel=ctx.cancel,
        title=title,
        stdin_data=json.dumps(args) + "\n",
    )
    if res.ok:
        progress(1.0, f"{repo} downloaded")
        return
    err = state["error"] or {}
    etype, message = str(err.get("type", "")), str(err.get("message", "")) or _uv_error_line(res.tail)
    if etype in ("GatedRepoError", "RepositoryNotFoundError") or re.search(r"\b(401|403)\b", message):
        raise AddonError(
            f"Access to the Hugging Face model {repo} was denied.",
            hint=f"Sign in at https://huggingface.co/{repo}, accept the license, create an access token "
            "(read) and enter it in Settings.",
        )
    raise AddonError(f"Downloading {repo} failed: {message[:300]}", hint=error_hint(message, res.tail_text())
                     or INSTALL_LOG_HINT)


def step_check(ctx: InstallContext, step: dict, progress: ProgressReporter) -> None:
    """Verify the install: ``worker --check`` (venv) or launch each tool (binary)."""
    progress(0.1, f"Verifying {ctx.manifest.title}")
    if ctx.manifest.kind == "binary":
        from ..core.errors import AddonNotInstalled
        from .manager import resolve_binary, tool_command  # local import: avoid a cycle

        for name, spec in ctx.manifest.binaries.items():
            try:
                exe = resolve_binary(ctx.manifest, name, base_dir=ctx.addon_dir)
            except AddonNotInstalled as e:
                raise AddonError(f"{name} was not found after installing {ctx.manifest.title}.",
                                 hint=ctx.manifest.notes or e.hint) from e
            res = run_process([*tool_command(exe), *spec.get("check_args", ["--help"])],
                              env=native_tool_env(exe), log_sink=ctx.sink, cancel=ctx.cancel,
                              timeout=120, title=f"{name} check")
            if res.returncode in (3221225781, -1073741515, 3221225794, -1073741502):
                raise AddonError(
                    f"{name} cannot start: a required system DLL is missing.",
                    hint="Install the latest Microsoft Visual C++ Redistributable (x64) and retry.",
                )
            if res.returncode != 0 and not res.tail:
                raise AddonError(f"{name} did not start (exit code {res.returncode}).",
                                 hint="See install.log; reinstalling the add-on may help.")
        progress(1.0, "Tools verified")
        return
    _require_venv(ctx)
    worker = ctx.manifest.worker_path()
    if worker is None or not worker.is_file():
        raise AddonError(f"Worker script for {ctx.manifest.title} is missing.", hint="Reinstall VR Photo Studio.")
    args = ["--check", *([] if ctx.check_gpu else ["--no-gpu"]), *[str(a) for a in ctx.expand(step.get("args", []))]]
    info = run_worker(
        ctx.python, worker, "", {}, {}, ctx.addon_dir,
        env=ctx.venv_env(), cwd=worker_cwd(ctx.manifest, ctx.addon_dir), log_sink=ctx.sink,
        progress=progress, cancel=ctx.cancel, timeout=1800, title=f"{ctx.manifest.title} check",
        args=args, send_request=False,
    )
    ctx.sink.write("check result: " + json.dumps(info, default=str))
    progress(1.0, "Verified")


def step_worker(ctx: InstallContext, step: dict, progress: ProgressReporter) -> None:
    """Run the worker with extra flags (e.g. ``--prefetch`` to download weights)."""
    _require_venv(ctx)
    worker = ctx.manifest.worker_path()
    if worker is None or not worker.is_file():
        raise AddonError(f"Worker script for {ctx.manifest.title} is missing.", hint="Reinstall VR Photo Studio.")
    run_worker(
        ctx.python, worker, "", {}, {}, ctx.addon_dir,
        env=ctx.venv_env(), cwd=worker_cwd(ctx.manifest, ctx.addon_dir), log_sink=ctx.sink,
        progress=progress, cancel=ctx.cancel, timeout=6 * 3600,
        title=step.get("title") or ctx.manifest.title,
        args=[str(a) for a in ctx.expand(step["args"])], send_request=False,
    )


def worker_cwd(manifest: AddonManifest, addon_dir: Path) -> Path:
    if manifest.cwd:
        p = Path(_expand(manifest.cwd, {"addon_dir": str(addon_dir), "src": str(addon_dir / "src")}))
        p = p if p.is_absolute() else addon_dir / p
        if p.is_dir():
            return p
    return addon_dir


STEP_RUNNERS: dict[str, StepFn] = {
    "venv": step_venv,
    "pip": step_pip,
    "pip_requirements": step_pip_requirements,
    "github_archive": step_github_archive,
    "archive": step_archive,
    "hf_download": step_hf_download,
    "check": step_check,
    "worker": step_worker,
}


def step_title(step: dict) -> str:
    if step.get("title"):
        return str(step["title"])
    t = step.get("type")
    return {
        "venv": "Creating Python environment",
        "pip": "Installing Python packages",
        "pip_requirements": "Installing Python packages",
        "github_archive": f"Downloading {step.get('repo', 'source code')}",
        "archive": f"Downloading {_download_name(str(step.get('url', '')), 'files')}",
        "hf_download": f"Downloading {step.get('repo', 'model')}",
        "check": "Verifying installation",
        "worker": "Preparing models",
    }.get(str(t), str(t))


def step_weight(step: dict) -> float:
    if "weight" in step:
        return max(0.01, float(step["weight"]))
    w = _DEFAULT_WEIGHTS.get(str(step.get("type")), 1.0)
    if step.get("type") == "pip" and any(str(p).lower().startswith(("torch", "xformers")) for p in step.get("packages", [])):
        w = 20.0
    if step.get("type") == "hf_download" and "size_gb" in step:
        w = max(2.0, 2.0 * float(step["size_gb"]))
    if step.get("type") == "archive" and "size_mb" in step:
        w = max(1.0, float(step["size_mb"]) / 50.0)
    return w


def resume_point(manifest: AddonManifest, steps_done: list[str]) -> tuple[int, list[str]]:
    """Index of the first step to run and the keys of the completed prefix.

    Only the unbroken prefix of completed steps counts: a step recorded as done *after*
    the first missing / changed one ran against an environment that is about to be
    rebuilt, so its record is stale (the caller must drop it from ``state.json``).
    """
    keys = [step_key(i, s) for i, s in enumerate(manifest.steps)]
    done = set(steps_done)
    start = 0
    while start < len(keys) and keys[start] in done:
        start += 1
    return start, keys[:start]


def run_steps(
    ctx: InstallContext,
    steps_done: list[str],
    *,
    progress: ProgressReporter,
    on_step_done: Callable[[str], None],
) -> list[str]:
    """Run the manifest steps, resuming after the completed prefix of ``steps_done``.

    Calls ``on_step_done(key)`` after each successful step (the caller persists it).
    Optional steps that fail are logged and skipped.  Returns the warnings.
    """
    steps = ctx.manifest.steps
    keys = [step_key(i, s) for i, s in enumerate(steps)]
    start, _prefix = resume_point(ctx.manifest, steps_done)
    if start == len(steps):  # everything installed already: only re-verify
        todo = [i for i, s in enumerate(steps) if s["type"] == "check"]
        ctx.sink.write("All steps already done; re-running the verification only")
    else:
        todo = list(range(start, len(steps)))
        if start:
            ctx.sink.write(f"Resuming at step {start + 1} of {len(steps)} (earlier steps already done)")
    weights = [step_weight(s) if step_applies(s) else 0.0 for s in steps]
    remaining = sum(weights[i] for i in todo) or 1.0
    acc = 0.0
    warnings: list[str] = []
    for i in todo:
        step = steps[i]
        if ctx.cancel is not None:
            ctx.cancel.check()
        title = step_title(step)
        lo, hi = acc / remaining, (acc + weights[i]) / remaining
        acc += weights[i]
        sub = progress.sub(lo, hi)
        if not step_applies(step):
            ctx.sink.write(f"[step {i + 1}/{len(steps)}] {title}: skipped on {sys.platform}")
            on_step_done(keys[i])
            continue
        ctx.sink.write(f"[step {i + 1}/{len(steps)}] {title}")
        sub(0.0, title)
        t0 = time.monotonic()
        try:
            STEP_RUNNERS[step["type"]](ctx, step, sub)
        except Cancelled:
            ctx.sink.write(f"[step {i + 1}] cancelled")
            raise
        except AddonError as e:
            if step.get("optional"):
                msg = f"Optional step '{title}' failed: {e}"
                ctx.sink.write("WARNING: " + msg)
                log.warning(msg)
                warnings.append(msg)
                on_step_done(keys[i])
                continue
            ctx.sink.write(f"[step {i + 1}] FAILED: {e}" + (f" ({e.hint})" if e.hint else ""))
            raise
        except OSError as e:
            if step.get("optional"):
                warnings.append(f"Optional step '{title}' failed: {e}")
                on_step_done(keys[i])
                continue
            ctx.sink.write(f"[step {i + 1}] FAILED: {e}")
            raise AddonError(f"{title} failed: {e}", hint=error_hint(str(e))) from e
        ctx.sink.write(f"[step {i + 1}] done in {time.monotonic() - t0:.1f}s")
        sub(1.0, title)
        on_step_done(keys[i])
    return warnings


__all__ = [
    "HF_SNIPPET",
    "INSTALL_LOG_HINT",
    "PYTHON_ENV",
    "STEP_RUNNERS",
    "InstallContext",
    "find_managed_python",
    "run_steps",
    "step_applies",
    "step_key",
    "require_uv",
    "resume_point",
    "step_title",
    "step_weight",
    "uv_version",
    "worker_cwd",
    "worker_env_extra",
]
