"""Add-on manifests: TOML files in ``vrphoto/addons/manifests/`` describing how to install
and run one add-on.  Loaded with :mod:`tomllib` and validated strictly so a broken
manifest fails at load time (and in the unit tests) rather than half-way through a
multi-gigabyte install.

Top-level keys::

    id, title, description, version, experimental, license, noncommercial,
    region_restricted (ISO 3166-1 alpha-2 codes where the license does NOT apply),
    min_vram_gb, min_ram_gb (system memory; advisory, shown in the UI), disk_gb,
    kind ("venv" | "binary"), python ("3.11"), hf_gated, homepage,
    tasks (list), worker (file in addons/workers; required for kind="venv"),
    notes, platforms (sys.platform values; empty = all), cwd (worker cwd, relative to
    the add-on folder), entry ("module:function" for kind="binary" tasks),
    [env] (extra worker environment), [binaries.<name>] (kind="binary" tools),
    [[steps]] (ordered install steps, see :data:`STEP_SCHEMAS`).

String values in steps / env may use the placeholders ``{addon_dir}``, ``{src}``
(= ``{addon_dir}/src``), ``{venv}``, ``{python}`` (venv interpreter) and ``{manifests}``.
"""

from __future__ import annotations

import hashlib
import re
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from ..core.config import resource_path
from ..core.errors import AddonError

KINDS = ("venv", "binary")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
COUNTRY_RE = re.compile(r"^[A-Z]{2}$")
PLATFORMS = ("win32", "linux", "darwin")
TORCH_BACKENDS_HINT = "e.g. 'cu128', 'cu126', 'cpu' or 'auto'"

# key -> (type, required)
_COMMON_STEP_KEYS: dict[str, tuple[type | tuple[type, ...], bool]] = {
    "type": (str, True),
    "title": (str, False),
    "platforms": (list, False),
    "optional": (bool, False),
    "weight": ((int, float), False),
}
_PIP_INDEX_KEYS: dict[str, tuple[type | tuple[type, ...], bool]] = {
    "index_url": (str, False),
    "extra_index_url": (list, False),
    "torch_backend": (str, False),
    "index_strategy": (str, False),
    "no_deps": (bool, False),
    "no_build_isolation": (bool, False),
    "reinstall": (bool, False),
    "pre": (bool, False),
}

#: Allowed keys per step type (in addition to the common keys).
STEP_SCHEMAS: dict[str, dict[str, tuple[type | tuple[type, ...], bool]]] = {
    "venv": {"seed": (bool, False)},
    "pip": {"packages": (list, True), **_PIP_INDEX_KEYS},
    "pip_requirements": {"file": (str, True), **_PIP_INDEX_KEYS},
    "github_archive": {
        "repo": (str, True),
        "ref": (str, True),
        "dest": (str, True),
        "pip_install": (bool, False),
        "editable": (bool, False),
        "sha256": (str, False),
        "url": (str, False),
        **_PIP_INDEX_KEYS,
    },
    "archive": {
        "url": (str, True),
        "dest": (str, True),
        "sha256": (str, False),
        "extract": (bool, False),
        "strip": (bool, False),
        "size_mb": ((int, float), False),
    },
    "hf_download": {
        "repo": (str, True),
        "revision": (str, False),
        "allow_patterns": (list, False),
        "ignore_patterns": (list, False),
        "local_dir": (str, False),
        "repo_type": (str, False),
        "gated": (bool, False),
        "size_gb": ((int, float), False),
    },
    "check": {"args": (list, False)},
    "worker": {"args": (list, True)},
}
STEP_TYPES = tuple(STEP_SCHEMAS)

_TOP_KEYS: dict[str, tuple[type | tuple[type, ...], bool]] = {
    "id": (str, True),
    "title": (str, True),
    "description": (str, True),
    "version": (str, False),
    "experimental": (bool, False),
    "license": (str, True),
    "noncommercial": (bool, False),
    "region_restricted": (list, False),
    "min_vram_gb": ((int, float), False),
    "min_ram_gb": ((int, float), False),
    "disk_gb": ((int, float), False),
    "kind": (str, True),
    "python": (str, False),
    "hf_gated": (bool, False),
    "homepage": (str, False),
    "tasks": (list, True),
    "worker": (str, False),
    "notes": (str, False),
    "platforms": (list, False),
    "cwd": (str, False),
    "entry": (str, False),
    "env": (dict, False),
    "binaries": (dict, False),
    "steps": (list, True),
}


def workers_dir() -> Path:
    """Folder holding the worker scripts (works from source and when frozen)."""
    return resource_path("addons", "workers")


def manifests_dir() -> Path:
    """Folder holding the bundled manifests."""
    return resource_path("addons", "manifests")


@dataclass
class AddonManifest:
    """One add-on, as described by its TOML manifest (see module docstring)."""

    id: str
    title: str
    description: str
    experimental: bool
    license: str
    noncommercial: bool
    region_restricted: list[str]
    min_vram_gb: float
    disk_gb: float
    python: str
    kind: str
    steps: list[dict]
    worker: str | None
    tasks: list[str]
    hf_gated: bool
    homepage: str
    version: str = "1"
    notes: str = ""
    platforms: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    entry: str = ""
    binaries: dict[str, dict] = field(default_factory=dict)
    path: Path | None = None
    sha256: str = ""
    #: System memory (GB) the add-on needs; advisory (the UI shows it, the worker enforces its own limit).
    min_ram_gb: float = 0.0

    # ----------------------------------------------------------------- derived
    @property
    def supported(self) -> bool:
        """True if the add-on can be installed on this operating system."""
        return not self.platforms or sys.platform in self.platforms

    @property
    def manifest_dir(self) -> Path:
        return self.path.parent if self.path is not None else manifests_dir()

    def worker_path(self) -> Path | None:
        """Absolute path of the worker script (bundled workers first, then next to the manifest)."""
        if not self.worker:
            return None
        cand = workers_dir() / self.worker
        if cand.is_file():
            return cand
        local = self.manifest_dir / self.worker
        return local if local.is_file() else cand

    def notices(self) -> list[str]:
        """Human-readable license / hardware warnings for the UI."""
        out: list[str] = []
        if self.experimental:
            out.append("Experimental: may fail to install or produce poor results.")
        if self.noncommercial:
            out.append(f"Non-commercial use only ({self.license}).")
        if self.region_restricted:
            out.append(
                "The license does not apply in: "
                + _region_text(self.region_restricted)
                + ". Do not install or use it there."
            )
        if self.hf_gated:
            out.append("Needs a free Hugging Face account and access token (gated model).")
        if self.min_vram_gb:
            gpu = "an NVIDIA GPU" if self.kind == "venv" else "a GPU"
            out.append(f"Needs {gpu} with at least {self.min_vram_gb:g} GB VRAM.")
        if self.min_ram_gb:
            out.append(f"Needs at least {self.min_ram_gb:g} GB of system memory (RAM).")
        if self.disk_gb:
            out.append(f"Download/disk space: about {self.disk_gb:g} GB.")
        if not self.supported:
            out.append("Not available on this operating system.")
        return out

    def requires_license_ack(self) -> bool:
        """True if the UI must ask the user to accept the license before installing."""
        return bool(self.noncommercial or self.region_restricted or self.hf_gated)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able summary (for the CLI / GUI)."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "version": self.version,
            "experimental": self.experimental,
            "license": self.license,
            "noncommercial": self.noncommercial,
            "region_restricted": list(self.region_restricted),
            "min_vram_gb": self.min_vram_gb,
            "min_ram_gb": self.min_ram_gb,
            "disk_gb": self.disk_gb,
            "kind": self.kind,
            "python": self.python,
            "tasks": list(self.tasks),
            "hf_gated": self.hf_gated,
            "homepage": self.homepage,
            "supported": self.supported,
            "notices": self.notices(),
        }


_EU = {
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
    "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
}


def _region_text(codes: list[str]) -> str:
    s = set(codes)
    parts: list[str] = []
    if _EU <= s:
        parts.append("the European Union")
        s -= _EU
    names = {"GB": "United Kingdom", "KR": "South Korea", "US": "United States", "CN": "China"}
    parts.extend(names.get(c, c) for c in sorted(s))
    return ", ".join(parts)


# --------------------------------------------------------------------------- loading
def _err(path: Path, msg: str) -> AddonError:
    return AddonError(f"Invalid add-on manifest {path.name}: {msg}")


def _check_keys(path: Path, where: str, data: dict, schema: dict) -> None:
    unknown = set(data) - set(schema)
    if unknown:
        raise _err(path, f"{where}: unknown key(s) {', '.join(sorted(unknown))}")
    for key, (typ, required) in schema.items():
        if key not in data:
            if required:
                raise _err(path, f"{where}: missing required key '{key}'")
            continue
        value = data[key]
        if typ in (int, float) or typ == (int, float):
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:
            ok = isinstance(value, typ)
        if not ok:
            raise _err(path, f"{where}: '{key}' has the wrong type ({type(value).__name__})")


def _str_list(path: Path, where: str, value: Any, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) and v.strip() for v in value):
        raise _err(path, f"{where} must be a list of non-empty strings")
    if nonempty and not value:
        raise _err(path, f"{where} must not be empty")
    return [v.strip() for v in value]


def _safe_relpath(path: Path, where: str, value: str) -> None:
    v = value.replace("\\", "/")
    if v.startswith("{"):
        # placeholder-based (expanded later): the rest must still not climb out
        v = re.sub(r"^\{[a-z_]+\}/?", "", v)
        if ".." in PurePosixPath(v).parts:
            raise _err(path, f"{where} must stay inside the add-on folder: {value!r}")
        return
    p = PurePosixPath(v)
    if p.is_absolute() or ".." in p.parts or (len(v) > 1 and v[1] == ":"):
        raise _err(path, f"{where} must be a relative path inside the add-on folder: {value!r}")


def _validate_step(path: Path, i: int, step: Any) -> dict:
    where = f"steps[{i}]"
    if not isinstance(step, dict):
        raise _err(path, f"{where} must be a table")
    stype = step.get("type")
    if stype not in STEP_SCHEMAS:
        raise _err(path, f"{where}: unknown step type {stype!r} (known: {', '.join(STEP_TYPES)})")
    _check_keys(path, f"{where} ({stype})", step, {**_COMMON_STEP_KEYS, **STEP_SCHEMAS[stype]})
    if "platforms" in step:
        plats = _str_list(path, f"{where}.platforms", step["platforms"], nonempty=True)
        bad = [p for p in plats if p not in PLATFORMS]
        if bad:
            raise _err(path, f"{where}.platforms: unknown platform(s) {bad} (use {', '.join(PLATFORMS)})")
    if "extra_index_url" in step:
        _str_list(path, f"{where}.extra_index_url", step["extra_index_url"])
    for key in ("index_url", "url"):
        if key in step and not str(step[key]).startswith(("https://", "http://")):
            raise _err(path, f"{where}.{key} must be an http(s) URL")
    if "sha256" in step and not SHA256_RE.match(step["sha256"]):
        raise _err(path, f"{where}.sha256 must be 64 hex characters")
    if stype == "pip":
        _str_list(path, f"{where}.packages", step["packages"], nonempty=True)
    elif stype == "github_archive":
        if not REPO_RE.match(step["repo"]):
            raise _err(path, f"{where}.repo must look like 'owner/name'")
        if not step["ref"].strip():
            raise _err(path, f"{where}.ref must not be empty")
        _safe_relpath(path, f"{where}.dest", step["dest"])
    elif stype == "archive":
        _safe_relpath(path, f"{where}.dest", step["dest"])
    elif stype == "hf_download":
        for key in ("allow_patterns", "ignore_patterns"):
            if key in step:
                _str_list(path, f"{where}.{key}", step[key])
        if "local_dir" in step:
            _safe_relpath(path, f"{where}.local_dir", step["local_dir"])
        if step.get("repo_type", "model") not in ("model", "dataset", "space"):
            raise _err(path, f"{where}.repo_type must be model, dataset or space")
    elif stype in ("check", "worker"):
        if "args" in step:
            _str_list(path, f"{where}.args", step["args"], nonempty=stype == "worker")
    return dict(step)


def _validate_binaries(path: Path, value: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    schema = {"patterns": (list, False), "path_names": (list, False), "check_args": (list, False),
              "env_var": (str, False)}
    for name, spec in value.items():
        if not isinstance(spec, dict):
            raise _err(path, f"binaries.{name} must be a table")
        _check_keys(path, f"binaries.{name}", spec, schema)
        for key in ("patterns", "path_names", "check_args"):
            if key in spec:
                _str_list(path, f"binaries.{name}.{key}", spec[key])
        if not spec.get("patterns") and not spec.get("path_names"):
            raise _err(path, f"binaries.{name} needs 'patterns' and/or 'path_names'")
        for pat in spec.get("patterns", []):
            _safe_relpath(path, f"binaries.{name}.patterns", pat)
        out[name] = dict(spec)
    return out


def parse_manifest(data: dict[str, Any], path: Path, *, sha256: str = "") -> AddonManifest:
    """Validate parsed TOML ``data`` and build an :class:`AddonManifest`."""
    if not isinstance(data, dict):
        raise _err(path, "top level must be a table")
    _check_keys(path, "top level", data, _TOP_KEYS)
    addon_id = data["id"].strip()
    if not ID_RE.match(addon_id):
        raise _err(path, f"id {addon_id!r} must be lowercase letters, digits, '-' or '_'")
    kind = data["kind"]
    if kind not in KINDS:
        raise _err(path, f"kind must be one of {', '.join(KINDS)}")
    tasks = _str_list(path, "tasks", data["tasks"], nonempty=True)
    region = _str_list(path, "region_restricted", data.get("region_restricted", []))
    bad = [c for c in region if not COUNTRY_RE.match(c) or c in ("EU", "UK")]
    if bad:
        raise _err(path, f"region_restricted must hold ISO 3166-1 alpha-2 country codes, e.g. GB not UK and "
                         f"the member states instead of EU (got {bad})")
    platforms = _str_list(path, "platforms", data.get("platforms", []))
    bad = [p for p in platforms if p not in PLATFORMS]
    if bad:
        raise _err(path, f"platforms: unknown platform(s) {bad}")
    for key in ("min_vram_gb", "min_ram_gb", "disk_gb"):
        if float(data.get(key, 0)) < 0:
            raise _err(path, f"{key} must not be negative")
    steps_raw = data["steps"]
    if not isinstance(steps_raw, list) or not steps_raw:
        raise _err(path, "steps must be a non-empty array of tables ([[steps]])")
    steps = [_validate_step(path, i, s) for i, s in enumerate(steps_raw)]
    python = str(data.get("python", "")).strip()
    worker = data.get("worker")
    if kind == "venv":
        if not re.match(r"^3\.\d{1,2}$", python):
            raise _err(path, "kind 'venv' needs python = \"3.X\"")
        if not worker:
            raise _err(path, "kind 'venv' needs a worker script")
        if steps[0]["type"] != "venv":
            raise _err(path, "the first step of a 'venv' add-on must be {type = \"venv\"}")
        if sum(1 for s in steps if s["type"] == "venv") != 1:
            raise _err(path, "exactly one venv step is allowed")
    else:
        venv_only = [s["type"] for s in steps if s["type"] in ("venv", "pip", "pip_requirements", "hf_download", "worker")]
        if venv_only:
            raise _err(path, f"kind 'binary' cannot use venv steps ({', '.join(sorted(set(venv_only)))})")
        if data.get("entry") and ":" not in data["entry"]:
            raise _err(path, "entry must look like 'package.module:function'")
    if worker is not None:
        if not isinstance(worker, str) or "/" in worker or "\\" in worker or not worker.endswith(".py"):
            raise _err(path, "worker must be a .py file name in vrphoto/addons/workers")
        if not ((workers_dir() / worker).is_file() or (path.parent / worker).is_file()):
            raise _err(path, f"worker script {worker} not found")
    env = data.get("env", {})
    if not all(isinstance(k, str) and isinstance(v, (str, int, float, bool)) for k, v in env.items()):
        raise _err(path, "[env] values must be strings or numbers")
    cwd = str(data.get("cwd", ""))
    if cwd:
        _safe_relpath(path, "cwd", cwd)
    binaries = _validate_binaries(path, data.get("binaries", {}))
    return AddonManifest(
        id=addon_id,
        title=data["title"].strip(),
        description=data["description"].strip(),
        experimental=bool(data.get("experimental", False)),
        license=data["license"].strip(),
        noncommercial=bool(data.get("noncommercial", False)),
        region_restricted=region,
        min_vram_gb=float(data.get("min_vram_gb", 0)),
        min_ram_gb=float(data.get("min_ram_gb", 0)),
        disk_gb=float(data.get("disk_gb", 0)),
        python=python,
        kind=kind,
        steps=steps,
        worker=worker,
        tasks=tasks,
        hf_gated=bool(data.get("hf_gated", False)),
        homepage=str(data.get("homepage", "")),
        version=str(data.get("version", "1")),
        notes=str(data.get("notes", "")).strip(),
        platforms=platforms,
        env={k: str(v) for k, v in env.items()},
        cwd=cwd,
        entry=str(data.get("entry", "")),
        binaries=binaries,
        path=path,
        sha256=sha256,
    )


def load_manifest(path: str | Path) -> AddonManifest:
    """Read and validate one manifest file (raises :class:`AddonError`)."""
    p = Path(path)
    try:
        raw = p.read_bytes()
    except OSError as e:
        raise AddonError(f"Cannot read add-on manifest {p.name}: {e}") from e
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise _err(p, str(e)) from e
    return parse_manifest(data, p, sha256=hashlib.sha256(raw).hexdigest())


__all__ = [
    "KINDS",
    "STEP_SCHEMAS",
    "STEP_TYPES",
    "AddonManifest",
    "load_manifest",
    "manifests_dir",
    "parse_manifest",
    "workers_dir",
]
