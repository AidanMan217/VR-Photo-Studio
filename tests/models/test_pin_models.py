"""scripts/pin_models.py with a fake Hugging Face / GitHub API."""

from __future__ import annotations

import json

import pytest

from vrphoto.models.registry import BASE_MODELS, apply_lock

SHA = "0123456789abcdef0123456789abcdef01234567"


def _lfs(path, size, oid):
    return {"type": "file", "path": path, "size": 134, "lfs": {"oid": oid, "size": size, "pointerSize": 134}}


def make_api(mod, monkeypatch, missing=()):
    monkeypatch.setenv("HF_ENDPOINT", "https://hf.test")
    h = "ab" * 32
    trees = {
        "onnx-community/depth-anything-v2-small": {
            "onnx": [
                _lfs("onnx/model.onnx", 99_000_000, h),
                _lfs("onnx/model_fp16.onnx", 50_000_000, "cd" * 32),
                {"type": "directory", "path": "onnx/sub"},
            ]
        },
        "Ruicheng/moge-2-vitl-normal-onnx": {
            "": [
                {"type": "file", "path": "README.md", "size": 1000},
                _lfs("moge-2-vitl-normal.onnx", 1_300_000_000, "ef" * 32),
                _lfs("moge-2-vitl-normal.onnx_data", 10, "aa" * 32),
            ]
        },
    }

    def fetch(url):
        if url.startswith("https://hf.test/api/models/"):
            rest = url[len("https://hf.test/api/models/") :]
            for repo, folders in trees.items():
                if rest.startswith(repo + "/revision/"):
                    if repo in missing:
                        raise mod.PinError(f"GET {url} -> HTTP 404")
                    return {"sha": SHA}
                if rest.startswith(f"{repo}/tree/{SHA}"):
                    folder = rest[len(f"{repo}/tree/{SHA}") :].lstrip("/")
                    return folders.get(folder, [])
            raise mod.PinError(f"GET {url} -> HTTP 404")
        if url == "https://api.github.com/repos/danielgatis/rembg/releases/tags/v0.0.0":
            return {
                "assets": [
                    {"name": "u2net.onnx", "size": 175997641, "digest": "sha256:" + "12" * 32},
                    {"name": "isnet-general-use.onnx", "size": 178648008},
                ]
            }
        raise mod.PinError(f"GET {url} -> HTTP 404")

    return fetch


def test_pin_hf_resolves_files_sizes_and_hashes(pin_models_mod, monkeypatch):
    fetch = make_api(pin_models_mod, monkeypatch)
    entry = pin_models_mod.pin_hf(BASE_MODELS["da2-small"], fetch)
    assert entry["revision"] == SHA
    assert entry["files"] == ["onnx/model.onnx"]
    assert entry["fp16_file"] == "onnx/model_fp16.onnx"
    assert entry["sizes"] == {"onnx/model.onnx": 99_000_000, "onnx/model_fp16.onnx": 50_000_000}
    assert entry["sha256"]["onnx/model.onnx"] == "ab" * 32
    assert entry["size_mb"] == 99.0


def test_pin_hf_resolves_todo_file_name_and_external_data(pin_models_mod, monkeypatch):
    fetch = make_api(pin_models_mod, monkeypatch)
    entry = pin_models_mod.pin_hf(BASE_MODELS["moge2-vitl"], fetch)
    assert entry["files"] == ["moge-2-vitl-normal.onnx", "moge-2-vitl-normal.onnx_data"]
    assert entry["fp16_file"] is None
    assert set(entry["sha256"]) == set(entry["files"])


def test_pin_github_release(pin_models_mod, monkeypatch):
    fetch = make_api(pin_models_mod, monkeypatch)
    u2 = pin_models_mod.pin_github(BASE_MODELS["u2net"], fetch)
    assert u2 == {
        "revision": "v0.0.0",
        "sizes": {"u2net.onnx": 175997641},
        "sha256": {"u2net.onnx": "12" * 32},
        "size_mb": 176.0,
    }
    isnet = pin_models_mod.pin_github(BASE_MODELS["isnet-general"], fetch)
    assert isnet["sha256"]["isnet-general-use.onnx"].startswith("md5:")  # no digest -> keep md5
    with pytest.raises(pin_models_mod.PinError):
        pin_models_mod.pin_github(BASE_MODELS["birefnet-portrait"], fetch)  # asset missing


def test_build_lock_and_overlay(pin_models_mod, monkeypatch):
    fetch = make_api(pin_models_mod, monkeypatch)
    doc, errors = pin_models_mod.build_lock(BASE_MODELS, fetch, ids=["da2-small", "u2net", "bogus"])
    assert errors == ["unknown model id 'bogus'"]
    assert set(doc["models"]) == {"da2-small", "u2net"}
    merged = apply_lock(BASE_MODELS, json.loads(json.dumps(doc)))
    assert merged["da2-small"].revision == SHA and merged["da2-small"].pinned
    assert merged["u2net"].sizes == {"u2net.onnx": 175997641}


def test_main_write_and_check(pin_models_mod, monkeypatch, tmp_path):
    fetch = make_api(pin_models_mod, monkeypatch)
    lock = tmp_path / "lock.json"
    assert pin_models_mod.main(["--lock", str(lock), "--models", "da2-small,u2net"], fetch=fetch) == 0
    doc = json.loads(lock.read_text(encoding="utf-8"))
    assert set(doc["models"]) == {"da2-small", "u2net"}
    assert (
        pin_models_mod.main(["--lock", str(lock), "--check", "--models", "da2-small,u2net"], fetch=fetch) == 0
    )
    # upstream changes -> --check fails
    doc["models"]["da2-small"]["revision"] = "f" * 40
    lock.write_text(json.dumps(doc), encoding="utf-8")
    assert pin_models_mod.main(["--lock", str(lock), "--check", "--models", "da2-small"], fetch=fetch) == 1


def test_failures_keep_old_entries(pin_models_mod, monkeypatch, tmp_path):
    fetch = make_api(pin_models_mod, monkeypatch, missing=("onnx-community/depth-anything-v2-small",))
    old = {"schema": 1, "models": {"da2-small": {"revision": "old"}}}
    doc, errors = pin_models_mod.build_lock(BASE_MODELS, fetch, ids=["da2-small"], old=old)
    assert errors and doc["models"]["da2-small"] == {"revision": "old"}


def test_diff_locks(pin_models_mod):
    a = {"models": {"x": {"revision": "1"}, "y": {"revision": "2"}}}
    b = {"models": {"x": {"revision": "3"}, "z": {"revision": "4"}}}
    diffs = pin_models_mod.diff_locks(a, b)
    assert "x.revision: '1' -> '3'" in diffs
    assert any(d.startswith("y:") for d in diffs) and any(d.startswith("z:") for d in diffs)
    assert pin_models_mod.diff_locks(a, a) == []


def test_a_commit_pinned_url_model_keeps_its_table_pins(pin_models_mod):
    """YuNet's URL is a Git LFS file at an exact opencv_zoo commit, not a GitHub release: the
    weekly job keeps the registry table's revision / checksum instead of failing."""
    spec = BASE_MODELS["yunet"]
    assert spec.url and spec.pinned and "/releases/download/" not in spec.url

    def no_fetch(url):
        raise AssertionError(f"nothing to fetch for a pinned URL model: {url}")

    out = pin_models_mod.pin_spec(spec, no_fetch)
    assert out["revision"] == spec.revision and out["sha256"] == spec.sha256 and out["sizes"] == spec.sizes
    doc, errors = pin_models_mod.build_lock(BASE_MODELS, no_fetch, ids=["yunet"])
    assert errors == [] and doc["models"]["yunet"]["revision"] == spec.revision
