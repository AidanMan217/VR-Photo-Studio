"""Model registry: consistency, URL building, lock-file overlay."""

from __future__ import annotations

import json

import pytest

from vrphoto.core.errors import ModelUnavailable
from vrphoto.models import registry
from vrphoto.models.registry import (
    BASE_MODELS,
    MODELS,
    ModelSpec,
    apply_lock,
    env_var_for,
    file_url,
    get_spec,
    hf_url,
    list_specs,
)

CONTRACT_IDS = {
    "da2-small",
    "da2-base",
    "da2-large",
    "moge2-vitl",
    "da3-mono-large",
    "lama",
    "migan",
    "birefnet-lite",
    "birefnet-portrait",
    "isnet-general",
    "u2net",
}


def test_contract_ids_present():
    assert CONTRACT_IDS <= set(MODELS)


@pytest.mark.parametrize("spec", list(BASE_MODELS.values()), ids=lambda s: s.id)
def test_every_spec_is_complete(spec: ModelSpec):
    assert spec.id and spec.title
    assert spec.task in registry.TASKS
    assert spec.license, f"{spec.id} has no license"
    assert spec.files and all(f and not f.startswith("/") and ".." not in f for f in spec.files)
    assert spec.files[0].endswith(".onnx")
    assert spec.repo or spec.url, "needs a repo or a direct URL"
    assert spec.revision
    assert spec.size_mb > 0
    for f, h in spec.sha256.items():
        assert f in spec.files
        algo, _, hexd = h.rpartition(":")
        assert algo in ("", "md5", "sha256")
        assert all(c in "0123456789abcdef" for c in hexd)
    if spec.input_size is not None:
        assert len(spec.input_size) == 2 and all(v > 0 for v in spec.input_size)
    # file_url works for every file
    for f in spec.files:
        assert file_url(spec, f).startswith("https://")


def test_licence_flags():
    assert get_spec("da2-small").license == "Apache-2.0" and not get_spec("da2-small").noncommercial
    assert get_spec("da2-base").noncommercial and get_spec("da2-large").noncommercial
    assert "NC" in get_spec("da2-large").license
    assert get_spec("lama").dml_ok is False
    assert get_spec("lama").input_size == (512, 512)
    assert get_spec("birefnet-lite").input_size == (1024, 1024)
    assert get_spec("da3-mono-large").experimental
    assert get_spec("moge2-vitl").license == "MIT"


def test_get_spec_unknown_raises_with_hint():
    with pytest.raises(ModelUnavailable) as ei:
        get_spec("nope")
    assert "nope" in str(ei.value)
    assert "da2-small" in (ei.value.hint or "")


def test_list_specs_by_task():
    depth = list_specs("depth")
    assert {s.id for s in depth} >= {"da2-small", "moge2-vitl", "da3-mono-large"}
    assert all(s.task == "depth" for s in depth)
    assert {s.id for s in list_specs("inpaint")} >= {"lama", "migan"}
    assert len(list_specs()) == len(MODELS)
    assert list_specs("bogus") == []


def test_hf_url_building(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert (
        hf_url("onnx-community/depth-anything-v2-small", "main", "onnx/model.onnx")
        == "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
    )
    # special characters are quoted, slashes in paths kept, revision fully quoted
    assert hf_url("a/b", "refs/pr/1", "dir/my file+1.onnx").endswith(
        "/a/b/resolve/refs%2Fpr%2F1/dir/my%20file%2B1.onnx"
    )
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.example/")
    assert hf_url("a/b", "abc", "m.onnx") == "https://hf-mirror.example/a/b/resolve/abc/m.onnx"


def test_file_url_override_and_hf(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    spec = get_spec("u2net")
    assert file_url(spec, "u2net.onnx") == spec.url
    assert "github.com/danielgatis/rembg/releases/download" in spec.url
    assert get_spec("birefnet-portrait").url.endswith("/BiRefNet-portrait-epoch_150.onnx")
    da2 = get_spec("da2-small")
    assert file_url(da2, "onnx/model_fp16.onnx").endswith("/resolve/main/onnx/model_fp16.onnx")
    base = ModelSpec("x", "depth", "X", "", "v1", ("a.onnx", "a.onnx_data"), url="https://h.example/dl/")
    assert file_url(base, "a.onnx_data") == "https://h.example/dl/a.onnx_data"
    single = ModelSpec("y", "depth", "Y", "", "v1", ("a.onnx", "b.bin"), url="https://h.example/a.onnx")
    with pytest.raises(ModelUnavailable):
        file_url(single, "b.bin")
    with pytest.raises(ModelUnavailable):
        file_url(ModelSpec("z", "depth", "Z", "", "v1", ("a.onnx",)), "a.onnx")


def test_env_var_names():
    assert env_var_for("da2-small") == "VRPHOTO_MODEL_DA2_SMALL"
    assert env_var_for("moge2-vitl") == "VRPHOTO_MODEL_MOGE2_VITL"
    assert env_var_for("x.y-z") == "VRPHOTO_MODEL_X_Y_Z"


def test_spec_hashable_and_properties():
    s = get_spec("da2-small")
    assert {s: 1}[s] == 1
    assert s.entry == "onnx/model.onnx"
    assert s.is_hf and not get_spec("u2net").is_hf
    assert not s.pinned  # revision "main" until pin_models runs
    assert "huggingface.co" in s.page_url or s.homepage


def test_apply_lock_overlays_pins():
    lock = {
        "schema": 1,
        "models": {
            "da2-small": {
                "revision": "a" * 40,
                "files": ["onnx/model.onnx"],
                "sha256": {"onnx/model.onnx": "AB" * 32},
                "sizes": {"onnx/model.onnx": 99_000_000},
                "size_mb": 99.0,
                "unknown_key": 1,
            },
            "moge2-vitl": {"files": ["onnx/moge.onnx", "onnx/moge.onnx_data"]},
            "not-a-model": {"revision": "x"},
        },
    }
    out = apply_lock(BASE_MODELS, lock)
    s = out["da2-small"]
    assert s.revision == "a" * 40
    assert s.sha256 == {"onnx/model.onnx": "ab" * 32}  # lower-cased
    assert s.sizes == {"onnx/model.onnx": 99_000_000}
    assert s.pinned
    assert out["moge2-vitl"].files == ("onnx/moge.onnx", "onnx/moge.onnx_data")
    assert out["moge2-vitl"].license == "MIT"  # untouched fields kept
    assert "not-a-model" not in out
    assert BASE_MODELS["da2-small"].revision == "main"  # base table not mutated


@pytest.mark.parametrize(
    "entry",
    [
        {"revision": ""},
        {"files": []},
        {"files": "model.onnx"},
        {"sha256": {"a": 5}},
        {"sizes": {"a": -1}},
        {"size_mb": "big"},
        "not-an-object",
    ],
)
def test_apply_lock_rejects_malformed_entries(entry):
    out = apply_lock(BASE_MODELS, {"schema": 1, "models": {"da2-small": entry}})
    assert out["da2-small"] == BASE_MODELS["da2-small"]


def test_apply_lock_rejects_bad_documents():
    assert apply_lock(BASE_MODELS, {"schema": 99, "models": {"da2-small": {"revision": "x"}}}) == BASE_MODELS
    assert apply_lock(BASE_MODELS, {"models": []}) == BASE_MODELS
    assert apply_lock(BASE_MODELS, []) == BASE_MODELS  # type: ignore[arg-type]


def test_reload_registry_from_file(tmp_path, monkeypatch):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"schema": 1, "models": {"lama": {"revision": "deadbeef"}}}), encoding="utf-8")
    try:
        registry.reload_registry(lock)
        assert get_spec("lama").revision == "deadbeef"
        monkeypatch.setenv("VRPHOTO_MODELS_LOCK", str(lock))
        assert registry.default_lock_path() == lock
        # corrupt file -> built-in values
        lock.write_text("{not json", encoding="utf-8")
        registry.reload_registry(lock)
        assert get_spec("lama").revision == BASE_MODELS["lama"].revision
    finally:
        monkeypatch.delenv("VRPHOTO_MODELS_LOCK", raising=False)
        registry.reload_registry()


def test_bundled_lock_file_is_valid():
    path = registry.default_lock_path()
    assert path.is_file()
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["schema"] == registry.LOCK_SCHEMA
    assert isinstance(doc["models"], dict)
    assert set(doc["models"]) <= set(BASE_MODELS)


def test_needs_license_ack():
    from vrphoto.core.config import get_settings

    assert registry.needs_license_ack("da2-large")
    assert not registry.needs_license_ack("da2-small")
    get_settings().accept_license("da2-large")
    assert not registry.needs_license_ack("da2-large")
