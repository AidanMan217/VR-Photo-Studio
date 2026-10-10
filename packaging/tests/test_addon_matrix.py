"""Tests for packaging/ci/addon_matrix.py (add-on CI matrix)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "ci" / "addon_matrix.py"
_spec = importlib.util.spec_from_file_location("addon_matrix", _PATH)
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


@pytest.fixture
def manifests(tmp_path) -> Path:
    d = tmp_path / "manifests"
    d.mkdir()
    (d / "capture.toml").write_text('id = "capture"\ntitle = "Capture to splat"\n', encoding="utf-8")
    (d / "trellis.toml").write_text('[addon]\nid = "trellis"\ntitle = "TRELLIS"\n', encoding="utf-8")
    (d / "hunyuan3d.toml").write_text('id = "hunyuan3d"\nexperimental = true\n', encoding="utf-8")
    (d / "sharp.toml").write_text('title = "SHARP"\n', encoding="utf-8")  # id from file name
    (d / "notes.txt").write_text("ignored", encoding="utf-8")
    return d


def test_read_manifests(manifests):
    got = M.read_manifests(manifests)
    assert [m["id"] for m in got] == ["capture", "sharp", "trellis", "hunyuan3d"]  # stable first
    by_id = {m["id"]: m for m in got}
    assert by_id["hunyuan3d"]["experimental"] is True
    assert by_id["trellis"]["experimental"] is False
    assert by_id["trellis"]["title"] == "TRELLIS"
    assert by_id["hunyuan3d"]["title"] == "hunyuan3d"


@pytest.mark.parametrize(
    ("selection", "ids"),
    [
        ("all", ["capture", "sharp", "trellis", "hunyuan3d"]),
        ("", ["capture", "sharp", "trellis", "hunyuan3d"]),
        ("trellis", ["trellis"]),
        ("sharp, capture", ["sharp", "capture"]),
        ("capture capture", ["capture"]),
    ],
)
def test_select(manifests, selection, ids):
    assert [m["id"] for m in M.build_matrix(selection, manifests)["include"]] == ids


def test_unknown_id(manifests):
    with pytest.raises(M.MatrixError, match="Unknown add-on id.*nope.*Known: capture"):
        M.build_matrix("nope", manifests)


def test_duplicate_id(manifests):
    (manifests / "trellis_copy.toml").write_text('id = "trellis"\n', encoding="utf-8")
    with pytest.raises(M.MatrixError, match="Duplicate"):
        M.read_manifests(manifests)


def test_bad_toml(manifests):
    (manifests / "broken.toml").write_text("id = \n", encoding="utf-8")
    with pytest.raises(M.MatrixError, match="broken.toml"):
        M.read_manifests(manifests)


def test_empty_and_missing_dirs(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(M.MatrixError, match="No add-on manifests"):
        M.build_matrix("all", tmp_path / "empty")
    with pytest.raises(M.MatrixError, match="not found"):
        M.read_manifests(tmp_path / "missing")


def test_main_github_output(manifests, tmp_path, capsys):
    out = tmp_path / "gh_output"
    out.write_text("other=1\n", encoding="utf-8")
    assert M.main(["capture,trellis", "--manifests", str(manifests), "--github-output", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "other=1"
    key, _, value = lines[1].partition("=")
    assert key == "matrix"
    assert [m["id"] for m in json.loads(value)["include"]] == ["capture", "trellis"]
    assert json.loads(capsys.readouterr().out)["include"][0]["id"] == "capture"


def test_main_error_exit_code(manifests, capsys):
    assert M.main(["nope", "--manifests", str(manifests)]) == 2
    assert "::error::" in capsys.readouterr().err


def test_real_manifest_dir_readable():
    if not M.MANIFEST_DIR.is_dir() or not any(M.MANIFEST_DIR.glob("*.toml")):
        pytest.skip("no add-on manifests in the source tree yet")
    got = M.read_manifests()
    assert got and all(m["id"] for m in got)
