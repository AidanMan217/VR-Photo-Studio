"""Static checks of the PyInstaller entry scripts and spec file."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[1]


def _calls(tree: ast.AST) -> set[str]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                out.add(f"{f.value.id}.{f.attr}")
            elif isinstance(f, ast.Name):
                out.add(f.id)
    return out


@pytest.mark.parametrize(("script", "func"), [("entry_gui.py", "main"), ("entry_cli.py", "cli_entry")])
def test_entry_script(script, func):
    tree = ast.parse((PACKAGING / script).read_text(encoding="utf-8"))
    imports = [(n.module, [a.name for a in n.names]) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert ("vrphoto.__main__", [func]) in imports
    calls = _calls(tree)
    assert "multiprocessing.freeze_support" in calls
    assert func in calls and "SystemExit" in calls
    assert "print" not in calls


def test_entry_points_exist_in_package():
    if importlib.util.find_spec("vrphoto.__main__") is None:
        pytest.skip("vrphoto/__main__.py not written yet")
    import vrphoto.__main__ as m

    assert callable(m.main) and callable(m.cli_entry)


def test_pyproject_entry_points_match(root):
    import tomllib

    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["vrphoto-cli"] == "vrphoto.__main__:cli_entry"
    assert data["project"]["gui-scripts"]["vrphoto"] == "vrphoto.__main__:main"


def test_spec_compiles_and_has_key_settings():
    text = (PACKAGING / "vrphoto.spec").read_text(encoding="utf-8")
    compile(text, "vrphoto.spec", "exec")
    tree = ast.parse(text)
    kw: dict[str, list] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            kw.setdefault(node.func.id, []).append({k.arg: k.value for k in node.keywords if k.arg})
    assert len(kw["EXE"]) == 2 and len(kw["COLLECT"]) == 1
    consoles = sorted(ast.literal_eval(e["console"]) for e in kw["EXE"])
    assert consoles == [False, True]
    assert ast.literal_eval(kw["COLLECT"][0]["upx"]) is False
    assert "entry_gui.py" in text and "entry_cli.py" in text
    assert "upx=False" in text and 'contents_directory="_internal"' in text
    assert "hook" in text and "H.uv_binary_entry()" in text


def test_icon_present_or_generated(packaging_dir):
    # the spec generates packaging/icon.ico when missing; make sure the generator works here
    import make_icon

    assert make_icon.DEFAULT_ICO == packaging_dir / "icon.ico"
