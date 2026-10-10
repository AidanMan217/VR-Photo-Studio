"""Checks of the GitHub Actions workflows (.github/workflows/*.yml).

Text checks always run; structural checks use PyYAML when it is installed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

WF_DIR = Path(__file__).resolve().parents[2] / ".github" / "workflows"
WORKFLOWS = ("ci.yml", "build.yml", "models.yml", "addons.yml")

#: Latest stable major versions, checked with `git ls-remote --tags` on 2026-10-10.
EXPECTED_MAJORS = {
    "actions/checkout": "v7",
    "actions/setup-python": "v7",
    "actions/upload-artifact": "v7",
    "actions/download-artifact": "v8",
    "actions/setup-node": "v7",
    "actions/cache/restore": "v6",
    "actions/cache/save": "v6",
    "actions/cache": "v6",
}
#: Third-party actions are pinned to a full commit SHA (with the tag in a comment).
PINNED_THIRD_PARTY = {
    "astral-sh/setup-uv": "v10.2.0",
    "softprops/action-gh-release": "v3.0.3",
}


def _text(name: str) -> str:
    return (WF_DIR / name).read_text(encoding="utf-8")


def _uses() -> list[tuple[str, str, str, str]]:
    out = []
    for name in WORKFLOWS:
        for m in re.finditer(r"uses:\s*([\w./-]+)@([\w.-]+)(?:\s*#\s*(\S+))?", _text(name)):
            out.append((name, m.group(1), m.group(2), m.group(3) or ""))
    return out


@pytest.fixture(scope="module")
def yaml():
    return pytest.importorskip("yaml")


def _load(yaml, name: str) -> dict:
    data = yaml.safe_load(_text(name))
    # YAML 1.1 parses the bare key "on" as boolean True
    if True in data and "on" not in data:
        data["on"] = data.pop(True)
    return data


@pytest.mark.parametrize("name", WORKFLOWS)
def test_workflow_exists_and_has_no_tabs(name):
    text = _text(name)
    assert "\t" not in text
    assert text.endswith("\n")


def test_action_versions():
    found = _uses()
    assert found
    for wf, action, ref, comment in found:
        if action in PINNED_THIRD_PARTY:
            assert re.fullmatch(r"[0-9a-f]{40}", ref), f"{wf}: {action} must be SHA-pinned"
            assert comment == PINNED_THIRD_PARTY[action], f"{wf}: {action} comment should name the tag"
        else:
            assert action in EXPECTED_MAJORS, f"{wf}: unexpected action {action}"
            assert ref == EXPECTED_MAJORS[action], f"{wf}: {action}@{ref}, expected {EXPECTED_MAJORS[action]}"


def test_no_floating_branch_refs():
    for wf, action, ref, _ in _uses():
        assert ref not in ("main", "master", "latest"), f"{wf}: {action}@{ref}"


def test_ci_text():
    t = _text("ci.yml")
    assert 'pytest -m "not models"' in t and "packaging/tests" in t
    assert "QT_QPA_PLATFORM: offscreen" in t
    for lib in ("libegl1", "libgl1", "libxkbcommon0", "libfontconfig1", "libdbus-1-3"):
        assert lib in t
    assert 'pip install -e ".[dev]"' in t
    assert "npx playwright install --with-deps chromium" in t and "npm test" in t and "npm ci" in t


def test_build_text():
    t = _text("build.yml")
    assert "vrphoto-cli.exe --selftest --json" in t
    assert r"Inno Setup 6\ISCC.exe" in t and "choco install innosetup" in t
    assert "packaging/vrphoto.spec" in t and "packaging/make_icon.py" in t
    assert "archive: false" in t


def test_models_text():
    t = _text("models.yml")
    assert "scripts/pin_models.py --check" in t
    assert 'VRPHOTO_REAL_MODELS: "1"' in t
    assert "pytest -m models" in t
    assert "hashFiles('src/vrphoto/models/models.lock.json')" in t


def test_addons_text():
    t = _text("addons.yml")
    assert "addons install" in t and "--no-check-gpu" in t
    assert "addons check" in t
    assert "continue-on-error: ${{ matrix.experimental }}" in t
    assert "packaging/ci/addon_matrix.py" in t


# ----------------------------------------------------------------------- structure (PyYAML)


@pytest.mark.parametrize("name", WORKFLOWS)
def test_parses(yaml, name):
    data = _load(yaml, name)
    assert data["name"] and data["jobs"] and data["on"]
    assert data.get("permissions", {}).get("contents") == "read"
    for job_id, job in data["jobs"].items():
        assert "runs-on" in job, job_id
        assert isinstance(job.get("timeout-minutes"), int), f"{name}:{job_id} needs timeout-minutes"
        assert job["steps"], job_id


def test_ci_structure(yaml):
    d = _load(yaml, "ci.yml")
    assert d["on"]["push"]["branches"] == ["**"]
    assert "pull_request" in d["on"]
    test = d["jobs"]["test"]
    assert set(test["strategy"]["matrix"]["os"]) == {"ubuntu-latest", "windows-latest"}
    assert d["env"]["PYTHON_VERSION"] == "3.12"
    assert test["strategy"]["fail-fast"] is False
    assert d["jobs"]["viewer"]["runs-on"] == "ubuntu-latest"


def test_build_structure(yaml):
    d = _load(yaml, "build.yml")
    on = d["on"]
    assert "main" in on["push"]["branches"] and "claude/**" in on["push"]["branches"]
    assert on["push"]["tags"] == ["v*"]
    assert "workflow_dispatch" in on
    win = d["jobs"]["windows"]
    assert win["runs-on"] == "windows-latest"
    rel = d["jobs"]["release"]
    assert rel["needs"] == "windows"
    assert "refs/tags/v" in rel["if"]
    assert rel["permissions"] == {"contents": "write"}
    names = [s.get("name", "") for s in win["steps"]]
    order = [
        next(i for i, n in enumerate(names) if n.startswith(prefix))
        for prefix in (
            "Draw icon",
            "Bundle with PyInstaller",
            "Self-test",
            "Build installer",
            "Silent per-user",
        )
    ]
    assert order == sorted(order), "build steps out of order"


def test_models_structure(yaml):
    d = _load(yaml, "models.yml")
    assert "workflow_dispatch" in d["on"] and d["on"]["schedule"][0]["cron"]
    job = d["jobs"]["models"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["env"]["VRPHOTO_REAL_MODELS"] == "1"


def test_addons_structure(yaml):
    d = _load(yaml, "addons.yml")
    inp = d["on"]["workflow_dispatch"]["inputs"]["addon"]
    assert inp["default"] == "all"
    job = d["jobs"]["install"]
    assert job["runs-on"] == "windows-latest"
    assert job["strategy"]["fail-fast"] is False
    assert "fromJSON(needs.plan.outputs.matrix)" in job["strategy"]["matrix"]


# ----------------------------------------------------------------------- regressions


def test_viewer_job_installs_python_for_the_fixture_server():
    """Playwright's webServer runs tests/viewer/serve_fixtures.py, which needs vrphoto + numpy."""
    t = _text("ci.yml")
    viewer = t[t.index("  viewer:") :]
    assert "actions/setup-python@" in viewer
    assert "pip install -e ." in viewer
    assert viewer.index("pip install -e .") < viewer.index("npm test")
    root = WF_DIR.parents[1]
    config = (root / "tests" / "viewer" / "playwright.config.mjs").read_text(encoding="utf-8")
    assert "serve_fixtures.py" in config  # still the reason the job needs Python


def test_models_job_points_every_test_cache_variable_at_the_cache():
    """Real-model tests pick their shared download folder from env vars; all must be the cache."""
    root = WF_DIR.parents[1]
    used = set()
    for path in (root / "tests").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "VRPHOTO_REAL_MODELS" in text:
            used |= set(re.findall(r"VRPHOTO_[A-Z_]*(?:HOME|DIR)\b", text))
    used.discard("VRPHOTO_HOME")  # the per-test isolated data root, not a cache
    assert used, "expected real-model tests that use a shared download folder"
    t = _text("models.yml")
    for var in sorted(used):
        assert re.search(rf"^\s+{var}: \$\{{\{{ github\.workspace \}}\}}/\.models-cache$", t, re.M), var
    assert "path: .models-cache/models" in t


def test_build_uninstall_waits_for_the_uninstaller_to_finish():
    t = _text("build.yml")
    step = t[t.index("Silent uninstall removes the app") :]
    step = step[: step.index("- name:", 1)]
    assert "(Test-Path $dir)" in step  # waits until the install folder is gone
    assert "$p.ExitCode -ne 0" in step
    assert step.index("Test-Path $dir") < step.index('ci-marker.txt"))) {')


def test_version_is_read_through_the_consistency_check():
    """``spec_helpers.py version`` fails on an __init__/pyproject mismatch; the step must stop then."""
    t = _text("build.yml")
    step = t[t.index("- name: Determine version") :]
    step = step[: step.index("- name:", 1)]
    assert "python packaging/spec_helpers.py version" in step
    assert step.index("$PSNativeCommandUseErrorActionPreference = $true") < step.index(
        "spec_helpers.py version"
    )


def test_ci_runs_the_frozen_bundle_smoke_test(yaml):
    d = _load(yaml, "ci.yml")
    job = d["jobs"]["bundle"]
    assert job["runs-on"] == "ubuntu-latest"
    assert job["env"]["VRPHOTO_TEST_PYINSTALLER"] == "1"
    assert job["env"]["VRPHOTO_TEST_FROZEN_SELFTEST"] == "1"
    run = "\n".join(s.get("run", "") for s in job["steps"])
    assert "packaging/tests/test_frozen_build.py" in run and 'pip install -e ".[dev]"' in run
    assert "libegl1" in run  # the frozen self-test imports Qt
