"""Optional pytest entry point for the Playwright browser tests of the WebXR viewer.

The browser suite normally runs on its own (``cd tests/viewer && npx playwright test``, see
README.md).  Set ``VRPHOTO_BROWSER_TESTS=1`` to run it from pytest as well; it is skipped when
Node.js or the npm dependencies (``npm ci`` in tests/viewer) are missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

pytestmark = [pytest.mark.slow]


@pytest.mark.skipif(
    os.environ.get("VRPHOTO_BROWSER_TESTS") != "1", reason="set VRPHOTO_BROWSER_TESTS=1 to run"
)
@pytest.mark.timeout(1800)
def test_playwright_suite():
    npx = shutil.which("npx.cmd" if os.name == "nt" else "npx")
    if npx is None:
        pytest.skip("Node.js / npx not installed")
    if not (HERE / "node_modules" / "@playwright" / "test").is_dir():
        pytest.skip("run `npm ci` in tests/viewer first")
    proc = subprocess.run(
        [npx, "playwright", "test", "--reporter=line"],
        cwd=HERE,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=1700,
        check=False,
    )
    assert proc.returncode == 0, f"Playwright failed:\n{proc.stdout[-8000:]}\n{proc.stderr[-4000:]}"
