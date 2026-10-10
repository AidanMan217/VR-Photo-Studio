"""Make the build helpers in ``packaging/`` importable for these tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[1]
ROOT = PACKAGING.parent
if str(PACKAGING) not in sys.path:
    sys.path.insert(0, str(PACKAGING))


@pytest.fixture
def root() -> Path:
    return ROOT


@pytest.fixture
def packaging_dir() -> Path:
    return PACKAGING
