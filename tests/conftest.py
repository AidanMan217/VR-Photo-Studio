"""Shared pytest fixtures: isolated data root and synthetic scenes with known geometry."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory, monkeypatch):
    """Every test gets its own VRPHOTO_HOME so nothing touches the real user profile."""
    home = tmp_path_factory.mktemp("vrphoto_home")
    monkeypatch.setenv("VRPHOTO_HOME", str(home))
    from vrphoto.core import config

    config.set_settings(None)
    yield home
    config.set_settings(None)


def _texture(h: int, w: int, seed: int) -> np.ndarray:
    """Deterministic high-frequency colour texture (good for warp tests)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    tex = np.kron(base, np.ones((8, 8, 1), dtype=np.uint8))[:h, :w]
    yy, xx = np.mgrid[0:h, 0:w]
    stripes = ((np.sin(xx / 3.0) + np.cos(yy / 5.0)) * 20).astype(np.int16)
    return np.clip(tex.astype(np.int16) + stripes[..., None], 0, 255).astype(np.uint8)


@dataclass
class TwoPlaneScene:
    """A near square (z = z_near) in front of a textured far plane (z = z_far).

    ``img`` is what a pinhole camera at the origin sees; ``inv_z`` is the exact inverse
    depth; ``far_rgb`` is the full far-plane texture (ground truth for look-behind);
    ``fg_mask`` marks the square.
    """

    img: np.ndarray
    inv_z: np.ndarray
    far_rgb: np.ndarray
    fg_mask: np.ndarray
    K: object
    z_near: float
    z_far: float


@pytest.fixture
def two_plane_scene() -> TwoPlaneScene:
    from vrphoto.core.types import Intrinsics

    h, w = 240, 320
    K = Intrinsics.from_hfov(w, h, 60.0)
    z_near, z_far = 1.5, 4.0
    far = _texture(h, w, 1)
    near = _texture(h, w, 2)
    mask = np.zeros((h, w), bool)
    mask[80:160, 120:200] = True
    img = far.copy()
    img[mask] = near[mask]
    inv_z = np.where(mask, 1.0 / z_near, 1.0 / z_far).astype(np.float32)
    return TwoPlaneScene(img, inv_z, far, mask, K, z_near, z_far)


@pytest.fixture
def gradient_photo(tmp_path) -> Path:
    """A 640x480 JPEG with EXIF FocalLengthIn35mmFilm = 26 mm (typical phone)."""
    from PIL import Image

    h, w = 480, 640
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx * 255 // w, yy * 255 // h, (xx + yy) * 255 // (w + h)], -1).astype(np.uint8)
    img[300:460, 250:390] = (200, 60, 60)  # a "subject" near the bottom
    pil = Image.fromarray(img)
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0xA405] = 26
    p = tmp_path / "photo.jpg"
    pil.save(p, "JPEG", quality=95, exif=exif)
    return p


@pytest.fixture
def photo_rgb() -> np.ndarray:
    """In-memory 320x240 RGB with a bright subject over a gradient background."""
    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx * 255 // w, yy * 255 // h, np.full_like(xx, 128)], -1).astype(np.uint8)
    img[120:220, 120:200] = (230, 80, 60)
    return img


@pytest.fixture
def person_alpha() -> np.ndarray:
    """Person-like silhouette alpha (head + torso + legs) on 256x192."""
    import cv2

    h, w = 256, 192
    a = np.zeros((h, w), np.uint8)
    cv2.circle(a, (96, 40), 22, 255, -1)  # head
    cv2.rectangle(a, (66, 62), (126, 150), 255, -1)  # torso
    cv2.rectangle(a, (46, 66), (64, 140), 255, -1)  # left arm
    cv2.rectangle(a, (128, 66), (146, 140), 255, -1)  # right arm
    cv2.rectangle(a, (70, 150), (92, 246), 255, -1)  # legs
    cv2.rectangle(a, (100, 150), (122, 246), 255, -1)
    return (a > 0).astype(np.float32)


@pytest.fixture
def checker_equirect() -> np.ndarray:
    """1024x512 equirect checkerboard with a coloured marker straight ahead (lon=0, lat=0)."""
    h, w = 512, 1024
    yy, xx = np.mgrid[0:h, 0:w]
    chk = (((xx // 32) + (yy // 32)) % 2 * 200 + 30).astype(np.uint8)
    img = np.stack([chk, chk, chk], -1)
    cx, cy = w // 2, h // 2
    img[cy - 6 : cy + 6, cx - 6 : cx + 6] = (255, 0, 0)
    return img


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    d = (a - b) ** 2
    if mask is not None:
        d = d[mask]
    mse = float(d.mean()) if d.size else 0.0
    return 99.0 if mse <= 1e-12 else 10 * math.log10(255.0**2 / mse)


@pytest.fixture
def psnr_fn():
    return psnr
