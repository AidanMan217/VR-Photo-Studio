"""core.imaging regressions: EXIF FOV (F6/R7), huge photos (R1), full disk (R4), ICC (R8)."""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
from _icc_profile import rgb_profile
from PIL import Image

from vrphoto.core import imaging
from vrphoto.core.errors import InputError


def _jpeg_with_f35(path, w, h, f35=26, orientation=None):
    im = Image.fromarray(np.full((h, w, 3), 128, np.uint8))
    ex = Image.Exif()
    ex.get_ifd(0x8769)[0xA405] = f35
    if orientation:
        ex[0x0112] = orientation
    im.save(path, "JPEG", exif=ex.tobytes())
    return path


# ------------------------------------------------------------------ F6 / R7: 35 mm-equivalent FOV
def test_exif_hfov_uses_35mm_diagonal(tmp_path):
    land = imaging.exif_hfov(_jpeg_with_f35(tmp_path / "l.jpg", 400, 300))
    port = imaging.exif_hfov(_jpeg_with_f35(tmp_path / "p.jpg", 300, 400))
    # 4:3 phone photo, 26 mm equivalent: 67.3 x 53.0 deg (CIPA DC-008 diagonal definition)
    assert land == pytest.approx(67.30, abs=0.05)
    assert port == pytest.approx(53.06, abs=0.05)


def test_exif_hfov_same_focal_px_for_both_orientations(tmp_path):
    """One camera must give one focal length in pixels whether held landscape or portrait."""
    land = imaging.exif_hfov(_jpeg_with_f35(tmp_path / "l.jpg", 400, 300))
    rot = imaging.exif_hfov(_jpeg_with_f35(tmp_path / "o.jpg", 400, 300, orientation=6))
    port = imaging.exif_hfov(_jpeg_with_f35(tmp_path / "p.jpg", 300, 400))
    assert rot == pytest.approx(port)
    f_land = 200 / math.tan(math.radians(land) / 2)
    f_port = 150 / math.tan(math.radians(port) / 2)
    assert f_land == pytest.approx(f_port, rel=1e-6)


def test_hfov_from_35mm_3x2_matches_classic_36mm():
    # for a 3:2 frame the diagonal definition equals the classic 36 mm width formula
    assert imaging.hfov_from_35mm(50.0, 6000, 4000) == pytest.approx(
        math.degrees(2 * math.atan(36.0 / 100.0)), abs=0.01
    )
    with pytest.raises(ValueError):
        imaging.hfov_from_35mm(0.0, 10, 10)


def test_exif_hfov_does_not_decode_pixels(tmp_path, monkeypatch):
    p = _jpeg_with_f35(tmp_path / "l.jpg", 400, 300)

    def no_load(self, *a, **k):  # pragma: no cover - would mean a full decode
        raise AssertionError("exif_hfov must only read the header")

    monkeypatch.setattr(Image.Image, "load", no_load)
    monkeypatch.setattr("PIL.ImageFile.ImageFile.load", no_load)
    assert imaging.exif_hfov(p) == pytest.approx(67.30, abs=0.05)


def test_exif_hfov_closes_the_file(tmp_path):
    """The header-only read must not leave the photo open (a lingering handle locks it on Windows)."""
    import gc
    import os
    import warnings

    p = _jpeg_with_f35(tmp_path / "l.jpg", 400, 300)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert imaging.exif_hfov(p) is not None
        assert imaging.exif_hfov(_jpeg_with_f35(tmp_path / "n.jpg", 40, 30, f35=0)) is None
        gc.collect()
    assert not [w for w in caught if issubclass(w.category, ResourceWarning)]
    os.replace(p, tmp_path / "moved.jpg")  # not locked


def test_exif_hfov_missing_and_unreadable(tmp_path):
    p = tmp_path / "plain.png"
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(p)
    assert imaging.exif_hfov(p) is None
    with pytest.raises(InputError):
        imaging.exif_hfov(tmp_path / "missing.jpg")


# ------------------------------------------------------------------ R1: 200 MP photos
def test_pillow_pixel_limit_allows_200mp_photos():
    assert Image.MAX_IMAGE_PIXELS is None or Image.MAX_IMAGE_PIXELS >= 16320 * 12240


def test_too_large_photo_gets_plain_message(tmp_path, monkeypatch):
    p = tmp_path / "huge.png"
    Image.fromarray(np.zeros((40, 50, 3), np.uint8)).save(p)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 500)  # 2000 px > 2x limit -> bomb error
    with pytest.raises(InputError) as ei:
        imaging.load_image(p)
    msg = str(ei.value)
    assert "too large" in msg and "decompression" not in msg.lower() and "DOS" not in msg
    assert ei.value.hint


def test_load_image_max_px_drafts_large_jpeg(tmp_path):
    rng = np.random.default_rng(1)
    p = tmp_path / "big.jpg"
    Image.fromarray(rng.integers(0, 255, (1200, 1600, 3), dtype=np.uint8)).save(p, quality=80)
    full = imaging.load_image(p)
    small = imaging.load_image(p, max_px=700)
    assert full.shape == (1200, 1600, 3)
    assert small.shape == (600, 800, 3)  # 1/2 scale: long side still >= 700
    assert imaging.load_image(p, max_px=5000).shape == full.shape
    # PNG ignores the hint (no reduced-scale decode)
    q = tmp_path / "big.png"
    Image.fromarray(full).save(q)
    assert imaging.load_image(q, max_px=700).shape == full.shape


def test_load_image_max_px_respects_exif_orientation(tmp_path):
    p = tmp_path / "rot.jpg"
    ex = Image.Exif()
    ex[0x0112] = 6
    Image.fromarray(np.zeros((800, 1200, 3), np.uint8)).save(p, exif=ex.tobytes())
    assert imaging.load_image(p, max_px=500).shape == (600, 400, 3)


# ------------------------------------------------------------------ R4: full disk / atomic save
def test_save_image_disk_full_raises_and_leaves_no_partial_file(tmp_path, monkeypatch):
    from vrphoto.core import fsutil

    real_open = open
    dest = tmp_path / "out_LR.jpg"

    class ShortWriter:
        """Simulates ENOSPC: Python's buffered writer raises on a short/failed write."""

        def __init__(self, fh):
            self.fh = fh

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.fh.close()

        def write(self, data):
            self.fh.write(bytes(data)[: len(data) // 3])
            raise OSError(28, "No space left on device")

        def flush(self):
            pass

        def fileno(self):
            return self.fh.fileno()

    def fake_open(path, mode="r", *a, **k):
        fh = real_open(path, mode, *a, **k)
        return ShortWriter(fh) if "x" in mode else fh

    monkeypatch.setattr("builtins.open", fake_open)
    img = np.random.default_rng(0).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    with pytest.raises(OSError):
        imaging.save_image(dest, img)
    monkeypatch.undo()
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []
    assert fsutil  # module used by save_image


def test_save_image_atomic_replace_and_formats(tmp_path):
    img = np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    for name in ("a.jpg", "a.png", "a.webp", "a.bmp", "a.tif"):
        p = imaging.save_image(tmp_path / name, img)
        with Image.open(p) as im:
            im.load()
            assert im.size == (80, 60)
    # overwriting keeps a valid file and leaves no temp files
    imaging.save_image(tmp_path / "a.jpg", img[::-1].copy(), xmp=imaging.gpano_xmp(80, 60, 80, 60, 0, 0))
    with Image.open(tmp_path / "a.jpg") as im:
        assert b"GPano" in im.info.get("xmp", b"")
    assert sorted(x.name for x in tmp_path.iterdir()) == sorted(
        ["a.jpg", "a.png", "a.webp", "a.bmp", "a.tif"]
    )
    with pytest.raises(ValueError):
        imaging.save_image(tmp_path / "a.unknownext", img)


def test_save_image_falls_back_without_optimize(tmp_path, monkeypatch):
    calls = []
    real = imaging._encode

    def flaky(im, fmt, **kw):
        calls.append(dict(kw))
        if kw.get("optimize"):
            raise OSError("Suspension not allowed here")
        return real(im, fmt, **kw)

    monkeypatch.setattr(imaging, "_encode", flaky)
    imaging.save_image(tmp_path / "x.jpg", np.zeros((10, 10, 3), np.uint8))
    assert len(calls) == 2 and "optimize" not in calls[1]
    assert (tmp_path / "x.jpg").stat().st_size > 0


# ------------------------------------------------------------------ R8: ICC colour profiles
P3_RED = (180, 60, 60)


def _p3_file(path, rgba=False, fmt=None):
    icc = rgb_profile("Display P3 (test)")
    shape = (6, 6, 4 if rgba else 3)
    a = np.zeros(shape, np.uint8)
    a[..., :3] = P3_RED
    if rgba:
        a[..., 3] = 100
    Image.fromarray(a).save(path, fmt, icc_profile=icc, **({"quality": 95} if path.suffix == ".jpg" else {}))
    return path


@pytest.mark.parametrize("name", ["p3.png", "p3.jpg"])
def test_load_image_converts_display_p3_to_srgb(tmp_path, name):
    px = imaging.load_image(_p3_file(tmp_path / name))[2, 2].astype(int)
    # P3 colours are more saturated than the same code values in sRGB
    assert px[0] > P3_RED[0] + 8 and px[1] < P3_RED[1] - 8


def test_load_image_icc_keeps_alpha(tmp_path):
    out = imaging.load_image(_p3_file(tmp_path / "p3a.png", rgba=True), with_alpha=True)
    assert out.shape[2] == 4 and int(out[0, 0, 3]) == 100
    assert int(out[0, 0, 0]) > P3_RED[0] + 8


def test_load_image_srgb_profile_unchanged(tmp_path):
    from PIL import ImageCms

    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    p = tmp_path / "s.png"
    Image.fromarray(np.full((4, 4, 3), P3_RED, np.uint8)).save(p, icc_profile=icc)
    assert tuple(imaging.load_image(p)[0, 0]) == P3_RED


def test_broken_icc_profile_is_ignored(tmp_path):
    p = tmp_path / "bad.png"
    Image.fromarray(np.full((4, 4, 3), P3_RED, np.uint8)).save(p, icc_profile=b"not a profile")
    assert tuple(imaging.load_image(p)[0, 0]) == P3_RED


def test_image_from_bytes_converts_icc(tmp_path):
    data = _p3_file(tmp_path / "p3.png").read_bytes()
    px = imaging.image_from_bytes(data)[0, 0].astype(int)
    assert px[0] > P3_RED[0] + 8


def test_to_srgb_public_helper(tmp_path):
    with Image.open(_p3_file(tmp_path / "p3.png")) as im:
        im.load()
        out = imaging.to_srgb(im)
    assert out.mode == "RGB" and "icc_profile" not in out.info
    plain = Image.new("RGB", (2, 2), P3_RED)
    assert imaging.to_srgb(plain) is plain
    assert io  # keep import used
