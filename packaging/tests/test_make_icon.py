"""Tests for packaging/make_icon.py (application icon)."""

from __future__ import annotations

import io

import make_icon
import pytest
from PIL import Image


@pytest.mark.parametrize("size", [16, 24, 32, 48, 256])
def test_draw_logo(size):
    img = make_icon.draw_logo(size)
    assert img.size == (size, size)
    assert img.mode == "RGBA"
    px = img.load()
    c = size // 2
    assert px[c, c][3] == 255, "centre must be opaque"
    if size >= 32:
        assert px[0, 0][3] < 40, "rounded corners must be (nearly) transparent"


def test_logo_has_white_headset_and_dark_lenses():
    img = make_icon.draw_logo(256).convert("RGB")
    # headset body between the lenses (nose bridge area, above the cut-out)
    r, g, b = img.getpixel((128, 100))
    assert min(r, g, b) > 200
    # inside the left lens: not white
    r, g, b = img.getpixel((80, 150))
    assert min(r, g, b) < 200


def test_draw_logo_deterministic():
    a = io.BytesIO()
    b = io.BytesIO()
    make_icon.draw_logo(64).save(a, "PNG")
    make_icon.draw_logo(64).save(b, "PNG")
    assert a.getvalue() == b.getvalue()


def test_draw_logo_rejects_tiny():
    with pytest.raises(ValueError):
        make_icon.draw_logo(4)


def test_write_ico_all_sizes(tmp_path):
    p = make_icon.write_ico(tmp_path / "icon.ico")
    with Image.open(p) as im:
        assert im.format == "ICO"
        sizes = {s[0] for s in im.info["sizes"]}
    assert sizes == set(make_icon.ICO_SIZES)
    assert 16 in sizes and 256 in sizes


def test_write_ico_custom_sizes_filters_invalid(tmp_path):
    p = make_icon.write_ico(tmp_path / "x.ico", [16, 32, 512, 4])
    with Image.open(p) as im:
        assert {s[0] for s in im.info["sizes"]} == {16, 32}


def test_write_ico_no_valid_sizes(tmp_path):
    with pytest.raises(ValueError):
        make_icon.write_ico(tmp_path / "x.ico", [1000])


def test_main_writes_ico_and_png(tmp_path, capsys):
    ico = tmp_path / "a" / "icon.ico"
    png = tmp_path / "b" / "logo.png"
    assert (
        make_icon.main(["--out", str(ico), "--png", str(png), "--png-size", "128", "--sizes", "16", "48"])
        == 0
    )
    with Image.open(png) as im:
        assert im.size == (128, 128)
    assert ico.is_file()
    assert "wrote" in capsys.readouterr().out
