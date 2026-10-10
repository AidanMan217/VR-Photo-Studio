"""Output names: Quest player tokens, Windows sanitising, unique paths."""

from __future__ import annotations

import threading

import pytest

from vrphoto.core.errors import InputError
from vrphoto.pipeline import naming
from vrphoto.pipeline.naming import (
    MAX_NAME_CHARS,
    output_name,
    safe_base,
    sanitize_filename,
    split_token,
    token_for,
    unique_path,
)


@pytest.mark.parametrize(
    ("kind", "ext", "expected"),
    [
        ("sbs", "jpg", "beach_LR.jpg"),
        ("half_sbs", "jpg", "beach_LR.jpg"),
        ("tb", "png", "beach_TB.png"),
        ("half_tb", ".jpg", "beach_TB.jpg"),
        ("vr180", "jpg", "beach_180_LR.jpg"),
        ("vr180", "mp4", "beach_180_LR.mp4"),
        ("360_tb", "jpg", "beach_360_TB.jpg"),
        ("360", "jpg", "beach_360.jpg"),
        ("anaglyph", "jpg", "beach_anaglyph.jpg"),
        ("parallax", "mp4", "beach_parallax.mp4"),
        ("parallax_sbs", "mp4", "beach_parallax_LR.mp4"),
        ("parallax_vr180", "mp4", "beach_parallax_180_LR.mp4"),
        ("depth", "png", "beach_depth.png"),
        ("mono", "jpg", "beach.jpg"),
        ("SBS", "JPG", "beach_LR.jpg"),
        ("_trellis", "glb", "beach_trellis.glb"),
    ],
)
def test_quest_tokens(kind, ext, expected):
    assert output_name("beach", kind, ext) == expected


def test_every_kind_has_a_token_and_vr_tokens_are_suffixes():
    for kind in naming.KIND_TOKENS:
        name = output_name("x", kind, "jpg")
        assert name.startswith("x") and name.endswith(".jpg")
    # the projection tokens Quest players look for must END the stem
    assert output_name("a", "vr180", "mp4")[:-4].endswith("_180_LR")
    assert output_name("a", "360_tb", "mp4")[:-4].endswith("_360_TB")


def test_unknown_kind_raises():
    with pytest.raises(InputError):
        token_for("hologram")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('a<b>c:d"e/f\\g|h?i*j.jpg', "a_b_c_d_e_f_g_h_i_j.jpg"),
        ("CON", "_CON"),
        ("con.jpg", "_con.jpg"),
        ("nul.tar.gz", "_nul.tar.gz"),
        ("COM1", "_COM1"),
        ("lpt9.png", "_lpt9.png"),
        ("console.jpg", "console.jpg"),  # only exact device names are reserved
        ("trailing dots...", "trailing dots"),
        ("  spaced  out  .jpg", "spaced out .jpg"),
        ("tab\tand\nnewline", "tab and newline"),
        ("ctrl\x01char", "ctrl_char"),
        ("", "untitled"),
        ("...", "untitled"),
        ("Über straße.jpg", "Über straße.jpg"),
    ],
)
def test_sanitize_filename(raw, expected):
    assert sanitize_filename(raw) == expected


def test_reserved_base_names_get_escaped_in_outputs():
    assert output_name("CON", "sbs", "jpg") == "_CON_LR.jpg"
    assert output_name("aux", "mono", "png") == "_aux.png"
    assert output_name("my:photo?", "vr180", "jpg") == "my_photo__180_LR.jpg"


def test_long_names_are_shortened_but_keep_token_and_extension():
    base = "x" * 400
    name = output_name(base, "vr180", "jpg")
    assert len(name) <= MAX_NAME_CHARS
    assert name.endswith("_180_LR.jpg")
    long = sanitize_filename("y" * 300 + ".jpeg")
    assert len(long) <= MAX_NAME_CHARS and long.endswith(".jpeg")


def test_safe_base_keeps_dots_inside():
    assert safe_base("v1.2 final") == "v1.2 final"
    assert safe_base("bad/name") == "bad_name"


def test_split_token():
    assert split_token("beach_180_LR") == ("beach", "_180_LR")
    assert split_token("beach_360_TB") == ("beach", "_360_TB")
    assert split_token("beach_LR") == ("beach", "_LR")
    assert split_token("beach") == ("beach", "")
    assert split_token("_LR") == ("_LR", "")  # nothing before the token


def test_unique_path_counter_goes_before_token(tmp_path):
    p = tmp_path / "beach_180_LR.jpg"
    assert unique_path(p, reserve=False) == p
    p.write_bytes(b"x")
    p2 = unique_path(p, reserve=False)
    assert p2.name == "beach (2)_180_LR.jpg"
    p2.write_bytes(b"x")
    assert unique_path(p, reserve=False).name == "beach (3)_180_LR.jpg"
    # asking for an already numbered name continues the sequence
    assert unique_path(p2, reserve=False).name == "beach (3)_180_LR.jpg"


def test_unique_path_plain_and_no_extension(tmp_path):
    (tmp_path / "photo.png").write_bytes(b"x")
    assert unique_path(tmp_path / "photo.png", reserve=False).name == "photo (2).png"
    (tmp_path / "noext").write_bytes(b"x")
    assert unique_path(tmp_path / "noext", reserve=False).name == "noext (2)"


def test_unique_path_does_not_strip_year_like_numbers(tmp_path):
    (tmp_path / "Party (1999).jpg").write_bytes(b"x")
    assert unique_path(tmp_path / "Party (1999).jpg", reserve=False).name == "Party (1999) (2).jpg"


def test_unique_path_reservation_prevents_parallel_collisions(tmp_path):
    target = tmp_path / "shared_LR.jpg"
    got: list[str] = []
    lock = threading.Lock()

    def worker():
        p = unique_path(target)
        with lock:
            got.append(p.name)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(got)) == 8
    assert "shared_LR.jpg" in got and "shared (2)_LR.jpg" in got


def test_release_reserved_paths(tmp_path):
    target = tmp_path / "x_LR.jpg"
    a = unique_path(target)
    assert unique_path(target).name == "x (2)_LR.jpg"  # reserved, although nothing is written
    naming.release_reserved([a])
    assert unique_path(target, reserve=False) == a
    naming.release_reserved()
    assert unique_path(tmp_path / "x (2)_LR.jpg", reserve=False).name == "x (2)_LR.jpg"
