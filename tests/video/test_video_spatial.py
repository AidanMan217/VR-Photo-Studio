"""Spherical / stereo MP4 metadata injection (algorithm K), verified with ffprobe/ffmpeg."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from video_helpers import (
    H,
    N,
    W,
    encode_clip,
    ffmpeg,
    ffmpeg_banner,
    ffprobe_side_data,
    make_tone,
    synth_frames,
    top_level_boxes,
)

from vrphoto.core.errors import EncodeError, InputError
from vrphoto.video import decode, spatial
from vrphoto.video.spatial import BOUNDS_180, inject, read

REPO = Path(__file__).resolve().parents[2]


def _side(path: Path, kind: str) -> dict:
    data = ffprobe_side_data(path)
    assert data is not None
    for sd in data:
        if sd.get("side_data_type") == kind:
            return sd
    raise AssertionError(f"no {kind!r} side data in {data}")


def _spherical_uuid_count(path: Path) -> int:
    from vrphoto.third_party.spatialmedia.metadata_utils import SPHERICAL_UUID_ID

    return path.read_bytes().count(SPHERICAL_UUID_ID)


def test_plain_file_has_no_metadata(h264_clip):
    info = read(h264_clip)
    assert info["spherical"] is False and info["stereo"] is None and info["coverage"] is None
    assert info["v1"] is None and info["v2"] is False


def test_inject_180_lr(h264_copy):
    before = list(decode.iter_frames(h264_copy))
    order = top_level_boxes(h264_copy)
    assert inject(h264_copy, coverage=180, stereo="lr") == h264_copy

    info = read(h264_copy)
    assert info["spherical"] is True and info["v2"] is True
    assert info["stereo"] == "lr" and info["coverage"] == 180
    assert info["projection"] == "equirectangular"
    assert info["bounds"] == {"top": 0, "bottom": 0, "left": BOUNDS_180, "right": BOUNDS_180}
    v1 = info["v1"]
    assert v1["Spherical"] == "true" and v1["ProjectionType"] == "equirectangular"
    assert v1["StereoMode"] == "left-right"
    eye = W // 2  # one eye of the side-by-side frame
    assert int(v1["CroppedAreaImageWidthPixels"]) == eye
    assert int(v1["CroppedAreaImageHeightPixels"]) == H
    assert int(v1["FullPanoWidthPixels"]) == 2 * eye
    assert int(v1["FullPanoHeightPixels"]) == H
    assert int(v1["CroppedAreaLeftPixels"]) == eye // 2
    assert int(v1["CroppedAreaTopPixels"]) == 0

    # ffmpeg's own demuxer sees the V2 boxes
    banner = ffmpeg_banner(h264_copy)
    assert "stereo3d: side by side" in banner
    assert "spherical: tiled equirectangular" in banner

    # still decodes, identical frames (chunk offsets were fixed), still faststart
    after = list(decode.iter_frames(h264_copy))
    assert len(after) == len(before) == N
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
    assert decode.probe(h264_copy).n_frames == N
    assert top_level_boxes(h264_copy) == order
    assert order.index(b"moov") < order.index(b"mdat")


def test_inject_180_lr_ffprobe(h264_copy):
    if ffprobe_side_data(h264_copy) is None:
        pytest.skip("ffprobe not installed")
    inject(h264_copy, coverage=180, stereo="lr")
    st = _side(h264_copy, "Stereo 3D")
    assert st["type"] == "side by side" and int(st.get("inverted", 0)) == 0
    sph = _side(h264_copy, "Spherical Mapping")
    assert sph["projection"] == "tiled equirectangular"
    # ffprobe prints the 0.32 bounds converted to pixels of the full pano: 25% each side
    assert abs(int(sph["bound_left"]) - W // 2) <= 2 and abs(int(sph["bound_right"]) - W // 2) <= 2
    assert int(sph["bound_left"]) + int(sph["bound_right"]) == W
    assert int(sph["bound_top"]) == 0 and int(sph["bound_bottom"]) == 0


def test_inject_360_tb_hevc(hevc_copy):
    before = list(decode.iter_frames(hevc_copy))
    inject(hevc_copy, coverage=360, stereo="tb")
    info = read(hevc_copy)
    assert info["spherical"] and info["stereo"] == "tb" and info["coverage"] == 360
    assert info["bounds"] == {"top": 0, "bottom": 0, "left": 0, "right": 0}
    assert info["v1"]["StereoMode"] == "top-bottom"
    assert "CroppedAreaImageWidthPixels" not in info["v1"]
    banner = ffmpeg_banner(hevc_copy)
    assert "stereo3d: top and bottom" in banner
    assert "spherical: equirectangular" in banner and "tiled" not in banner
    assert "hvc1" in banner
    after = list(decode.iter_frames(hevc_copy))
    assert len(after) == N and all(np.array_equal(a, b) for a, b in zip(before, after))
    if ffprobe_side_data(hevc_copy) is not None:
        assert _side(hevc_copy, "Stereo 3D")["type"] == "top and bottom"
        sph = _side(hevc_copy, "Spherical Mapping")
        assert sph["projection"] == "equirectangular"
        assert int(sph.get("bound_left", 0)) == 0


def test_inject_360_mono_and_180_tb_crop(h264_copy):
    inject(h264_copy, coverage=360, stereo="mono")
    info = read(h264_copy)
    assert info["spherical"] and info["stereo"] == "mono" and info["coverage"] == 360
    assert "stereo3d: 2D" in ffmpeg_banner(h264_copy)
    xml = spatial.spherical_xml(180, "tb", 2048, 2048)
    assert "<GSpherical:CroppedAreaImageWidthPixels>2048<" in xml
    assert "<GSpherical:CroppedAreaImageHeightPixels>1024<" in xml
    assert "<GSpherical:FullPanoWidthPixels>4096<" in xml
    assert "<GSpherical:CroppedAreaLeftPixels>1024<" in xml
    assert "<GSpherical:StereoMode>top-bottom<" in xml


def test_reinject_replaces_metadata(h264_copy):
    inject(h264_copy, coverage=180, stereo="lr")
    size1 = h264_copy.stat().st_size
    inject(h264_copy, coverage=180, stereo="lr")
    assert h264_copy.stat().st_size == size1  # idempotent: no duplicate boxes
    assert _spherical_uuid_count(h264_copy) == 1
    inject(h264_copy, coverage=360, stereo="tb")
    info = read(h264_copy)
    assert info["stereo"] == "tb" and info["coverage"] == 360
    assert _spherical_uuid_count(h264_copy) == 1
    assert len(list(decode.iter_frames(h264_copy))) == N


def test_non_faststart_file(tmp_path, h264_clip):
    """moov after mdat: offsets unchanged, layout preserved, frames identical."""
    slow = tmp_path / "slow.mp4"
    subprocess.run([ffmpeg(), "-v", "error", "-i", str(h264_clip), "-c", "copy", str(slow)], check=True)
    order = top_level_boxes(slow)
    assert order.index(b"mdat") < order.index(b"moov")
    before = list(decode.iter_frames(slow))
    inject(slow, coverage=180, stereo="lr")
    assert top_level_boxes(slow) == order
    after = list(decode.iter_frames(slow))
    assert len(after) == N and all(np.array_equal(a, b) for a, b in zip(before, after))
    assert read(slow)["coverage"] == 180


def test_file_with_audio_and_mov(tmp_path):
    tone = make_tone(tmp_path / "t.m4a", 3.0)
    out = encode_clip(
        tmp_path / "av_180_LR.mov", synth_frames(30, 128, 64), audio_from=tone, encoder="libx264"
    )
    inject(out, coverage=180, stereo="sbs")  # alias for lr
    info = decode.probe(out)
    assert info.has_audio and info.n_frames == 30
    assert read(out)["stereo"] == "lr"
    pcm = subprocess.run(
        [ffmpeg(), "-v", "error", "-i", str(out), "-map", "0:a:0", "-f", "s16le", "-"],
        capture_output=True,
        check=True,
    ).stdout
    assert len(pcm) > 10000


def test_invalid_arguments(tmp_path, h264_copy):
    with pytest.raises(InputError):
        inject(h264_copy, coverage=270, stereo="lr")
    with pytest.raises(InputError):
        inject(h264_copy, coverage=180, stereo="interlaced")
    with pytest.raises(InputError):
        inject(tmp_path / "missing.mp4", coverage=180, stereo="lr")
    mkv = tmp_path / "x.mkv"
    mkv.write_bytes(b"\x1a\x45\xdf\xa3")
    with pytest.raises(InputError):
        inject(mkv, coverage=360, stereo="mono")
    with pytest.raises(InputError):
        read(mkv)


def test_garbage_file_is_left_untouched(tmp_path, capsys):
    junk = tmp_path / "junk.mp4"
    payload = b"\x00\x00\x00\x08free" + b"\xff" * 64
    junk.write_bytes(payload)
    with pytest.raises(EncodeError):
        inject(junk, coverage=360, stereo="mono")
    assert junk.read_bytes() == payload
    assert not list(tmp_path.glob(".*spatial*"))
    assert capsys.readouterr().out == ""  # vendored library diagnostics go to logging, not stdout


def test_read_v1_only_file(h264_copy):
    """A file carrying only the V1 XML (older tools) is reported from the XML."""
    from vrphoto.third_party.spatialmedia import metadata_utils, mpeg

    xml = spatial.spherical_xml(180, "lr", W, H)
    tmp = h264_copy.with_name("v1.mp4")
    with open(h264_copy, "rb") as fh:
        mp4 = mpeg.load(fh)
        assert metadata_utils.mpeg4_add_spherical_xml_v1(mp4, fh, xml)
        with open(tmp, "wb") as out:
            mp4.save(fh, out)
    info = read(tmp)
    assert info["v2"] is False and info["spherical"] is True
    assert info["stereo"] == "lr" and info["coverage"] == 180
    assert len(list(decode.iter_frames(tmp))) == N


def test_vendored_library_metadata():
    root = REPO / "src" / "vrphoto" / "third_party" / "spatialmedia"
    txt = (root / "VENDORED.txt").read_text(encoding="utf-8")
    assert "github.com/google/spatial-media" in txt
    assert "Commit:       " in txt and "Commit date:" in txt
    assert (root / "LICENSE").read_text(encoding="utf-8").lstrip().startswith("Copyright")
    assert "Apache License" in (root / "LICENSE").read_text(encoding="utf-8")
    import vrphoto.third_party.spatialmedia as sm

    assert sm.mpeg.sv3d.EQUIBox(bounds=[0, 0, 1, 2]).bounds_right == 2
    assert "spatialmedia" not in sys.modules  # no absolute top-level package leaked


def test_vendor_script_check_passes():
    script = REPO / "scripts" / "vendor_spatialmedia.py"
    spec = importlib.util.spec_from_file_location("vendor_spatialmedia_test", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    assert mod.check() == 0
    # the transform is pure and rewrites imports + adds the print shim
    out = mod.transform("mpeg/sa3d.py", "import io\nfrom spatialmedia.mpeg import constants\nprint('x')\n")
    assert "from . import constants" in out and "from .._log import log_print as print" in out


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_inject_keeps_file_permissions(h264_copy):
    import os
    import stat

    os.chmod(h264_copy, 0o640)
    inject(h264_copy, coverage=180, stereo="lr")
    assert stat.S_IMODE(h264_copy.stat().st_mode) == 0o640
    assert read(h264_copy)["coverage"] == 180


def test_inject_after_two_pass_audio_mux(tmp_path):
    """The pipeline's VR180 flow: encode with audio (second mux pass) -> inject -> decode."""
    tone = make_tone(tmp_path / "t.m4a", 1.0)
    out = encode_clip(
        tmp_path / "x_180_LR.mp4", synth_frames(45, 128, 64), audio_from=tone, encoder="libx264"
    )
    inject(out, coverage=180, stereo="lr")
    info = decode.probe(out)
    assert info.n_frames == 45 and info.has_audio
    meta = read(out)
    assert meta["spherical"] and meta["stereo"] == "lr" and meta["coverage"] == 180
    boxes = top_level_boxes(out)
    assert boxes.index(b"moov") < boxes.index(b"mdat")
