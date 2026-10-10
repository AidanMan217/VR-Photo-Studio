"""Video probing and frame decoding."""

from __future__ import annotations

import subprocess
import time

import numpy as np
import pytest
from video_helpers import H, N, W, encode_clip, ffmpeg, make_tone, synth_frames

from vrphoto.core.errors import Cancelled, EncodeError, InputError
from vrphoto.core.types import CancelToken
from vrphoto.video import decode
from vrphoto.video.decode import VideoInfo, iter_frames, parse_ffmpeg_info, probe


def test_probe(h264_clip):
    info = probe(h264_clip)
    assert isinstance(info, VideoInfo)
    assert (info.width, info.height, info.n_frames) == (W, H, N)
    assert info.fps == pytest.approx(30.0)
    assert info.duration == pytest.approx(N / 30.0, abs=0.05)
    assert info.has_audio is False


def test_probe_other_rate_and_audio(tmp_path):
    tone = make_tone(tmp_path / "t.m4a", 5.0)
    out = encode_clip(
        tmp_path / "a.mp4", synth_frames(24, 96, 48), fps=24.0, audio_from=tone, encoder="libx264"
    )
    info = probe(out)
    assert (info.width, info.height, info.n_frames) == (96, 48, 24)
    assert info.fps == pytest.approx(24.0)
    assert info.has_audio


def test_iter_frames_values(h264_clip):
    ref = synth_frames()
    frames = list(iter_frames(h264_clip))
    assert len(frames) == N
    assert all(f.shape == (H, W, 3) and f.dtype == np.uint8 for f in frames)
    assert frames[0] is not frames[1] and not np.shares_memory(frames[0], frames[1])
    for i in (0, 31, 63):
        assert np.abs(frames[i].astype(int) - ref[i].astype(int)).mean() < 3


def test_iter_frames_downscale(h264_clip):
    frames = list(iter_frames(h264_clip, max_long=64))
    assert len(frames) == N
    assert frames[0].shape == (32, 64, 3)
    # no-op when already small enough
    assert next(iter_frames(h264_clip, max_long=1000)).shape == (H, W, 3)


def test_iter_frames_time_range(h264_clip):
    frames = list(iter_frames(h264_clip, start=0.5, end=1.5))
    assert abs(len(frames) - 30) <= 1
    # the first decoded frame is frame ~15 (red channel encodes the index * 4)
    assert abs(float(frames[0][..., 0].mean()) - 15 * 4) <= 8
    tail = list(iter_frames(h264_clip, start=1.9))
    assert 5 <= len(tail) <= 8  # 2.133 s - 1.9 s = 7 frames


def test_iter_frames_constant_fps(h264_clip):
    frames = list(iter_frames(h264_clip, fps=15))
    assert abs(len(frames) - N // 2) <= 1


def test_early_close_stops_ffmpeg(h264_clip):
    t = time.monotonic()
    gen = iter_frames(h264_clip)
    for i, _f in enumerate(gen):
        if i == 2:
            break
    gen.close()
    assert time.monotonic() - t < 5


def test_cancel(h264_clip):
    tok = CancelToken()
    got = 0
    with pytest.raises(Cancelled):
        for _f in iter_frames(h264_clip, cancel=tok):
            got += 1
            if got == 3:
                tok.cancel()
    assert got == 3


def test_errors(tmp_path, h264_clip):
    with pytest.raises(InputError):
        probe(tmp_path / "missing.mp4")
    with pytest.raises(InputError):
        list(iter_frames(tmp_path / "missing.mp4"))
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"\x00not a video" * 100)
    with pytest.raises(EncodeError):
        probe(junk)
    with pytest.raises(InputError):
        list(iter_frames(h264_clip, start=1.0, end=0.5))
    with pytest.raises(InputError):
        list(iter_frames(h264_clip, max_long=1))
    with pytest.raises(EncodeError):  # range past the end -> no frames
        list(iter_frames(h264_clip, start=30.0))


def test_truncated_file_decodes_what_it_can(tmp_path, h264_clip):
    """A faststart file cut in half still yields its first frames (no hang); a file cut
    inside the moov box is a clean EncodeError."""
    data = h264_clip.read_bytes()
    cut = tmp_path / "cut.mp4"
    cut.write_bytes(data[: len(data) // 2])
    frames = list(iter_frames(cut))
    assert 0 < len(frames) < N
    head = tmp_path / "head.mp4"
    head.write_bytes(data[:200])
    with pytest.raises(EncodeError):
        list(iter_frames(head))


BANNER = """Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'IMG_1234.MOV':
  Metadata:
    creation_time   : 2024-05-01T10:00:00.000000Z
  Duration: 00:01:02.50, start: 0.000000, bitrate: 15890 kb/s
  Stream #0:0[0x1](und): Video: hevc (Main) (hvc1 / 0x31637668), yuv420p(tv, bt709), 1920x1080, 15000 kb/s, 29.97 fps, 29.97 tbr, 600 tbn (default)
      Metadata:
        handler_name    : Core Media Video
      Side data:
        displaymatrix: rotation of -90.00 degrees
  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 160 kb/s (default)
  Stream #0:2[0x3](und): Data: none (mebx / 0x7862656D), 0 kb/s (default)
At least one output file must be specified
"""


def test_parse_banner_rotation_fps_audio():
    info = parse_ffmpeg_info(BANNER)
    assert info["duration"] == pytest.approx(62.5)
    assert (info["width"], info["height"]) == (1080, 1920)  # rotated portrait video
    assert info["fps"] == 30000 / 1001  # banner "29.97" snapped to the exact NTSC rate
    assert info["rotation"] == pytest.approx(-90.0)
    assert info["has_audio"] is True
    assert info["codec"] == "hevc"


def test_parse_banner_tbr_fallback_and_attached_pic():
    text = (
        "  Duration: 00:00:10.00, start: 0.000000, bitrate: 100 kb/s\n"
        "  Stream #0:0: Video: mjpeg (Baseline), yuvj420p(pc, bt470bg/unknown/unknown), 600x600, "
        "90k tbr, 90k tbn (attached pic)\n"
        "  Stream #0:1: Video: vp9 (Profile 0), yuv420p(tv), 640x360, SAR 1:1 DAR 16:9, 1k tbr, 1k tbn\n"
    )
    info = parse_ffmpeg_info(text)
    assert (info["width"], info["height"]) == (640, 360)  # cover art skipped
    assert info["fps"] == pytest.approx(1000.0)
    assert info["has_audio"] is False
    assert info["rotation"] == 0.0


def test_probe_estimates_frames_when_count_fails(monkeypatch, h264_clip):
    monkeypatch.setattr(decode, "_count_packets", lambda *a, **k: None)
    info = probe(h264_clip)
    assert info.n_frames == round(info.duration * 30)


def test_iter_frames_validates_eagerly(tmp_path, h264_clip):
    """Errors surface when the iterator is created, before any frame is requested."""
    with pytest.raises(InputError):
        iter_frames(tmp_path / "missing.mp4")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"\x00garbage" * 50)
    with pytest.raises(EncodeError) as ei:
        iter_frames(junk)
    assert "junk.mp4" in str(ei.value)
    with pytest.raises(InputError):
        iter_frames(h264_clip, fps=0)
    it = iter_frames(h264_clip, max_long=32)
    assert next(it).shape == (16, 32, 3)
    it.close()


def test_rotated_phone_video(tmp_path, h264_clip):
    """Display-matrix rotation: probe reports the displayed size and frames come rotated."""
    import subprocess

    from video_helpers import ffmpeg

    rot = tmp_path / "rot.mp4"
    res = subprocess.run(
        [
            ffmpeg(),
            "-v",
            "error",
            "-y",
            "-display_rotation",
            "90",
            "-i",
            str(h264_clip),
            "-c",
            "copy",
            str(rot),
        ],
        capture_output=True,
    )
    if res.returncode != 0:
        pytest.skip("this ffmpeg cannot write rotation metadata")
    info = probe(rot)
    assert (info.width, info.height, info.n_frames) == (H, W, N)
    assert next(iter_frames(rot)).shape == (W, H, 3)
    assert next(iter_frames(rot, max_long=64)).shape == (64, 32, 3)


# --------------------------------------------------------------------------- display corrections
def _ff(*args: str) -> subprocess.CompletedProcess:
    from video_helpers import ffmpeg

    return subprocess.run([ffmpeg(), "-v", "error", "-y", *args], capture_output=True)


def test_parse_banner_colour_sar_and_stream_index():
    text = (
        "  Duration: 00:00:05.00, start: 0.000000, bitrate: 900 kb/s\n"
        "  Stream #0:0[0x1]: Video: mjpeg (Baseline), yuvj420p(pc, bt470bg/unknown/unknown), 60x60 "
        "[SAR 1:1 DAR 1:1], 90k tbr, 90k tbn (attached pic)\n"
        "  Stream #0:1[0x2](und): Video: hevc (Main 10) (hvc1 / 0x31637668), "
        "yuv420p10le(tv, bt2020nc/bt2020/arib-std-b67, progressive), 1440x1080 [SAR 4:3 DAR 16:9], "
        "8000 kb/s, 25 fps, 25 tbr, 90k tbn (default)\n"
    )
    info = parse_ffmpeg_info(text)
    assert info["stream"] == 1  # cover art at index 0 skipped
    assert (info["coded_width"], info["coded_height"]) == (1440, 1080)
    assert (info["width"], info["height"]) == (1920, 1080)  # square-pixel display size
    assert info["sar"] == (4, 3)
    assert info["pix_fmt"] == "yuv420p10le" and info["color_range"] == "tv"
    assert (info["color_space"], info["color_primaries"], info["color_trc"]) == (
        "bt2020nc",
        "bt2020",
        "arib-std-b67",
    )
    assert info["hdr"] is True
    sdr = parse_ffmpeg_info(
        "  Stream #0:0: Video: h264 (High), yuv420p(tv, bt709, progressive), 640x360, 30 fps\n"
    )
    assert (sdr["color_space"], sdr["color_trc"], sdr["hdr"]) == ("bt709", "bt709", False)
    untagged = parse_ffmpeg_info("  Stream #0:0: Video: h264, yuv420p(tv, progressive), 1280x720, 30 fps\n")
    assert untagged["color_space"] is None and untagged["color_range"] == "tv"
    assert (
        parse_ffmpeg_info("  Stream #0:0: Video: h264, yuv420p, 1280x720, 30 fps\n")["pix_fmt"] == "yuv420p"
    )
    # rotation after SAR: display size of a 90 degree anamorphic clip
    rot = parse_ffmpeg_info(
        "  Stream #0:0: Video: h264, yuv420p, 1440x1080 [SAR 4:3 DAR 16:9], 25 fps\n"
        "      Side data:\n        displaymatrix: rotation of 90.00 degrees\n"
    )
    assert (rot["width"], rot["height"]) == (1080, 1920)


def test_anamorphic_video_is_decoded_to_square_pixels(tmp_path, h264_clip):
    sar = tmp_path / "sar.mp4"
    res = _ff("-i", str(h264_clip), "-c", "copy", "-bsf:v", "h264_metadata=sample_aspect_ratio=2/1", str(sar))
    if res.returncode != 0:
        pytest.skip("this ffmpeg cannot rewrite the sample aspect ratio")
    info = probe(sar)
    assert (info.width, info.height, info.n_frames) == (2 * W, H, N)
    f = next(iter_frames(sar))
    assert f.shape == (H, 2 * W, 3)
    ref = synth_frames(1)[0]
    # horizontally stretched x2: column 2x of the output ~ column x of the source
    assert np.abs(f[:, ::2].astype(int) - ref.astype(int)).mean() < 4
    assert next(iter_frames(sar, max_long=128)).shape == (32, 128, 3)


def _primaries_frame(w: int, h: int) -> np.ndarray:
    f = np.zeros((h, w, 3), np.uint8)
    f[:, : w // 3] = (255, 0, 0)
    f[:, w // 3 : 2 * w // 3] = (0, 255, 0)
    f[:, 2 * w // 3 :] = (0, 0, 255)
    return f


def test_untagged_hd_video_uses_bt709(tmp_path):
    """Untagged HD video is decoded with BT.709 (what players assume), SD keeps BT.601."""
    from vrphoto.video.encode import VideoWriter

    for w, h in ((1280, 720), (320, 180)):
        src = tmp_path / f"tagged_{w}.mp4"
        with VideoWriter(src, w, h, 30, "preview", encoder="libx264") as vw:
            vw.write(_primaries_frame(w, h))
        untagged = tmp_path / f"untagged_{w}.mp4"
        bsf = "h264_metadata=colour_primaries=2:transfer_characteristics=2:matrix_coefficients=2"
        if _ff("-i", str(src), "-c", "copy", "-bsf:v", bsf, str(untagged)).returncode != 0:
            pytest.skip("this ffmpeg cannot strip colour tags")
        assert decode._header(untagged)[1]["color_space"] is None
        f = next(iter_frames(untagged)).astype(int)
        red, green = f[h // 2, w // 6], f[h // 2, w // 2]
        if w >= 1280:
            assert np.abs(red - (255, 0, 0)).max() <= 4 and np.abs(green - (0, 255, 0)).max() <= 4
        else:  # SD: BT.601 like players -> a BT.709 encode decodes visibly off
            assert np.abs(red - (255, 0, 0)).max() > 10


def _hdr_clip(path, trc: str) -> None:
    """160x90 10-bit HEVC with a PQ / HLG transfer made from an SDR test pattern."""
    from vrphoto.video.decode import available_filters

    if "zscale" not in available_filters():
        pytest.skip("ffmpeg without zscale")
    vf = (f"zscale=tin=bt709:min=bt709:pin=bt709:rin=tv:t={trc}:m=bt2020nc:p=bt2020:r=tv:npl=100,"
          "format=yuv420p10le")  # fmt: skip
    res = _ff("-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10:duration=0.3", "-vf", vf, "-c:v", "libx265",
              "-x265-params", "log-level=error", "-colorspace", "bt2020nc", "-color_primaries", "bt2020",
              "-color_trc", trc, str(path))  # fmt: skip
    if res.returncode != 0:
        pytest.skip(f"cannot make an HDR test clip: {res.stderr[-200:]!r}")


@pytest.mark.parametrize("trc", ["arib-std-b67", "smpte2084"])
def test_hdr_video_is_tone_mapped(tmp_path, trc):
    """PQ / HLG frames are tone-mapped to SDR instead of decoded washed out."""
    clip = tmp_path / "hdr.mp4"
    _hdr_clip(clip, trc)
    ref_path = tmp_path / "ref.png"
    assert (
        _ff("-f", "lavfi", "-i", "testsrc2=size=160x90:rate=10", "-frames:v", "1", str(ref_path)).returncode
        == 0
    )
    from PIL import Image

    ref = np.asarray(Image.open(ref_path).convert("RGB")).astype(float)
    assert decode._header(clip)[1]["hdr"] is True
    tm = next(iter_frames(clip)).astype(float)
    flat = next(iter_frames(clip, hdr_to_sdr=False)).astype(float)

    def sat(a):
        return float((a.max(-1) - a.min(-1)).mean())

    assert np.abs(tm - ref).mean() < 0.6 * np.abs(flat - ref).mean()
    assert sat(tm) > 1.5 * sat(flat) and sat(tm) > 0.65 * sat(ref)


def test_hdr_without_zscale_decodes_plain(tmp_path, monkeypatch, caplog):
    clip = tmp_path / "hdr.mp4"
    _hdr_clip(clip, "arib-std-b67")
    flat = next(iter_frames(clip, hdr_to_sdr=False))
    monkeypatch.setattr(decode, "available_filters", lambda exe=None: set())
    with caplog.at_level("WARNING"):
        got = next(iter_frames(clip))
    assert np.array_equal(got, flat)
    assert "tonemap" in caplog.text


def test_available_filters_parses_and_caches(monkeypatch):
    names = decode.available_filters()
    assert {"scale", "setsar", "fps"} <= names
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("re-ran")))
    assert decode.available_filters() == names


def test_interlaced_video_is_deinterlaced(tmp_path):
    """A field-interlaced clip (moving pattern woven from two fields) decodes without combing."""
    clip = tmp_path / "interlaced.mp4"
    res = _ff("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=50:duration=0.5",
              "-vf", "tinterlace=interleave_top,fieldorder=tff", "-c:v", "libx264", "-pix_fmt", "yuv420p",
              "-flags", "+ildct+ilme", "-x264opts", "tff=1", "-field_order", "tt", str(clip))  # fmt: skip
    if res.returncode != 0:
        pytest.skip("cannot make an interlaced test clip")
    assert decode._header(clip)[1]["interlaced"] is True

    def combing(f):
        g = f.astype(float).mean(-1)
        between = np.abs(g[1:-1] - 0.5 * (g[:-2] + g[2:])).mean()  # odd vs neighbouring even rows
        return between

    from video_helpers import ffmpeg

    raw = subprocess.run([ffmpeg(), "-v", "error", "-i", str(clip), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout  # fmt: skip
    woven = np.frombuffer(raw, np.uint8).reshape(-1, 240, 320, 3)
    frames = list(iter_frames(clip))
    assert len(frames) == len(woven)  # one output frame per input frame
    assert np.mean([combing(f) for f in frames]) < 0.6 * np.mean([combing(f) for f in woven])
    flags = parse_ffmpeg_info(
        "  Stream #0:0: Video: h264, yuv420p(tv, bt709, top coded first (swapped)), 720x576\n"
    )
    assert flags["interlaced"] is True and flags["color_space"] == "bt709"


@pytest.mark.parametrize(("w", "h"), [(2, 2), (4, 6), (16, 8)])
def test_tiny_videos_round_trip(tmp_path, w, h):
    """Single-digit frame sizes (the writer accepts >= 2x2) are probed and decoded."""
    from vrphoto.video.encode import VideoWriter

    out = tmp_path / f"tiny_{w}x{h}.mp4"
    with VideoWriter(out, w, h, 10, encoder="libx264") as vw:
        for _ in range(3):
            vw.write(np.full((h, w, 3), 100, np.uint8))
    info = probe(out)
    assert (info.width, info.height, info.n_frames) == (w, h, 3)
    frames = list(iter_frames(out))
    assert len(frames) == 3 and frames[0].shape == (h, w, 3)
    assert np.abs(frames[0].astype(int) - 100).max() <= 3
    # the codec tag "0x31637661" must never be mistaken for a frame size
    tag_only = parse_ffmpeg_info(
        "  Stream #0:0: Video: h264 (High) (avc1 / 0x31637661), yuv420p, 6x4, 30 fps\n"
    )
    assert (tag_only["width"], tag_only["height"]) == (6, 4)


# --------------------------------------------------------------------------- review fixes
def test_ntsc_24p_decodes_without_drift(tmp_path):
    """F4: a 24000/1001 source shows as "23.98 fps"; decoding at 23.980 made extra frames
    (the picture then drifted behind the sound).  probe() returns the exact rate and the
    decode filter uses the same fraction as the encoder, so the frame count is exact."""
    src = tmp_path / "ntsc24.mp4"
    subprocess.run(
        [ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "color=c=gray:s=16x16:r=24000/1001:d=300", "-c:v", "libx264", "-preset", "ultrafast", str(src)],
        check=True, capture_output=True,
    )  # fmt: skip
    info = probe(src)
    assert info.fps == 24000 / 1001
    assert info.n_frames == 7193
    filters = decode._decode_filters("ffmpeg", {"width": 16, "height": 16}, 16, 16, info.fps, True, "x")
    assert filters[0] == "fps=24000/1001"
    assert sum(1 for _ in iter_frames(src, fps=info.fps)) == info.n_frames  # was 7194 at 23.980


def test_audio_only_file_says_so(tmp_path):
    """R6: no raw ffmpeg banner as the hint; tell the user what the file is."""
    tone = make_tone(tmp_path / "sound.mp4", 1.0)
    for call in (probe, iter_frames):
        with pytest.raises(InputError) as ei:
            call(tone)
        assert "only sound" in str(ei.value) and "sound.mp4" in str(ei.value)
        text = f"{ei.value} {ei.value.hint}"
        assert "Metadata" not in text and "vendor_id" not in text and "Stream #" not in text


def test_damaged_or_unreadable_file_messages(tmp_path):
    """R6: a partly copied video is called damaged with a remedy; ffmpeg's own output
    stays in the log, not in the message or hint."""
    full = tmp_path / "phone.mp4"  # like a phone recording: index (moov) at the end
    subprocess.run(
        [ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", "testsrc2=s=64x64:r=30:d=2", "-c:v", "libx264", str(full)],
        check=True, capture_output=True,
    )  # fmt: skip
    head = tmp_path / "partial.mp4"
    head.write_bytes(full.read_bytes()[: full.stat().st_size // 2])  # copy interrupted
    with pytest.raises(EncodeError) as ei:
        probe(head)
    assert "damaged or was not copied completely" in str(ei.value)
    assert "Copy the video" in ei.value.hint
    for text in (str(ei.value), ei.value.hint):
        assert "[mov" not in text and "Error opening" not in text and "@ 0x" not in text
    # unknown failure: short generic hint pointing at the log
    err = decode._no_video_error(tmp_path / "x.mkv", "Something odd happened\n", {"has_audio": False})
    assert isinstance(err, EncodeError) and "Help > Log" in err.hint and "odd" not in str(err)
    # a broken video stream next to an audio stream is not "only sound"
    err = decode._no_video_error(tmp_path / "x.mp4", "", {"has_audio": True, "stream": 0})
    assert isinstance(err, EncodeError) and "only sound" not in str(err)
