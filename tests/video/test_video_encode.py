"""ffmpeg encoding: presets, encoder probing, round trips, audio passthrough, failures."""

from __future__ import annotations

import os
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
    ffmpeg_banner,
    make_tone,
    synth_frame,
    synth_frames,
    top_level_boxes,
)

from vrphoto.core.errors import EncodeError, InputError
from vrphoto.video import decode, encode
from vrphoto.video.encode import PRESETS, VideoWriter


def test_ffmpeg_exe_runs():
    exe = encode.ffmpeg_exe()
    assert Path(exe).is_file() or os.path.basename(exe).startswith("ffmpeg")
    out = subprocess.run([exe, "-version"], capture_output=True, check=True).stdout.decode()
    assert "ffmpeg version" in out


def test_presets_contract():
    assert set(PRESETS) >= {"quest3_hevc", "quest2_hevc", "h264_compat", "preview"}
    assert PRESETS["quest3_hevc"]["encoders"][0] == "hevc_nvenc"
    assert PRESETS["quest3_hevc"]["encoders"][-1] == "libx265"
    nv = PRESETS["quest3_hevc"]["args"]["hevc_nvenc"]
    # constant quality with a peak cap (E2E-1 / R3: no fixed 60 Mbit/s target for small clips)
    for flag, val in (("-preset", "p5"), ("-rc", "vbr"), ("-cq", "19"), ("-b:v", "0"), ("-maxrate", "100M")):
        assert nv[nv.index(flag) + 1] == val
    x265 = PRESETS["quest3_hevc"]["args"]["libx265"]
    assert x265[x265.index("-preset") + 1] == "fast" and x265[x265.index("-crf") + 1] == "18"
    assert PRESETS["quest2_hevc"]["max_width"] == 5760
    assert PRESETS["h264_compat"]["max_width"] == 4096
    assert set(PRESETS["h264_compat"]["encoders"]) == {"libx264", "h264_nvenc"}
    x264 = PRESETS["h264_compat"]["args"]["libx264"]
    assert x264[x264.index("-crf") + 1] == "18"


def test_probe_encoders_and_cache(monkeypatch):
    res = encode.probe_encoders(refresh=True)
    assert set(res) == set(encode.ENCODER_CANDIDATES)
    assert res["libx264"] is True and res["libx265"] is True
    assert all(isinstance(v, bool) for v in res.values())

    def boom(*a, **k):  # cached: no subprocess may run now
        raise AssertionError("probe_encoders re-ran ffmpeg")

    monkeypatch.setattr(subprocess, "run", boom)
    assert encode.probe_encoders() == res


def test_probe_encoders_without_ffmpeg(monkeypatch):
    def missing():
        raise EncodeError("ffmpeg was not found.")

    monkeypatch.setattr(encode, "ffmpeg_exe", missing)
    res = encode.probe_encoders(refresh=True)
    assert res == dict.fromkeys(encode.ENCODER_CANDIDATES, False)
    monkeypatch.undo()
    encode.probe_encoders(refresh=True)  # restore a real cache for later tests


def test_choose_encoder():
    avail = {"libx264": True, "libx265": True, "hevc_nvenc": False, "h264_nvenc": False}
    assert encode.choose_encoder("quest3_hevc", "auto", avail) == "libx265"
    assert encode.choose_encoder("quest3_hevc", "auto", {**avail, "hevc_nvenc": True}) == "hevc_nvenc"
    assert encode.choose_encoder("h264_compat", "auto", avail) == "libx264"
    assert encode.choose_encoder("h264_compat", "auto", {**avail, "h264_nvenc": True}) == "h264_nvenc"
    assert encode.choose_encoder("preview", "auto", {**avail, "h264_nvenc": True}) == "libx264"
    assert encode.choose_encoder("quest3_hevc", "nvenc", {**avail, "hevc_nvenc": True}) == "hevc_nvenc"
    assert encode.choose_encoder("h264_compat", "nvenc", {**avail, "h264_nvenc": True}) == "h264_nvenc"
    assert encode.choose_encoder("quest3_hevc", "nvenc", avail) == "libx265"  # unavailable -> auto
    assert encode.choose_encoder("h264_compat", "amf", avail) == "h264_amf"  # not probed -> as-is
    assert encode.choose_encoder("quest3_hevc", "x264", avail) == "libx265"  # preset decides the codec
    assert encode.choose_encoder("h264_compat", "libx265", avail) == "libx265"  # explicit name as-is
    assert encode.choose_encoder("quest3_hevc", "auto", {}) == "libx265"  # nothing probed -> software


def test_resolve_preset_auto_uses_settings():
    from vrphoto.core.config import get_settings

    s = get_settings()
    s.quest_model = "quest2"
    assert encode.resolve_preset("auto") == "quest2_hevc"
    assert encode.resolve_preset("auto", "quest3") == "quest3_hevc"
    assert encode.resolve_preset("auto", "pcvr") == "h264_compat"
    with pytest.raises(InputError):
        encode.resolve_preset("dvd")


def test_output_size_caps_and_even():
    assert encode.output_size(255, 129, "h264_compat") == (254, 128)
    assert encode.output_size(7680, 3840, "quest3_hevc") == (7680, 3840)
    assert encode.output_size(7680, 3840, "quest2_hevc") == (5760, 2880)
    assert encode.output_size(8192, 4096, "h264_compat") == (4096, 2048)
    assert encode.output_size(3840, 7680, "h264_compat") == (2048, 4096)
    assert encode.output_size(2560, 1440, "preview") == (1280, 720)


@pytest.mark.parametrize(
    ("encoder", "preset", "codec", "tag"),
    [("libx264", "h264_compat", "h264", "avc1"), ("libx265", "quest3_hevc", "hevc", "hvc1")],
)
def test_round_trip(tmp_path, encoder, preset, codec, tag):
    frames = synth_frames()
    out = encode_clip(tmp_path / f"rt_{encoder}.mp4", frames, preset=preset, encoder=encoder)
    assert out.is_file()
    assert not list(tmp_path.glob(".*encoding*")), "temporary file left behind"
    info = decode.probe(out)
    assert (info.width, info.height, info.n_frames) == (W, H, N)
    assert info.fps == pytest.approx(30.0)
    assert info.duration == pytest.approx(N / 30.0, abs=0.05)
    assert not info.has_audio
    banner = ffmpeg_banner(out)
    assert f"Video: {codec}" in banner and tag in banner and "yuv420p" in banner and "bt709" in banner
    dec = list(decode.iter_frames(out))
    assert len(dec) == N
    for i in (0, 17, 40, N - 1):
        err = np.abs(dec[i].astype(np.int16) - frames[i].astype(np.int16))
        assert err.mean() < 3.0, (i, err.mean())
        # frame index encoded in the red channel survives
        assert abs(float(dec[i][..., 0].mean()) - (i * 4) % 256) < 4
    # faststart: moov before mdat
    boxes = top_level_boxes(out)
    assert boxes.index(b"moov") < boxes.index(b"mdat")


def test_audio_passthrough(tmp_path):
    tone = make_tone(tmp_path / "tone.m4a", seconds=4.0)
    out = encode_clip(tmp_path / "with_audio.mp4", synth_frames(), audio_from=tone, encoder="libx264")
    info = decode.probe(out)
    assert info.has_audio
    assert info.n_frames == N
    # -shortest: the 4 s tone is cut to the ~2.1 s video
    assert info.duration == pytest.approx(N / 30.0, abs=0.25)
    banner = ffmpeg_banner(out)
    assert "Audio: aac" in banner
    # the audio decodes and is a non-silent tone
    pcm = subprocess.run(
        [encode.ffmpeg_exe(), "-v", "error", "-i", str(out), "-map", "0:a:0", "-f", "s16le", "-ac", "1", "-"],
        capture_output=True,
        check=True,
    ).stdout
    samples = np.frombuffer(pcm, np.int16)
    assert samples.size > 48000 and np.abs(samples).max() > 1000


def _pcm(path: Path) -> np.ndarray:
    """Mono s16 samples (48 kHz) of the first audio stream."""
    raw = subprocess.run(
        [encode.ffmpeg_exe(), "-v", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le", "-ac", "1",
         "-ar", "48000", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, np.int16)


def _silence_then_tone(path: Path, silence_s: float, seconds: float) -> Path:
    expr = f"if(lt(t\\,{silence_s})\\,0\\,0.5*sin(2*PI*440*t))"
    subprocess.run(
        [encode.ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i",
         f"aevalsrc={expr}:s=48000:d={seconds}", "-c:a", "aac", str(path)],
        check=True, capture_output=True,
    )  # fmt: skip
    return path


def test_audio_offset_and_source_without_audio(tmp_path, h264_clip):
    """audio_start_s skips into the source audio (trimmed video stays in sync)."""
    src = _silence_then_tone(tmp_path / "late.m4a", 1.5, 4.0)
    plain = encode_clip(tmp_path / "plain.mp4", synth_frames(20), fps=10.0, audio_from=src, encoder="libx264")
    shifted = VideoWriter(tmp_path / "o.mp4", W, H, 10, "preview", audio_from=src, audio_start_s=1.5,
                          encoder="libx264")  # fmt: skip
    for f in synth_frames(20):
        shifted.write(f)
    shifted.close()
    head = slice(2400, 9600)  # 0.05 .. 0.2 s
    assert np.abs(_pcm(plain)[head]).max() < 200  # still silent at the start
    assert np.abs(_pcm(tmp_path / "o.mp4")[head]).max() > 3000  # the tone starts right away
    # a source video without an audio stream: no failure, no audio, single encoding pass
    vw = VideoWriter(tmp_path / "noaudio.mp4", W, H, 30, audio_from=h264_clip, encoder="libx264")
    assert vw._mux_audio is False and "+faststart" in vw.command
    for f in synth_frames(8):
        vw.write(f)
    out = vw.close()
    assert not decode.probe(out).has_audio and vw.warnings == []
    with pytest.raises(InputError):
        VideoWriter(tmp_path / "x.mp4", W, H, 30, audio_from=src, audio_start_s=-1.0)


def test_audio_shorter_than_video_keeps_every_frame(tmp_path):
    """Regression: with '-shortest' on the live pipe ffmpeg stopped reading frames when a
    shorter audio track ended (BrokenPipe -> the whole encode failed).  The audio is now
    padded with silence to the exact video length."""
    tone = make_tone(tmp_path / "short.m4a", seconds=1.0)
    out = encode_clip(tmp_path / "long_video.mp4", synth_frames(90), audio_from=tone, encoder="libx264")
    info = decode.probe(out)
    assert info.n_frames == 90 and info.has_audio
    assert info.duration == pytest.approx(3.0, abs=0.05)
    pcm = _pcm(out)
    assert pcm.size == pytest.approx(3.0 * 48000, rel=0.02)  # padded to the video length
    assert np.abs(pcm[: 48000 // 2]).max() > 1000  # tone at the start
    assert np.abs(pcm[-48000:]).max() < 50  # silence after the 1 s tone
    boxes = top_level_boxes(out)
    assert boxes.index(b"moov") < boxes.index(b"mdat")  # the mux pass keeps faststart
    assert not list(tmp_path.glob(".*"))  # no temporary files left


@pytest.mark.parametrize(
    ("encoder", "preset", "fps"), [("libx265", "quest3_hevc", 29.97), ("libx264", "h264_compat", 24)]
)
def test_audio_mux_keeps_codec_tag_and_rate(tmp_path, encoder, preset, fps):
    tone = make_tone(tmp_path / "t.m4a", seconds=5.0)
    out = encode_clip(
        tmp_path / "a.mp4", synth_frames(40), fps=fps, preset=preset, audio_from=tone, encoder=encoder
    )
    info = decode.probe(out)
    assert info.n_frames == 40 and info.fps == pytest.approx(fps, abs=0.01)
    assert info.duration == pytest.approx(40 / fps, abs=0.05)
    banner = ffmpeg_banner(out)
    assert ("hvc1" if encoder == "libx265" else "avc1") in banner and "bt709" in banner


def test_audio_mux_failure_keeps_silent_video(tmp_path, monkeypatch):
    tone = make_tone(tmp_path / "t.m4a", seconds=2.0)
    real = VideoWriter._mux_cmd

    def broken(self, with_audio):
        cmd = real(self, with_audio)
        if with_audio:
            cmd[cmd.index("-c:a") + 1] = "no_such_audio_codec"
        return cmd

    monkeypatch.setattr(VideoWriter, "_mux_cmd", broken)
    vw = VideoWriter(tmp_path / "v.mp4", W, H, 30, audio_from=tone, encoder="libx264")
    for f in synth_frames(10):
        vw.write(f)
    out = vw.close()
    info = decode.probe(out)
    assert info.n_frames == 10 and not info.has_audio
    assert len(vw.warnings) == 1 and "without sound" in vw.warnings[0]
    boxes = top_level_boxes(out)
    assert boxes.index(b"moov") < boxes.index(b"mdat")
    assert not list(tmp_path.glob(".*"))


def test_bad_encoder_raises_encode_error(tmp_path):
    out = tmp_path / "bad.mp4"
    with pytest.raises(EncodeError) as ei:
        with VideoWriter(out, W, H, 30, encoder="definitely_not_an_encoder") as vw:
            for f in synth_frames(8):
                vw.write(f)
    msg = str(ei.value)
    assert "definitely_not_an_encoder" in msg
    assert "ffmpeg said" in msg  # tail of the ffmpeg log
    assert not out.exists()
    assert not list(tmp_path.glob(".*encoding*"))


def test_odd_dimensions_are_made_even(tmp_path):
    frames = [synth_frame(i, 255, 129) for i in range(6)]
    out = encode_clip(tmp_path / "odd.mp4", frames, encoder="libx264")
    info = decode.probe(out)
    assert (info.width, info.height, info.n_frames) == (254, 128, 6)
    dec = next(decode.iter_frames(out))
    assert np.abs(dec.astype(int) - frames[0][:128, :254].astype(int)).mean() < 3


def test_preset_scales_down_wide_input(tmp_path):
    frames = [synth_frame(i, 1600, 64) for i in range(3)]
    vw = VideoWriter(tmp_path / "wide.mp4", 1600, 64, 24, "preview", encoder="libx264")
    assert (vw.out_width, vw.out_height) == (1280, 50)
    for f in frames:
        vw.write(f)
    out = vw.close()
    info = decode.probe(out)
    assert (info.width, info.height) == (1280, 50)
    assert vw.close() == out  # idempotent


def test_bitrate_override_and_fractional_fps(tmp_path):
    vw = VideoWriter(tmp_path / "b.mp4", 64, 32, 29.97, "quest3_hevc", encoder="libx265", bitrate="8M")
    cmd = vw.command
    assert cmd[cmd.index("-r") + 1] == "30000/1001"
    assert cmd[cmd.index("-b:v") + 1] == "8M" and "-crf" not in cmd
    assert cmd[cmd.index("-bufsize") + 1] == "16M"
    assert cmd[cmd.index("-tag:v") + 1] == "hvc1"
    assert "+faststart" in cmd and "-shortest" not in cmd
    vw.write(np.zeros((32, 64, 3), np.uint8))
    out = vw.close()
    assert decode.probe(out).fps == pytest.approx(29.97, abs=0.01)


def test_float_gray_rgba_frames_are_converted(tmp_path):
    """Floats follow the project convention ([0, 1], like core.imaging.to_uint8); RGBA
    drops alpha; grayscale is replicated; other integer types are clipped."""
    ref = synth_frame(3, 64, 32)
    f = ref.astype(np.float32) / 255.0
    f[0, 0] = np.nan
    rgba = np.dstack([ref, np.full((32, 64), 7, np.uint8)])
    gray = ref[..., 1].copy()
    wide = ref.astype(np.int64) * 4 - 300  # out-of-range ints are clipped
    frames = [f, rgba, gray, gray[..., None], wide]
    out = encode_clip(tmp_path / "f.mp4", frames, encoder="libx264")
    dec = list(decode.iter_frames(out))
    assert len(dec) == 5
    assert np.abs(dec[0][4:, 4:].astype(float) - ref[4:, 4:]).mean() < 3
    assert np.abs(dec[1].astype(float) - ref).mean() < 3
    for d in dec[2:4]:
        assert np.abs(d.astype(float) - gray[..., None]).mean() < 3
    assert np.abs(dec[4].astype(float) - np.clip(wide, 0, 255)).mean() < 4
    a = encode.as_rgb24(np.full((2, 2), True), 2, 2)
    assert a.dtype == np.uint8 and a.shape == (2, 2, 3) and (a == 255).all()
    assert encode.as_rgb24(np.array([[[np.inf, -np.inf, 0.5]]]), 1, 1).tolist() == [[[255, 0, 128]]]
    assert encode.as_rgb24(np.array([[[65535, 0, 32896]]], np.uint16), 1, 1).tolist() == [[[255, 0, 128]]]


def test_mov_container(tmp_path):
    out = encode_clip(tmp_path / "clip.mov", synth_frames(5), encoder="libx264")
    assert decode.probe(out).n_frames == 5
    assert "mov,mp4" in ffmpeg_banner(out)


def test_writer_validation(tmp_path):
    with pytest.raises(InputError):
        VideoWriter(tmp_path / "x.avi", 64, 32, 30)
    with pytest.raises(InputError):
        VideoWriter(tmp_path / "x.mp4", 1, 32, 30)
    with pytest.raises(InputError):
        VideoWriter(tmp_path / "x.mp4", 64, 32, 0)
    with pytest.raises(InputError):
        VideoWriter(tmp_path / "x.mp4", 64, 32, 30, "nope")
    with pytest.raises(InputError):
        VideoWriter(tmp_path / "x.mp4", 64, 32, 30, audio_from=tmp_path / "missing.wav")
    vw = VideoWriter(tmp_path / "x.mp4", 64, 32, 30, encoder="libx264")
    with pytest.raises(InputError):
        vw.write(np.zeros((32, 63, 3), np.uint8))
    with pytest.raises(InputError):
        vw.write(np.zeros((32, 64, 5), np.uint8))
    with pytest.raises(InputError):
        vw.write(np.zeros((32, 64, 3), np.complex64))
    with pytest.raises(InputError):
        vw.write(np.zeros(32 * 64 * 3, np.uint8))
    vw.write(np.zeros((32, 64, 3), np.uint8))
    vw.close()
    with pytest.raises(EncodeError):
        vw.write(np.zeros((32, 64, 3), np.uint8))


def test_no_frames_is_an_error(tmp_path):
    vw = VideoWriter(tmp_path / "empty.mp4", 64, 32, 30, encoder="libx264")
    with pytest.raises(EncodeError):
        vw.close()
    assert not (tmp_path / "empty.mp4").exists()


def test_exception_in_block_aborts_and_keeps_old_file(tmp_path):
    out = tmp_path / "keep.mp4"
    out.write_bytes(b"previous")

    with pytest.raises(RuntimeError), VideoWriter(out, 64, 32, 30, encoder="libx264") as vw:
        vw.write(np.zeros((32, 64, 3), np.uint8))
        raise RuntimeError("user cancelled")
    assert out.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".*encoding*"))
    with pytest.raises(EncodeError):
        vw.close()  # aborted writers cannot be closed


def test_encode_frames_helper(tmp_path):
    out = encode.encode_frames(iter(synth_frames(4, 64, 32)), tmp_path / "h.mp4", 12, encoder="libx264")
    assert decode.probe(out).n_frames == 4
    with pytest.raises(InputError):
        encode.encode_frames([], tmp_path / "e.mp4", 12)


def test_popen_flags_windows_branch(monkeypatch):
    monkeypatch.setattr(encode.sys, "platform", "win32")
    flags = encode.popen_flags()
    assert flags["creationflags"] == getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    monkeypatch.setattr(encode.sys, "platform", "linux")
    assert encode.popen_flags() == {}


def test_colour_matrix_round_trip(tmp_path):
    """Golden: BT.709 tagged on encode and honoured on decode - saturated primaries survive
    (a BT.601/709 mix-up shifts pure red by ~20 levels)."""
    cols = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 255),
        (0, 0, 0),
        (128, 128, 128),
        (200, 50, 120),
    ]
    f = np.zeros((64, 64 * len(cols), 3), np.uint8)
    for i, c in enumerate(cols):
        f[:, i * 64 : (i + 1) * 64] = c
    for enc, preset in (("libx264", "h264_compat"), ("libx265", "quest3_hevc")):
        out = encode_clip(tmp_path / f"c_{enc}.mp4", [f, f, f], preset=preset, encoder=enc)
        d = next(decode.iter_frames(out))
        for i, c in enumerate(cols):
            got = d[32, i * 64 + 32].astype(int)
            assert np.abs(got - np.array(c)).max() <= 3, (enc, c, got)


@pytest.mark.skipif(sys.platform == "win32", reason="':' is not allowed in Windows file names")
def test_relative_path_with_colon(tmp_path, monkeypatch):
    """ffmpeg would read 'name:rest' as a protocol URL; paths are passed absolute."""
    monkeypatch.chdir(tmp_path)
    tone = make_tone(tmp_path / "t.m4a", seconds=1.0)
    (tmp_path / "12:30 tone.m4a").write_bytes(tone.read_bytes())
    out = encode_clip(Path("12:30 clip.mp4"), synth_frames(6), audio_from="12:30 tone.m4a", encoder="libx264")
    info = decode.probe(Path("12:30 clip.mp4"))
    assert out.is_file() and info.n_frames == 6 and info.has_audio
    assert len(list(decode.iter_frames("12:30 clip.mp4"))) == 6


def test_hardware_encoder_falls_back_when_size_unsupported(tmp_path, monkeypatch):
    """A GPU encoder that passed the small probe but rejects the real frame size (e.g. a
    7680-wide VR180 video on AMF) is replaced by the next working encoder of the preset."""
    avail = dict.fromkeys(encode.ENCODER_CANDIDATES, False) | {
        "libx264": True,
        "libx265": True,
        "hevc_nvenc": True,
    }
    monkeypatch.setattr(encode, "probe_encoders", lambda refresh=False: dict(avail))
    tested = []

    def works_at(enc, w, h):
        tested.append((enc, w, h))
        return enc != "hevc_nvenc" or w <= 4096

    monkeypatch.setattr(encode, "encoder_works_at", works_at)
    small = VideoWriter(tmp_path / "s.mp4", 256, 128, 30, "quest3_hevc")
    assert small.encoder == "hevc_nvenc"  # fits: kept (not started for real below)
    small.abort()
    big = VideoWriter(tmp_path / "b.mp4", 7680, 3840, 30, "quest3_hevc", encoder="nvenc")
    assert big.encoder == "libx265" and ("hevc_nvenc", 7680, 3840) in tested
    big.abort()
    tested.clear()
    explicit = VideoWriter(tmp_path / "e.mp4", 7680, 3840, 30, "quest3_hevc", encoder="libx265")
    assert explicit.encoder == "libx265" and tested == []  # software / explicit names: no test
    explicit.abort()
    assert not list(tmp_path.iterdir())


def test_encoder_works_at_real_ffmpeg():
    assert encode.encoder_works_at("libx264", 320, 240) is True
    assert encode.encoder_works_at("libx264", 320, 240) is True  # cached
    assert encode.encoder_works_at("definitely_not_an_encoder", 64, 64) is False


# --------------------------------------------------------------------------- review fixes
def test_flat_output_auto_preset_is_h264():
    """E2E-1 / R3: a flat video is for phones / PCs (no HEVC decoder on stock Windows):
    "auto" gives H.264 there; headset outputs keep the Quest HEVC preset."""
    assert encode.resolve_preset("auto", "quest3", output="flat") == "h264_compat"
    assert encode.resolve_preset("auto", "quest2", output=" FLAT ") == "h264_compat"
    assert encode.resolve_preset("", "quest3", output="flat") == "h264_compat"
    for out in ("sbs", "vr180", "tb", None):
        assert encode.resolve_preset("auto", "quest3", output=out) == "quest3_hevc"
    # an explicit preset always wins
    assert encode.resolve_preset("quest3_hevc", "quest3", output="flat") == "quest3_hevc"
    assert encode.resolve_preset("preview", output="flat") == "preview"


def test_hardware_encoders_use_constant_quality():
    """E2E-1: no fixed bitrate target (60 Mbit/s made a 6 s clip ~45 MB); every hardware
    encoder uses quality-based rate control, NVENC keeps a peak cap for the Quest."""
    for preset in ("quest3_hevc", "quest2_hevc", "h264_compat"):
        for enc, args in PRESETS[preset]["args"].items():
            if enc in ("libx264", "libx265"):
                assert "-crf" in args
                continue
            assert any(q in args for q in ("-cq", "-qp_i", "-q:v")), (preset, enc, args)
            if "-b:v" in args:
                assert args[args.index("-b:v") + 1] == "0", (preset, enc)
            if enc.endswith("_nvenc"):
                assert args[args.index("-rc") + 1] == "vbr" and "-maxrate" in args
            if enc.endswith("_amf"):
                assert args[args.index("-rc") + 1] == "cqp"


def test_bitrate_override_switches_hardware_rate_control():
    spec = PRESETS["quest3_hevc"]
    amf = encode._rate_args(spec, "hevc_amf", "20M")
    assert amf[amf.index("-rc") + 1] == "vbr_peak" and "cqp" not in amf
    assert amf[amf.index("-b:v") + 1] == "20M" and "-qp_i" not in amf and "-qp_p" not in amf
    nv = encode._rate_args(spec, "hevc_nvenc", "20M")
    assert "-cq" not in nv and nv.count("-b:v") == 1 and nv[nv.index("-b:v") + 1] == "20M"
    qsv = encode._rate_args(spec, "hevc_qsv", "20M")
    assert "-q:v" not in qsv and qsv[qsv.index("-b:v") + 1] == "20M"
    assert encode._rate_args(spec, "hevc_nvenc", None) == spec["args"]["hevc_nvenc"]


def test_ntsc_rates_are_exact():
    """F4: the banner's rounded "23.98" means 24000/1001 for decoding and encoding."""
    from fractions import Fraction

    assert encode.ntsc_rate(23.98) == Fraction(24000, 1001)
    assert encode.ntsc_rate(24000 / 1001) == Fraction(24000, 1001)
    assert encode.ntsc_rate(29.97) == Fraction(30000, 1001)
    assert encode.ntsc_rate(59.94) == Fraction(60000, 1001)
    assert encode.ntsc_rate(47.95) == Fraction(48000, 1001)
    assert encode.ntsc_rate(24.0) is None and encode.ntsc_rate(25.0) is None
    assert encode.fps_arg(23.98) == "24000/1001"
    assert encode.fps_arg(30.0) == "30/1"
    assert encode.fps_arg(12.5) == "25/2"
    # the decoder now uses fps_arg too: a very slow rate must not round to "0/1"
    slow = encode.fps_arg(0.0004)
    assert slow != "0/1" and float(Fraction(slow)) == pytest.approx(0.0004)


def test_close_retries_a_briefly_locked_destination(tmp_path, monkeypatch):
    """WIN-2: antivirus / thumbnailer briefly lock the fresh file on Windows; the final
    rename waits that out instead of failing the whole encode."""
    from vrphoto.core import fsutil

    out = tmp_path / "locked.mp4"
    vw = VideoWriter(out, 64, 32, 30, encoder="libx264")
    vw.write(np.zeros((32, 64, 3), np.uint8))
    real = os.replace
    calls = []

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) <= 2:
            raise PermissionError(32, "The process cannot access the file", str(dst))
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(fsutil.time, "sleep", lambda s: None)
    assert vw.close() == out
    monkeypatch.undo()
    assert len(calls) == 3 and out.is_file() and decode.probe(out).n_frames == 1
    assert not list(tmp_path.glob(".*encoding*"))


def test_close_reports_a_busy_file_in_plain_words(tmp_path, monkeypatch):
    from vrphoto.core import fsutil

    monkeypatch.setattr(fsutil.time, "sleep", lambda s: None)  # Windows CI retries; do not wait
    out = tmp_path / "busy.mp4"
    vw = VideoWriter(out, 64, 32, 30, encoder="libx264")
    vw.write(np.zeros((32, 64, 3), np.uint8))

    def locked(src, dst):
        raise PermissionError(32, "The process cannot access the file", str(dst))

    monkeypatch.setattr(os, "replace", locked)
    with pytest.raises(EncodeError) as ei:
        vw.close()
    monkeypatch.undo()
    assert "busy" in str(ei.value) and "antivirus" in str(ei.value)
    assert not out.exists() and not list(tmp_path.glob(".*encoding*"))
