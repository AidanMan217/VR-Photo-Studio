"""vrphoto-cli: generated options, job specs, exit codes, JSON output and subcommands."""

from __future__ import annotations

import dataclasses
import io
import json
import os
import shutil
import subprocess
import sys
import typing
from pathlib import Path

import numpy as np
import pytest

from vrphoto import cli
from vrphoto.core.errors import AddonError, Cancelled, InputError
from vrphoto.pipeline.jobs import JOB_TYPES, CommonOptions, JobResult, job_to_dict

NO_MODELS = ["--depth-model", "heuristic", "--inpaint-model", "opencv", "--segment-model", "grabcut"]


@pytest.fixture(autouse=True)
def _settings(tmp_path, isolated_home):
    from vrphoto.core import config

    config.set_settings(config.Settings(output_dir=str(tmp_path / "default_out")))
    yield
    config.set_settings(None)


def run_cli(args, monkeypatch=None, stdin: str | None = None) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    old = sys.stdout, sys.stderr, sys.stdin
    sys.stdout, sys.stderr = out, err
    if stdin is not None:
        sys.stdin = io.StringIO(stdin)
    try:
        code = cli.main([str(a) for a in args])
    finally:
        sys.stdout, sys.stderr, sys.stdin = old
    return code, out.getvalue(), err.getvalue()


def json_lines(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip().startswith("{")]


@pytest.fixture
def photo(tmp_path) -> Path:
    from PIL import Image

    h, w = 120, 160
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx * 255 // w, yy * 255 // h, np.full_like(xx, 90)], -1).astype(np.uint8)
    img[60:110, 60:100] = (220, 60, 60)
    p = tmp_path / "pic.jpg"
    Image.fromarray(img).save(p, quality=95)
    return p


# --------------------------------------------------------------------------- option generation
def _subparser(kind: str):
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if isinstance(a, __import__("argparse")._SubParsersAction))
    return sub.choices[kind]


@pytest.mark.parametrize("kind", cli.MODE_KINDS)
def test_every_field_has_an_option(kind):
    sp = _subparser(kind)
    flags = {s for a in sp._actions for s in a.option_strings}
    cls = JOB_TYPES[kind]
    for f in dataclasses.fields(cls):
        if f.name in ("inputs", "common"):
            continue
        flag = "--" + f.name.replace("_", "-")
        assert flag in flags, f"{kind}: missing {flag}"
        hints = typing.get_type_hints(cls)
        if cli._unwrap_optional(hints[f.name])[0] is bool:
            assert "--no-" + f.name.replace("_", "-") in flags
    for f in dataclasses.fields(CommonOptions):
        flag = "--" + f.name.replace("_", "-")
        assert flag in flags, f"{kind}: missing common {flag}"
    assert "-o" in flags and "--dry-run" in flags


def _sample_value(kind: str, name: str, tp, default, choices):
    inner, optional = cli._unwrap_optional(tp)
    if inner is bool:
        return (not default), ["--no-" if default else "--", name]
    if choices:
        v = next(c for c in choices if c != default)
        return v, [v]
    if inner is int:
        v = (default or 0) + 7 if default is not None else 123
        return v, [str(v)]
    if inner is float:
        v = round((default or 0.0) + 0.25, 4) if default is not None else 1.75
        return v, [str(v)]
    if inner is str:
        return "abc", ["abc"]
    v = [{"t": 0, "x": 0.01}, {"t": 1, "x": -0.01}]
    return v, [json.dumps(v)]


@pytest.mark.parametrize("kind", cli.MODE_KINDS)
def test_every_option_reaches_the_spec(kind, tmp_path):
    """Give every option a non-default value and check it arrives in the job spec."""
    choices = cli._choices_table()
    expected: dict[str, object] = {}
    expected_common: dict[str, object] = {}
    args: list[str] = [kind, str(tmp_path / "in.jpg")]
    for owner, store, ckind in (
        (JOB_TYPES[kind], expected, kind),
        (CommonOptions, expected_common, "common"),
    ):
        hints = typing.get_type_hints(owner)
        for f in dataclasses.fields(owner):
            if f.name in ("inputs", "common"):
                continue
            default = cli._field_default(f)
            value, toks = _sample_value(kind, f.name, hints[f.name], default, choices.get((ckind, f.name)))
            flag = f.name.replace("_", "-")
            if cli._unwrap_optional(hints[f.name])[0] is bool:
                args.append(toks[0] + flag)
            else:
                args += [f"--{flag}", *toks]
            store[f.name] = value
    ns = cli.build_parser().parse_args(args)
    spec = cli.spec_from_namespace(kind, ns)
    for k, v in expected.items():
        assert getattr(spec, k) == v, k
    for k, v in expected_common.items():
        got = getattr(spec.common, k)
        if k == "output_dir":
            assert got == os.path.abspath("abc")
        else:
            assert got == v, k
    assert spec.inputs == [str(tmp_path / "in.jpg")]


def test_unset_options_keep_dataclass_defaults(tmp_path):
    ns = cli.build_parser().parse_args(["stereo", "a.jpg"])
    spec = cli.spec_from_namespace("stereo", ns)
    assert job_to_dict(spec) == job_to_dict(JOB_TYPES["stereo"](inputs=[os.path.abspath("a.jpg")]))


def test_auto_and_json_values():
    ns = cli.build_parser().parse_args(["stereo", "a.jpg", "--convergence-m", "auto", "--hfov-deg", "none"])
    spec = cli.spec_from_namespace("stereo", ns)
    assert spec.convergence_m is None and spec.common.hfov_deg is None
    ns = cli.build_parser().parse_args(
        ["parallax", "a.jpg", "--path", "custom", "--keyframes", '[{"t":0,"x":0.1}]']
    )
    assert cli.spec_from_namespace("parallax", ns).keyframes == [{"t": 0, "x": 0.1}]


def test_help_texts_mention_defaults_units_and_choices():
    sp = _subparser("stereo")
    helps = {a.dest: a.help for a in sp._actions}
    assert "Default: sbs" in helps["job__layout"] and "anaglyph" in helps["job__layout"]
    assert "Unit: metres" in helps["job__ipd_m"]
    assert "%%" in helps["job__max_parallax_pct"] or "%" not in helps["job__max_parallax_pct"].replace(
        "%%", ""
    )
    text = sp.format_help()  # must not raise on '%' in tooltips
    assert "--max-parallax-pct" in text


@pytest.mark.parametrize("kind", cli.MODE_KINDS)
def test_mode_help_renders(kind):
    code, out, _ = run_cli([kind, "--help"])
    assert code == 0 and "--output-dir" in out


# --------------------------------------------------------------------------- running jobs
def test_stereo_job_end_to_end(photo, tmp_path):
    out = tmp_path / "o"
    code, stdout, stderr = run_cli(["stereo", photo, *NO_MODELS, "-o", out, "--layout", "half_sbs"])
    assert code == 0, stderr
    assert (out / "pic_LR.jpg").is_file()
    assert "Done in" in stdout and "pic_LR.jpg" in stdout


def test_json_output_lines(photo, tmp_path):
    code, stdout, stderr = run_cli(["depth", photo, *NO_MODELS, "-o", tmp_path / "o", "--json"])
    assert code == 0, stderr
    lines = json_lines(stdout)
    assert len(lines) == len(stdout.strip().splitlines())  # stdout is pure JSON lines
    assert lines[0]["type"] == "progress" and lines[-1]["type"] == "result"
    assert lines[-1]["ok"] is True and any("Depth map" in k for k in lines[-1]["outputs"])
    fr = [x["frac"] for x in lines if x["type"] == "progress"]
    assert fr[-1] == 1.0


def test_dry_run_and_run_spec(photo, tmp_path):
    code, stdout, _ = run_cli(
        [
            "vr180",
            photo,
            "--eye-px",
            "128",
            "--no-export-scene",
            "--dry-run",
            *NO_MODELS,
            "-o",
            tmp_path / "o",
        ]
    )
    assert code == 0
    d = json.loads(stdout)
    assert d["kind"] == "vr180" and d["eye_px"] == 128 and d["common"]["export_scene"] is False
    assert not (tmp_path / "o").exists()  # nothing ran
    # relative inputs are resolved against the job file's folder
    d["inputs"] = ["pic.jpg"]
    job = tmp_path / "job.json"
    job.write_text(json.dumps(d), encoding="utf-8")
    code, stdout, stderr = run_cli(["run", "--spec", job, "--json"])
    assert code == 0, stderr
    res = json_lines(stdout)[-1]
    assert Path(res["outputs"]["VR180 photo"]).name == "pic_180_LR.jpg"


def test_run_spec_from_stdin_with_input_override(photo, tmp_path):
    d = job_to_dict(JOB_TYPES["depth"](inputs=["ignored.jpg"]))
    d["common"].update(depth_model="heuristic", output_dir=str(tmp_path / "o"))
    code, stdout, stderr = run_cli(["run", "--spec", "-", photo], stdin=json.dumps(d))
    assert code == 0, stderr
    assert (tmp_path / "o" / "pic_depth.png").is_file()


def test_run_spec_errors(tmp_path):
    code, _, err = run_cli(["run", "--spec", tmp_path / "missing.json"])
    assert code == 1 and "not found" in err
    bad = tmp_path / "bad.json"
    bad.write_text('{"kind": "stereo", "bogus": 1}', encoding="utf-8")
    code, _, err = run_cli(["run", "--spec", bad])
    assert code == 1 and "bogus" in err


# --------------------------------------------------------------------------- exit codes
def test_usage_errors_exit_2():
    assert run_cli(["stereo"])[0] == 2  # no input
    assert run_cli(["stereo", "a.jpg", "--layout", "nope"])[0] == 2
    assert run_cli(["frobnicate"])[0] == 2
    assert run_cli(["stereo", "a.jpg", "--ipd-m", "wide"])[0] == 2
    code, out, _ = run_cli([])
    assert code == 2 and "<command>" in out
    assert run_cli(["models"])[0] == 2


def test_help_and_version_exit_0():
    code, out, _ = run_cli(["--help"])
    assert code == 0 and "selftest" in out
    code, out, _ = run_cli(["--version"])
    assert code == 0 and "VR Photo Studio" in out


def test_error_exit_1_with_hint(tmp_path):
    code, _, err = run_cli(["stereo", tmp_path / "missing.jpg"])
    assert code == 1
    assert "Error: File not found" in err and "Hint:" in err
    code, out, err = run_cli(["stereo", tmp_path / "missing.jpg", "--json"])
    assert code == 1
    e = json_lines(out)[-1]
    assert e["type"] == "error" and e["kind"] == "InputError" and e["hint"]


def test_cancelled_exit_130(photo, monkeypatch):
    import vrphoto.pipeline.run as R

    def cancelled(*a, **k):
        raise Cancelled("Operation cancelled")

    monkeypatch.setattr(R, "run_job", cancelled)
    code, _, err = run_cli(["stereo", photo])
    assert code == 130 and "Cancelled" in err


def test_ctrl_c_cancels_the_job_gracefully(photo, monkeypatch):
    """A KeyboardInterrupt in the waiting main thread cancels the token; the job stops."""
    import threading

    import vrphoto.pipeline.run as R

    seen = {}
    started = threading.Event()

    def slow_job(spec, ctx, *, progress, cancel):
        seen["cancel"] = cancel
        started.set()
        while True:
            cancel.check()
            threading.Event().wait(0.01)

    monkeypatch.setattr(R, "run_job", slow_job)
    real_join = threading.Thread.join
    calls = {"n": 0}

    def join(self, timeout=None):
        if self.name == "vrphoto-cli-task" and calls["n"] == 0:
            started.wait(5)
            calls["n"] += 1
            raise KeyboardInterrupt
        return real_join(self, timeout)

    monkeypatch.setattr(threading.Thread, "join", join)
    code, _, err = run_cli(["stereo", photo])
    assert code == 130
    assert seen["cancel"].cancelled
    assert "Cancelling" in err


def test_unexpected_error_exit_1(photo, monkeypatch):
    import vrphoto.pipeline.run as R

    def bug(*a, **k):
        raise ZeroDivisionError("oops")

    monkeypatch.setattr(R, "run_job", bug)
    code, _, err = run_cli(["stereo", photo])
    assert code == 1 and "Unexpected error" in err and "ZeroDivisionError" in err


def test_warnings_are_printed(photo, monkeypatch):
    import vrphoto.pipeline.run as R

    monkeypatch.setattr(R, "run_job", lambda *a, **k: JobResult(outputs={"x": "y"}, warnings=["careful"]))
    code, out, err = run_cli(["stereo", photo])
    assert code == 0 and "Warning: careful" in err and "x: y" in out


# --------------------------------------------------------------------------- selftest
def test_selftest_flag_maps_to_command():
    assert cli._prepare_argv(["--selftest", "--json"]) == ["selftest", "--json"]
    assert cli._prepare_argv(["--json", "--selftest"]) == ["selftest", "--json"]
    assert cli._prepare_argv(["stereo", "a.jpg"]) == ["stereo", "a.jpg"]


def test_selftest_command_uses_report(monkeypatch, tmp_path):
    import vrphoto.selftest as st

    report = {
        "ok": False,
        "version": "x",
        "python": "3",
        "platform": "p",
        "failed": ["ffmpeg"],
        "checks": {
            "ffmpeg": {"ok": False, "required": True, "seconds": 0.1, "error": "no ffmpeg"},
            "imports": {"ok": True, "required": True, "seconds": 0.2},
        },
    }
    got = {}

    def fake(out=None, *, models=False, progress=None):
        got.update(out=out, models=models)
        return report

    monkeypatch.setattr(st, "run_selftest", fake)
    code, out, _ = run_cli(["--selftest", "--json", "--models", "--out", tmp_path / "r.json"])
    assert code == 1
    assert json.loads(out)["failed"] == ["ffmpeg"]
    assert got["models"] is True and got["out"] == tmp_path / "r.json"
    code, out, _ = run_cli(["selftest"])
    assert code == 1 and "FAIL" in out and "no ffmpeg" in out


# --------------------------------------------------------------------------- serve
def test_serve_prints_urls_and_qr(tmp_path):
    code, out, err = run_cli(
        ["serve", "--https-port", "0", "--http-port", "0", "--bind", "127.0.0.1", "--duration", "0.3"]
    )
    assert code == 0, err
    assert "https://127.0.0.1:" in out and "http://localhost:" in out
    assert "Ctrl+C" in out
    assert any(ch in out for ch in ("█", "##"))  # the QR code


def test_serve_json(tmp_path):
    code, out, err = run_cli(
        [
            "serve",
            "--https-port",
            "0",
            "--http-port",
            "0",
            "--bind",
            "127.0.0.1",
            "--duration",
            "0.2",
            "--json",
        ]
    )
    assert code == 0, err
    lines = json_lines(out)
    assert lines[0]["type"] == "server" and lines[0]["urls"]["https"].startswith("https://127.0.0.1:")
    assert lines[-1]["type"] == "stopped"


def test_qr_text_ascii_fallback():
    t = cli._qr_text("https://192.168.1.5:8443/", "ascii")
    assert "#" in t and "█" not in t
    u = cli._qr_text("https://192.168.1.5:8443/", "utf-8")
    assert "█" in u


# --------------------------------------------------------------------------- models
def test_models_list_and_path():
    code, out, _ = run_cli(["models", "list", "--json"])
    assert code == 0
    rows = json.loads(out)
    ids = {r["id"] for r in rows}
    assert "da2-small" in ids and all(r["downloaded"] is False for r in rows)
    code, out, _ = run_cli(["models", "list"])
    assert code == 0 and "da2-small" in out and "non-commercial" in out
    code, out, _ = run_cli(["models", "path", "da2-small"])
    assert code == 0 and "da2-small" in out
    code, _, err = run_cli(["models", "path", "no-such-model"])
    assert code == 1 and "Error" in err


def test_models_download_and_delete(monkeypatch, tmp_path):
    from vrphoto.core.config import get_settings
    from vrphoto.models import download

    calls = []

    def fake_ensure(model_id, *, progress=None, cancel=None, **kw):
        calls.append(model_id)
        progress(0.5, "Downloading")
        return tmp_path / f"{model_id}.onnx"

    monkeypatch.setattr(download, "ensure_model", fake_ensure)
    monkeypatch.setattr(download, "delete_model", lambda mid: calls.append(("del", mid)))
    code, out, err = run_cli(["models", "download", "da2-large", "--accept-license"])
    assert code == 0, err
    assert calls == ["da2-large"] and "ready" in out
    assert "non-commercial" in err
    assert get_settings().license_accepted("da2-large")
    code, out, _ = run_cli(["models", "delete", "da2-large", "--json"])
    assert code == 0 and json_lines(out)[-1]["deleted"] is True
    assert ("del", "da2-large") in calls


# --------------------------------------------------------------------------- addons
def test_addons_list_and_status():
    code, out, _ = run_cli(["addons", "list", "--json"])
    assert code == 0
    ids = {r["id"] for r in json.loads(out)}
    assert {"trellis", "capture", "sdxl360"} <= ids
    code, out, _ = run_cli(["addons", "list", "--stable"])
    assert code == 0 and "trellis2" not in out
    code, out, _ = run_cli(["addons", "status", "trellis", "--json"])
    assert code == 0 and json.loads(out)["installed"] is False
    code, _, err = run_cli(["addons", "status", "nope"])
    assert code == 1 and "Unknown add-on" in err


def test_addons_install_uninstall_check(monkeypatch):
    from vrphoto.addons import manager

    calls = []

    def fake_install(addon_id, *, progress=None, cancel=None, hf_token=None, check_gpu=True, force=False):
        calls.append(("install", addon_id, check_gpu, force, hf_token))
        progress(1.0, "done")

    monkeypatch.setattr(manager, "install", fake_install)
    monkeypatch.setattr(manager, "uninstall", lambda a: calls.append(("uninstall", a)))
    monkeypatch.setattr(
        manager,
        "check",
        lambda a, *, check_gpu=True, cancel=None, **k: (
            calls.append(("check", a, check_gpu)) or {"torch": "2.7"}
        ),
    )
    code, out, err = run_cli(["addons", "install", "trellis", "--no-check-gpu", "--force"])
    assert code == 0, err
    assert calls[-1] == ("install", "trellis", False, True, None)
    assert "Non-commercial" in err  # license notices are shown
    code, _, _ = run_cli(["addons", "install", "sdxl360", "--hf-token", "tok"])
    assert calls[-1] == ("install", "sdxl360", True, False, "tok")
    monkeypatch.setattr(cli, "_nvidia_present", lambda: False)
    code, out, _ = run_cli(["addons", "check", "trellis"])
    assert code == 0 and calls[-1] == ("check", "trellis", False) and "torch: 2.7" in out
    code, out, _ = run_cli(["addons", "check", "trellis", "--check-gpu", "--json"])
    assert calls[-1] == ("check", "trellis", True) and json_lines(out)[-1]["info"] == {"torch": "2.7"}
    code, out, _ = run_cli(["addons", "uninstall", "trellis"])
    assert code == 0 and calls[-1] == ("uninstall", "trellis")


def test_addons_check_failure_exit_1(monkeypatch):
    from vrphoto.addons import manager

    def broken(*a, **k):
        raise AddonError("CUDA is not available", hint="Install the NVIDIA driver")

    monkeypatch.setattr(manager, "check", broken)
    code, _, err = run_cli(["addons", "check", "trellis", "--no-check-gpu"])
    assert code == 1 and "CUDA is not available" in err and "NVIDIA driver" in err


# --------------------------------------------------------------------------- misc commands
def test_devices_and_version():
    code, out, _ = run_cli(["devices", "--json", "--no-encoders"])
    assert code == 0
    d = json.loads(out)
    assert "CPUExecutionProvider" in d["onnxruntime_providers"] and "video_encoders" not in d
    code, out, _ = run_cli(["devices"])
    assert code == 0 and "ONNX Runtime providers" in out
    code, out, _ = run_cli(["version", "--json"])
    assert code == 0 and json.loads(out)["app"] == "VR Photo Studio"
    code, out, _ = run_cli(["version"])
    assert code == 0 and "VR Photo Studio" in out


def test_quiet_and_verbose_flags(photo, tmp_path):
    code, out, err = run_cli(["depth", photo, "--depth-model", "heuristic", "-o", tmp_path / "o", "-q"])
    assert code == 0 and "[" not in err  # no progress output in quiet mode
    code, out, err = run_cli(["-v", "version"])
    assert code == 0


def test_console_survives_unencodable_output():
    class AsciiOut(io.StringIO):
        def write(self, s):
            s.encode("ascii")
            return super().write(s)

    c = cli.Console(AsciiOut(), AsciiOut())
    c.info("café █")  # falls back to replacement characters instead of crashing
    assert "caf?" in c.out.getvalue()


# --------------------------------------------------------------------------- __main__
def test_main_dispatch_gui_vs_cli(monkeypatch, photo):
    import vrphoto.__main__ as m
    import vrphoto.gui.app as app

    calls = []
    monkeypatch.setattr(app, "run_gui", lambda argv=None: calls.append(argv) or 0)
    assert m.main([]) == 0
    assert calls[-1][1:] == []
    assert m.main([str(photo)]) == 0  # file association / drag & drop
    assert calls[-1][1:] == [str(photo)]
    n = len(calls)
    code = m.main(["version"])
    assert code == 0 and len(calls) == n
    assert m.main(["--help"]) == 0 and len(calls) == n
    assert m.main([str(photo) + ".missing"]) == 2  # not a file -> CLI (unknown command)
    assert m.cli_entry(["version"]) == 0


def test_main_without_stdout_does_not_crash(monkeypatch, isolated_home):
    import vrphoto.__main__ as m

    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    assert m.cli_entry(["version"]) == 0
    assert sys.stdout is not None and sys.stderr is not None
    stream = sys.stdout
    stream.flush()
    log = Path(isolated_home) / "logs" / "cli-output.log"
    assert "VR Photo Studio" in log.read_text(encoding="utf-8")
    stream.close()  # release the file (Windows cannot delete open files)


def test_main_gui_import_failure_is_reported(monkeypatch):
    import vrphoto.__main__ as m

    def fail(files):
        raise ImportError("no Qt")

    monkeypatch.setattr(m, "_launch_gui", fail)
    assert m.main([]) == 1


def _cli_command() -> list[str]:
    exe = shutil.which("vrphoto-cli")
    return [exe] if exe else [sys.executable, "-m", "vrphoto"]


@pytest.mark.timeout(240)
def test_cli_subprocess_version_and_selftest(tmp_path):
    """The installed console entry point works and 'selftest --json' passes here."""
    env = dict(os.environ, VRPHOTO_HOME=str(tmp_path / "home"), QT_QPA_PLATFORM="offscreen")
    proc = subprocess.run([*_cli_command(), "version"], capture_output=True, text=True, timeout=120, env=env)
    assert proc.returncode == 0, proc.stderr
    assert "VR Photo Studio" in proc.stdout
    proc = subprocess.run(
        [*_cli_command(), "selftest", "--json"], capture_output=True, text=True, timeout=200, env=env
    )
    report = json.loads(proc.stdout)
    assert proc.returncode == 0 and report["ok"] is True, json.dumps(report.get("checks"), indent=1)[-3000:]
    proc = subprocess.run(
        [*_cli_command(), "--selftest", "--json"], capture_output=True, text=True, timeout=200, env=env
    )
    start = proc.stdout.find("{")
    result, _ = json.JSONDecoder().raw_decode(proc.stdout[start:])
    assert proc.returncode == 0 and result["ok"] is True


def test_input_error_class_names_in_json(tmp_path):
    code, out, _ = run_cli(["pano", tmp_path / "x.jpg", "--json"])
    assert code == 1 and json_lines(out)[-1]["kind"] == InputError.__name__


def test_wildcards_and_folders_are_expanded(tmp_path, photo):
    folder = tmp_path / "album"
    folder.mkdir()
    for n in ("b.jpg", "a.png", "notes.txt"):
        (folder / n).write_bytes(photo.read_bytes() if n != "notes.txt" else b"x")
    got = cli.expand_inputs("stereo", [str(folder)])
    assert [Path(p).name for p in got] == ["a.png", "b.jpg"]
    got = cli.expand_inputs("depth", [str(folder / "*.jpg")])
    assert [Path(p).name for p in got] == ["b.jpg"]
    assert cli.expand_inputs("capture", [str(folder)]) == [str(folder)]
    with pytest.raises(InputError, match="No files match"):
        cli.expand_inputs("stereo", [str(folder / "*.heic")])
    with pytest.raises(InputError, match="No videos"):
        cli.expand_inputs("video", [str(folder)])
    code, out, err = run_cli(["depth", folder, "--depth-model", "heuristic", "-o", tmp_path / "o", "--json"])
    assert code == 0, err
    res = json_lines(out)[-1]
    assert len(res["outputs"]) == 4  # 2 photos x (16-bit + colour)
