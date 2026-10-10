"""Command line interface: ``vrphoto-cli <command> [options]``.

Used by power users, scripts and CI.  Every output mode has a subcommand whose options are
generated from the job dataclasses in :mod:`vrphoto.pipeline.jobs` (so the CLI always
matches what the GUI can do)::

    vrphoto-cli stereo photo.jpg --layout half_sbs --strength 1.2
    vrphoto-cli vr180 *.jpg --eye-px 2880 --no-export-scene -o out/
    vrphoto-cli run --spec job.json
    vrphoto-cli serve                      # start the Quest viewer, prints URL + QR code
    vrphoto-cli models list | download da2-small | delete da2-small | path da2-small
    vrphoto-cli addons list | status trellis | install trellis | check trellis
    vrphoto-cli selftest --json            # same as: vrphoto-cli --selftest --json
    vrphoto-cli devices | version

Progress is a single updating line on stderr (or JSON lines on stdout with ``--json``).
Exit codes: 0 success, 1 error (message and hint on stderr), 2 usage error, 130 cancelled
(Ctrl+C cancels the running job gracefully; press it twice to quit at once).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import shutil
import sys
import threading
import time
import types
import typing
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO

from . import APP_NAME, __version__
from .core.errors import Cancelled, VRPhotoError
from .core.log import get_logger

log = get_logger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_CANCELLED = 130

MODE_KINDS = ("depth", "stereo", "parallax", "vr180", "pano", "scene", "object", "capture", "video")
COMMANDS = (*MODE_KINDS, "run", "serve", "models", "addons", "selftest", "devices", "version")
_STRUCTURAL = ("inputs", "common")

#: Help text of the INPUT argument per mode (default: photos).
_INPUT_HELP = {
    "video": "video files or folders of videos (wildcards like *.mp4 work)",
    "capture": "photos, phone videos or folders with them - all used together for one 3D scene",
}

#: Short descriptions of the modes (shown in ``vrphoto-cli --help``).
MODE_SUMMARY = {
    "depth": "save depth maps (16-bit PNG + colour preview)",
    "stereo": "3D photo for Quest players / 3D TVs / anaglyph glasses",
    "parallax": "'living photo' video with a moving virtual camera",
    "vr180": "VR180 stereo photo (or short video) from a normal photo",
    "pano": "360 photo: stereo from a 360 input, or a normal photo in a 360 sphere",
    "scene": "interactive 6DoF 3D photo for the WebXR viewer",
    "object": "person / object you can walk around (passthrough AR)",
    "capture": "many photos or a phone video -> Gaussian splat (add-on)",
    "video": "convert a normal video to 3D (side-by-side) or VR180",
}


# --------------------------------------------------------------------------- console output
class Console:
    """Thread-safe progress line / messages / JSON lines."""

    def __init__(self, out: TextIO, err: TextIO, *, json_mode: bool = False, quiet: bool = False) -> None:
        self.out = out
        self.err = err
        self.json_mode = json_mode
        self.quiet = quiet
        self._lock = threading.Lock()
        self._line_len = 0
        self._last_t = 0.0
        self._last_msg: str | None = None
        self._last_pct = -1
        try:
            self.tty = bool(err.isatty())
        except (AttributeError, ValueError, OSError):
            self.tty = False

    # ---------------------------------------------------------------- low level
    def _write(self, stream: TextIO, text: str) -> None:
        try:
            stream.write(text)
            stream.flush()
        except (OSError, ValueError, UnicodeEncodeError):
            try:
                stream.write(text.encode("ascii", "replace").decode("ascii"))
                stream.flush()
            except (OSError, ValueError):
                pass

    def _clear_line(self) -> None:
        if self._line_len:
            self._write(self.err, "\r" + " " * self._line_len + "\r")
            self._line_len = 0

    # ---------------------------------------------------------------- API
    def progress(self, frac: float, msg: str = "") -> None:
        """Report progress (throttled)."""
        now = time.monotonic()
        pct = int(max(0.0, min(1.0, float(frac))) * 100)
        with self._lock:
            if self.json_mode:
                if msg == self._last_msg and (
                    pct == self._last_pct or (now - self._last_t < 0.1 and pct < 100)
                ):
                    return
                self._last_t, self._last_msg, self._last_pct = now, msg, pct
                self._write(
                    self.out,
                    json.dumps({"type": "progress", "frac": round(float(frac), 4), "msg": msg}) + "\n",
                )
                return
            if self.quiet:
                return
            if self.tty:
                if now - self._last_t < 0.1 and pct < 100 and msg == self._last_msg:
                    return
                self._last_t, self._last_msg = now, msg
                width = max(20, shutil.get_terminal_size((100, 20)).columns - 1)
                text = f"[{pct:3d}%] {msg}"[:width]
                pad = max(0, self._line_len - len(text))
                self._write(self.err, "\r" + text + " " * pad)
                self._line_len = len(text)
            else:
                step = pct // 10
                if msg == self._last_msg and step == self._last_pct:
                    return
                if msg != self._last_msg and now - self._last_t < 1.0 and step == self._last_pct:
                    return
                self._last_t, self._last_msg, self._last_pct = now, msg, step
                self._write(self.err, f"[{pct:3d}%] {msg}\n")

    def end_progress(self) -> None:
        with self._lock:
            self._clear_line()
            self._last_msg = None
            self._last_pct = -1

    def info(self, text: str = "") -> None:
        """Human-readable output (stdout); suppressed in JSON mode."""
        if self.json_mode:
            return
        with self._lock:
            self._clear_line()
            self._write(self.out, text + "\n")

    def note(self, text: str) -> None:
        """Side message on stderr (not in quiet mode)."""
        if self.quiet and not self.json_mode:
            return
        with self._lock:
            self._clear_line()
            self._write(self.err, text + "\n")

    def emit(self, obj: dict[str, Any]) -> None:
        """A JSON line on stdout (JSON mode only)."""
        if not self.json_mode:
            return
        with self._lock:
            self._write(self.out, json.dumps(obj, default=str) + "\n")

    def data(self, obj: Any) -> None:
        """Print a JSON document (JSON mode) - used by list/status style commands."""
        with self._lock:
            self._clear_line()
            self._write(self.out, json.dumps(obj, indent=2, default=str) + "\n")

    def error(self, message: str, hint: str | None = None, *, kind: str = "error") -> None:
        with self._lock:
            self._clear_line()
            if self.json_mode:
                self._write(
                    self.out,
                    json.dumps({"type": "error", "kind": kind, "message": message, "hint": hint}) + "\n",
                )
            self._write(self.err, f"Error: {message}\n")
            if hint:
                self._write(self.err, f"Hint: {hint}\n")


class _ConsoleLogHandler(logging.Handler):
    """Log records to the console's stderr without breaking the progress line."""

    def __init__(self, console: Console, level: int) -> None:
        super().__init__(level)
        self.console = console
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def filter(self, record: logging.LogRecord) -> bool:
        # job warnings are reported in the result summary; do not print them twice
        if record.name == "vrphoto.pipeline.run" and record.levelno == logging.WARNING:
            return False
        return super().filter(record)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:  # pragma: no cover - logging must never raise
            return
        with self.console._lock:
            self.console._clear_line()
            self.console._write(self.console.err, msg + "\n")


# --------------------------------------------------------------------------- option generation
def _escape(text: str) -> str:
    return text.replace("%", "%%")


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    args = typing.get_args(tp)
    origin = typing.get_origin(tp)
    if origin in (typing.Union, types.UnionType) and type(None) in args:
        rest = [a for a in args if a is not type(None)]
        return (rest[0] if len(rest) == 1 else tp), True
    return tp, False


def _none_or(conv: Callable[[str], Any], what: str) -> Callable[[str], Any]:
    def parse(text: str) -> Any:
        t = str(text).strip()
        if t.lower() in ("", "none", "auto", "null"):
            return None
        try:
            return conv(t)
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(f"expected {what} or 'auto', got {text!r}") from None

    return parse


def _typed(conv: Callable[[str], Any], what: str) -> Callable[[str], Any]:
    def parse(text: str) -> Any:
        try:
            return conv(str(text).strip())
        except (TypeError, ValueError):
            raise argparse.ArgumentTypeError(f"expected {what}, got {text!r}") from None

    return parse


def _json_value(text: str) -> Any:
    t = str(text).strip()
    if t.startswith("@"):
        try:
            t = Path(t[1:]).read_text(encoding="utf-8")
        except OSError as e:
            raise argparse.ArgumentTypeError(f"cannot read {t[1:]}: {e}") from None
    if t.lower() in ("", "none", "null"):
        return None
    try:
        return json.loads(t)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid JSON ({e})") from None


def _converter(tp: Any) -> tuple[Callable[[str], Any] | None, bool, str]:
    """``(converter, is_bool, metavar)`` for a dataclass field type."""
    inner, optional = _unwrap_optional(tp)
    origin = typing.get_origin(inner) or inner
    if inner is bool:
        return None, True, ""
    if inner is int:
        return (_none_or(int, "an integer") if optional else _typed(int, "an integer")), False, "N"
    if inner is float:
        return (_none_or(float, "a number") if optional else _typed(float, "a number")), False, "X"
    if inner is str:
        return str, False, "TEXT"
    if origin in (list, dict, tuple):
        return _json_value, False, "JSON"
    return str, False, "VALUE"


def _choices_table() -> dict[tuple[str, str], tuple[str, ...]]:
    """Allowed values per ``(kind, field)``; ``kind`` ``"common"`` for CommonOptions."""
    from .pipeline import run as R

    try:
        from .video.paths import PATH_KINDS

        paths = tuple(PATH_KINDS)
    except Exception:  # pragma: no cover - video module missing
        paths = ("dolly", "circle", "swing", "zoom", "kenburns", "orbit", "custom")
    try:
        from .video.encode import PRESETS

        presets = ("auto", *PRESETS)
    except Exception:  # pragma: no cover
        presets = ("auto", "quest3_hevc", "quest2_hevc", "h264_compat", "preview")
    fmt = ("jpg", "png")
    return {
        ("stereo", "layout"): R.STEREO_LAYOUTS,
        ("stereo", "look_behind"): R.LOOK_BEHIND,
        ("stereo", "image_format"): fmt,
        ("parallax", "path"): paths,
        ("parallax", "output"): R.PARALLAX_OUTPUTS,
        ("parallax", "look_behind"): R.LOOK_BEHIND,
        ("parallax", "encoder_preset"): presets,
        ("vr180", "fill"): R.FILLS,
        ("vr180", "look_behind"): R.LOOK_BEHIND,
        ("vr180", "image_format"): fmt,
        ("pano", "fill"): R.FILLS,
        ("scene", "look_behind"): R.SCENE_LOOK_BEHIND,
        ("object", "method"): R.OBJECT_METHODS,
        ("object", "back"): R.BACKS,
        ("object", "placement"): R.PLACEMENTS,
        ("video", "output"): R.VIDEO_OUTPUTS,
        ("video", "encoder_preset"): presets,
        ("common", "quest_model"): R.QUEST_MODELS,
    }


def _known_values() -> dict[str, list[str]]:
    """Accepted model names for help texts (not enforced: aliases are allowed)."""
    out: dict[str, list[str]] = {}
    for key, mod, fn in (
        ("depth_model", "vrphoto.depth.base", "list_estimators"),
        ("inpaint_model", "vrphoto.inpaint.base", "list_inpainters"),
        ("segment_model", "vrphoto.segment.base", "list_segmenters"),
    ):
        try:
            import importlib

            out[key] = [str(i) for i, _ in getattr(importlib.import_module(mod), fn)()]
        except Exception:  # noqa: BLE001 - help text only
            continue
    out["device"] = ["auto", "cpu", "0", "1"]
    return out


def _meta_tooltip(kind: str, name: str) -> str:
    try:
        from .gui.options_meta import COMMON_META, JOB_META  # Qt-free metadata
    except Exception:  # noqa: BLE001 - optional (GUI metadata)
        return ""
    meta = COMMON_META.get(name) if kind == "common" else JOB_META.get(kind, {}).get(name)
    return str(getattr(meta, "tooltip", "") or "")


#: Units of numeric fields by name suffix (the CLI uses the stored units, e.g. metres not mm).
_UNITS = {
    "_m": "metres",
    "_s": "seconds",
    "_deg": "degrees",
    "_pct": "percent of the image width",
    "_px": "pixels",
}
_METAVARS = {"output_dir": "FOLDER", "name": "NAME", "keyframes": "JSON"}


def _help_for(
    kind: str, f: dataclasses.Field, default: Any, choices: Sequence[str] | None, known: Sequence[str] | None
) -> str:
    text = _meta_tooltip(kind, f.name) or f.name.replace("_", " ").capitalize() + "."
    for suffix, unit in _UNITS.items():
        if f.name.endswith(suffix):
            text += f" Unit: {unit}."
            break
    if choices:
        text += f" Choices: {', '.join(choices)}."
    elif known:
        text += f" Known values: {', '.join(known)}."
    if default is None:
        text += " Default: auto."
    elif isinstance(default, bool):
        text += f" Default: {'on' if default else 'off'}."
    elif default != "":
        text += f" Default: {default}."
    return _escape(text)


def _field_default(f: dataclasses.Field) -> Any:
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    return None


def add_job_options(p: argparse.ArgumentParser, kind: str) -> dict[str, str]:
    """Add one ``--option-name`` per job field and per CommonOptions field.

    Returns ``{dest: "job.<field>" | "common.<field>"}``.  Defaults are not stored in the
    namespace (``argparse.SUPPRESS``), so only options given on the command line override
    the job's defaults.
    """
    from .pipeline.jobs import JOB_TYPES, CommonOptions

    cls = JOB_TYPES[kind]
    choices = _choices_table()
    known = _known_values()
    mapping: dict[str, str] = {}
    job_fields = [f for f in dataclasses.fields(cls) if f.name not in _STRUCTURAL]
    job_names = {f.name for f in job_fields}
    groups = (
        (p.add_argument_group(f"{kind} options"), cls, job_fields, "job", kind),
        (
            p.add_argument_group("common options (all modes)"),
            CommonOptions,
            list(dataclasses.fields(CommonOptions)),
            "common",
            "common",
        ),
    )
    for group, owner, fields, prefix, ckind in groups:
        hints = typing.get_type_hints(owner)
        for f in fields:
            flag_name = f.name.replace("_", "-")
            if prefix == "common" and f.name in job_names:
                flag_name = "common-" + flag_name
            flags = [f"--{flag_name}"]
            if prefix == "common" and f.name == "output_dir":
                flags.insert(0, "-o")
            dest = f"{prefix}__{f.name}"
            default = _field_default(f)
            conv, is_bool, metavar = _converter(hints.get(f.name, str))
            metavar = _METAVARS.get(f.name, metavar)
            ch = choices.get((ckind, f.name))
            help_text = _help_for(ckind, f, default, ch, known.get(f.name) if prefix == "common" else None)
            if is_bool:
                group.add_argument(
                    *flags,
                    dest=dest,
                    action=argparse.BooleanOptionalAction,
                    default=argparse.SUPPRESS,
                    help=help_text,
                )
            else:
                group.add_argument(
                    *flags,
                    dest=dest,
                    type=conv,
                    choices=ch,
                    default=argparse.SUPPRESS,
                    metavar=metavar if not ch else None,
                    help=help_text,
                )
            mapping[dest] = f"{prefix}.{f.name}"
    return mapping


def expand_inputs(kind: str, raw: Sequence[str]) -> list[str]:
    """Absolute input paths; wildcards are expanded (Windows shells do not do it) and, except
    for ``capture`` (which takes folders), folders are replaced by the photos (or videos for
    ``video``) inside them."""
    import glob

    from .core.errors import InputError
    from .core.imaging import is_image, is_video

    want = is_video if kind == "video" else is_image
    out: list[str] = []
    for item in raw:
        r = os.path.expanduser(str(item))
        if any(ch in r for ch in "*?[") and not os.path.exists(r):
            hits = sorted(
                h for h in glob.glob(r) if os.path.isfile(h) or (kind == "capture" and os.path.isdir(h))
            )
            if not hits:
                raise InputError(f"No files match {r}.")
            out += hits
            continue
        if kind != "capture" and os.path.isdir(r):
            files = sorted(str(p) for p in Path(r).iterdir() if p.is_file() and want(p))
            if not files:
                what = "videos" if kind == "video" else "photos"
                raise InputError(f"No {what} were found in the folder {r}.")
            out += files
            continue
        out.append(r)
    return [os.path.abspath(p) for p in out]


def spec_from_namespace(kind: str, ns: argparse.Namespace) -> Any:
    """Build a job spec from parsed arguments (validated by ``job_from_dict``)."""
    from .pipeline.jobs import job_from_dict

    d: dict[str, Any] = {"kind": kind, "inputs": expand_inputs(kind, ns.inputs)}
    common: dict[str, Any] = {}
    for key, value in vars(ns).items():
        if key.startswith("job__"):
            d[key[5:]] = value
        elif key.startswith("common__"):
            common[key[8:]] = value
    if common.get("output_dir"):
        common["output_dir"] = os.path.abspath(os.path.expanduser(str(common["output_dir"])))
    d["common"] = common
    return job_from_dict(d)


# --------------------------------------------------------------------------- parser
class _Parser(argparse.ArgumentParser):
    """ArgumentParser that raises instead of exiting (so main() can return codes)."""

    def error(self, message: str) -> typing.NoReturn:  # type: ignore[override]
        raise _UsageError(self, message)


class _UsageError(Exception):
    def __init__(self, parser: argparse.ArgumentParser, message: str) -> None:
        super().__init__(message)
        self.parser = parser


def _global_parent() -> argparse.ArgumentParser:
    g = argparse.ArgumentParser(add_help=False)
    g.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="machine-readable output: JSON lines on stdout",
    )
    g.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=argparse.SUPPRESS,
        help="show more log messages (-vv for debug)",
    )
    g.add_argument(
        "-q", "--quiet", action="store_true", default=argparse.SUPPRESS, help="only print results and errors"
    )
    return g


def build_parser() -> argparse.ArgumentParser:
    """The full argument parser (mode options generated from the job dataclasses)."""
    from .pipeline.jobs import JOB_TYPES

    parent = _global_parent()
    p = _Parser(
        prog="vrphoto-cli",
        description=f"{APP_NAME} {__version__} - turn photos into VR content for Meta Quest.",
        epilog="Run 'vrphoto-cli <command> --help' for the options of a command.",
        parents=[parent],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    p.add_argument("--selftest", action="store_true", help="same as the 'selftest' command")
    sub = p.add_subparsers(dest="command", metavar="<command>", parser_class=_Parser)
    for kind in MODE_KINDS:
        cls = JOB_TYPES[kind]
        doc = (cls.__doc__ or "").strip()
        sp = sub.add_parser(kind, parents=[parent], help=MODE_SUMMARY.get(kind, doc), description=doc)
        sp.add_argument(
            "inputs", nargs="+", metavar="INPUT",
            help=_INPUT_HELP.get(kind, "photo files or folders of photos (wildcards like *.jpg work)"),
        )  # fmt: skip
        sp.add_argument(
            "--dry-run", action="store_true", help="print the job as JSON and exit without running it"
        )
        add_job_options(sp, kind)
        sp.set_defaults(_mode=kind)

    rp = sub.add_parser(
        "run",
        parents=[parent],
        help="run a job described in a JSON file",
        description="Run a job saved as JSON (see 'vrphoto-cli <mode> ... --dry-run').",
    )
    rp.add_argument("--spec", required=True, metavar="JOB.json", help="job file ('-' reads standard input)")
    rp.add_argument("inputs", nargs="*", metavar="INPUT", help="replace the job's input files")
    rp.add_argument("--dry-run", action="store_true", help="print the job and exit")

    sv = sub.add_parser(
        "serve",
        parents=[parent],
        help="start the WebXR viewer server for the Quest",
        description="Serve the viewer library to the Quest Browser (HTTPS on your network).",
    )
    sv.add_argument("--https-port", type=int, default=None, help="HTTPS port (default from Settings, 8443)")
    sv.add_argument(
        "--http-port", type=int, default=None, help="local HTTP port (default from Settings, 8080)"
    )
    sv.add_argument("--bind", default="0.0.0.0", help="address to listen on (default: all networks)")
    sv.add_argument("--no-qr", action="store_true", help="do not print the QR code")
    sv.add_argument("--duration", type=float, default=None, help=argparse.SUPPRESS)  # tests / automation

    mp = sub.add_parser("models", parents=[parent], help="list / download / delete AI models")
    msub = mp.add_subparsers(dest="action", metavar="<action>", parser_class=_Parser, required=True)
    msub.add_parser("list", parents=[parent], help="list models and whether they are downloaded")
    md = msub.add_parser("download", parents=[parent], help="download a model")
    md.add_argument("model_id")
    md.add_argument(
        "--accept-license",
        action="store_true",
        help="record that you accept a non-commercial model license (as the app asks)",
    )
    mdel = msub.add_parser("delete", parents=[parent], help="delete a downloaded model")
    mdel.add_argument("model_id")
    mpath = msub.add_parser("path", parents=[parent], help="print where a model is stored")
    mpath.add_argument("model_id")

    ap = sub.add_parser("addons", parents=[parent], help="manage optional add-ons (TRELLIS, capture, ...)")
    asub = ap.add_subparsers(dest="action", metavar="<action>", parser_class=_Parser, required=True)
    al = asub.add_parser("list", parents=[parent], help="list add-ons")
    al.add_argument("--stable", action="store_true", help="hide experimental add-ons")
    ast = asub.add_parser("status", parents=[parent], help="show the install status of an add-on")
    ast.add_argument("addon_id")
    ai = asub.add_parser("install", parents=[parent], help="install (or resume / update) an add-on")
    ai.add_argument("addon_id")
    ai.add_argument(
        "--no-check-gpu",
        action="store_true",
        help="verify the install without using the GPU (machines without NVIDIA GPU, CI)",
    )
    ai.add_argument("--force", action="store_true", help="delete the add-on folder first (clean reinstall)")
    ai.add_argument("--hf-token", default=None, help="Hugging Face token for gated models")
    au = asub.add_parser("uninstall", parents=[parent], help="delete an add-on")
    au.add_argument("addon_id")
    ac = asub.add_parser("check", parents=[parent], help="verify an installed add-on (worker --check)")
    ac.add_argument("addon_id")
    ac.add_argument(
        "--check-gpu",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="also test CUDA (default: only when an NVIDIA GPU is present)",
    )

    st = sub.add_parser("selftest", parents=[parent], help="check that the installation works")
    st.add_argument("--out", type=Path, default=None, help="also write the JSON report to this file")
    st.add_argument("--models", action="store_true", help="also download and run the default depth model")

    dv = sub.add_parser(
        "devices", parents=[parent], help="list GPUs, ONNX Runtime providers and video encoders"
    )
    dv.add_argument("--no-encoders", action="store_true", help="skip the (slower) video encoder probe")

    sub.add_parser("version", parents=[parent], help="print version information")
    return p


# --------------------------------------------------------------------------- execution helpers
def _run_cancellable(console: Console, fn: Callable[[Any, Any], Any]) -> Any:
    """Run ``fn(progress, cancel)`` in a worker thread; Ctrl+C cancels it gracefully.

    Returns fn's result; re-raises its exception; raises :class:`Cancelled` after a
    second Ctrl+C without waiting for the worker.
    """
    from .core.types import CancelToken, ProgressReporter

    token = CancelToken()
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["result"] = fn(ProgressReporter(console.progress), token)
        except BaseException as e:  # noqa: BLE001 - re-raised in the main thread
            box["error"] = e

    t = threading.Thread(target=target, name="vrphoto-cli-task", daemon=True)
    t.start()
    interrupts = 0
    while t.is_alive():
        try:
            t.join(0.2)
        except KeyboardInterrupt:
            interrupts += 1
            token.cancel()
            if interrupts >= 2:
                console.end_progress()
                raise Cancelled("Stopped by the user") from None
            console.end_progress()
            console.note("Cancelling... (press Ctrl+C again to stop immediately)")
    console.end_progress()
    if "error" in box:
        raise box["error"]
    return box.get("result")


def _print_result(console: Console, result: Any) -> None:
    d = result.to_dict()
    if console.json_mode:
        console.emit({"type": "result", "ok": True, **d})
        return
    console.info(f"Done in {d.get('elapsed_s', 0):.1f} s.")
    for label, path in d.get("outputs", {}).items():
        console.info(f"  {label}: {path}")
    if d.get("scene_id"):
        console.info(
            f"Added to the VR viewer library (scene '{d['scene_id']}'). Run 'vrphoto-cli serve' to view it on the Quest."
        )
    for w in d.get("warnings", []):
        console.note(f"Warning: {w}")


def _cmd_job(console: Console, spec: Any, dry_run: bool) -> int:
    from .pipeline.context import RunContext
    from .pipeline.jobs import job_to_dict
    from .pipeline.run import run_job

    if dry_run:
        console.data(job_to_dict(spec))
        return EXIT_OK
    ctx = RunContext()
    result = _run_cancellable(
        console, lambda progress, cancel: run_job(spec, ctx, progress=progress, cancel=cancel)
    )
    _print_result(console, result)
    return EXIT_OK


def _cmd_run(console: Console, ns: argparse.Namespace) -> int:
    from .core.errors import InputError
    from .pipeline.jobs import job_from_json

    src = ns.spec
    if src == "-":
        text = sys.stdin.read() if sys.stdin is not None else ""
        spec = job_from_json(text.strip() or "{}")
        base = Path.cwd()
    else:
        p = Path(src).expanduser()
        if not p.is_file():
            raise InputError(f"Job file not found: {p}")
        spec = job_from_json(p)
        base = p.resolve().parent
    if ns.inputs:
        spec.inputs = expand_inputs(spec.kind, ns.inputs)
    else:
        fixed = []
        for i in spec.inputs:
            q = Path(i).expanduser()
            if not q.is_absolute() and not q.exists() and (base / q).exists():
                q = base / q
            fixed.append(str(q))
        spec.inputs = fixed
    return _cmd_job(console, spec, ns.dry_run)


def _qr_text(url: str, encoding: str | None) -> str:
    """QR code as text (light modules drawn, for dark terminals)."""
    import qrcode

    qr = qrcode.QRCode(border=2, error_correction=qrcode.constants.ERROR_CORRECT_L)
    qr.add_data(url)
    qr.make(fit=True)
    m = [[bool(v) for v in row] for row in qr.get_matrix()]
    try:
        "█▀▄".encode(encoding or "ascii")
        blocks = True
    except (UnicodeEncodeError, LookupError):
        blocks = False
    lines: list[str] = []
    if blocks:
        if len(m) % 2:
            m.append([False] * len(m[0]))
        for y in range(0, len(m), 2):
            row = []
            for top, bottom in zip(m[y], m[y + 1]):
                lt, lb = not top, not bottom  # light modules
                row.append("█" if lt and lb else "▀" if lt else "▄" if lb else " ")
            lines.append("".join(row))
    else:
        for r in m:
            lines.append("".join("  " if v else "##" for v in r))
    return "\n".join(lines)


def _cmd_serve(console: Console, ns: argparse.Namespace) -> int:
    from .core.config import get_settings
    from .webxr.server import ViewerServer

    s = get_settings()
    https_port = ns.https_port if ns.https_port is not None else s.https_port
    http_port = ns.http_port if ns.http_port is not None else s.http_port
    kwargs: dict[str, Any] = {"https_port": https_port, "http_port": http_port, "bind": ns.bind}
    try:
        import inspect

        if "files_dir" in inspect.signature(ViewerServer).parameters:
            kwargs["files_dir"] = s.outputs_dir
    except (TypeError, ValueError):  # pragma: no cover
        pass
    srv = ViewerServer(s.library_dir, **kwargs)
    srv.start()
    try:
        urls = srv.urls()
        if console.json_mode:
            console.emit({"type": "server", "urls": urls, "library": str(s.library_dir)})
        else:
            console.info(f"{APP_NAME} viewer is running. Library: {s.library_dir}")
            for k, v in urls.items():
                console.info(f"  {k:8s} {v}")
            main_url = urls.get("https") or urls.get("local") or ""
            if main_url and not ns.no_qr:
                console.info("")
                console.info("Open this address in the Quest Browser (or scan the QR code with your phone):")
                console.info(_qr_text(main_url, getattr(console.out, "encoding", None)))
            console.info("")
            console.info("The browser warns about the certificate the first time: choose Advanced > Proceed.")
            console.info("Press Ctrl+C to stop.")
            _firewall_hint(console, srv)
        t_end = None if ns.duration is None else time.monotonic() + max(0.0, ns.duration)
        try:
            while t_end is None or time.monotonic() < t_end:
                time.sleep(0.2 if t_end is None else min(0.2, max(0.0, t_end - time.monotonic())))
        except KeyboardInterrupt:
            console.note("Stopping the viewer server...")
    finally:
        srv.stop()
    console.emit({"type": "stopped"})
    return EXIT_OK


def _firewall_hint(console: Console, srv: Any) -> None:
    if sys.platform != "win32" or srv.https_port is None:
        return
    try:
        from .webxr.net import firewall_status

        if firewall_status(int(srv.https_port)) == "blocked":
            console.note(
                "Warning: Windows Firewall may block the Quest. Allow 'VR Photo Studio' for private networks "
                "in Windows Security, or use 'Allow through firewall' in the app."
            )
    except Exception:  # noqa: BLE001 - hint only
        log.debug("Firewall check failed", exc_info=True)


def _cmd_models(console: Console, ns: argparse.Namespace) -> int:
    from .core.config import get_settings
    from .models import download, registry

    if ns.action == "list":
        rows = []
        for spec in registry.list_specs():
            rows.append({
                "id": spec.id, "task": spec.task, "title": spec.title, "size_mb": spec.size_mb,
                "license": spec.license, "noncommercial": bool(spec.noncommercial),
                "downloaded": bool(download.is_downloaded(spec.id)),
            })  # fmt: skip
        if console.json_mode:
            console.data(rows)
        else:
            console.info(f"{'ID':20s} {'TASK':9s} {'SIZE':>8s}  {'STATUS':11s} TITLE")
            for r in rows:
                status = "downloaded" if r["downloaded"] else "-"
                nc = "  [non-commercial]" if r["noncommercial"] else ""
                console.info(
                    f"{r['id']:20s} {r['task']:9s} {r['size_mb']:7.0f}M  {status:11s} {r['title']}{nc}"
                )
            console.info(f"Models folder: {get_settings().models_dir}")
        return EXIT_OK
    spec = registry.get_spec(ns.model_id)
    if ns.action == "download":
        if spec.noncommercial:
            s = get_settings()
            if getattr(ns, "accept_license", False):
                s.accept_license(spec.id)
                s.save()
            console.note(f"Note: {spec.title} is for non-commercial use only ({spec.license}).")
        path = _run_cancellable(
            console, lambda progress, cancel: download.ensure_model(spec.id, progress=progress, cancel=cancel)
        )
        if console.json_mode:
            console.emit({"type": "result", "ok": True, "model": spec.id, "path": str(path)})
        else:
            console.info(f"{spec.title} is ready: {path}")
        return EXIT_OK
    if ns.action == "delete":
        download.delete_model(spec.id)
        if console.json_mode:
            console.emit({"type": "result", "ok": True, "model": spec.id, "deleted": True})
        else:
            console.info(f"Deleted {spec.title}.")
        return EXIT_OK
    # path
    local = getattr(download, "local_path", None)
    path = (
        local(spec.id)
        if (local is not None and download.is_downloaded(spec.id))
        else download.model_dir(spec.id)
    )
    if console.json_mode:
        console.emit(
            {
                "type": "result",
                "ok": True,
                "model": spec.id,
                "path": str(path),
                "downloaded": bool(download.is_downloaded(spec.id)),
            }
        )
    else:
        console.info(str(path))
    return EXIT_OK


def _nvidia_present() -> bool:
    try:
        from .models.gpu import nvidia_vram_mb

        return nvidia_vram_mb() is not None
    except Exception:  # noqa: BLE001
        return False


def _cmd_addons(console: Console, ns: argparse.Namespace) -> int:
    from .addons import manager

    if ns.action == "list":
        rows = []
        for m in manager.list_addons(include_experimental=not ns.stable):
            st = manager.status(m.id)
            rows.append({**m.to_dict(), "installed": bool(st.get("installed")), "size_mb": st.get("size_mb")})
        if console.json_mode:
            console.data(rows)
        else:
            console.info(f"{'ID':10s} {'STATUS':10s} TITLE")
            for r in rows:
                status = "installed" if r["installed"] else "-"
                flags = " [experimental]" if r["experimental"] else ""
                flags += " [non-commercial]" if r["noncommercial"] else ""
                console.info(f"{r['id']:10s} {status:10s} {r['title']}{flags}")
        return EXIT_OK
    m = manager.get_manifest(ns.addon_id)
    if ns.action == "status":
        st = {"id": m.id, "title": m.title, **manager.status(m.id)}
        if console.json_mode:
            console.data(st)
        else:
            for k, v in st.items():
                console.info(f"{k:16s} {v}")
        return EXIT_OK
    if ns.action == "install":
        for n in m.notices():
            console.note(f"Note: {n}")
        from .pipeline.run import _call_opt

        _run_cancellable(
            console,
            lambda progress, cancel: _call_opt(
                manager.install,
                m.id,
                progress=progress,
                cancel=cancel,
                hf_token=ns.hf_token,
                check_gpu=not ns.no_check_gpu,
                force=ns.force,
            ),
        )
        if console.json_mode:
            console.emit({"type": "result", "ok": True, "addon": m.id, "status": manager.status(m.id)})
        else:
            console.info(f"{m.title} is installed.")
        return EXIT_OK
    if ns.action == "uninstall":
        manager.uninstall(m.id)
        if console.json_mode:
            console.emit({"type": "result", "ok": True, "addon": m.id, "uninstalled": True})
        else:
            console.info(f"{m.title} was removed.")
        return EXIT_OK
    # check
    check_gpu = ns.check_gpu if ns.check_gpu is not None else _nvidia_present()
    info = _run_cancellable(
        console, lambda progress, cancel: manager.check(m.id, check_gpu=check_gpu, cancel=cancel)
    )
    if console.json_mode:
        console.emit({"type": "result", "ok": True, "addon": m.id, "check_gpu": check_gpu, "info": info})
    else:
        console.info(f"{m.title} works{' (GPU tested)' if check_gpu else ' (import check, GPU not tested)'}.")
        for k, v in (info or {}).items():
            console.info(f"  {k}: {v}")
    return EXIT_OK


def _cmd_selftest(console: Console, ns: argparse.Namespace) -> int:
    from .selftest import check_names, run_selftest

    names = check_names(bool(ns.models))

    def step(name: str) -> None:
        if not console.json_mode:
            i = names.index(name) if name in names else 0
            console.progress(i / max(1, len(names)), f"Checking {name.replace('_', ' ')}")

    report = run_selftest(ns.out, models=bool(ns.models), progress=step)
    console.end_progress()
    if console.json_mode:
        console.data(report)
    else:
        console.info(
            f"{APP_NAME} {report['version']} self-test (Python {report['python']}, {report['platform']})"
        )
        for name, c in report["checks"].items():
            mark = "OK  " if c["ok"] else ("FAIL" if c.get("required", True) else "info")
            line = f"  [{mark}] {name:18s} {c['seconds']:6.2f} s"
            if not c["ok"]:
                line += f"  {c.get('error', '')}"
            console.info(line)
        console.info(
            "Result: "
            + ("OK - everything works." if report["ok"] else "PROBLEMS FOUND: " + ", ".join(report["failed"]))
        )
        if ns.out:
            console.info(f"Report written to {ns.out}")
    return EXIT_OK if report["ok"] else EXIT_ERROR


def _cmd_devices(console: Console, ns: argparse.Namespace) -> int:
    from dataclasses import asdict

    from .models.gpu import list_devices, nvidia_vram_mb
    from .models.ort import available_providers

    info: dict[str, Any] = {
        "gpus": [asdict(d) for d in list_devices()],
        "nvidia_vram_mb": nvidia_vram_mb(),
        "onnxruntime_providers": available_providers(),
    }
    if not ns.no_encoders:
        try:
            from .video.encode import probe_encoders

            info["video_encoders"] = probe_encoders()
        except VRPhotoError as e:
            info["video_encoders"] = {"error": str(e)}
    if console.json_mode:
        console.data(info)
        return EXIT_OK
    console.info("GPUs:")
    if info["gpus"]:
        for d in info["gpus"]:
            vram = f", {d['vram_mb'] / 1024:.0f} GB" if d.get("vram_mb") else ""
            console.info(f"  [{d['index']}] {d['name']} ({d['vendor']}{vram})")
    else:
        console.info(
            "  (none detected"
            + (" - GPU listing is only available on Windows)" if sys.platform != "win32" else ")")
        )
    if info["nvidia_vram_mb"]:
        console.info(f"NVIDIA VRAM: {info['nvidia_vram_mb'] / 1024:.1f} GB")
    console.info("ONNX Runtime providers: " + (", ".join(info["onnxruntime_providers"]) or "none"))
    enc = info.get("video_encoders")
    if isinstance(enc, dict) and "error" not in enc:
        console.info("Video encoders: " + (", ".join(k for k, v in enc.items() if v) or "none"))
    return EXIT_OK


def _cmd_version(console: Console, ns: argparse.Namespace) -> int:
    import platform

    from .core.config import data_root, is_frozen

    info = {
        "app": APP_NAME,
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "frozen": is_frozen(),
        "data_folder": str(data_root()),
    }
    if console.json_mode:
        console.data(info)
    else:
        console.info(
            f"{APP_NAME} {__version__} (Python {info['python']}, {info['platform']}"
            f"{', packaged' if info['frozen'] else ''})"
        )
        console.info(f"Data folder: {info['data_folder']}")
    return EXIT_OK


# --------------------------------------------------------------------------- main
def _prepare_argv(argv: Sequence[str]) -> list[str]:
    """Map the top-level ``--selftest`` flag to the ``selftest`` command."""
    args = list(argv)
    if "--selftest" in args:
        first_cmd = next((a for a in args if a in COMMANDS), None)
        idx = args.index("--selftest")
        if first_cmd is None or args.index(first_cmd) > idx:
            args.pop(idx)
            return ["selftest", *args]
    return args


def _safe_stream(stream: TextIO | None) -> TextIO:
    if stream is None:
        return open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 - lives for the process
    try:
        stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError, OSError):
        pass
    return stream


def _setup_logging(console: Console, verbose: int, quiet: bool) -> _ConsoleLogHandler:
    from .core.log import setup_logging

    setup_logging(logging.DEBUG if verbose >= 2 else logging.INFO, console=False)
    if verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    level = (
        logging.DEBUG
        if verbose >= 2
        else logging.INFO
        if verbose == 1
        else logging.ERROR
        if quiet
        else logging.WARNING
    )
    handler = _ConsoleLogHandler(console, level)
    logging.getLogger().addHandler(handler)
    return handler


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code (never calls ``sys.exit``)."""
    args = _prepare_argv(sys.argv[1:] if argv is None else argv)
    out = _safe_stream(sys.stdout)
    err = _safe_stream(sys.stderr)
    try:
        parser = build_parser()
    except Exception as e:  # pragma: no cover - broken installation
        err.write(f"Error: the command line could not be set up: {e}\n")
        return EXIT_ERROR
    try:
        ns = parser.parse_args(args)
    except _UsageError as e:
        e.parser.print_usage(err)
        err.write(f"{e.parser.prog}: error: {e}\n")
        return EXIT_USAGE
    except SystemExit as e:  # --help / --version
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else EXIT_USAGE)
        return int(code)
    json_mode = bool(getattr(ns, "json", False))
    quiet = bool(getattr(ns, "quiet", False))
    console = Console(out, err, json_mode=json_mode, quiet=quiet)
    if ns.command is None:
        if getattr(ns, "selftest", False):
            ns = parser.parse_args(["selftest", *[a for a in args if a != "--selftest"]])
        else:
            parser.print_help(out)
            return EXIT_USAGE
    handler = _setup_logging(console, int(getattr(ns, "verbose", 0) or 0), quiet)
    try:
        return _dispatch(console, ns)
    except Cancelled:
        console.end_progress()
        console.error("Cancelled.", kind="cancelled")
        return EXIT_CANCELLED
    except KeyboardInterrupt:
        console.end_progress()
        console.error("Cancelled.", kind="cancelled")
        return EXIT_CANCELLED
    except VRPhotoError as e:
        console.end_progress()
        console.error(str(e), getattr(e, "hint", None), kind=type(e).__name__)
        return EXIT_ERROR
    except Exception as e:  # noqa: BLE001 - last resort: friendly message + log
        console.end_progress()
        log.exception("Unexpected error in the command line tool")
        from .core.log import setup_logging

        path = setup_logging()
        console.error(
            f"Unexpected error: {type(e).__name__}: {e}",
            f"Details are in the log file: {path}" if path else None,
            kind="unexpected",
        )
        return EXIT_ERROR
    finally:
        logging.getLogger().removeHandler(handler)


def _dispatch(console: Console, ns: argparse.Namespace) -> int:
    cmd = ns.command
    if cmd in MODE_KINDS:
        spec = spec_from_namespace(cmd, ns)
        return _cmd_job(console, spec, bool(ns.dry_run))
    handlers: dict[str, Callable[[Console, argparse.Namespace], int]] = {
        "run": _cmd_run,
        "serve": _cmd_serve,
        "models": _cmd_models,
        "addons": _cmd_addons,
        "selftest": _cmd_selftest,
        "devices": _cmd_devices,
        "version": _cmd_version,
    }
    return handlers[cmd](console, ns)


__all__ = [
    "main",
    "build_parser",
    "add_job_options",
    "spec_from_namespace",
    "Console",
    "MODE_KINDS",
    "COMMANDS",
    "EXIT_OK",
    "EXIT_ERROR",
    "EXIT_USAGE",
    "EXIT_CANCELLED",
]
