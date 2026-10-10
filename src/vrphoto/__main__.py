"""Program entry points.

* :func:`main` - ``VRPhotoStudio.exe`` / ``vrphoto`` / ``python -m vrphoto``: without
  arguments (or with only file paths, e.g. a photo double-clicked through a file
  association) the GUI starts; any other arguments run the command line interface.
* :func:`cli_entry` - ``vrphoto-cli.exe`` / ``vrphoto-cli``: always the CLI.

In the windowed (``console=False``) PyInstaller exe ``sys.stdout``/``sys.stderr`` are
``None``; CLI output is then written to ``<data folder>/logs/cli-output.log`` (or
discarded) so nothing crashes.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
from collections.abc import Sequence
from pathlib import Path

_SUBCOMMANDS = frozenset(
    {"depth", "stereo", "parallax", "vr180", "pano", "scene", "object", "capture", "video",
     "run", "serve", "models", "addons", "selftest", "devices", "version"}
)  # fmt: skip


def _ensure_streams() -> None:
    """Give a windowed (frozen) process usable stdout/stderr."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    stream = None
    try:
        from .core.config import data_root

        logs = data_root() / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        stream = open(logs / "cli-output.log", "w", encoding="utf-8", errors="replace")  # noqa: SIM115
    except OSError:
        stream = None
    if stream is None:
        stream = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115 - process lifetime
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


def _wants_gui(args: Sequence[str]) -> bool:
    """True for no arguments, or only existing file/folder paths (file association / drag & drop)."""
    if not args:
        return True
    if args[0] in _SUBCOMMANDS:
        return False
    for a in args:
        if not a or a.startswith("-"):
            return False
        try:
            if not Path(a).exists():
                return False
        except (OSError, ValueError):
            return False
    return True


def _launch_gui(files: list[str]) -> int:
    import inspect

    from .gui.app import run_gui

    try:
        params = [
            p for p in inspect.signature(run_gui).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
        ]  # fmt: skip
    except (TypeError, ValueError):
        params = []
    # run_gui takes a full argument vector (argv[0] = program, then files to open)
    prog = sys.argv[0] if sys.argv and sys.argv[0] else "vrphoto"
    rc = run_gui([prog, *files]) if params else run_gui()
    return int(rc or 0)


def main(argv: Sequence[str] | None = None) -> int:
    """GUI without arguments (or with only file paths); otherwise the CLI.  Returns the exit code."""
    multiprocessing.freeze_support()
    args = list(sys.argv[1:] if argv is None else argv)
    if _wants_gui(args):
        try:
            return _launch_gui(args)
        except ImportError as e:
            _ensure_streams()
            sys.stderr.write(
                f"The VR Photo Studio window could not be started ({e}).\n"
                "Run 'vrphoto-cli --help' for the command line, or reinstall the app.\n"
            )
            return 1
    _ensure_streams()
    from .cli import main as cli_main

    return cli_main(args)


def cli_entry(argv: Sequence[str] | None = None) -> int:
    """Console entry point: always the CLI.  Returns the exit code."""
    multiprocessing.freeze_support()
    _ensure_streams()
    from .cli import main as cli_main

    return cli_main(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    sys.exit(main())
