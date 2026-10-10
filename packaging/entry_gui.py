"""PyInstaller entry point for the windowed ``VRPhotoStudio.exe``.

Starts the GUI when launched without arguments (double-click / Start menu); with
arguments it behaves like the CLI (see ``vrphoto.__main__.main``).  Note that in this
windowed executable ``sys.stdout``/``sys.stderr`` are ``None``.
"""

from __future__ import annotations

import multiprocessing

if __name__ == "__main__":
    # Must run before anything else in a frozen Windows app that may use multiprocessing.
    multiprocessing.freeze_support()

    from vrphoto.__main__ import main

    raise SystemExit(main())
