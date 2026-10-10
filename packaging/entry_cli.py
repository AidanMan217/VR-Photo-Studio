"""PyInstaller entry point for the console ``vrphoto-cli.exe``.

Examples: ``vrphoto-cli --help``, ``vrphoto-cli --selftest --json``,
``vrphoto-cli addons install capture``.
"""

from __future__ import annotations

import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()

    from vrphoto.__main__ import cli_entry

    raise SystemExit(cli_entry())
