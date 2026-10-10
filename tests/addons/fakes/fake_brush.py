"""Fake Brush CLI for the capture tests: prints training progress and exports a tiny PLY.

Environment: FAKE_TOOL_LOG (jsonl argv log), FAKE_BRUSH_SLEEP (seconds to hang),
FAKE_BRUSH_FAIL (exit 1 without output), FAKE_BRUSH_PLY (PLY file to export instead).
"""

import json
import os
import struct
import sys
import time
from pathlib import Path

args = sys.argv[1:]
log = os.environ.get("FAKE_TOOL_LOG")
if log:
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(["brush", *args]) + "\n")
if "--help" in args or "--version" in args:
    print("Brush - universal splats")
    sys.exit(0)


def opt(name, default=None):
    key = "--" + name
    return args[args.index(key) + 1] if key in args else default


if os.environ.get("FAKE_BRUSH_FAIL"):
    print("thread 'main' panicked: out of memory allocation", file=sys.stderr)
    sys.exit(1)
total = int(opt("total-steps", "30000"))
dataset = Path(args[0])
assert (dataset / "sparse" / "cameras.bin").is_file(), "dataset layout"
for it in range(200, total + 1, 200):
    print(f"[INFO brush_cli] Refine iter {it}, {it * 10} splats.")
    sys.stdout.flush()
if os.environ.get("FAKE_BRUSH_SLEEP"):
    time.sleep(float(os.environ["FAKE_BRUSH_SLEEP"]))
if os.environ.get("FAKE_BRUSH_PLY"):  # export a prepared scene instead of the tiny default
    import shutil

    out = Path(opt("export-path", "."))
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(os.environ["FAKE_BRUSH_PLY"], out / opt("export-name", "export.ply"))
    print("Done training! Took 1s.")
    sys.exit(0)
props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
         "rot_0", "rot_1", "rot_2", "rot_3"]
pts = [(float(i), float(i % 3), float(-i), 0.1, 0.2, 0.3, 2.0, -3.0, -3.0, -3.0, 1.0, 0.0, 0.0, 0.0) for i in range(10)]
header = f"ply\nformat binary_little_endian 1.0\ncomment Exported from Brush\nelement vertex {len(pts)}\n"
header += "".join(f"property float {p}\n" for p in props) + "end_header\n"
out = Path(opt("export-path", "."))
out.mkdir(parents=True, exist_ok=True)
with open(out / opt("export-name", "export.ply"), "wb") as fh:
    fh.write(header.encode("ascii"))
    for p in pts:
        fh.write(struct.pack("<14f", *p))
print("Done training! Took 1s.")
