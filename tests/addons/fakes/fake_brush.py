"""Fake Brush CLI for the capture tests: prints training progress and exports a tiny PLY.

Environment: FAKE_TOOL_LOG (jsonl argv log), FAKE_BRUSH_SLEEP (seconds to hang),
FAKE_BRUSH_FAIL (exit 1 without output), FAKE_BRUSH_PLY (PLY file to export instead),
FAKE_BRUSH_ECHO_INIT (export the dataset's init.ply with every mean moved by up to 2 mm),
FAKE_BRUSH_FAIL_SH0 (exit 1 when ``--sh-degree 0`` is passed, like Brush 0.3.0's SH0 bug).
SHARP 360 polish switches: FAKE_BRUSH_OOM_ABOVE=<px> (exit 1 with an out-of-memory panic
unless ``--max-resolution`` is given and at most <px>), FAKE_BRUSH_REJECT_FLAGS (comma
separated options this Brush "does not know": exit 2 with clap's "unexpected argument"
error when one is passed), FAKE_BRUSH_ECHO_SCALE=<k> (with FAKE_BRUSH_ECHO_INIT: the echoed
means are also scaled by k about their mean, a "diverged" training) and FAKE_BRUSH_ECHO_NAN
(with FAKE_BRUSH_ECHO_INIT: the first rows' x becomes NaN).

The dataset must hold a COLMAP model (``cameras.bin`` or ``cameras.txt`` anywhere below it,
e.g. ``sparse/cameras.bin`` from the undistorter or ``sparse/0/cameras.txt`` from
``capture.write_colmap_text``).  With an ``init.ply`` (a known-camera dataset) every image
listed in ``images.txt`` must exist in ``images/``, every mask must belong to an image,
``init.ply`` must be the only ``.ply`` in the dataset folder and the export folder must lie
outside it (an export there would be read back as an input next time).
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
if "--version" in args:
    print("brush_app 0.3.0 - universal splats (fake)")
    sys.exit(0)
if "--help" in args:
    print("Brush - universal splats")
    sys.exit(0)


def opt(name, default=None):
    key = "--" + name
    return args[args.index(key) + 1] if key in args else default


if os.environ.get("FAKE_BRUSH_FAIL"):
    print("thread 'main' panicked: out of memory allocation", file=sys.stderr)
    sys.exit(1)
if os.environ.get("FAKE_BRUSH_FAIL_SH0") and opt("sh-degree") == "0":
    print("thread 'main' panicked: index out of bounds (sh degree 0)", file=sys.stderr)
    sys.exit(101)
for flag in filter(None, os.environ.get("FAKE_BRUSH_REJECT_FLAGS", "").split(",")):
    if flag.strip() in args:
        print(f"error: unexpected argument '{flag.strip()}' found", file=sys.stderr)
        sys.exit(2)
if os.environ.get("FAKE_BRUSH_OOM_ABOVE"):
    res = opt("max-resolution")
    if res is None or int(res) > int(os.environ["FAKE_BRUSH_OOM_ABOVE"]):
        print("thread 'main' panicked: wgpu error: Out of Memory (allocation failed)", file=sys.stderr)
        sys.exit(1)
total = int(opt("total-steps", "30000"))
dataset = Path(args[0])
cameras = [p for p in dataset.rglob("cameras.*") if p.name in ("cameras.bin", "cameras.txt")]
assert cameras, "dataset layout: no COLMAP cameras.bin / cameras.txt"


def listed_images(images_txt):
    """Names in a COLMAP images.txt: data lines come in pairs (image line, 2D points line)."""
    lines = images_txt.read_text(encoding="utf-8").splitlines()
    while lines and lines[0].startswith("#"):
        lines.pop(0)
    return [ln.split()[9] for ln in lines[0::2] if ln.strip()]


init = dataset / "init.ply"
if init.is_file():
    images = {p.name for p in (dataset / "images").iterdir()}
    for txt in dataset.rglob("images.txt"):
        names = listed_images(txt)
        assert names, "images.txt lists no images"
        missing = sorted(set(names) - images)
        assert not missing, f"images listed in {txt} are missing: {missing}"
    masks = dataset / "masks"
    if masks.is_dir():
        stems = {Path(n).stem for n in images}
        orphans = sorted(p.name for p in masks.iterdir() if p.stem not in stems)
        assert not orphans, f"masks without an image: {orphans}"
    plys = sorted(str(p.relative_to(dataset)) for p in dataset.rglob("*.ply"))
    assert plys == ["init.ply"], f"init.ply must be the only .ply in the dataset folder: {plys}"
    export = Path(opt("export-path", ".")).resolve()
    assert not export.is_relative_to(dataset.resolve()), f"the export folder {export} is inside the dataset"

for it in range(200, total + 1, 200):
    print(f"[INFO brush_cli] Refine iter {it}, {it * 10} splats.")
    sys.stdout.flush()
if os.environ.get("FAKE_BRUSH_SLEEP"):
    time.sleep(float(os.environ["FAKE_BRUSH_SLEEP"]))
out = Path(opt("export-path", "."))
out.mkdir(parents=True, exist_ok=True)
target = out / opt("export-name", "export.ply")
if os.environ.get("FAKE_BRUSH_PLY"):  # export a prepared scene instead of the tiny default
    import shutil

    shutil.copyfile(os.environ["FAKE_BRUSH_PLY"], target)
    print("Done training! Took 1s.")
    sys.exit(0)
if os.environ.get("FAKE_BRUSH_ECHO_INIT"):  # "training" that barely moves the initial splats
    import numpy as np

    assert init.is_file(), "FAKE_BRUSH_ECHO_INIT needs an init.ply in the dataset"
    raw = init.read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii").splitlines()
    assert "format binary_little_endian 1.0" in header
    n = next(int(ln.split()[2]) for ln in header if ln.startswith("element vertex"))
    props = [ln.split()[2] for ln in header if ln.startswith("property float")]
    rec = np.frombuffer(raw[end:], dtype=[(p, "<f4") for p in props], count=n).copy()
    rng = np.random.default_rng(int(opt("seed", "0")))
    for k in ("x", "y", "z"):
        rec[k] += rng.uniform(-0.002, 0.002, n).astype(np.float32)
    if os.environ.get("FAKE_BRUSH_ECHO_SCALE"):  # a "diverged" training: the cloud blows up
        k_scale = float(os.environ["FAKE_BRUSH_ECHO_SCALE"])
        for k in ("x", "y", "z"):
            rec[k] = (rec[k].mean() + k_scale * (rec[k] - rec[k].mean())).astype(np.float32)
    if os.environ.get("FAKE_BRUSH_ECHO_NAN"):
        rec["x"][: max(1, n // 100)] = np.nan
    target.write_bytes(raw[:end] + rec.tobytes())
    print("Done training! Took 1s.")
    sys.exit(0)
props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity", "scale_0", "scale_1", "scale_2",
         "rot_0", "rot_1", "rot_2", "rot_3"]
pts = [(float(i), float(i % 3), float(-i), 0.1, 0.2, 0.3, 2.0, -3.0, -3.0, -3.0, 1.0, 0.0, 0.0, 0.0) for i in range(10)]
header = f"ply\nformat binary_little_endian 1.0\ncomment Exported from Brush\nelement vertex {len(pts)}\n"
header += "".join(f"property float {p}\n" for p in props) + "end_header\n"
with open(target, "wb") as fh:
    fh.write(header.encode("ascii"))
    for p in pts:
        fh.write(struct.pack("<14f", *p))
print("Done training! Took 1s.")
