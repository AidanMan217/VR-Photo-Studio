"""Fake COLMAP CLI for the capture tests (records argv, writes plausible outputs).

Behaviour switches (environment): FAKE_TOOL_LOG (jsonl argv log), FAKE_REG (images to
register), FAKE_GLOBAL_REG (images the global mapper registers), FAKE_NO_GLOBAL (no
global_mapper), FAKE_GLOBAL_FAIL, FAKE_GPU_FAIL, FAKE_LEGACY (COLMAP 3.x option names).
"""

import json
import os
import shutil
import struct
import sys
from pathlib import Path

args = sys.argv[1:]
log = os.environ.get("FAKE_TOOL_LOG")
if log:
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(["colmap", *args]) + "\n")
cmd = args[0] if args else "help"


def opt(name, default=None):
    key = "--" + name
    return args[args.index(key) + 1] if key in args else default


def list_images(root):
    root = Path(root)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def write_model(folder: Path, names):
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "images.bin", "wb") as fh:
        fh.write(struct.pack("<Q", len(names)))
        for i, n in enumerate(names):
            fh.write(struct.pack("<I7dI", i + 1, 1.0, 0.0, 0.0, 0.0, 0.1 * i, 0.0, 0.0, 1))
            fh.write(n.encode() + b"\0")
            fh.write(struct.pack("<Q", 1))
            fh.write(struct.pack("<ddq", 1.0, 2.0, -1))
    (folder / "cameras.bin").write_bytes(struct.pack("<Q", 1))
    (folder / "points3D.bin").write_bytes(struct.pack("<Q", 0))


if cmd == "help":
    # like the real one: version header first, then ~70 lines (longer than the runner's
    # 60-line output tail, so a probe must read the whole output)
    print("COLMAP 3.9.1" if os.environ.get("FAKE_LEGACY") else "COLMAP 4.2.1 -- Structure-from-Motion")
    print("Usage:\n  colmap [command] [options]\nExample usage:\n  colmap mapper --image_path IMAGES")
    print("Available commands:\n  help\n  version\n  automatic_reconstructor\n  feature_extractor")
    if not os.environ.get("FAKE_NO_GLOBAL") and not os.environ.get("FAKE_LEGACY"):
        print("  global_mapper")
    for i in range(70):
        print(f"  zz_command_{i:02d}")
    print("  exhaustive_matcher\n  sequential_matcher\n  mapper")
    sys.exit(0)
if "-h" in args:
    if os.environ.get("FAKE_LEGACY"):
        print("  --random_seed arg (=0)\n  --SiftExtraction.use_gpu arg (=1)")
    else:
        print("  --log_target arg (=stderr_and_file)\n  --FeatureExtraction.use_gpu arg (=1)")
    for i in range(80):  # the interesting options come first, like in COLMAP's help
        print(f"  --SiftExtraction.option_{i:02d} arg (=1)")
    sys.exit(0)
if cmd == "feature_extractor":
    gpu = opt("FeatureExtraction.use_gpu", opt("SiftExtraction.use_gpu"))
    if os.environ.get("FAKE_GPU_FAIL") and gpu == "1":
        print("ERROR: SiftGPU not fully supported (CUDA error)", file=sys.stderr)
        sys.exit(1)
    images = list_images(opt("image_path"))
    if opt("ImageReader.single_camera_per_folder") != "1":
        sys.exit(3)
    for i in range(len(images)):
        print(f"I20261010 feature_extraction.cc:271] Processed file [{i + 1}/{len(images)}]", file=sys.stderr)
    Path(opt("database_path")).write_bytes(b"sqlite")
    sys.exit(0)
if cmd in ("exhaustive_matcher", "sequential_matcher"):
    if not Path(opt("database_path")).is_file():
        sys.exit(2)
    print("Processing block [1/1, 1/1]" if cmd == "exhaustive_matcher" else "Processing image [1/1]", file=sys.stderr)
    sys.exit(0)
if cmd in ("mapper", "global_mapper"):
    if cmd == "global_mapper" and os.environ.get("FAKE_GLOBAL_FAIL"):
        print("Failed to create sparse model", file=sys.stderr)
        sys.exit(1)
    images = list_images(opt("image_path"))
    reg = os.environ.get("FAKE_GLOBAL_REG") if cmd == "global_mapper" else None
    n = min(len(images), int(reg or os.environ.get("FAKE_REG", len(images))))
    for i in range(n):
        print(f"Registering image #{i + 1} ({i + 1})", file=sys.stderr)
    if n:
        write_model(Path(opt("output_path")) / "0", images[:n])
    sys.exit(0)
if cmd == "image_undistorter":
    src = Path(opt("input_path"))
    out = Path(opt("output_path"))
    (out / "sparse").mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        shutil.copy2(f, out / "sparse" / f.name)
    shutil.copytree(opt("image_path"), out / "images", dirs_exist_ok=True)
    sys.exit(0)
print(f"unknown command {cmd}", file=sys.stderr)
sys.exit(1)
