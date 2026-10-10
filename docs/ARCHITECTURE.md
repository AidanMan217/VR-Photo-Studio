# VR Photo Studio - architecture (developer overview)

This document is for people who want to work on the code. The frozen interfaces between modules
(signatures, conventions, the scene manifest and the core algorithms) are in
[CONTRACTS.md](CONTRACTS.md) - read that first; this page explains how the pieces fit together.

## 1. Big picture

```
                 ┌──────────────── Windows PC ────────────────────────────────────────────┐
 photos/videos → │ GUI (PySide6)  ─┐                                                      │
                 │ CLI (argparse) ─┴─> JobSpec (pipeline/jobs.py) ──> JobQueue ──> run_job │
                 │                                                     │                   │
                 │      models/ (ONNX Runtime + DirectML)  <───────────┤                   │
                 │      depth/  segment/  inpaint/                     │                   │
                 │      geometry/ (layers, warp, stereo, equirect, ODS, mesh, splat, glTF)  │
                 │      video/ (ffmpeg encode/decode, paths, spatial metadata)              │
                 │      addons/ ──> uv venvs (PyTorch/CUDA workers, JSON-lines protocol)    │
                 │                                                     ▼                   │
                 │   outputs: Pictures\VR Photo Studio\*_LR.jpg, *_180_LR.mp4, *.glb ...    │
                 │   viewer library: %LOCALAPPDATA%\VRPhotoStudio\library\<id>\scene.json   │
                 │                                                     │                   │
                 │   webxr/server.py (HTTPS :8443 self-signed, HTTP :8080) ── viewer/ (JS)  │
                 └─────────────────────────────────────────────────────┼───────────────────┘
                                         Wi-Fi (https://<pc-ip>:8443) or USB (adb reverse :8080)
                                                                       ▼
                                                    Quest Browser (WebXR: Enter VR / Enter AR)
```

* All heavy work happens in **worker threads** (the GUI never blocks); every long-running
  function takes `progress=` and `cancel=` (see `core/progress.py`).
* Core AI models are **ONNX** files run with **onnxruntime-directml** (any DirectX 12 GPU) and
  downloaded on first use. Large PyTorch/CUDA tools are **add-ons**, each installed by the app
  into its own `uv` virtual environment and run as a separate process.

## 2. Packages (`src/vrphoto/`)

| Package | Responsibility |
|---|---|
| `core/` | Shared types (`Intrinsics`, `Pose`, `DepthResult`, `LayeredImage`, `Mesh`, `SplatCloud`, ...), errors, settings and data folders (`config.py`), logging, image I/O incl. HEIC/EXIF/XMP (`imaging.py`), push-pull hole filling (`fill.py`), progress/cancel tokens. |
| `models/` | Model registry (`registry.py` + pinned `models.lock.json`), resumable verified downloads (`download.py`), ONNX Runtime sessions with DirectML → CPU fallback (`ort.py`), GPU discovery (`gpu.py`). |
| `depth/` | Depth estimators (Depth Anything V2/3, MoGe-2, heuristic fallback), post-processing (normalise, guided filter, temporal smoothing), 360 depth via cube faces (`pano.py`). |
| `segment/` | Subject cut-outs (BiRefNet, IS-Net, U2-Net, GrabCut) and matte clean-up. |
| `inpaint/` | "Look behind" inpainting (LaMa, MI-GAN, OpenCV, push-pull) with crop-based processing. |
| `geometry/` | Camera maths, layered depth images and background hallucination, backward warping, stereo pairs and layouts, equirectangular/cubemap/VR180 projection, ODS 360 stereo, depth meshes, billboards, Gaussian splats, glTF/OBJ writers. |
| `video/` | Camera paths, ffmpeg encoding presets (NVENC/x265/x264), decoding, frame renderers, Google spatial-media (V1/V2) metadata injection. |
| `webxr/` | Scene library (`scene.py`), self-signed certificate (`cert.py`), LAN helpers + firewall (`net.py`), threaded HTTPS/HTTP server with Range support (`server.py`), adb/MTP helpers (`quest.py`), and the static three.js/Spark viewer (`viewer/`, validated by `scene.schema.json`). |
| `addons/` | Add-on manifests (`manifests/*.toml`), install/uninstall/run via `uv` (`manager.py`), clean subprocess environments (`env.py`), the worker JSON-lines protocol (`protocol.py`), worker scripts that run inside the add-on venvs (`workers/`), capture-to-splat orchestration (`capture.py`). |
| `pipeline/` | Job specifications (`jobs.py`), run context with cached models (`context.py`), `run_job` for every mode (`run.py`), Quest file naming (`naming.py`), a Qt-free job queue (`queue.py`). |
| `gui/` | PySide6 main window, panels and dialogs; marshals queue events to the Qt thread. |
| `cli.py`, `__main__.py`, `selftest.py` | Command line (`vrphoto-cli <mode> ...`, `run --spec job.json`, `addons ...`, `serve`, `models`, `selftest`); `__main__.main()` starts the GUI without arguments and the CLI with arguments. |
| `third_party/` | Vendored third-party code/assets (with their licenses). |

## 3. Data flow of a typical job (3D Scene)

1. GUI/CLI builds a `SceneJob` (`pipeline/jobs.py`), JSON round-trippable, and submits it to the
   `JobQueue`.
2. `run_job` loads the image (`core.imaging`, EXIF orientation, HEIC), limits its size and asks the
   `RunContext` for the depth estimator, segmenter and inpainter (models are cached per process).
3. Depth → inverse depth (algorithm A), intrinsics from user/MoGe/EXIF/65° (priority order).
4. `geometry.layers.build_layers` splits foreground/background and inpaints a band behind depth
   edges (algorithm C).
5. `geometry.mesh.layered_meshes` turns the layers into textured meshes (algorithm H) → `write_glb`.
6. `webxr.scene.export_scene` writes `library/<id>/scene.json` + assets + thumbnail.
7. The `ViewerServer` serves the library and the viewer; the Quest Browser loads
   `view.html?scene=<id>`, which reads `scene.json` and renders it with three.js (meshes) or Spark
   (splats), in WebXR `immersive-vr` or `immersive-ar`.

Flat outputs (3D Photo, VR180, Living Photo, Video) skip steps 5-7 and render views with
`geometry.warp.render_view` (algorithm D), compose layouts (`geometry.stereo.compose`), project to
equirect (algorithm E) and encode with `video.encode.VideoWriter`; file names carry the Quest
tokens and videos get spatial metadata (algorithm K).

## 4. Add-ons

* A manifest (`addons/manifests/<id>.toml`) describes the add-on: title, license (non-commercial /
  region flags), VRAM and disk needs, Python version, install steps (uv pip installs, downloads,
  binaries) and the worker script.
* `addons.manager.install()` uses the bundled **uv** (`_internal/bin/uv.exe` in the frozen app,
  `uv.find_uv_bin()` from source) to create `%LOCALAPPDATA%\VRPhotoStudio\addons\<id>\` and install
  the packages.
* `run()` starts `<venv>\python.exe <worker.py>`, writes one JSON request line and reads JSON-lines
  messages (`hello`, `progress`, `log`, `result`/`error`) - see `addons/protocol.py`. `--check`
  verifies an install by importing the heavy packages only.
* CI checks real installs on Windows with `.github/workflows/addons.yml`
  (`vrphoto-cli addons install <id> --no-check-gpu`, then `vrphoto-cli addons check <id>`).

## 5. Where things live at run time

| | From source | Frozen (installed app) |
|---|---|---|
| Package data (viewer, manifests, workers, model lock) | `src/vrphoto/...` | `<install>\_internal\vrphoto\...` |
| Finding package data | `core.config.resource_path(...)` works in both cases | |
| uv | `uv.find_uv_bin()` | `<install>\_internal\bin\uv.exe` |
| ffmpeg | imageio-ffmpeg's binary | `<install>\_internal\imageio_ffmpeg\binaries\ffmpeg-*.exe` |
| User data (settings, models, add-ons, library, certs, logs, cache) | `data_root()` = `$VRPHOTO_HOME` or `%LOCALAPPDATA%\VRPhotoStudio` (`~/.local/share/vrphoto` on Linux) | same |
| Outputs | `Pictures\VR Photo Studio` (setting `output_dir`) | same |

In the windowed `VRPhotoStudio.exe`, `sys.stdout`/`sys.stderr` are `None`; never `print()` - use
`core.log.get_logger(__name__)` (logs go to `<data_root>\logs\vrphoto.log`).

## 6. Running from source

Requirements: Python 3.12 (3.11-3.13 work), [uv](https://docs.astral.sh/uv/) (or pip), Git.
On Linux, Qt needs `libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3`.

```powershell
git clone https://github.com/AidanMan217/VR-Photo-Studio
cd VR-Photo-Studio
uv sync --python 3.12 --extra dev    # .venv: the app (editable) + test/build tools; writes uv.lock
uv run vrphoto                       # GUI
uv run vrphoto-cli --help            # CLI
uv run vrphoto-cli --selftest --json # health check
```

Without uv: `python -m venv .venv`, activate it, `pip install -e ".[dev]"`, then run `vrphoto` /
`vrphoto-cli` / `pytest` directly.

On Linux the app installs `onnxruntime` (CPU) instead of `onnxruntime-directml`. Set
`VRPHOTO_HOME` to keep a development profile separate from your real one.

## 7. Tests

```powershell
uv run pytest -m "not models"            # everything that runs offline (what CI runs)
uv run pytest tests/geometry -q          # one area
uv run pytest packaging/tests -q         # packaging/CI checks
$env:VRPHOTO_TEST_PYINSTALLER = "1"; $env:VRPHOTO_TEST_FROZEN_SELFTEST = "1"; uv run pytest packaging/tests/test_frozen_build.py  # real build (slow)
cd tests/viewer; npm install; npx playwright install chromium; npm test   # WebXR viewer in a browser
```

* `tests/conftest.py` gives every test its own `VRPHOTO_HOME` and synthetic scenes with known
  geometry (two planes, gradients, a person silhouette, a 360 checkerboard).
* Real model weights are never needed for the normal suite: `scripts/make_test_onnx.py` builds tiny
  stand-in ONNX models with the real input/output names, wired in through
  `VRPHOTO_MODEL_<ID>=<path>` environment overrides.
* Markers: `gui` (Qt offscreen; set `QT_QPA_PLATFORM=offscreen`), `slow`, `models` (real downloads,
  run by `models.yml` with `VRPHOTO_REAL_MODELS=1`), `windows`, `network`.
* Lint: `uv run ruff check .` (configuration in `pyproject.toml`).

## 8. Packaging

| File | Purpose |
|---|---|
| `packaging/vrphoto.spec` | PyInstaller 6 onedir build: **two executables** (`VRPhotoStudio.exe` windowed, `vrphoto-cli.exe` console) sharing one `VRPhotoStudio\_internal` folder. No UPX. |
| `packaging/spec_helpers.py` | Unit-tested logic used by the spec: version, package data list, hidden imports, excluded modules, Qt library filtering, licence collection, uv location, bundle check (`python packaging/spec_helpers.py check dist/VRPhotoStudio`). |
| `packaging/entry_gui.py`, `entry_cli.py` | Entry scripts (`multiprocessing.freeze_support()` then `vrphoto.__main__.main` / `cli_entry`). |
| `packaging/hooks/hook-pillow_heif.py` | Collects libheif/libde265/x265 DLLs for HEIC support. |
| `packaging/version_info.py` | Windows version resource generated from `vrphoto.__version__`. |
| `packaging/make_icon.py` | Draws `packaging/icon.ico` (16-256 px) with Pillow; `--png` for other uses. |
| `packaging/licenses/` | Full GPL/LGPL/Apache texts shipped in `_internal/licenses/` (with each dependency's own licence files). |
| `packaging/installer.iss` | Inno Setup 6 installer: per-user by default, optional firewall rule (elevated only), asks before deleting user data on uninstall. |
| `packaging/build_windows.ps1` | One-command local build: venv → pip install → icon → PyInstaller → bundle check → self-test → installer (→ zip). |
| `packaging/ci/addon_matrix.py` | Builds the add-on CI matrix from the manifests. |

Local Windows build:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 -Clean -Zip
# -> dist\VRPhotoStudio\ and packaging\Output\VRPhotoStudio-Setup-<ver>.exe
```

The spec also builds on Linux (used to smoke-test the bundle):
`pyinstaller packaging/vrphoto.spec --noconfirm --distpath /tmp/dist --workpath /tmp/build`.
Set `VRPHOTO_STRICT_BUNDLE=1` to make the build fail if the bundle check finds a problem.

## 9. Continuous integration (`.github/workflows/`)

| Workflow | When | What |
|---|---|---|
| `ci.yml` | every push and pull request | ruff; pytest on ubuntu-latest and windows-latest (Python 3.12, Qt offscreen, `-m "not models"`, plus `packaging/tests`); a Linux PyInstaller bundle smoke test (`packaging/tests/test_frozen_build.py`: build, frozen `--selftest`, GUI start); WebXR viewer tests with Playwright/Chromium (`tests/viewer`, with Python for the fixture server). |
| `build.yml` | push to `main` and `claude/**`, tags `v*`, manual | windows-latest: icon → PyInstaller → bundle check → frozen `vrphoto-cli.exe --selftest --json` → Inno Setup → silent per-user install → installed self-test → silent uninstall (keeps user data) → upload installer, portable zip and `SHA256SUMS.txt`; on `v*` tags a GitHub release. |
| `models.yml` | weekly + manual | `scripts/pin_models.py --check` against Hugging Face/GitHub and the real-model tests (`pytest -m models`), with the model cache keyed by `models.lock.json`. Manual runs can re-pin and upload a fresh lock file. |
| `addons.yml` | manual (`addon` = id, list or `all`) | windows-latest matrix over the add-on manifests: `vrphoto-cli addons install <id> --no-check-gpu` + `addons check <id>`; experimental add-ons may fail without failing the run. |

Third-party actions are pinned to commit SHAs; GitHub's own actions to their current major versions.

## 10. Release process

1. Update the version in **both** `src/vrphoto/__init__.py` (`__version__`) and `pyproject.toml`
   (`[project] version`), e.g. `0.2.0` (pre-releases: `0.2.0rc1`). The build stops with a clear
   error if the two differ (`python packaging/spec_helpers.py version` checks it).
2. Move the "Unreleased" notes in `CHANGELOG.md` under the new version with today's date.
3. Make sure `ci.yml` and `build.yml` are green on `main` (download the build artifact and run
   through [QUEST_CHECKLIST.md](QUEST_CHECKLIST.md) on a real PC + Quest for bigger releases).
4. If models changed, run the **Models** workflow with *refresh_lock* and commit the new
   `models.lock.json`.
5. Tag and push: `git tag v0.2.0 && git push origin v0.2.0`. `build.yml` checks that the tag matches
   `__version__`, builds, tests and publishes the GitHub release with the installer, the portable zip
   and checksums (versions containing letters are marked as pre-release).

## 11. Conventions (summary)

* Images are `HxWx3 uint8 RGB` numpy arrays; masks/alpha `float32` in `[0, 1]`.
* Source camera frame: OpenCV (x right, y down, z forward, metres); exported 3D: glTF (y up,
  camera looks down -z).
* Errors: subclasses of `core.errors.VRPhotoError` with messages a non-technical user understands.
* No GPU, network or model weights at import time; heavy imports inside functions.
* Code must run on Windows and Linux and inside the frozen app (use `pathlib`, `resource_path`).
