# Changelog

All notable changes to VR Photo Studio are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

**SHARP 360°: SHARP relief, face check (M6) and the AI photo-editor backs (M7, experimental).**
Design: `docs/design/SHARP360.md` (sections 3.5.6 and 4.4). Nothing here has run with the real models
on a GeForce GTX 1080 yet; the times are estimates.

### Added

- **Qwen photo editor add-on (`qwenedit`, experimental, slow, about 30 GB):** Qwen-Image-Edit-2511
  (Q4_K_M GGUF), the Qwen2.5-VL-7B text encoder (GGUF), fal's Multiple-Angles camera LoRA and the
  4-step Lightning LoRA, run by a pinned, headless ComfyUI (v0.39.2 with ComfyUI-GGUF) that the add-on
  starts as a separate program. Needs 16 GB of system memory (32 GB recommended; the manifest's new
  `min_ram_gb`, shown in the Add-ons window); several minutes per view on a GTX 1080. Out-of-memory
  ladder (less graphics memory, smaller pictures, skip the view), a full-precision retry for black
  pictures, and every finished view kept when a run fails or is cancelled.
- **SHARP 360° back made by the Qwen photo editor** (*Back made by: Qwen photo editor*, experimental,
  never *Automatic*): the plain cut-out on light grey -> back views (one per *AI attempt*) -> the one
  that fits the photo's outline best and shows no face -> left / right side views (with *Best* also
  the back quarters) made with the chosen back as a second picture -> lined up and lifted like back
  photos. The views go to the view cache, so a re-run with other stitching options skips the
  minutes of painting. 15-60 minutes per person (estimate).
- **SHARP 360° back from your own ComfyUI** (*Back made by: Your own ComfyUI*, expert): the app
  uploads the cut-out to your ComfyUI server and queues your API-format workflow once per view with
  `{{IMAGE}} {{PROMPT}} {{SEED}} {{AZIMUTH}} {{WIDTH}} {{HEIGHT}}` filled in (new options *ComfyUI
  address* and *ComfyUI workflow*, also `--comfyui-url` / `--comfyui-workflow`). A workflow file
  that is missing, saved in ComfyUI's normal format or lacks `{{IMAGE}}` stops the job before any
  work; a server that does not answer or rejects the workflow is a note and the built-in back.
- **Face check (YuNet):** an AI back view that shows a frontal face (the AI painted the front onto
  the back of the head) is not used - Qwen / ComfyUI back candidates and MV-Adapter sets; a face on
  your own back photo is only a note. The 0.23 MB OpenCV model downloads on first use; without it
  the check is skipped.
- **Back detail: Sharp:** SHARP runs once more on the lined-up back view (the SHARP add-on now takes
  a list of pictures) and its 3D surface shapes the back's relief; *Automatic* uses it with *Best*
  quality. If SHARP cannot run, the depth model is used.

### Changed

- The SHARP add-on's worker accepts several pictures in one run (the model loads once); a picture
  that runs out of graphics memory is made on the CPU and the next one is tried on the GPU again.
- `read_splats(..., return_rows=True)` also returns the file rows of the Gaussians it kept and the
  file's own row count, `(cloud, rows, n_file)` (provenance in diagnostics; dropped rows at the end
  of SHARP's file are counted too).
- The frozen app includes the add-ons' standard-library ComfyUI client
  (`vrphoto.addons.workers._comfy`), which the *Your own ComfyUI* source uses.

### Fixed

- Add-on installs: a Hugging Face download step whose file names match nothing in the model's
  repository now fails with the missing file and `repo@revision` named, instead of downloading
  nothing and being recorded as done.
- SHARP 360° with MV-Adapter and several *AI attempts*: the relief model (and its licence notice)
  recorded for the back is that of the attempt that was kept, not of the last one lined up.

## [0.3.0b1] - 2026-10-13

**SHARP 360°: MV-Adapter AI views and the Brush polish.** Design: `docs/design/SHARP360.md`
(milestones M4 and M5). Not yet tested on a GeForce GTX 1080 with the real models; the times below
are estimates.

### Added

- **MV-Adapter add-on (experimental, about 14 GB):** MV-Adapter on Stable Diffusion 2.1 base paints
  a cut-out person or object from six directions (front, both sides, straight from behind and the
  two front quarters) with exact orthographic cameras. fp16 weights (fp32 math on GeForce
  10-series cards), an fp32 VAE, model CPU offload below 7 GB of free graphics memory, an
  out-of-memory ladder (free memory, then sequential offload) and an fp32 retry when a picture
  comes out black or broken (with sequential offload on 8 GB cards, where the fp32 model does not
  fit with model offload). Licences: Apache-2.0 (MV-Adapter) and CreativeML Open RAIL++-M
  (Stable Diffusion 2.1).
- **SHARP 360° back made by MV-Adapter** (*Back made by: MV-Adapter AI views*): the app gives
  MV-Adapter a levelled, straight-on picture of the photo's front and lifts the back and sides
  from the six views: the AI's front view must match the photo's outline (a small shift or size
  difference is corrected), every view must show the person at the same height (a view more than
  5 % off is left out), the back view is locked onto the mirrored front outline, the colours are
  matched to the photo, the side views give the body's depth and a reversed left/right camera
  convention is noticed (the views are turned round only when renders of SHARP's front at the
  in-between angles agree, so side views the AI drew turned are not mirrored). A set that does
  not fit is not used (note, then TRELLIS or the built-in
  back). With two attempts (*Best*) the better-fitting set is kept. *Back made by: Automatic* now
  tries MV-Adapter first, then TRELLIS, then the built-in shape (not with *Fast*). About 6-10
  minutes with *Balanced* (estimate).
- **Describe the hidden side** (*Advanced*): a few words for MV-Adapter about what the photo
  cannot show (for example "long brown ponytail, black backpack").
- **View cache:** the AI views are kept (up to 2 GB in the app's cache folder, least recently used
  removed first), so a re-run of the same photo with another seam, colour match, relief or polish
  setting - or another back quality with *AI steps* set - skips MV-Adapter. MV-Adapter saves each
  attempt as soon as it is made: attempts finished before a failure (for example out of memory in
  the second attempt) are used in the same run, those finished before a cancel are kept for the
  next one. A saved set that cannot be read any more is made again.
- **Brush polish** (*Advanced*: *Brush polish* Automatic / Off / On and *Brush steps*): Brush from
  the Capture add-on trains the finished splat against the photo, renders of SHARP's front and the
  views of a back made from views (MV-Adapter or your own back photo), so they agree and the join
  softens; SHARP's photographed front is put back exactly afterwards (Brush's own copies of it are
  removed first, so the outline is not drawn twice). *Automatic* polishes only
  with *Best* back quality and after a passed *Test Brush on this PC*; a failed polish keeps the
  unpolished result (Brush's log is saved next to the outputs). *Best* with MV-Adapter, two
  attempts and the polish: about 15-30 minutes (estimate).
- **Editable AI views folder:** with *Also save parts and debug data*, an MV-Adapter run also saves
  `<name>_sharp360_views/` (`back.png`, `right.png`, ... and `views.json` with the exact cameras).
  Edit a view and run again with *Your own back photo* and that `back.png`.
- `<name>_sharp360_views.png` (the contact sheet) is also written for MV-Adapter's views.
- Command line: `--back-source mvadapter`, `--back-prompt`, `--polish auto|off|on` and
  `--polish-steps` for `--method sharp360`.
- When *MV-Adapter* is chosen but not installed, or *Brush polish* is *On* without the Capture
  add-on, the app asks whether to open the Add-ons window, continue without it, or cancel.

### Changed

- *AI attempts* and *AI steps* also apply to MV-Adapter (*Automatic*: 20 / 30 / 50 steps for
  *Fast* / *Balanced* / *Best*). A too long *Describe the hidden side*, an unknown *Brush polish* or
  *Brush steps* outside 1000-30000 stop the job before any slow work.

## [0.2.0b2] - 2026-10-13

Second beta of **SHARP 360°: AI and photo backs, previews.** Design: `docs/design/SHARP360.md` (milestones M2
and M3).

### Added

- **SHARP 360° back made by TRELLIS:** with the TRELLIS add-on installed (no new download), an
  AI builds the whole person in 3D from the photo; the app finds how that shape is turned and
  sized, lines it up exactly behind SHARP's front, matches its colours to the photo and uses
  only its back half. Real volume (hair, shoulders, a bag), soft texture; about 4-8 minutes on
  a GeForce GTX 1080 (estimate). *Back made by: Automatic* now uses TRELLIS when it is installed
  (except with *Fast* back quality); if it cannot make a usable back the built-in shape is used,
  with a note saying why.
- **SHARP 360° back from your own photo** (*Back made by: Your own back photo* and *Back
  photo*): a photo of the person from behind - or a back view made with any AI photo tool - is
  cut out, bent onto the outline of the front photo and lifted onto a back surface, with its
  colours matched to the photo. Optional side photos next to it (`right`, `left`, `back_right`,
  `back_left`, `front_right`, `front_left`) shape and colour the sides; left and right named the
  wrong way round are often noticed and fixed (both pictures together; otherwise a note says the
  naming could not be confirmed). A phone photo taken at eye level and tilted down is
  straightened before the pose check; side photos with the feet cut off keep their real size. A
  back in a different pose is not used (note, built-in back); a back or side photo that cannot
  be read, or where the person is too small, is named in the note. No download, about 1-3
  minutes (estimate).
- **Preview pictures:** every SHARP 360° job saves `<name>_sharp360_preview.jpg`, the finished
  splat seen from 8 directions (rendered on the CPU), and with your own back photo
  `<name>_sharp360_views.png`, showing how your photos were lined up.
- New SHARP 360° options: *Back photo*, *Back detail* (relief from the depth model or smooth),
  *Match colours to the photo*, *AI attempts* and *AI steps* (TRELLIS; the best-fitting attempt
  is kept). *Variation* now also applies to SHARP 360° (a different TRELLIS back and seam
  pattern; the photographed front never changes).
- Command line: `--back-source trellis|images`, `--back-images FILE`, `--back-relief`,
  `--no-color-match`, `--back-tries`, `--back-steps` and `--seed` for `--method sharp360`.
- When *TRELLIS* is chosen as the back but not installed, the app asks whether to open the
  Add-ons window, continue with the built-in back, or cancel.

### Changed

- Bad SHARP 360° options (a missing or unreadable back photo, an unknown *Back detail*, out-of-range
  attempts / steps / *Variation*) stop the job before any slow work, with a plain message.
- The *Back photo* is checked when you click *Create*, exactly as the job checks it (a folder of
  views and a path pasted with quotes are fine). `vrphoto-cli run --spec job.json` finds the
  back photo next to the job file, like the inputs.

## [0.2.0b1] - 2026-10-12

First beta of **SHARP 360°**. Design: `docs/design/SHARP360.md`.

### Added

- **SHARP 360° (experimental Person / Object method):** keeps SHARP's photo-real front of a person
  or object, adds a back so you can walk all the way round, and saves it **life-size** (feet on the
  floor, shown 1:1 in the viewer). It needs only the SHARP add-on. SHARP's result is cleaned first:
  the wall and floor blobs SHARP puts behind the person and the streaks at the outline are removed,
  without touching what the photo shows. In this version the back is the **built-in shape** (a
  smooth, rounded back in the colours around the outline; no download, seconds); AI-made backs
  follow in later versions.
- New options for SHARP 360°: *Back made by*, *Back quality*, *Keep the photo up to* (where the
  photo ends and the back begins), *Size from* (your height or SHARP's own measurement, automatic
  when the subject is cut off by the photo edge) and *Also save parts and debug data*.
  *Thickness* and *Back side* also apply to SHARP 360° (*Thickness* is kept within 0.06-0.18 there,
  with a note, because the back is life-size and the presets set it for Billboard). The presets set
  *Back quality*.
- Command line: `--method sharp360`, `--back-source`, `--back-quality`, `--seam-deg`,
  `--size-from`, `--save-extras`; `vrphoto-cli addons brush-test` checks that Brush (Capture
  add-on) works on this PC and remembers the result. The Add-ons window has the same test
  (*Test Brush on this PC*).

### Changed

- Before an add-on starts (including Capture mode's COLMAP and Brush), the app now closes its own
  loaded AI models (depth, cut-out) so the add-on gets the whole graphics memory - fewer "CUDA out
  of memory" errors on 8 GB cards. They are re-loaded automatically when needed.
- Settings files with a damaged list or dictionary entry now load that entry's default instead
  of failing.

### Fixed

- **MoGe-2 depth model** always failed ("Ranks of input data are different, cannot concatenate
  them"), so jobs that asked for it (e.g. the 3D scene *Max quality* preset) used the simple
  built-in depth estimate instead, with a warning; it works again.

### Notes

- SHARP 360° is a beta: from behind it shows a believable "mannequin" back, not the person's real
  back; loose hair, bags and arms held away from the body come out flattened. Like SHARP itself it
  is for **non-commercial** use only.

## [0.1.1] - 2026-10-11

Fixes from the first test on a real PC (Windows 11, GeForce GTX 1080 8 GB).

### Fixed

- **Hunyuan3D add-on:** jobs crashed with "Boolean value of Tensor with more than one value is
  ambiguous" (a bug in the add-on progress reporting; also affected TRELLIS/TRELLIS.2 samplers).
- **TRELLIS add-on:** the install check failed with "No module named 'setuptools'"; all PyTorch
  add-ons now install setuptools.
- **Add-on install "untrusted mount point" (os error 448):** Python environments are created from
  the exact Python folder, and the installer no longer passes Windows' RedirectionGuard to the app
  when it is started from Setup's last page.
- **Graphics memory:** when an AI model runs out of GPU memory, other loaded models are released
  and it tries again on the GPU instead of switching to the (much slower) CPU for the rest of the
  session.
- **One graphics card listed twice** in the log and the Settings device list.
- Add-on errors no longer tell you to check your internet connection when the problem is
  something else; expected errors are logged as one line instead of a long traceback.

### Added

- PyTorch add-ons check at install time that the installed PyTorch build supports your graphics
  card and say so clearly if it doesn't (SHARP falls back to the CPU instead).

### Notes

- Installed TRELLIS, TRELLIS.2, SHARP and SDXL 360 add-ons re-run from their dependency step the
  next time you click *Install* (one package was added); downloaded models are kept.
- When two graphics cards have the same name, the device list now shows "GPU 0: …", "GPU 1: …".

## [0.1.0] - 2026-10-11

First public version.

### Added

- **3D Photo:** stereo pictures from any photo - side-by-side, half side-by-side, top-bottom,
  red/cyan anaglyph and cross-eyed layouts, automatic convergence and comfort limits, optional
  wiggle GIF.
- **Living Photo:** looping parallax videos (circle, swing, dolly, zoom, Ken Burns, orbit, custom
  key frames) as flat, side-by-side 3D or VR180 video.
- **VR180:** immersive 180° stereo photos (and optional still videos) that keep the photo's real
  field of view, with blurred, black or AI-outpainted edges.
- **360:** stereo top-bottom 360 from 360-camera photos (omni-directional stereo) and 360 spheres
  built around normal photos.
- **3D Scene (6DoF):** layered meshes with AI "look behind" (background band or clean plate) for the
  WebXR viewer; GLB/OBJ export.
- **Person / Object:** cut-out billboards with an inflated 3D shape that stand on your floor in
  passthrough AR; optional AI image-to-3D through add-ons (TRELLIS, SHARP; experimental TRELLIS.2
  and Hunyuan3D).
- **Capture:** photos or a walk-around video → Gaussian splat (COLMAP + Brush add-on).
- **3D Video:** side-by-side, top-bottom or VR180 conversion with temporal smoothing and audio.
- **Depth Map:** 16-bit PNG and colour previews.
- AI models via ONNX Runtime + DirectML (NVIDIA, AMD, Intel GPUs or CPU), downloaded and verified on
  first use: Depth Anything V2 (Small/Base/Large), MoGe-2, Depth Anything 3 (experimental), LaMa,
  MI-GAN, BiRefNet, IS-Net, U2-Net. Non-commercial models are clearly marked.
- **Live WebXR viewer** served from the PC to the Quest Browser over self-signed HTTPS (Wi-Fi) or
  USB (adb reverse): gallery, Enter VR / Enter AR, meshes, splats, VR180/360 and 3D photos. It shares
  only the viewer and its scene library (never the output folder); the firewall rule opens only the
  HTTPS port.
- Quest-friendly output names (`_LR`, `_TB`, `_180_LR`, `_360_TB`, `_360`), spatial video metadata
  and GPano photo metadata; HEVC presets for Quest 3 and Quest 2 (NVENC when available) and H.264 for
  flat videos that play on phones and PCs.
- Add-on manager that installs heavy PyTorch/CUDA tools into private uv environments: Capture to 3D
  (COLMAP + Brush), TRELLIS, AI outpainting for VR180/360 (SDXL), and the experimental SHARP,
  TRELLIS.2 and Hunyuan3D-2.
- Windows installer (per-user, optional firewall rule, keeps or removes user data on request),
  portable zip, command-line tool `vrphoto-cli` with `--selftest`.
- User guide, Quest 3 test checklist and developer documentation.

[0.1.1]: https://github.com/AidanMan217/VR-Photo-Studio/releases/tag/v0.1.1
[0.1.0]: https://github.com/AidanMan217/VR-Photo-Studio/releases/tag/v0.1.0
