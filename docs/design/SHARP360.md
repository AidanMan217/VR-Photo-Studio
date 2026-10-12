# SHARP 360°: final design (a SHARP person/object splat with an AI-predicted back)

This document merges the three designs. The two judges both picked **ship-first** as the backbone. On top of it I added:
- from **geometry-consistency**: the exact orthographic checks, the "mould" back surface, the rim wall, side re-texturing and the Brush recipe;
- from **photoreal-person**: the user-photo source, the editable views folder, optional add-on prompts, the cut-off handling, YuNet, and the experimental Qwen/ComfyUI tier.

I checked every fact used here against the reports in the research notes or the repo, and **critic.txt overrides the other reports wherever they disagree**. Runtimes marked "est." are estimates, not measurements.

---

## 0. Facts, decisions, conflicts resolved, errors fixed

### 0.1 Facts this design relies on

**Pipeline and viewer**
- **SHARP path in `pipeline/run.py`.**
  - `ADDON_METHODS` is at lines 121-126, and `OBJECT_METHODS = ("billboard", *ADDON_METHODS)` is at line 135.
  - The SHARP branch runs from 1712 to 1772: `sharp_input.png` → `_subject_splats` (1806-1863) → `_splats_cv_to_gltf` → `crop_splats(radius_pct=98)` → `normalize_orientation` → `decimate_splats(750k)`. The scene block it writes has **no `scale`**.
  - `_subject_splats` keeps Gaussians inside a matte dilated by 1 % and with z ≤ 1.5·p25 + 2(p25 − p5). This keeps the hidden-layer Gaussians (critic R6).
- **Viewer (`object.js`).**
  - It fits height with `fitScale = cfg.scale || height_m / nativeHeight`, where `nativeHeight` comes from the **full**, non-robust box (R4).
  - The pivot and floor come from robust bounds.
  - The scene schema accepts `object.scale` (`webxr/scene.py:105`), and the capture mode already writes `scale: 1.0` (`run.py:2024-2028`).

**GPU and add-ons**
- **SHARP and new torch add-ons run on the GTX 1080.**
  - On Windows, torch 2.8.0+cu128 includes sm_61 (`tests/addons/test_workers.py:458`), and `sharp._device("auto") == "cuda"` with that arch list (line 594).
  - `trellis.toml` uses torch 2.7.0+cu128, is `platforms=["win32"]` and `noncommercial=true`.
- **TRELLIS needs no worker change.** `trellis_worker` already accepts `glb=False, ply=True` and then runs `formats=["gaussian"]` (lines 101-133).
- **Freeing app GPU memory before add-ons is safe.**
  - `models.ort.unload_model()` exists and is never called before add-ons (R5).
  - Depth and segmentation sessions reload lazily afterwards (`depth/da2.py:127`, `segment/_runtime.py` LazyModel).
- **`Cancelled` is a `VRPhotoError` subclass** (`core/errors.py:23`). Every fallback `except` must re-raise `Cancelled` first.
- **Worker imports.** `test_worker_top_level_imports_are_light` only allows stdlib, `_common` and numpy at worker top level.
- **Manifests.** `manifest.py` rejects unknown top-level keys, so `min_ram_gb` needs a code change. `hf_download` steps must carry a `revision`. torch must come from cu128 and be below 2.12 (`test_manifest_pins`).
- **Fake Brush** asserts `sparse/cameras.bin` (`tests/addons/fakes/fake_brush.py:34`).

**Main app libraries**
- The dependencies are numpy, OpenCV 5.0.0, Pillow, onnxruntime and PySide6. There is **no scipy**.
- `cv2.FaceDetectorYN` exists in the installed OpenCV 5.0.0.

**GUI and helpers**
- **`gui/options_meta.py`.**
  - `FieldMeta.enabled_if` can test only one sibling field.
  - `required_addon_candidates` ignores `enabled_if`, and `_check_addons` treats candidates as any-of.
  - `KINDS` includes `file`, `folder` and `text`.
  - Tests pin the exact seed set (`test_seed_is_not_offered_for_sharp`). They require add-on ids to exist as manifests (`test_addon_choice_ids_match_manifests`) and the three preset names per mode (`test_every_mode_has_presets`).
- **`geometry/billboard.py`:**
  - `local_thickness(dt, mask)` takes **two** arguments.
  - `_dist` and `_masked_blur` are helpers.
  - Constants: `_BAND_PX=8`, `_BACK_DARKEN=0.9`, `_KAPPA_MAX=4`, `_DETAIL_M=0.03`.
- **Output names.** `naming.PLAYER_TOKENS` are matched at the **end** of the name, so `photo_sharp360.splat` and `photo_sharp360_preview.jpg` do not end in `_360`.

**Brush 0.3.0 (critic R3)**
- The training background is fixed black.
- A black mask means "ignore", and a masked image does not train alpha.
- Nothing can be frozen.
- `init.ply` in the dataset folder is always used, while a `ply_file_path` that fails to read is silently ignored.
- There is no `--invert-masks`, and the SH0 fix (#392) is not in 0.3.0.

**The user's own setup**
- They use SHARP's person/object mode, so the SHARP add-on works on their PC.
- They have run TRELLIS ("super low quality"), so **TRELLIS is probably already installed**. This needs confirming.

### 0.2 Key decisions
1. **A new experimental object method, `sharp360`.**
   - Label: "SHARP 360° – photo front + AI back".
   - `sharp` stays byte-for-byte unchanged.
   - Add `ADDON_METHODS["sharp360"] = ("sharp", "image_to_splat")`, so a missing SHARP fails fast exactly as today. `_mode_object` branches to `pipeline/sharp360.py` right after the matte check.
2. **SHARP is the only required add-on.** Every back source is optional and sits in a **fallback chain that warns and never fails the job**. If even the built-in back fails, the job writes the cleaned SHARP front.
3. **A pure numpy/cv2 fusion core in the main app** (CPU, testable in CI). It is a fixed sequence:
   - clean the SHARP front;
   - find an upright subject frame;
   - get a back part from one of the pluggable sources;
   - one shared **mould back builder**: a screened-Poisson back height field along the photo's camera rays, plus the rim wall;
   - partition by visibility and dither the seam;
   - match colours and feather them into SHARP;
   - remove floaters;
   - set an **explicit life size** (`scale: 1.0`).
4. **The photographed front is protected.** A SHARP Gaussian is never edited or removed if it faces the camera at less than (seam − 10°) or lies in the outline band. The only exceptions are:
   - "card" collapse along its own ray (with a size correction, so it looks the same from the original camera);
   - removing the hidden layer and background;
   - the existing opacity < 0.04 prune.
5. **Back sources, by release.**
   - `shell` (built in).
   - `trellis` (existing add-on).
   - `images` (your own back/side photos, or AI edits made elsewhere).
   - `mvadapter` (new add-on).
   - `qwen` and `comfyui` (experimental tier, gated by an on-PC spike, never chosen by `auto`).
   - Brush polish is an optional refinement for view-based sources only.
6. **The CPU splat renderer is off the critical path.** It is used only for the turntable preview, Brush anchor views and an optional yaw check.

### 0.3 Conflicts resolved

| Topic | Options in the designs | Choice | Why |
|---|---|---|---|
| Method id | `sharp360` / `sharp_full` | `sharp360`, output `_sharp360` | Two of three designs and both judges chose it. It does not end in the player token `_360`. |
| How the method is wired | Own branch with `_require_addon` / entry in `ADDON_METHODS` | `ADDON_METHODS` entry plus an early branch | Fail-fast for SHARP and the existing `test_object_addon_not_installed_fails_fast` cover it for free. |
| Seed | Separate `back_seed` / reuse `seed` | Reuse `seed` ("Variation"); the exact-set test becomes `{trellis, trellis2, hunyuan, sharp360}` | One knob with one meaning ("different AI result"). The test's intent (plain `sharp` takes no seed) is kept. |
| Front depth | Min z / min z with opacity ≥ 0.3 / depth where accumulated alpha reaches 0.5 | Alpha reaches 0.5 (transmittance) | A translucent floater cannot define the surface, so the floater test stays meaningful (the photoreal and geometry methods had errors here). |
| Edge "cards" | Drop in the outer band / collapse | Collapse along their own ray, with angular-size correction | The outline and hair wisps stay pixel-identical from the photo view (a judge flagged geometry's drop). |
| Soft edge | Hard 0.5 matte / 1 % dilated ring | 1 % ring, kept only when its depth matches the subject | Matches plain `sharp`'s outline but drops wall blobs (a judge flagged photoreal's hard matte). |
| Generator input | Orthographic render of the SHARP splat / levelled forward-splat of the photo | Levelled forward-splat (§3.4) | No renderer on the critical path, full photo colour resolution, and the pitch and perspective mismatch (about 12 %) are removed. |
| Lifting MV-Adapter views | Ray-march a voxel hull (ship-first) / orthographic height-field mould (geometry) | **Mould on the photo's isotropic perspective grid**: a Poisson back bounded by per-ray hull intervals, plus the rim wall, side re-texturing and hole lifting | Both judges asked for one shared back builder. Working along camera rays closes the shape exactly at the photo silhouette and needs no renderer. |
| Hull carving | Empty if outside **any** view / "all but one" voting | All-but-one voting; the photo matte stays exact | One inconsistent generated silhouette can no longer over-carve the body. |
| Relief scale `a` | Rim-ring LSQ (geometry) / per-row fit against side-view back extents (ship-first) | Per-row fit when side views exist; otherwise bounded high-pass detail on the thickness prior | The rim-ring fit is ill-conditioned (judge). Without side views, the thickness prior is the only well-posed constraint. |
| Missing explicit source, or `polish=on` without Capture | Raise `AddonNotInstalled` (geometry) / warn and fall back | Warn and fall back. The GUI asks first with three buttons. | This is a hard constraint: fall back rather than fail. |
| Auto order | mvadapter > sv3d > zero123pp > trellis / qwen first / mvadapter > trellis > shell | mvadapter > trellis > shell. `fast` always uses the shell. `images`, `qwen` and `comfyui` are never automatic. | Every source in the auto chain has known cameras or a known shape and fits 8 GB. Qwen on Pascal has never been measured. |
| Brush dataset format | nerfstudio / COLMAP text | COLMAP text plus `init.ply` in the dataset folder | Brush tries COLMAP first, it uses OpenCV conventions, and it avoids the silently ignored `ply_file_path`. |
| Polish default | On for balanced and best / best only | `auto` = best only, view-based sources only, and only after the Brush self-test passes on the user's PC | Brush on Pascal is unverified (critic O2). It is pointless for TRELLIS or the shell, which have no views to train against. |
| Seam | 60 ± 12 / 65 ± 9 / 55-75 | 60° ± 10° (option 35-80°) | The middle of the proposals. SHARP stays fully kept up to 50°. |
| Export order | crop → normalize → scale / normalize → scale → box crop | `normalize(up)` → `to_life_size` → AABB crop → decimate | The box is only meaningful when the subject is upright and metric. The 98 % radius crop would clip head and feet. |
| SHARP on several images | New task `images_to_splats` / list input on the same task | List input on `image_to_splat` | The manifest hash does not change, so there is no "update available" prompt. Workers ship with the app. |
| Qwen | Default source / out of scope / M6 experimental | M7 experimental `qwenedit` add-on, gated by a spike on the user's PC, never `auto` | The judges' recommendation. Speed and RAM on Pascal are unknown (critic R9). |
| Zero123++ / SV3D | Sources (geometry) / plan B | Not in scope | Zero123++ has no 180° view, so the mould cannot use it (judge). SV3D's framing is unverified. Can be revisited later. |
| Where generated views are saved | Always a folder / opt-in | A small contact-sheet PNG always; an editable `images`-style folder only with `save_extras` | Keeps the output folder tidy while keeping the "edit a view and re-run" workflow. |

### 0.4 Technical errors fixed
1. **Hull over-carving:** all-but-one voting instead of empty-if-outside-any (ship-first 3.4c(7)).
2. **Degenerate front-edge test** (`σ·x·ẑ < 0` is never negative). It is now `u_f = argmin over silhouette columns of (sin φ·u)`, with `sin φ` taken from the view's camera (§3.5.2 V5).
3. **Shell Gaussian frames:** the finite-difference tangents are Gram–Schmidt orthonormalised and `n = t1 × t2` (det +1). Ship-first used the raw tangents.
4. **`local_thickness(dt, mask)`** takes two arguments. The new helper `billboard.inflation_profile` wraps lines 269-272.
5. **Anisotropic distance transforms:** ship-first ran `dt`, Poisson and normals on SHARP's anisotropic 768² grid. These now run on an **isotropic mould grid**. The SHARP grid is used only for the front z-buffer.
6. **Choices tied to manifests:** `addon_choices` for `mvadapter` and `qwen` are added only in the milestone that ships the manifest (`test_addon_choice_ids_match_manifests`).
7. **MV-Adapter licence:** `noncommercial = true` until the pinned LICENSE is verified (critic R12: the README states none).
8. **SD2.1 base repository:** one `hf_download` step whose repo is decided at M4 (stabilityai, or the `Manojb` mirror if it has gone), pinned by revision. Ship-first's "optional mirror" step would download 2.6 GB twice.
9. **Photoreal design errors:**
   - **z-buffers:** "sort then scatter, last write wins" is not guaranteed by numpy. Use `np.minimum.at` or lexsort with `minimum.reduceat`.
   - **Duplicate image names:** the photo listed three times in the COLMAP model needs three distinct file names.
   - **Invalid TOML:** `[[steps]] type="venv"` written on one line.
   - **Unpinned quant:** the Qwen quant was downloaded by the worker's `--prefetch`, outside the manifest pin tests. It becomes a pinned `hf_download` step.
   - **Rigid registration:** a 7-DoF fit of SHARP-lifted views cannot absorb a wrong FOV, because depth and lateral size couple. Nominal-pose views now go through the silhouette lock plus the relief mould instead.
10. **Geometry design errors:**
    - **Floater test:** the min-z depth made the floater test vacuous. Replaced by the transmittance depth.
    - **Ill-conditioned relief fit:** the rim-ring fit is replaced (see 0.3).
    - **Fail-fast:** raising on missing add-ons is replaced by fallback.
    - **Licence default:** `noncommercial=false` before verification is replaced by `true`.
11. **Report errors (all three designs already overrode them):** "SHARP runs on the CPU on the 1080", "new add-ons need cu126", scipy for the Poisson solve, and the `--invert-masks` flag are all wrong (critic R1-R3).
12. **New in this synthesis:** `Cancelled` derives from `VRPhotoError`, so every fallback chain lists `except Cancelled: raise` first. Without this, a cancel would be swallowed as "source failed".

---

## (a) In plain words: how the mode works and what quality to expect

- **The front stays SHARP's.** SHARP turns your photo into a cloud of tiny coloured blobs (a "splat"). The blobs the camera actually saw look photo-real. SHARP has nothing behind the person, so today you can only look from roughly where the photo was taken.
- **SHARP 360° keeps that front exactly as it is,** cleans off the smears at the edges, and then **builds a back**. You choose who "imagines" the back:
  - **Built-in** (no download, about 1-2 min): a smooth, rounded back in the colours from around the person's outline. It looks like the person from behind in soft focus. It is always closed, never see-through, and right-sized.
  - **TRELLIS** (you probably have it already, about 5 min): an AI-made 3D shape of the whole person. Only its back half is used, fitted exactly behind your photo. You get real volume (hair, shoulders, a backpack), but the texture is soft.
  - **MV-Adapter** (new download of about 10 GB, about 6-10 min): an AI paints the person from six directions (front, both sides, back and two in-between). The app turns those pictures into the back and sides. Clothing and hair look plausible but softer than the photo. Faces seen from the side may not quite match.
  - **Your own back photo** (no download, about 2 min): if you have a photo of the person from behind, or made one with any AI tool you like, the app bends it onto the right outline and uses it. This can look the most real.
  - **Experimental, later — Qwen photo editor** (about 30 GB, 15-60 min): the most photo-like AI backs, but slow on a GTX 1080, needs plenty of RAM, and is only offered if a test on your PC passes.
- **The join.** The app joins front and back where the body turns away from the camera (about 60° by default). It matches the back's colours to the photo and removes stray blobs. It then sizes the person to the height you type, so in the Quest they stand life-size on your floor and you can walk all the way round.
- **What to expect:**
  - From the photo's angle it looks exactly like plain SHARP.
  - From the side you will see the join; it softens with Best quality.
  - From behind it is believable but invented. It will never be the person's real back unless you give it a real back photo.

---

## 1. User-facing behaviour and options

### 1.1 Where it lives
- In *Person / Object*, the **Method** combo gets `("sharp360", "SHARP 360° – photo front + AI back (experimental add-on)")`, added to `experimental_choices`, with `addon_choices["sharp360"] = ("sharp",)`.
- The Method tooltip gains: "SHARP 360° keeps SHARP's photo-real front and adds a back predicted by the built-in shape, TRELLIS, MV-Adapter or your own back photo, so you can walk all the way round."

### 1.2 Fields
Every field has a default, so old job JSON files still load.

| Field | Type, default | Choices / range | GUI (enabled when) | Meaning |
|---|---|---|---|---|
| `back_source` | str `"auto"` | auto, shell, trellis, images, mvadapter (M4); qwen, comfyui (M7, experimental choices) | combo, basic (method = sharp360). `addon_choices` = `{trellis: ("trellis",)}`, plus mvadapter (M4) and qwenedit (M7). New `FieldMeta.addon_optional=True` | **"Back made by"**. Auto = MV-Adapter if installed, else TRELLIS, else built-in. Fast quality always uses built-in. |
| `back_quality` | `"balanced"` | fast, balanced, best | combo, basic | What each "Automatic" setting means (§1.3). |
| `back_images` | str\|None `None` | file | file (back_source = images) | "Back photo". The picture of the back. Pictures in the same folder named `left`, `right`, `back_left`, … (§4.6) are used as side views. |
| `back_prompt` | `""` | text | text, advanced (back_source ∈ {auto, mvadapter, qwen, comfyui}) | "Describe the hidden side", e.g. "long brown ponytail, black backpack". |
| `back_tries` | int 0 | 0 (Automatic) to 4 | spin, advanced (back_source ∈ {auto, trellis, mvadapter, qwen, comfyui}) | How many AI attempts to make; the best-fitting is kept. Automatic = 1, or 2 for best. |
| `back_steps` | int 0 | 0 (Automatic) to 100 | spin, advanced (same as above) | Generation steps. |
| `back_relief` | `"auto"` | auto, depth, flat; `sharp` added in M6 | combo, advanced | Surface detail on the back: from the depth model on the back view, or smooth. |
| `seam_deg` | float 60 | 35-80, step 5, unit ° | slider, advanced | "Keep the photo up to": the photo is used where the body faces the camera within this angle. |
| `color_match` | bool True | | check, advanced | Match the back's colours and exposure to the photo. |
| `polish` | `"auto"` | auto, off, on | combo, advanced | Brush polish (Capture add-on). Auto = best quality only. |
| `polish_steps` | int 5000 | 1000-30000, step 500 | slider, advanced (polish ∈ {auto, on}, plus method) | Brush training steps. |
| `size_from` | `"auto"` | auto, height, photo | combo, advanced | Life size from "Real-world height", or SHARP's own measured size. Auto = height, unless the subject is cut off by the photo edge. |
| `save_extras` | bool False | | check, advanced | Also save the AI views folder (re-runnable as "Back photo"), front/back parts, PLY and debug JSON. |
| `comfyui_url` (M7) | `"http://127.0.0.1:8188"` | text | advanced (back_source = comfyui) | Expert: your own ComfyUI server. |
| `comfyui_workflow` (M7) | str\|None `None` | file | advanced (back_source = comfyui) | API-format workflow with placeholders `{{IMAGE}} {{PROMPT}} {{SEED}} {{AZIMUTH}} {{WIDTH}} {{HEIGHT}}`. |

**Reused fields:**
- `height_m`.
- `thickness`: `enabled_if` gains `sharp360`; it sets the thickness of the built-in back.
- `back` (diffuse | mirror): gains `sharp360`; it sets the built-in back's colours. The tooltip warns that mirror puts a face on the back of a head.
- `seed`: gains `sharp360`; it is the generator seed and the seam dither seed.
- `placement`.

**`FieldMeta` additions:**
- `enabled_if_all: tuple[tuple[str, tuple], ...] = ()` (an all-of list of extra conditions).
- `addon_optional: bool = False`.
- A pure `field_applies(kind, name, values) -> bool` used by both `panels/options.py:_applies` and the add-on check, so they cannot disagree.

**Presets.** `PRESETS["object"]` adds `back_quality`: Comfortable → fast, Strong 3D → balanced, Max quality → best. The preset names are unchanged.

### 1.3 What "Automatic" means per quality

| Knob | fast | balanced (default) | best |
|---|---|---|---|
| Source when `auto` | shell | mvadapter > trellis > shell | same as balanced |
| Mould grid long side | 512 | 768 | 1024 |
| Back Gaussian stride | 2 | 1 | 1 |
| MV-Adapter steps / guidance | 20 / 3.0 | 30 / 3.0 | 50 / 3.0 |
| TRELLIS ss / slat steps | 12 / 12 | 12 / 12 | 25 / 25 |
| Tries (seeds) | 1 | 1 | 2 (kept by quality-check cost, §3.5.3) |
| Relief | flat | depth | depth (`sharp` from M6) |
| Side re-texture and hole fill | off | on | on |
| Brush polish (when `polish=auto`) | off | off | on, 5000 steps, view-based sources only |
| Turntable preview | 256 px | 384 px | 512 px |

### 1.4 While it runs
**Progress messages:**
- 0-0.04 "Loading".
- 0.04-0.10 "Finding the subject".
- 0.10-0.25 "Running SHARP (front)", followed by the worker's own messages.
- 0.25-0.30 "Cleaning the SHARP front (removing background layers and edge streaks)".
- 0.30-0.72, depending on the source: "Generating the back and sides with MV-Adapter (try 1/2, step 12/30)", "TRELLIS: generating the shape", "Fitting your back photo to the outline" or "Building the back".
- 0.72-0.82 "Lining up the AI views", "Building the back surface", "Filling the sides", "Matching colours", "Stitching front and back".
- 0.82-0.92 "Polishing with Brush (step 1200/5000)". When polish is skipped, this range goes to the earlier stages.
- 0.92-1.0 "Removing floaters", "Making it life-size", "Rendering a turntable preview", then "Adding to the VR viewer library".

**Warnings** (one plain sentence per fallback), for example:
- "MV-Adapter could not make the back (CUDA out of memory); TRELLIS was used instead."
- "The AI side views did not line up with the photo (height differs 9 %); the sides use the built-in shape."
- "The person is cut off at the bottom of the photo, so SHARP's own size estimate was used. Set 'Size from' to 'Real-world height' to force your height."
- "The camera looks down steeply (34°); the back may be less accurate."
- "Brush polish was skipped (Capture add-on not installed)."

**Non-commercial notices:** SHARP always; TRELLIS when used; MV-Adapter until its licence is verified. They work as today, through `run.addon_notice`. The output label names the source, e.g. "3D object (Gaussian splat, SHARP 360° – back by TRELLIS)".

**Totals on a GTX 1080 (est.):**

| Setting | Time |
|---|---|
| fast (built-in back) | 1-2 min |
| your own back photo | 1-3 min |
| balanced + TRELLIS | 4-8 min |
| balanced + MV-Adapter | 6-10 min |
| best + MV-Adapter (2 tries + Brush) | 15-30 min |
| Qwen, experimental | 15-60 min |

### 1.5 Outputs
- `<base>_cutout.png` (as today).
- `<base>_sharp360.splat`: the viewer asset. Feet at y = 0, front facing +z, life-size, at most 750k splats.
- `<base>_sharp360_preview.jpg`: a 2×4 turntable sheet at 0, 45, …, 315° (from M3).
- `<base>_sharp360_views.png`: a contact sheet of the AI or user views with the photo's outline drawn on top. Only written when a view-based source ran.
- **With `save_extras`:**
  - `_sharp360.ply`;
  - `_sharp360_front.splat` (cleaned SHARP only, for A/B comparison);
  - `_sharp360_back.splat`;
  - `_sharp360_views/` (`back.png`, `right.png`, … plus `views.json` with the cameras), which can be re-run with "Back photo";
  - `_sharp360_debug.json`.

  `debug.json` is always written to the work folder, and a one-line summary goes to `vrphoto.log`.
- **Scene block:**
  ```json
  {"asset": "object.splat", "format": "splat", "height_m": H, "scale": 1.0, "pivot": "base", "placement": "..."}
  ```

### 1.6 CLI
Flags are generated automatically. Add to `cli._choices_table`:
- `("object","back_source"): R.BACK_SOURCES`
- `back_quality`, `back_relief`
- `("object","polish"): ("auto","off","on")`
- `("object","size_from"): ("auto","height","photo")`

Examples:
```
vrphoto-cli object me.jpg --method sharp360
vrphoto-cli object me.jpg --method sharp360 --back-source mvadapter --back-quality best --back-prompt "grey hoodie, short hair" --seam-deg 55 --height-m 1.78 --save-extras
vrphoto-cli object front.jpg --method sharp360 --back-source images --back-images D:\pics\back.jpg
```

### 1.7 GUI add-on check
- **New helper.** `options_meta.required_addon_groups(kind, values) -> tuple[tuple[tuple[str, ...], bool], ...]` returns `(any-of ids, optional)` groups and skips fields that `field_applies` says are greyed out. So a remembered `back_source=trellis` on a billboard job asks for nothing. `required_addon_candidates` stays as the flattened wrapper for the existing tests.
- **`main_window._check_addons`:**
  - A missing **required** group shows today's dialog.
  - A missing **optional** group shows a new `util.ask_choice` with "Open Add-ons" / "Continue (the built-in back is used)" / "Cancel".
  - The job itself always warns and falls back, so the CLI behaves the same way.

---

## 2. Back sources ("what predicts the hidden back")

| Source | Model / add-on | Install | VRAM / RAM | GTX 1080 time (est.) | Licence flag | Expected quality on a real person |
|---|---|---|---|---|---|---|
| **`shell`** (M1) | Built-in mould: the silhouette inflated behind SHARP's front (billboard profile); colours pushed in from the outline band (or mirrored) | 0 | CPU, < 1.5 GB | 10-30 s | app (MIT) | Closed and smooth, "mannequin" back, no clothing detail. Always works and is the final fallback. |
| **`trellis`** (M2) | TRELLIS v1 Gaussians (`glb=False, ply=True`) from the levelled input, registered behind SHARP | 0 if installed (20 GB) | 8 GB fp16 / about 4 GB | 2-5 min (no GLB baking) | `noncommercial=true` (already) | Real 3D volume (hair, shoulders, bag), soft and flat-lit texture. Its front is discarded, so SHARP owns the face. |
| **`images`** (M2) | Your own back photo (and optional side photos) or any AI edit made outside the app. Silhouette-locked and lifted by the mould. | 0 | CPU | 1-3 min | User's responsibility | A real back photo of the same pose gives a truly photographic back. Pose differences beyond the gate are rejected with a warning. |
| **`mvadapter`** (M4) | MV-Adapter i2mv SD2.1 (`mvadapter_i2mv_sd21.safetensors` on SD2.1-base): 6 orthographic views at 512 px, azimuths 0/45/90/180/270/315, elevation 0; the worker exports the cameras | About 10 GB | 5-7 GB peak fp16 with model offload / 4-6 GB | 4-7 min per try | `noncommercial=true` until the LICENSE is verified; SD2.1 OpenRAIL++ | Consistent views all round, including an exact 180°. Object-trained: soft (about 3.7 mm/px for 1.7 m), identity drifts on the sides. Better than shell or TRELLIS for clothing and hair outlines. No published evaluation on real people. |
| Brush polish (M5) | Brush 0.3.0 (existing Capture add-on), known cameras, `init.ply` | 0 (Capture) | 2-3 GB | 5k steps ≈ 4-8 min (**Pascal unverified**) | Apache-2.0 | Makes views mutually consistent and blends the seam. The SHARP front is re-inserted afterwards. |
| **`qwen`** (M7, experimental, never auto) | Qwen-Image-Edit-2511 GGUF, Multiple-Angles LoRA and Lightning LoRA, in a pinned headless ComfyUI subprocess (`qwenedit` add-on) | 25-30 GB | About 7 GB VRAM; 32 GB RAM recommended, refuses below 16 GB | 5-15 min per view (critic R9 arithmetic; unmeasured) | Apache-2.0 (verify the fal LoRA card); ComfyUI GPL-3.0 as a separate program | The most photo-like backs. Views are independent (nominal poses, possible pose drift or a face on the back of the head), so silhouette lock, candidates and the YuNet face check are needed. |
| **`comfyui`** (M7, experimental) | Your own ComfyUI server plus workflow | 0 | Yours | Yours | Yours | Anything you run (FLUX Kontext, LongCat, Qwen with other LoRAs). |
| *Later* | SHARP relief (M6); TripoSplat (MIT; VRAM unknown; needs a bf16→fp16 patch); LongCat-Image-Edit-Turbo (6B Apache, inside `qwenedit`) | | | | | Behind the same `BackPart` interface. |

**Rejected for this PC:**
- LHM++, Era3D, SiTH: CUDA extensions compiled on the user's PC.
- PSHuman (> 40 GB), MagicMan (23 GB), IDOL (24 GB), SEVA and ViewCrafter (about 20 GB), SAM 3D (32 GB / Linux): too much VRAM.
- Hunyuan3D-2mv paint: region-restricted, more than 8 GB.
- Iterative SDXL orbit-inpaint: inconsistent, and 0.5-1.5 h.
- Zero123++: no 180° view (§0.3).

---

## 3. Pipeline and algorithms

### 3.0 Frames, grids and conventions

**SHARP camera frame C**
- OpenCV axes (x right, y down, z forward), metres, camera at the origin.
- From the worker outputs: `f = f_px`, W×H = `width` × `height` (= `photo.rgb`).
- Pixel coordinates are continuous: `u = f·X/Z + W/2`, `v = f·Y/Z + H/2`. The pixel index is `⌊u⌋`, which equals `_subject_splats`' `round(u − 0.5)`.
- `geometry.camera` (CONTRACTS §0) puts pixel centres at integers, so subtract 0.5 when calling `project`/`unproject`.

**Two grids**
- **SHARP grid, G×G (anisotropic).** Used only for the front z-buffer.
  - G = 768 if N ≈ L·768² for some L in 1..3; otherwise G = round(√(N/2)). The detected value is logged.
  - A Gaussian's cell is `(⌊v·G/H⌋, ⌊u·G/W⌋)`; each cell holds about one Gaussian per layer.
- **Mould grid, Hm×Wm (isotropic).** Used for distance transforms, normals, Poisson, hull intervals and back Gaussians.
  - The long side is `Lm` (512 / 768 / 1024) and a cell is `W/Wm` photo pixels square.
  - The matte is resized with `INTER_AREA`; D_f is resampled bilinearly from the filled SHARP-grid map.

**Tolerances and rays**
- Depth tolerance `τ(z) = max(0.02 m, 0.01·z)`.
- The ray of a point μ is `r̂ = μ/|μ|`.

**Upright subject frame S**
- Axes expressed in C: `ŷ = −up` (down), `ẑ` = the horizontal viewing direction (away from the camera), `x̂ = ŷ × ẑ` (right).
- Origin `c = (x0, y0, z_axis)`. Points convert with `P_S = Rᵀ(P_C − c)`, where `R = [x̂ ŷ ẑ]`.
- `z_axis` is the orbit-axis depth. It is solved from side views, from TRELLIS registration, or from the prior `median(front z_S) + T_prior/2`.

**Azimuth φ (contract)**
- φ is the direction of the camera from the subject, measured from the original camera (φ = 0, at −ẑ) toward +x̂. So φ = 90 means "you walked to your right"; φ = 180 is behind.
- View camera: `forward(φ) = cos φ·ẑ − sin φ·x̂`, `right(φ) = cos φ·x̂ + sin φ·ẑ`, `down = ŷ`.
  - Check: φ = 180 gives right = −x̂, the mirror image. φ = 90 gives forward = −x̂.
- Orthographic view φ with m metres per pixel and size N:
  `u = (P_S − a)·right/m + N/2`, `v = (P_S − a)·ŷ/m + N/2`, where `a = (x0, y0, z_axis)` is the orbit axis and projects to the image centre.

### 3.1 SHARP
1. Free the app's GPU memory: `_free_app_gpu()`, which is `ort.unload_model(wait=True)` wrapped in try/log. It is placed inside `_run_addon`, so it runs before every add-on, and also before Brush.
2. Factor `run.py` 1713-1720 into `_sharp_inputs(run, item, photo) -> (image_path, params)`, shared with `sharp`.
3. `outs = _run_addon(run, item, "sharp", "image_to_splat", …)`, then `sc_full = read_splats(outs["ply"])`.
4. A SHARP failure fails the job, as today.

### 3.2 Front clean-up: `sharp_front(sc_full, alpha, cam, *, mould_long, progress, cancel) -> FrontModel`
1. **Candidates.**
   - Keep Gaussians with z > 0.05 whose centres project inside the matte dilated by 1 % of the long side (the same dilation as `_subject_splats`).
   - The **ring** is that dilation minus the hard matte (α ≥ 0.5).
2. **Cards first.** For each candidate, take `a1` = the axis of the largest scale `s1` (from `quat_to_rotmat`) and `s2` = the middle scale.
   - A card satisfies `s1 > 4·s2`, `|a1·r̂| > 0.8` and `s1 > 1 cm`.
   - Cards are excluded from the depth statistics in step 3 and collapsed in step 6.
3. **Front depth by transmittance (vectorised).**
   - Sort non-card candidates with `np.lexsort((z, cell))`.
   - Per cell, `acc = 1 − exp(segmented cumsum of log1p(−α))`.
   - Find the first index where `acc ≥ 0.5` with `searchsorted` on the globally monotone key `rank(cell)·2 + acc`, looking up `rank·2 + 0.5`. That Gaussian's z is `D_f(cell)`.
   - If the threshold is never reached, use the z of the largest `α·T`. `A_f(cell)` is the final `acc`.
   - Fill holes inside the matte with `push_pull_fill`, then apply a 3×3 median.
   - Derive (OpenCV float morphology):
     - `D_near = cv2.erode(D_f, 5×5)`;
     - `D_far = cv2.dilate(D_f, 9×9)`;
     - `D_ext` = `D_near` push-pull-extrapolated into the ring.
4. **Hidden layer and background (fixes critic R6).** For matte-interior Gaussians with `z > D_f + τ`:
   - **keep** only if `D_far − D_f > 5 cm` **and** `z ≤ D_far + τ` (disocclusion fill near an internal edge, e.g. torso behind an arm);
   - **drop** everything else (wall blobs).
   - Also drop `z < D_f − 3τ` (front floaters).
5. **Ring Gaussians.** Keep a ring Gaussian iff `|z − D_ext| ≤ 3τ` (hair wisps and soft edges at subject depth). This keeps plain SHARP's outline but drops background.
6. **Collapse cards.** Slide each card along its own ray onto `D_near(cell)` (or `D_ext` in the ring): `μ' = μ·D_near/μ_z`.
   - Set `s1' = 1.5·s2`, and multiply `s2, s3` by `D_near/μ_z`, so the angular footprint and the original-view look are unchanged.
7. **Mould-grid maps.** Compute `D_f_m`, `D_ext_m`, the matte, the boundary ring and `dt`.
   - Normals `n_f`: unproject the 4-neighbours, take the cross product, orient toward the camera.
   - `cosθ = n_f·(−r̂)`.
   - Per Gaussian: `θ` from its mould cell, and `rim = dist_to_matte_boundary ≤ max(2 cells, 0.4 % of the long side)`. The outline band is **protected**.
8. **Fallback.** If fewer than max(500, 1 %) Gaussians remain, use the `_subject_splats` result as F (with the same maps from its z-buffer) and warn.
9. **Stats** go to `debug.json`: kept, hidden dropped, cards collapsed, wisps kept, detected G, and `N_raw` vs L·768².

### 3.3 Canonical frame: `canonical_frame(front, sc_full, alpha, *, thickness) -> Canon`
1. **Up vector:** the first of these that applies.
   1. **Floor.**
      - Take Gaussians in `sc_full` that are **not** on the subject, project below the lowest matte row (horizontally within 1.5× the subject width), have opacity > 0.3 and depth within [0.5, 2]·median subject depth.
      - Fit with `splat._ransac_plane` (made public as `fit_plane_ransac`), tolerance 1 cm.
      - Accept if the normal is within 35° of −y_C, inliers are ≥ 25 %, and the subject's lowest front points are within 10 cm of the plane.
   2. **PCA.** The opacity-weighted PCA major axis of F, if it is within 25° of −y_C and the elongation is > 1.8.
   3. **Camera.** `up = (0, −1, 0)`.
   - Record `pitch_deg` and warn above 30°.
2. **Axes.** With `d` = the normalised centroid ray of reliable front points: `ŷ = −up`, `ẑ = normalize(d − (d·ŷ)ŷ)`, `x̂ = ŷ × ẑ`.
3. **Sizes.**
   - `height_s` = P99.5 − P0.5 of y_S over F with α > 0.5.
   - `T_prior = thickness·height_s`.
   - `x0, y0` = the bounding-box centre of front points in S.
4. **Cut-off.** `cut_bottom` / `cut_top` is set if matte pixels within 1 % of that image border span more than 15 % of the subject width.

### 3.4 Levelled orthographic input: `levelled_ortho_input(rgb, alpha, front, canon, *, size=512, ref=1024) -> OrthoInput`
1. For photo pixels with α > 0.02 (photo downscaled to a long side ≤ 2048):
   - depth comes from `D_f_m`, or `D_ext_m` in the ring, upsampled bilinearly;
   - convert to `P_C` and then `P_S`.
2. **Scale and centre.**
   - `L_S` = the robust long extent of the subject in S.
   - `m_in = L_S/(0.9·size)`.
   - The target pixel is `((x_S − x0)/m_in + size/2, (y_S − y0)/m_in + size/2)`.
3. **Z-buffer.** `lexsort` by (target, z_S), then `minimum.reduceat` gives the per-pixel front z. Average the colours within 2 cm of it with `bincount`. Alpha is forward-splatted α.
4. Morphological close, then push-pull fill inside the closed coverage. Output RGBA at 512 (generator) and 1024 (colour reference, TRELLIS input).
5. `alpha0` (512) is the **orthographic front silhouette**, used for:
   - the back lock;
   - user-image registration;
   - the MV-Adapter view-0 check.
6. The worker composites on 0.5 grey and does **not** re-crop, so an input pixel maps to S exactly. The step also removes pitch and the perspective magnification (about 12 %).

### 3.5 Back sources: `fn(ctx, rep) -> BackPart`

```
BackPart(sc: SplatCloud (C frame, generated only), kind: "mould"|"cloud", source: str, ok: bool, reason: str,
         normals: (N,3), hull: HullIntervals|None, back_z: (Hm,Wm)|None, views: list[GenView], diag: dict)
```

#### 3.5.1 `shell`
`mould_back(front, canon, target=Z_prior, z_rim=None, hull=None)` with no solve:
- **Prior.** `Z_prior(p) = D_smooth(p) + T(p)`.
  - `D_smooth` = `_masked_blur(D_f_m, σ = 4 % of the silhouette height)`.
  - `T = κ·prof·mpp`, where `prof, dt = billboard.inflation_profile(mask)` on the isotropic mould grid and `mpp(p) = D_f_m(p)/f_m`.
  - κ is chosen so `median T = T_prior`, with κ ≤ 4.
- **Clamps.** `z_b ≥ D_f_m + 5 mm`, and `z_b ≤ clip − 5 mm` when a clip map is given (the TRELLIS under-layer).
- **Colours.**
  - `diffuse`: push-pull of an 8-cell band inside the silhouette, × 0.9.
  - `mirror`: the photo colour.

#### 3.5.2 Shared view-based pipeline (mvadapter, images, qwen, comfyui)
`GenView(name, phi_deg, rgb, alpha, kind: "ortho"|"nominal", m, depth=None, meta)`.

**V1. Segmentation.**
- Run `run.ctx.segmenter(common.segment_model)` on each view.
- If it covers less than 2 % or fails, key out the flat background instead (`|rgb − bg| > 0.06`, background estimated from a 4 px border ring).
- Then keep the largest component and apply `clean_matte`.
- Drop views whose matte covers less than 2 %.

**V2. Registration to S.**
- **Orthographic (mvadapter):**
  - **Lateral check.** `IoU(α_view0, alpha0) ≥ 0.85`. Otherwise fit a 2D similarity (s, tx, ty) by symmetric chamfer: `E = mean DT_a0(s·p + t) + mean DT_v0((q − t)/s)`, using `cv2.distanceTransform` and the numpy Nelder–Mead. Apply it to all views (they share the framing). Below 0.75, reject the whole set.
  - **Invariant rows.** At elevation 0 in orthographic projection, the top and bottom silhouette rows are identical in every view. For each view, `s_k = (b0 − t0)/(b_k − t_k)`. If `|s_k − 1| > 5 %`, drop that view; otherwise rescale it about its centre column and align the bottom row (`cv2.warpAffine`).
- **Nominal (user, Qwen, ComfyUI):**
  - **Back (φ = 180).** Mirror the alpha horizontally and fit a similarity (s, tx, ty, roll ≤ 5°) to `alpha0` by chamfer plus Nelder–Mead. Gate: IoU ≥ 0.75.
  - **Side views.** Invariant rows give `s` and `ty`. The column offset is absorbed in V5.

**V3. Back lock** (an exact orthographic invariant: back silhouette = mirrored front silhouette).
1. Require `IoU(flip(α180), alpha0) ≥ 0.75`; otherwise reject the back view. That rejects the set for mvadapter and fails the source for nominal ones.
2. Warp the back RGB and alpha with a thin-plate spline so `α180 := flip(alpha0)`:
   - 64 control points sampled by arc length on both contours, starting at the topmost point with the same orientation;
   - backward TPS from target to source, `K = r² log r`, λ = 1e-3, solved as a 67×67 numpy system;
   - then `cv2.remap`.
3. Reject if the maximum displacement is above 6 % of `L` (the pose is too different).

**V4. Colour fit** (if `color_match`).
- **mvadapter:**
  - Pixel pairs between view 0 and the levelled input: both α > 0.9, eroded 4 px, gradient below its 30th percentile.
  - In linear RGB, solve `min Σ ρ_Huber(‖A c_g + b − c_s‖) + λ‖A − I‖²` (5 IRLS iterations, λ = 0.1·n).
  - Clamp `‖A − I‖_F ≤ 0.35`, `|b| ≤ 0.12`, `diag A ∈ [0.6, 1.6]`. Fall back to Reinhard if the residual does not beat identity.
  - Apply the same fit to every view (they share one exposure).
- **Nominal:** LAB mean/std (Reinhard) transfer between the view's rim band and the photo's rim band (within 3 % of the silhouette), at strength 0.5.

**V5. Orbit axis and azimuth sign** (side views).
- In view k (φ = ±90), the depth of column u along the ray is constant: `z_S = z_axis + (u − N/2)·m·sin φ`.
- Front-edge column: `u_f(j) = argmin over silhouette columns of (sin φ·u)` in row j. Back edge: `u_b = argmax`.
- With `zf(j)` = P2 of the front points' z_S in row j: `z_axis(j) = zf(j) − (u_f(j) − N/2)·m·sin φ`.
- **mvadapter** (shared axis column):
  - `z_axis` = the median over valid rows of both views.
  - Accept if the two views agree within `max(4 cm, 2 % H)` and the row MAD is ≤ 3 cm; otherwise drop the side views and warn.
  - **Sign safety net.** Swapping the sign makes both views pick the back edge, so they still agree with each other but their row-wise MAD grows; MAD is the discriminator. If the flipped hypothesis gives < 0.5× the cost, flip and warn.
  - **From M3,** also compare the ±45° views with SHARP renders at ±40° (IoU + 0.5·masked SSIM; margin ≥ 0.05).
  - Rule: flip only if both tests contradict the worker cameras.
- **Nominal sides:** each view gets its own offset, matched to `zf(j)`. QC: the back edges implied by left and right views must agree within `max(4 cm, 3 % H)` (median).
- **Without side views:** `z_axis` uses the prior.

**V6. Hull intervals per mould-grid ray** (when at least one side or quarter view is accepted).
1. Sample 96 points from `D_f_m(p)` to `D_f_m(p) + max(0.25 m, 0.5·height_s)`.
2. A point is inside view k iff it projects into `dilate(α_k, 3 px)`. The photo matte holds by construction; pixels outside the photo frame count as unknown.
3. **Voting:** inside iff it is inside at least K − 1 of the K constraining views (view 0 excluded); if K ≤ 2, all are required.
4. `[z_in, z_out]` is the first contiguous inside run. Refine `z_out` with 3 bisection steps.
5. Process rays in chunks, with a cancel check per chunk.

**V7. Back depth target.**
- **Relief `d`.**
  - Run `run.ctx.depth(view180_rgb, estimator=…)` (call `model_notice`). Larger disparity means nearer the back camera, i.e. larger z_S.
  - If the result is the heuristic (no model), the relief is `flat`.
  - High-pass: `d_hp = d − masked_blur(d, 4 %)`.
- **With side views:**
  - Rows j share pixel rows across views.
  - `z_back(j)` = the maximum over side views of `z_axis + (u_b − N/2)·m·sin φ`.
  - `d_max(j)` = P99 of `d` in row j within α180.
  - Huber least squares (δ = 1 cm) for `b + a·d_max(j) ≈ z_back(j)`. Require ≥ 20 % valid rows; if `a ≤ 0`, set `a = 0`.
  - `b(j) = b + G_σ(residual)` with σ = 5 % of the rows.
  - `Z_lf(q) = b(j) + a·d(q)`.
- **Without side views:** `Z_target = Z_prior + A·clip(d_hp/P99|d_hp|, −1, 1)·taper(dt/3)`, with `A = min(3 cm, 0.25·T_prior)` (the billboard detail rule).
- **From the back-view pixel q to camera ray p.**
  - Start at `z⁰` = `z_out` or the prior.
  - Look up `Z_S = Z_lf(q(z⁰))`, then solve `z¹ = (Z_S + ẑ·c)/(ẑ·r_p)`.
  - Two iterations.
- **Clamp** to `[D_f + 1.5 cm, z_out + 2 cm]` and `≥ z_out − 0.4·(z_out − D_f)` (carving limit).
- **Rim depth.** `Z_rim` = `(z_in + z_out)/2` on boundary rays when the hull is valid there, else `D_ext_m`.

**V8. Poisson mould** (§3.6) gives `z_b`. Then build back Gaussians, and a **rim wall** where `Z_rim − D_ext > 1 cm`.

**V9. Colours.**
- Project back and rim Gaussians into each accepted view k. Visibility is a **point z-buffer**: centres splatted with radius `clip(2σ_t/m, 1, 4)` px, `minimum.at`, visible if `depth ≤ zbuf + 1.5 cm`.
- Weight `w_k = max(0, n·(−forward_k))⁸`; the colour is the weighted mean.
- The 180° view is always eligible. Views are sampled bilinearly from images bled 8 px outside their alpha (push-pull), so edges never pick up the grey background.

**V10. Hole fill from side views** (balanced and best; φ ∈ {90, 270, 45, 315}, plus 135 and 225 for nominal sources).
1. Point-splat F, back and rim into view k.
2. Uncovered silhouette pixels (α_k eroded 2 px) are lifted at the **hull entry** along the view's ray. Entry means: inside the photo matte through C, plus the voting rule.
3. Normal `−forward_k`, scale `0.7·m`, colour from view k.
4. If holes exceed 20 % of the back count, the view is inconsistent: skip it and warn.

**V11. QC cost** (to choose between tries): `(1 − IoU_back)` + height spread + z_axis disagreement + MAD/0.03 + yaw-margin penalty + a face penalty (from M6).

#### 3.5.3 mvadapter specifics
- **Inputs.** The levelled 512 RGBA. Params: `seeds=[seed + i for i in range(tries)]` (one worker call, no reload), steps, guidance, prompt (default "a person, full body, high quality, realistic photo").
- **Selection.** Keep the set with the lowest V11 cost. If every set is rejected, fall to the next source.
- **Cache.** Views go to `pipeline/view_cache.py`, keyed by `content_hash(levelled input)` + params + worker version, with a 2 GB LRU. A re-run with a different seam or quality skips generation.

#### 3.5.4 images specifics
- The `back_images` file is φ = 180.
- Siblings in the same folder are used if named `right` (φ 90), `left` (270), `back_right` (135), `back_left` (225), `front_right` (45), `front_left` (315), or `azNNN`. "Right" means you walked to your right.
- A `views.json` from a saved views folder restores exact orthographic cameras.
- Downscale views to a long side ≤ 1024.
- Without side photos, the result is the mould from the prior plus relief detail, with colour from the back photo.

#### 3.5.5 trellis
- **Run.** Input: the levelled 1024 RGBA. Call `_run_addon("trellis", "image_to_3d", {"image"}, {"seed", "ss_steps", "slat_steps", "glb": False, "ply": True})`, then `read_splats(ply)`.
- **Registration**, without trusting TRELLIS's axis convention:
  1. Take opaque centres (α > 0.3), subsampled to 20k.
  2. Try up ∈ {±x, ±y, ±z} × 48 yaws (7.5° steps). For each: scale `s0 = height_s / candidate height`; translate so the projected centre hits the matte centroid and the median near-surface depth equals `median(D_f)`; z-buffer-project at 128 px into C.
  3. Score = `IoU − 0.5·min(1, med|z − D_f|/0.1 m)`.
  4. Refine the best three with Nelder–Mead over (yaw, pitch, roll ≤ 15°, log s, t) at 192 px. Cost = `1 − IoU + 0.5·resid/0.1 + 0.2·chamfer`.
  5. **Projective ICP**, 8 iterations:
     - TRELLIS points that win the z-buffer are paired with `unproject(cell, D_f)`;
     - trim the worst 30 %;
     - Umeyama with scale: `Σ = UDVᵀ`, `S = diag(1, 1, det UVᵀ)`, `R = USVᵀ`, `s = tr(DS)/σ_p²`, `t = μ_q − sRμ_p`;
     - reject any step that rotates more than 10°.
  6. **Gates:** IoU ≥ 0.70, median residual ≤ 5 cm, `s ∈ [0.5, 2]·s0`.
- **Colour.** Fit the V4 affine on TRELLIS Gaussians that sit on the photographed surface against SHARP's colours at the same cells, then apply it to the rest.
- **Back depth for the under-layer.** `back_z(p)` = the maximum z of kept TRELLIS Gaussians projecting to mould cell p.
- **Tries.** Each try is a separate TRELLIS run; keep the best registration score.

#### 3.5.6 qwen / comfyui (M7, experimental)
- **Input.** The plain cut-out on #c8c8c8, which keeps hair edges more readable than black. Views are nominal, so they go through V1-V11.
- **Phases:**
  1. Back candidates.
  2. Choose by V3 IoU, YuNet frontal-face rejection on the 135/180/225 views (`cv2.FaceDetectorYN`, score > 0.8 in the head region) and the χ² of rim-band HSV histograms.
  3. Side views with references `[input, chosen back]`.
- **Side sign.** Test both ±90° hypotheses (V5 MAD). The vocabulary-to-φ mapping (`AZ_SIGN`) is measured in the spike.

### 3.6 Mould solver and Gaussians from a height field
**Poisson back (`poisson_heightfield`).** Unknowns: `z` on matte cells Ω of the mould grid. Energy:

`Σ_(p,q)∈E w_pq (z_q − z_p − (T_q − T_p))² + λ_r Σ_∂Ω (z_p − Z_rim,p)² + λ_s Σ_Ω (z_p − T_p)²`

- `T` = `Z_target`. `w_pq = 0.1` where `|∇d| > P99`, else 1. λ_r = 10, λ_s = 0.01; z in metres.
- **Solver.** Matrix-free, Jacobi-preconditioned conjugate gradient. The operator is built from shifted-slice Laplacians with edge masks.
  - Coarse to fine: Lm/4 → Lm/2 → Lm, 150 iterations per level, warm start by upsampling.
- **Clamps.** Clamp to `[D_f + 1 cm, z_out + 2 cm]`, then re-solve once with the clamped cells as soft constraints (λ = 10).
- **Sanity.** Without a hull, require `median(z_b − D_f) ∈ [0.5, 2]·T_prior`; otherwise use `Z_prior` and warn.

**Gaussians (`gaussians_from_heightfield`)**, stride s:
- Position: `P(p) = ray(p)·z(p)` with `ray = ((u − W/2)/f, (v − H/2)/f, 1)` at mould cell centres.
- Tangents by central differences `t_u, t_v`, then Gram–Schmidt:
  - `t1 = t̂_u`, `t2 = normalize(t_v − (t_v·t1)t1)`, `n = t1 × t2`;
  - if `n·r̂ < 0` (the back faces away from the camera), negate `t2` and `n` (determinant stays +1).
- Scales `σ1 = 0.6·s·|t_u|`, `σ2 = 0.6·s·|t_v⊥|`, clamped to [1 mm, 5 cm]; `σ3 = max(0.5 mm, 0.1·min(σ1, σ2))`.
  - Steep slopes near the outline get long tangent axes, which closes the sides.
- Quaternion = `rotmat_to_quat([t1 t2 n])`, opacity 0.95.

**Rim wall.** Along `cv2.findContours` of the mould matte:
- Gaussians every `m` from `D_ext` to `Z_rim` along each boundary ray.
- Axes: along the ray and along the contour (0.6 × spacing); the normal is the outward contour normal.
- Colour from the ±90 views, else a lerp of the front and back edge colours.

### 3.7 Merge and seam: `merge_parts(front, part, *, seam_deg, seed, budget) -> (sc, is_front, diag)`
- `w_s(θ) = smoothstep(cos(seam + 10°), cos(seam − 10°), cos θ)`; RNG = `np.random.default_rng(seed)`.
- **SHARP (F) Gaussians:**
  - **Protected** (bit-identical) if `θ < seam − 10°` or `rim`.
  - Otherwise kept with probability `w_s`, but only dropped where **covered**: at least one generated Gaussian within 2 cm (2 cm voxel hash, 27-neighbourhood). Otherwise always kept.
- **Generated Gaussians.** Test the centre and both ends `μ ± 2σ1·a1`:
  - drop if any test point projects outside `dilate(matte, 1 %)` or outside the image, so nothing new shows around the outline from the original view;
  - drop if `z < D_f − τ` (in front of the photo);
  - if `|z − D_f| ≤ τ` (on the photographed surface), keep with probability `1 − w_s(θ_cell)`;
  - else keep (hidden).
- Dithering keeps about one layer. Cross-fading opacities would not: two 0.95 layers at 50 % each give about 0.72.
- **Budget.** Decimate the **generated** part first to `750k − |F| − 5k` (`decimate_splats`). Only if `|F| > 600k` is F decimated (logged).

### 3.8 Closure under-layer (`cloud` sources, i.e. TRELLIS)
- A shell with `inset 5 mm`, clipped per ray to `back_z − 5 mm`.
- Keep shell Gaussians that are more than 1.5 cm from any other Gaussian. Holes in TRELLIS's back then show matching soft colour instead of see-through.
- Mould sources are closed by construction.

### 3.9 Colour feathering (never changes F)
- For generated Gaussians within 3 cm of reliable F: `c += w·(c̄_F − c)`, with `w = 1 − smoothstep(0, 3 cm, d)`.
- `c̄_F` = the mean colour of F over the 27 neighbouring 1.5 cm voxels; `d` = the distance to the nearest F voxel centroid.

### 3.10 Floaters: `remove_floaters(sc, is_front, front, canon, *, hull) -> (sc, is_front, diag)`
- **Generated Gaussians (and post-Brush ones) are dropped if:**
  - opacity < 0.05;
  - max scale > 0.035·height_s;
  - anisotropy > 25 with max scale > 0.012·height_s;
  - they lie outside the 1 %-dilated matte in C;
  - they lie more than 2 cm outside the hull;
  - they are radius outliers: fewer than 3 neighbours in 27 cells of 1.5 cm.
- **Everything: connected components** on 1.5 cm voxels with 26-connectivity.
  - Edges via `searchsorted` over sorted int64 voxel keys for 13 offsets.
  - Labels by `np.minimum.at` propagation plus pointer jumping (`lab = lab[lab]`).
  - Keep components with ≥ 1 % of the opacity mass or within 5 cm of the largest.

### 3.11 Brush polish (M5; view-based sources; `capture.train_splat`)
**Dataset** in `work/brush_ds`, world frame = C:
- `images/` and `masks/` (black = ignore);
- `sparse/0/cameras.txt` (PINHOLE `W H fx fy cx cy` per image), `images.txt` (`qw qx qy qz tx ty tz` with `R_wc = R_kᵀ`, `t_wc = −R_kᵀC_k`), `points3D.txt` (20k init means);
- `init.ply` (from `write_splats`, read back to validate) as the only `.ply`. The export folder sits outside the dataset folder.

**Frames:**
- The photo as 3 distinct copies (`p0_`, `p1_`, `p2_`): RGBA, long side ≤ 1024, cropped, K adjusted, identity pose.
- SHARP **anchors** (needs the M3 renderer): renders of F at yaw ±12°, ±24° and pitch ±10° around `c` (best adds ±18°/−8° and 0°/+8°), RGB on black.
  - Mask = (θ < seam ∧ render alpha ≥ 0.95) ∪ (more than 3 % outside the silhouette, as a black target that suppresses floaters).
- Generated φ = 180, ±90: RGBA with locked alpha (alpha is trained).
- φ = ±45: RGB with mask = NOT(front-reliable, dilated 5 px).
- Orthographic views become telephoto pinholes: centre `a − 40 m·forward`, `f = 40/m`, `cx = cy = N/2`. Brush takes bounds from the initial splats (R3), so the distant cameras are harmless.

**Command:**

```
brush_args(ds, out, steps=N, max_splats=clamp(1.3·n_init, 300k, 2.5M), export_name="polished.ply", seed=seed,
           extra=["--sh-degree", "0", "--growth-stop-iter", str(N//2), "--match-alpha-weight", "0.5", "--max-resolution", "1024"])
```

**Retries and checks:**
- A non-zero exit with `--sh-degree 0` retries once with 1.
- Out of memory retries with `--max-resolution 768` and the cap × 0.6.
- Reject the output if it has fewer than 1000 splats, a box more than 1.5× the init's, or NaNs. The result is then the unpolished cloud plus a warning.

**After training:**
- Apply §3.7 to the trained cloud against F (trained Gaussians count as generated), so it drops in-front, front-visible and outside-matte Gaussians.
- Re-insert the **exact kept F rows**, then run §3.9 and §3.10 again.
- If the spike shows that masks cannot be mixed with RGBA images, drop the anchors (F is still re-inserted).

### 3.12 Export
1. `sc = _splats_cv_to_gltf(merged)`.
2. `normalize_orientation(sc, up=cv_to_gltf(canon.up_C))`. With a known up, the floor is P0.5 and x/z are centred on the median.
3. `to_life_size(sc, mode, height_m, cut)`:
   - Measured height `h = P99.8(y | α > 0.5)`.
   - `height`: `k = height_m/h`.
   - `photo`: `k = 1` and `height_m := h`.
   - `auto`: height, unless `cut_top` or `cut_bottom`, then photo with a warning.
   - Scale means and scales by `k` (new `splat.scale_splats`).
   - Return the box: x, z ∈ [P0.2 − 3 cm, P99.8 + 3 cm], y ∈ [−2 cm, h' + 3 cm].
4. `crop_splats(sc, radius_pct=None, aabb=box)`.
5. `decimate_splats(max_n=750k)`.
6. `write_splats`.
7. Block with `scale: 1.0` and `height_m = h'`.

### 3.13 Previews
- **Contact sheet** (no renderer): input and views with the `alpha0` / mirrored outline drawn on top.
- **Turntable** (M3): `turntable_sheet(sc, n=8, size)`, orthographic renders around y. Failure is only a warning.

---

## 4. Code plan

### 4.1 Main app (numpy/cv2, lazy imports, `progress=`/`cancel=` on long functions)

| File | New / changed | Contents |
|---|---|---|
| `geometry/splat_fusion.py` | new (M1) | `SharpCam`, `MouldGrid`, `FrontModel`, `Canon`, `BackPart`; `sharp_front`, `canonical_frame`, `seam_weights`, `merge_parts`, `add_underlayer`, `feather_colors`, `remove_floaters`, `voxel_components`, `to_life_size` |
| `geometry/back_mould.py` | new (M1) | `inflation_prior`, `shell_back`, `poisson_heightfield`, `gaussians_from_heightfield`, `rim_wall`, `mould_back`, `HullIntervals` |
| `geometry/register.py` | new (M2) | `nelder_mead`, `umeyama`, `point_zbuffer`, `silhouette_iou`, `chamfer_similarity_fit`, `tps_fit` / `tps_warp`, `projective_icp`, `register_cloud_to_front`, `color_affine_fit` / `apply_color_affine`, `reinhard_lab` |
| `geometry/views_lift.py` | new (M2, extended in M4) | `GenView`, `OrthoInput`, `levelled_ortho_input`, `views_from_folder`, `views_from_cameras`, `segment_views`, `check_view_set`, `register_nominal_view`, `lock_back`, `color_fit_views`, `solve_orbit_axis`, `hull_intervals`, `back_target`, `colour_from_views`, `fill_view_holes`, `view_based_back`, `contact_sheet` |
| `geometry/splat_render.py` | new (M3) | `SplatCamera(R, center, width, height, kind, fx, fy, m, cx, cy)`, `render_splats(...) -> SplatRender(rgb, alpha, depth, depth_median)`, `orbit_camera`, `turntable_sheet` |
| `geometry/splat.py` | changed | Public `transform_splats(sc, R, t, s=1.0)` (rotates quats and SH), `scale_splats`, `concat_splats`, `subset_splats`, `fit_plane_ransac`; M6: `read_splats(path, *, return_rows=False)` (diagnostics only) |
| `geometry/billboard.py` | changed | Extract `inflation_profile(mask) -> (prof, dt_eff)` from lines 269-272; behaviour unchanged |
| `pipeline/sharp360.py` | new (M1) | `BACK_SOURCES`, `QUALITY` table, `Opts`, `resolve_options(spec, *, installed)`, `mode_sharp360(...)`, sources `_src_shell`, `_src_trellis`, `_src_images`, `_src_mvadapter`, `_src_qwen`, `_src_comfyui`, `_polish_with_brush`, `_write_extras`, `_debug` |
| `pipeline/view_cache.py` | new (M4) | `get(key)`, `put(key, files)`, LRU 2 GB in `<cache_dir>/backviews`, `salvage(dir)` |
| `pipeline/run.py` | changed | `ADDON_METHODS["sharp360"] = ("sharp", "image_to_splat")`; constants `BACK_SOURCES`, `BACK_QUALITIES`, `BACK_RELIEFS`, `POLISH_MODES`, `SIZE_FROM`; `_sharp_inputs()`; `_free_app_gpu()` called inside `_run_addon`; in `_mode_object` after the matte check: `if method == "sharp360": return mode_sharp360(...)` |
| `pipeline/jobs.py` | changed | §1.2 fields; method comment |
| `addons/capture.py` | changed | `brush_args(..., extra: Sequence[str] = ())`; `write_colmap_text(ds, cams: list[(name, K, R_wc, t_wc)], points, colors)`; `train_splat(dataset, out_dir, *, steps, max_splats, extra_args=(), export_name, seed, progress, cancel, timeout, log_name) -> Path`, built from block 3 of `capture_to_splat` (`_Runner`, `find_brush`, `_BRUSH_ITER_RE`, export discovery, OOM hint); `brush_selftest()` (M0) |
| `addons/manifest.py` | changed (M7) | Optional top-level `min_ram_gb` (advisory), shown in the add-ons dialog |
| `addons/workers/sharp_worker.py` | changed (M6) | `image_to_splat` also accepts `inputs["images"]` (list) and `hfov_deg` (float or list) → `plys`; per-image OOM → `free_cuda`, then CPU for that image; missing both inputs → "Missing input 'image'" |
| `models/registry.py` | changed (M6) | `yunet` (MIT, about 0.3 MB, sha256), optional |
| `gui/options_meta.py`, `gui/panels/options.py`, `gui/main_window.py`, `gui/util.py`, `gui/dialogs/addons_dialog.py` | changed | §1.2 / §1.7: `enabled_if_all`, `addon_optional`, `field_applies`, `required_addon_groups`, `ask_choice`, presets, show `min_ram_gb` |
| `cli.py` | changed | `_choices_table` rows; `addons brush-test` action (M0) |

### 4.2 Orchestration (`pipeline/sharp360.py`)

```python
def mode_sharp360(run, item, photo, alpha, spec, *, placement, height, rep):
    opts = resolve_options(spec, installed=_addon_installed)          # InputError on bad values, before work
    run.addon_notice(item, _addon_manifest("sharp")); _save_cutout(run, item, photo, alpha)
    outs, params = _run_sharp(run, item, photo, rep.sub(.10, .25))    # AddonError -> job fails (as today)
    sc_full = read_splats(outs["ply"]); cam = SharpCam.from_outputs(outs, params, n=len(sc_full))
    front = sharp_front(sc_full, alpha, cam, mould_long=opts.mould_long, cancel=run.cancel)  # internal fallback
    canon = canonical_frame(front, sc_full, alpha, thickness=spec.thickness)
    part = None
    for i, src in enumerate(opts.sources):                            # e.g. ["mvadapter", "trellis", "shell"]
        nxt = _title(opts.sources[i + 1]) if i + 1 < len(opts.sources) else "the photo front only"
        try:
            part = SOURCES[src](Ctx(run, item, photo, alpha, front, canon, opts), rep.sub(.30, .72))
            if part.ok:
                break
            run.warn(f"{_title(src)} could not make a usable back ({part.reason}); {nxt} was used instead.")
        except Cancelled:
            raise                                                       # Cancelled is a VRPhotoError!
        except AddonNotInstalled:
            run.warn(f"{_title(src)} is not installed; {nxt} was used instead.")
        except (AddonError, VRPhotoError, MemoryError, OSError) as e:
            run.warn(f"{_title(src)} could not make the back ({e}); {nxt} was used instead.")
        except Exception as e:                                          # fusion bug: degrade, keep traceback
            log.exception("back source %s", src)
            run.warn(f"{_title(src)} failed unexpectedly ({type(e).__name__}); {nxt} was used instead.")
        part = None
    try:
        merged, is_front = merge_parts(front, part, seam_deg=opts.seam, seed=spec.seed)
        if part is not None and part.kind == "cloud":
            merged, is_front = add_underlayer(merged, is_front, front, canon, part)
        merged = feather_colors(merged, is_front, front)
        if opts.polish and part is not None and part.views:
            try:
                merged, is_front = _polish_with_brush(run, item, merged, is_front, front, canon, part, opts)
            except Cancelled:
                raise
            except (AddonError, VRPhotoError, OSError) as e:
                run.warn(f"Brush polish was skipped ({e}); the unpolished result was kept.")
        merged, is_front = remove_floaters(merged, is_front, front, canon, hull=getattr(part, "hull", None))
    except Cancelled:
        raise
    except Exception:
        log.exception("stitching")
        run.warn("Stitching front and back failed; the cleaned SHARP front was saved instead.")
        merged = front.sc
    # §3.12 export -> previews (non-fatal) -> extras -> notices for used add-ons -> _export_scene(block with scale 1.0)
```

- Add-on calls go through `_run_addon`, which calls `_free_app_gpu()` and writes to `work_dir(item)/<addon_id>`. Workers handle one request per process, so their VRAM is freed when they exit.
- Brush runs in-process like `_mode_capture`.

### 4.3 New add-on `mvadapter` (M4)

**`addons/manifests/mvadapter.toml`**

```toml
# MV-Adapter image-to-multiview (SD 2.1 base): 6 orthographic views at 512 px, used by 'SHARP 360'.
id = "mvadapter"
title = "MV-Adapter (AI back and side views, experimental)"
description = """Generates 6 matching views (front, sides, back) of a cut-out person or object; \
'SHARP 360' builds the back of the splat from them. A few minutes per run on GTX 10-series GPUs."""
version = "1.0.0"
experimental = true
license = "MV-Adapter: see LICENSE at the pinned commit (model card: Apache-2.0); Stable Diffusion 2.1 base: CreativeML Open RAIL++-M"
noncommercial = true        # until the pinned LICENSE is verified (critic R12); then false
region_restricted = []
min_vram_gb = 6
disk_gb = 10
kind = "venv"
python = "3.11"
hf_gated = false
homepage = "https://github.com/huanngzh/MV-Adapter"
tasks = ["image_to_views"]
worker = "mvadapter_worker.py"
platforms = ["win32", "linux"]
notes = "fp16 weights (fp32 math on GTX 10-series), fp32 VAE; model CPU offload below 7 GB free VRAM; orthographic cameras, elevation 0."

[env]
MVADAPTER_BASE = "{addon_dir}/MODELS/sd21-base"
MVADAPTER_WEIGHTS = "{addon_dir}/MODELS/mv-adapter"
PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.65"
TORCHDYNAMO_DISABLE = "1"

[[steps]]
type = "venv"

[[steps]]
type = "pip"
title = "Installing PyTorch 2.8.0 (CUDA 12.8)"
packages = ["torch==2.8.0", "torchvision==0.23.0"]
index_url = "https://download.pytorch.org/whl/cu128"     # Windows build includes sm_61

[[steps]]
type = "pip"
title = "Installing diffusers"
packages = ["numpy==2.3.3", "pillow==11.3.0", "diffusers==<pin>", "transformers==<pin>", "accelerate==<pin>",
            "safetensors==0.6.2", "huggingface-hub==0.35.3", "peft==<pin>", "einops==<pin>", "setuptools==75.8.0"]
            # exact pins = MV-Adapter requirements at the pinned commit; confirmed by the CI addons.yml job

[[steps]]
type = "github_archive"
title = "Downloading MV-Adapter"
repo = "huanngzh/MV-Adapter"
ref = "<40-hex commit>"
dest = "src/mvadapter"
pip_install = false

[[steps]]
type = "hf_download"
title = "Downloading Stable Diffusion 2.1 base"
repo = "stabilityai/stable-diffusion-2-1-base"   # decided in M4: this id or the Manojb mirror (critic O5)
revision = "<40-hex>"
local_dir = "MODELS/sd21-base"
allow_patterns = ["model_index.json", "scheduler/*", "tokenizer/*", "*/config.json",
                  "unet/*fp16.safetensors", "text_encoder/*fp16.safetensors", "vae/diffusion_pytorch_model.safetensors"]
size_gb = 2.8

[[steps]]
type = "hf_download"
title = "Downloading the MV-Adapter weights"
repo = "huanngzh/mv-adapter"
revision = "<40-hex>"
local_dir = "MODELS/mv-adapter"
allow_patterns = ["mvadapter_i2mv_sd21.safetensors"]
size_gb = 0.4

[[steps]]
type = "check"
```

The fp16 file names must be verified at the pinned revision.

**`addons/workers/mvadapter_worker.py`.** Top-level imports are stdlib, `_common` and numpy only.
- **Task `image_to_views`.**
  - Inputs: `image` (512 RGBA, already framed). It is checked with `require_input` **before** torch is imported.
  - Params:
    - `steps` (30, range 5-100);
    - `guidance` (3.0, range 1-10);
    - `seeds` (list) or `seed`;
    - `prompt`, `negative_prompt`;
    - `offload` (auto | model | sequential | none);
    - `name`.
  - Outputs: `sets: [{seed, views: [6 png], cameras: path}]`, `frame: "opencv"`, `warnings`.
- **`cameras.json`:**
  ```json
  {"projection": "orthographic", "ortho_half_extent": 0.55, "distance": 1.8, "width": 512, "height": 512,
   "input_fill": 0.9, "pixel_center": 0.5, "convention": "c2w_opencv",
   "views": [{"file": "view_0.png", "azimuth_deg": 0, "elevation_deg": 0, "c2w": [[...]]}, ...]}
  ```
  `c2w` is converted from MV-Adapter's own `get_orthogonal_camera` (azimuth passed as `az − 90`, as the script does). The app derives φ from `c2w` and never guesses the convention.
- **Runtime.**
  - Composite on 0.5 grey with no crop.
  - Call `cuda_info(require=True)`.
  - fp16 UNet and text encoder; fp32 VAE; PyTorch SDPA; `enable_vae_slicing()`; `enable_model_cpu_offload()` below 7 GB free.
  - `callback_on_step_end` drives progress.
  - OOM ladder: `free_cuda` → sequential offload → `WorkerError` with a hint.
  - NaN or black output: retry once with an fp32 UNet.
  - Stub `nvdiffrast` / `cvcuda` in `sys.modules` if `mvadapter.utils` imports them at package load (the i2mv path never calls them).
  - `check()`: `cuda_info` first, then import diffusers, transformers, peft and `mvadapter.pipelines.pipeline_mvadapter_i2mv_sd`, then look for the weight files.

### 4.4 `qwenedit` add-on (M7, sketch with fixes)
**Manifest.**
- Same skeleton as above, plus `min_ram_gb = 16` and `disk_gb ≈ 28`.
- Licence text: Apache-2.0 (Qwen, Lightning, fal LoRA — verify the card); ComfyUI GPL-3.0 as a separate program; ComfyUI-GGUF Apache-2.0.
- **Steps:**
  - torch 2.8.0/cu128;
  - ComfyUI requirements with exact pins (torch removed from the list);
  - `github_archive` ComfyUI and ComfyUI-GGUF at pinned commits;
  - `hf_download` steps with revisions for the VAE, the Qwen2.5-VL GGUF plus mmproj, the fal Multiple-Angles LoRA, the Lightning LoRA (`optional=true`), and **one transformer GGUF quant** chosen from the user's RAM answer (Q3_K_M or Q4_K_M) — a pinned step, not worker prefetch;
  - `check`.

**Worker** (`qwenedit_worker.py` plus a stdlib-only `_comfy.py`, imported inside functions):
1. RAM tier: refuse below 16 GB with the hint "use 'Back photo' or the built-in back".
2. Start `python main.py --listen 127.0.0.1 --port <free> --lowvram --reserve-vram 1.0 --disable-auto-launch` (`--cache-none` on low RAM) with `HF_HUB_OFFLINE=1`.
3. Upload the input, queue a graph per view and candidate, and poll `/history`. Progress is parsed from ComfyUI's tqdm stderr.
4. Write each view as soon as it is finished and append it to `views.partial.jsonl`.
5. OOM: restart with `--novram`, then try the long side × 0.8, then skip the view.
6. Stop ComfyUI in `finally`. It is a child process, so it sits inside the kill-on-close Job object.
7. `--check` verifies `/object_info` contains every node used.

`comfyui` (expert) reuses `_comfy.ComfyClient` and `substitute(graph, mapping)` from the app.

### 4.5 Brush self-test (M0): `vrphoto-cli addons brush-test`
- Writes an analytic known-camera dataset to a temp folder: a coloured sphere seen by 8 orthographic-as-telephoto views, two of them with masks, plus `init.ply`.
- Runs Brush for 300 steps with `--sh-degree 0`, then with 1 if that fails.
- Prints steps per second, peak messages, and whether the export, SH0 and mixed masks worked.
- The result is stored in settings (`brush_ok`); `polish=auto` uses it.

### 4.6 `docs/CONTRACTS.md` §13 "SHARP 360" (plus updates to §5, §9 and §12)
- **Conventions:**
  - frames C and S;
  - the half-pixel convention;
  - the SHARP and mould grids;
  - the φ formula and orthographic projection;
  - the user file-naming rule (`back|behind|rear` = 180, `right` = 90 "you walked to your right", `left` = 270, `back_right` = 135, `back_left` = 225, `front_right` = 45, `front_left` = 315, `azNNN`).
- **Interfaces:**
  - §4.1 signatures and `BackPart`;
  - the front guarantee (§0.2 item 4);
  - the `image_to_views` worker contract and the `cameras.json` / `views.json` schemas;
  - `sharp_worker` list input;
  - `capture.train_splat`, `write_colmap_text`, `brush_args(extra)`;
  - `splat.transform_splats`, `scale_splats`, `fit_plane_ransac`, `billboard.inflation_profile`.
- `sharp360` scenes always carry `scale: 1.0` plus `height_m`.
- **Other docs:**
  - USER_GUIDE §4.6 (new method and options), §8 (MV-Adapter row; note that SHARP runs on the GPU of GTX 10-series cards on Windows) and the troubleshooting section;
  - CHANGELOG 0.2.0 / 0.3.0;
  - README;
  - THIRD_PARTY_NOTICES (MV-Adapter, SD2.1; Qwen and ComfyUI in M7).

---

## 5. Tests (CI: Linux + Windows, no GPU, no torch)

### 5.1 Synthetic fixtures: `tests/geometry/synth_person.py` (analytic, small grids for speed)
- **`Body`:** ellipsoids for the torso, head, legs, a raised **left** arm and a **bag on the right shoulder** (for asymmetry). Coloured by side: front red, back blue, left green, right yellow. `raycast(o, d)` is vectorised closed-form.
- **`sharp_like_cloud(body, cam, G=96)`:**
  - layer-0 Gaussians at ray hits (scale ∝ z/f);
  - layer-1 Gaussians on a wall at 3 m and on a floor pitched 10°;
  - ray-aligned cards at depth jumps;
  - the matte comes from projection.
- **`ortho_view(body, phi, m, N)`:** RGB, alpha and depth. `mvadapter_like_set(...)` adds perturbations (scale 0.97, a 3 px offset, flipped-sign cameras, grey background) and writes a `cameras.json`.
- **`trellis_like_cloud(body, up_axis, yaw, s)`:** a full-surface Gaussian cloud in a unit cube.

### 5.2 Unit tests
- **`test_splat_fusion.py`:**
  - At least 95 % of wall and layer-1 Gaussians inside the matte are dropped; at least 98 % of layer 0 is kept.
  - Torso-behind-arm disocclusion Gaussians are kept.
  - Cards are collapsed with (u, v) unchanged (< 0.01 px) and projected footprint unchanged (< 2 %).
  - Ring wisps at subject depth are kept; ring background is dropped.
  - Up is recovered within 1° from the tilted floor; with no floor it is −y_C.
  - `cut_bottom` is detected.
  - **Merge:** generated points on the non-grazing photographed surface are always dropped; protected F rows are **bit-identical**; the same seed gives the same output; the budget is ≤ 750k and F is never merged; outline (rim) Gaussians are never dropped.
  - `remove_floaters` removes 200 injected floaters and keeps more than 99.5 % of the body.
  - `to_life_size` is within 0.5 %, also with 50 floaters added.
  - `size_from=auto` with a cut-off subject keeps the metric size.
- **`test_back_mould.py`:**
  - The shell is closed from 8 azimuths (point coverage ≥ 97 % of the true silhouette); median thickness = `thickness·height` ± 10 %; frames are orthonormal with det +1.
  - Poisson recovers an ellipsoid back within 1 cm and equals `Z_rim` on the boundary.
  - The rim wall spans `D_ext` to `Z_rim`.
- **`test_register.py`:**
  - Umeyama is exact on noisy points; Nelder–Mead solves Rosenbrock.
  - Projective ICP converges from 5° / 5 % / 3 cm.
  - The up/yaw search recovers (up = −z, yaw = 137°, s = 0.37).
  - The gates reject a mismatched shape.
  - Chamfer similarity recovers (s, t); TPS lock reaches IoU ≥ 0.98.
  - The colour affine fit recovers gain 1.2 / offset −0.05 within 1 % with 20 % outliers, and clamps.
- **`test_views_lift.py`:**
  - `levelled_ortho_input` keeps `m_in` and the centre exact.
  - Lateral and invariant-row checks scale views and drop one at 9 %.
  - `solve_orbit_axis` is within 5 mm; flipped cameras are detected via MAD.
  - Hull voting tolerates one corrupted view (volume within 10 % of the truth); a strict any-view carve would fail this test.
  - Lifted back colours are blue and sides green/yellow (the bag appears on the correct side).
  - Relief carving is ≤ 40 % of the thickness.
  - A nominal back with roll 3° and scale 1.1 locks.
  - `views_from_folder` follows the naming rules.
- **`test_splat_render.py`** (M3):
  - One Gaussian matches the analytic 2D Gaussian (peak and σ within 2 %); depth order is correct.
  - Orthographic ≈ telephoto; band sizes give identical images; matches a brute-force loop on 20 Gaussians (< 1e-3).
  - Empty cloud and cancel are handled.
  - **Front untouched:** rendering the fused result from the original camera matches the SHARP-only render inside the eroded matte (mean difference < 2/255).
- **Performance** (`slow` marker): front clean-up on 1.2M Gaussians in under 30 s.

### 5.3 Pipeline tests: `tests/pipeline/test_run_sharp360.py`
They use `FakeManager`, which dispatches on `addon_id`, and `box_segmenter`. The fake SHARP writes `sharp_like_cloud`, fitted to the box matte (rows 300-460, cols 250-390 of 640×480), and returns `f_px`, `width` and `height`.

- **Only `sharp` installed:**
  - writes `photo_sharp360.splat`;
  - `check_scene` passes with `scale == 1.0` and `height_m`;
  - the robust y extent equals `height_m` ± 1 %;
  - the SHARP non-commercial notice is present;
  - `photo_sharp.splat` from `method="sharp"` is unchanged (regression).
- **`back_source=mvadapter`, not installed** → warning, then TRELLIS or shell, and the job succeeds. `polish=on` without Capture → warning.
- **mvadapter installed** (the fake returns `mvadapter_like_set`):
  - calls run in the order sharp → mvadapter;
  - a spy on `ort.unload_model` sees a call before each;
  - `_views.png` is written;
  - the second run hits the view cache (mvadapter is not called).
- **mvadapter raises `AddonError("CUDA out of memory")`** → falls back with the message in the warning. A garbage view set is rejected by the gates and falls back.
- **TRELLIS fake** (cloud at a random yaw):
  - registered, with IoU ≥ 0.7 in `debug.json`;
  - TRELLIS notice present;
  - the request had `glb=False, ply=True`.
- **images source:** a back PNG plus `right.png` siblings; a missing file is an `InputError` before any work.
- **Failure and cancel cases:**
  - SHARP fails → `AddonError` (as today).
  - SHARP missing → `AddonNotInstalled` before segmentation (`test_object_addon_not_installed_fails_fast` parametrised with `sharp360`).
  - `Cancelled` during the back stage propagates and is **not** turned into a fallback warning.
  - A monkeypatched exception in `merge_parts` → front-only output plus a warning.
- **Brush with the fake:**
  - `FAKE_BRUSH_ECHO_INIT` → polished path, and the exact F rows are re-inserted;
  - `FAKE_BRUSH_FAIL_SH0` → retries with `--sh-degree 1`;
  - `FAKE_BRUSH_FAIL` → warning, unpolished result kept.

### 5.4 Add-on tests
- **`test_manifests.py`:** add `mvadapter` (M4) and `qwenedit` (M7) to `EXPECTED_IDS` and to the exact torch set; the pins pass; `min_ram_gb` parses and a negative value is rejected.
- **`test_workers.py`:**
  - the light-import, unknown-task, missing-input and no-torch check tests pick the new workers up automatically;
  - add `mvadapter_worker` and `qwenedit_worker` to the GPU-guard parametrisation (`WIN_TORCH_280_CU128`);
  - pure helpers: the c2w conversion per azimuth (camera at +x for φ = 90), compositing without re-crop, parameter validation, `ram_tier`, `angle_prompt`, `build_workflow`, `parse_tqdm`;
  - `_comfy` against `fakes/fake_comfyui.py` (an `http.server` thread), including timeout and cancel → `/interrupt`;
  - `sharp_worker` list-input validation.
- **`fakes/fake_brush.py`:**
  - accept `sparse/0/cameras.txt` or `cameras.bin` anywhere;
  - when `init.ply` exists, assert the listed images exist and every mask stem matches an image;
  - new switches `FAKE_BRUSH_ECHO_INIT` (copy init, jitter means by 2 mm) and `FAKE_BRUSH_FAIL_SH0`.
- **`test_capture.py`:** `brush_args(extra)`; a `write_colmap_text` round trip (project a known point, error < 1e-6 px); `train_splat` progress and the OOM hint; `brush_selftest` with the fake.

### 5.5 GUI and CLI tests
- **`test_options_meta.py`:**
  - metadata exists for every new field (automatic);
  - the seed set is `{trellis, trellis2, hunyuan, sharp360}`;
  - `enabled_if_all` refers to sibling fields;
  - `required_addon_groups("object", {"method": "sharp360", "back_source": "mvadapter"}) == ((("sharp",), False), (("mvadapter",), True))`, and billboard plus a remembered trellis back gives `()`;
  - presets are valid.
- **`test_options_panel.py`:**
  - `sharp360` fields are greyed out for other methods;
  - `polish_steps` is greyed when polish = off;
  - `back_images` is enabled only for images;
  - `thickness` and `seed` are enabled for `sharp360` (extend `test_seed_and_texture_follow_the_method`).
- **`test_main_window.py`:** the optional-group prompt with a monkeypatched `ask_choice` (continue / open / cancel); a missing SHARP still shows the required dialog.
- **`test_cli.py`:** the §1.6 examples parse; a bad `--back-source` is rejected; `--no-color-match` and `--save-extras` work.

### 5.6 Manual protocol on the user's PC (GTX 1080, Windows 11, Quest 3)
Record times and peak values from Task Manager in a checklist file.

1. **Pre-flight:**
   - note system RAM and free disk;
   - run `vrphoto-cli addons check sharp` and confirm a SHARP job logs "Model loaded (cuda)";
   - run `vrphoto-cli addons brush-test` (Capture installed) and note steps/s, SH0 and masks.
2. **Photos:**
   - (a) full body, level camera, plain background;
   - (b) 3/4 view with the left arm raised;
   - (c) bust;
   - (d) black hair and black clothes;
   - (e) chair;
   - (f) camera looking down about 25°;
   - (g) person 30 cm in front of a wall;
   - (h) bag on one shoulder.
3. **Fast (built-in) on all photos.** In the Quest:
   - the front matches plain `sharp` (switch between the two scenes);
   - no streaks or wall blobs from 90°;
   - the back is closed;
   - height matches a door frame or tape measure within 2 cm;
   - no floaters.
4. **TRELLIS on (a), (b), (h):** time; `debug.json` (IoU, residual, yaw); doubled shoulders? colour seam?
5. **Back photo:** a real phone photo of the same person from behind for (a); check the seam at 60-120° from 1 m.
6. **Install MV-Adapter** (time, disk). Balanced on (a), (b), (d), (h):
   - peak "Dedicated GPU memory";
   - `_views.png` plausibility;
   - **the bag is on the correct shoulder seen from the side** (checks the azimuth sign);
   - `z_axis` MAD in `debug.json`.
7. **Best plus Brush on (a):** A/B with and without polish. If it fails, keep the `*_polish.log`.
8. **Robustness:**
   - offline run;
   - cancel mid-generation: stops within 5 s, no leftover `python.exe` or `brush_app.exe`;
   - two queued jobs;
   - low disk space.
9. **Send back:** `vrphoto.log`, the `debug.json` files and Quest screenshots, for threshold tuning (τ, seam, gates).

---

## 6. Milestones, effort and risks

| # | Milestone | Effort (dev-days) | Ships |
|---|---|---|---|
| M0 | Plumbing (fields, method, metadata, `field_applies`, `required_addon_groups`, `ask_choice`, `_free_app_gpu` in `_run_addon`, CLI rows, test skeletons and synthetic fixtures); `addons brush-test`; ask the user the §(b) questions | 1.5-2 | — |
| M1 | Fusion core (front clean-up, canonical frame, merge and seam, floaters, life-size export) plus the `shell` mould; pipeline tests; `debug.json` | 4-5 | 0.2.0-beta1 (user checks the front and the life size) |
| M2 | `register.py`; levelled input; **TRELLIS** source (likely already installed); **images** source (similarity, TPS lock, depth-model relief, Reinhard colours, views folder); contact sheet | 4-5 | **0.2.0** (first AI back) |
| M3 | CPU renderer; turntable preview; front-untouched regression test; yaw-check hook | 2 | 0.2.1 |
| M4 | MV-Adapter add-on (manifest, worker, CI install, licence and SD2.1-base check, camera export); side-view machinery (V5, V6, V10, V11), tries and QC, view cache; manual PC test | 5-6 | **0.3.0** |
| M5 | Brush polish (`train_splat`, COLMAP writer, anchors and masks, re-insertion, ladder) — only if `brush-test` passes | 3 | 0.3.1 |
| M6 | SHARP list input plus `back_relief=sharp`; YuNet face check; `read_splats(return_rows)` provenance diagnostics; tuning from the user's logs | 2-3 | 0.3.2 |
| M7 | (Gated by a Qwen spike on the user's PC: s/step, peak RAM, LoRA wording, AZ_SIGN, two-reference chaining) `qwenedit` add-on, `comfyui` source, `min_ram_gb` | 5-8 | 0.4.0, experimental |
| M8 | Docs (CONTRACTS §13, USER_GUIDE, CHANGELOG, notices); second tuning round | 2 | — |

**Total:** about 21-26 dev-days through M6 (the core feature), and 28-36 including the Qwen tier.

### Top risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| 1 | Object-trained MV-Adapter, and TRELLIS, give soft, identity-drifting backs on real people | Generated content is used only where the photo did not see the subject; prompt, tries and seed options; the user-photo source; the experimental Qwen tier; honest UI wording ("plausible, softer back") |
| 2 | Wrong azimuth sign or orbit depth (mirrored bag, gaps) | Worker-exported cameras; MAD sign test plus the ±45 render check; ±90 `z_axis` agreement; gates with fallback; asymmetric synthetic test; "bag on shoulder" manual check |
| 3 | Brush on Pascal unknown; black background; no freezing; SH0 bug | `brush-test` spike before enabling; off by default (best only); SH1 retry; anchors and masks; exact F rows re-inserted; failure keeps the unpolished result |
| 4 | MV-Adapter install fragility (diffusers pins, nvdiffrast imported at package load, SD2.1-base removed, unknown licence) | Pins from the pinned commit, checked by the CI install job; import stub; repo chosen at M4 and pinned by revision; `noncommercial=true` until verified |
| 5 | Floaters shrinking the person (R4) | Explicit `scale: 1.0`; outline cone test; hull; outlier and component filters; box crop |
| 6 | Seams from lighting differences | Huber affine or Reinhard colour fit; feathering; dithered seam band; adjustable `seam_deg`; Brush |
| 7 | 8 GB VRAM contention | `_free_app_gpu()` before every add-on; one process per stage; offload and an OOM ladder in workers |
| 8 | Steep pitch, cut-off or seated subjects that two height fields cannot represent | Levelling plus a warning; cut-off → metric size; hull and side hole lifting; documented limitation; shell fallback |
| 9 | CPU time and RAM of the numpy stages | Chunked and vectorised; cancel checks; grid size by quality; performance test |
| 10 | Qwen is slow or out of RAM on Pascal | Gated spike; RAM refusal; one pinned quant; partial results and view cache; never auto |

### Honest unknowns (resolved during milestones)
- MV-Adapter's real peak VRAM and its back quality on real people (M4).
- Whether Brush 0.3.0 can mix masked and RGBA images (M0 `brush-test`).
- Whether the app's depth model gives useful relief on grey-background generated views (M4; otherwise `flat` or `sharp` relief in M6).
- SHARP's shipped grid and layer count (logged from M1).

---

## (b) Open questions only the user can answer

| # | Question | How the answer changes the plan |
|---|---|---|
| 1 | How much system RAM does the PC have (16 / 32 / 64 GB)? | ≥ 32 GB: the Qwen tier (M7) is viable with the Q4_K_M quant. 16-24 GB: Q3_K_M with `--cache-none`, slower, lower priority. Below 16 GB: drop M7. MV-Adapter and everything else fit in 16 GB. |
| 2 | How much free disk space is on the drive holding `%LOCALAPPDATA%\VRPhotoStudio\addons`? | MV-Adapter needs about 10 GB and Qwen about 28 GB. With under 15 GB free, prioritise TRELLIS and back photos and keep MV-Adapter optional. |
| 3 | Is TRELLIS still installed and working (you mentioned trying it)? | Yes: the TRELLIS back works with no download from 0.2.0, and `auto` uses it until MV-Adapter is installed. No: 0.2.0 offers the built-in back and back photos, and MV-Adapter becomes the main AI route. |
| 4 | Has a Capture-mode job (Brush) ever finished on this PC? Will you run the 2-minute `brush-test`? | A pass enables Brush polish (M5) and `auto` polish for best. A fail drops M5 and saves about 3 dev-days. |
| 5 | Do your photos usually show the full body (feet visible), or half bodies? Standing or sitting? | Half bodies: size from SHARP's own measurement becomes the default advice, and side views matter more than feet and floor detection. Sitting: more weight on hull and side views; the built-in back looks worse. |
| 6 | Do you have (or could you take) photos of the same person from behind, or would you make back views yourself with ChatGPT, Gemini or ComfyUI? | Yes: the "Back photo" source is the most photo-real option and moves up in priority, with the naming guide in the user guide. No: MV-Adapter or Qwen carry the quality. |
| 7 | Is 15-60 minutes per person and about 30 GB of download acceptable for a more photo-like experimental back? | Yes plus enough RAM: schedule the M7 spike on your PC. No: stop at M6. |
| 8 | Do you already run ComfyUI yourself? | Yes: the cheap expert `comfyui` source can come earlier (about 2 days, independent of Qwen packaging). |
| 9 | Is any of this for commercial use? | SHARP's weights are non-commercial, so the whole mode is personal/research use whatever back source you pick. Answering "yes" would need a different front model (out of scope). |

### Critical Files for Implementation
- /home/user/VR-Photo-Studio/src/vrphoto/pipeline/run.py (plus new /home/user/VR-Photo-Studio/src/vrphoto/pipeline/sharp360.py)
- /home/user/VR-Photo-Studio/src/vrphoto/geometry/splat.py (plus new splat_fusion.py, back_mould.py, views_lift.py, register.py and splat_render.py in /home/user/VR-Photo-Studio/src/vrphoto/geometry/)
- /home/user/VR-Photo-Studio/src/vrphoto/addons/capture.py (plus /home/user/VR-Photo-Studio/tests/addons/fakes/fake_brush.py)
- /home/user/VR-Photo-Studio/src/vrphoto/gui/options_meta.py (plus /home/user/VR-Photo-Studio/src/vrphoto/pipeline/jobs.py, /home/user/VR-Photo-Studio/src/vrphoto/gui/main_window.py, /home/user/VR-Photo-Studio/src/vrphoto/gui/panels/options.py)
- /home/user/VR-Photo-Studio/src/vrphoto/addons/manifests/sharp.toml and /home/user/VR-Photo-Studio/src/vrphoto/addons/workers/sharp_worker.py (templates for the new mvadapter.toml and mvadapter_worker.py; also /home/user/VR-Photo-Studio/docs/CONTRACTS.md)