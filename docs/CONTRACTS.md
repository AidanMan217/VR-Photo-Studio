# VR Photo Studio - module contracts (v1)

This document is the **frozen interface** between modules. Each module owner implements
exactly these public names and signatures; extra private helpers are fine. Shared
dataclasses live in `src/vrphoto/core/types.py` (read it - it is the source of truth).
If a contract must change, change it here AND in every caller in the same commit.

## 0. Conventions (non-negotiable)

* Images: `numpy.ndarray`, `HxWx3 uint8 RGB` (never BGR outside a cv2 call). Alpha/masks:
  `float32 HxW in [0,1]` unless named `*_mask` / typed `bool`.
* Source camera frame (OpenCV): x right, **y down**, z forward, metres.
  `u = fx*X/Z + cx`, `v = fy*Y/Z + cy`. Pixel centres are integer coordinates (pixel (0,0) is
  centred at u=0,v=0), consistent with `cv2.remap`.
* Exported meshes/splats/scenes: glTF frame, x right, **y up**, z toward viewer (camera looks
  down -z), metres. `geometry.camera.cv_to_gltf(P) = P * [1, -1, -1]`.
* Disparity/inverse depth: larger = nearer. `DepthResult.disparity` in [0,1] (1 = nearest).
* `Pose(R, t)`: new camera expressed in the source frame. `P_new = R.T @ (P_src - t)`.
  `Pose.translate(x,0,0)` with x>0 moves the camera RIGHT, so scene content shifts LEFT
  in the new image.
* Left eye sits at x = -B/2, right eye at x = +B/2.
* Every long-running public function accepts `progress: ProgressReporter | None = None` and
  `cancel: CancelToken | None = None` (keyword), calls `cancel.check()` regularly, never
  blocks the GUI thread itself (callers run it in a worker thread).
* Errors: raise subclasses of `core.errors.VRPhotoError` with a human-readable message.
  Never `print()`; log with `core.log.get_logger(__name__)`.
* No module may require a GPU, network or real model weights at import time. Heavy optional
  imports (onnxruntime, PySide6, fast_simplification) are imported lazily inside functions
  where practical.
* Python >= 3.11; code must run on Windows and Linux. Use `pathlib`, never hard-code `/`.

## 1. core (lead - done)

* `core.types`: `Intrinsics, Pose, DepthResult, DepthScale, Layer, LayeredImage, StereoParams,
  RenderResult, Mesh, SplatCloud, CancelToken, ProgressReporter`.
* `core.errors`: `VRPhotoError, Cancelled, ModelUnavailable, DownloadError, InferenceError,
  AddonError, AddonNotInstalled, EncodeError, InputError`.
* `core.config`: `Settings` (dataclass, `.models_dir .addons_dir .cache_dir .certs_dir .logs_dir
  .library_dir .outputs_dir`, `load()/save()`, `license_accepted(id)`), `get_settings()`,
  `set_settings()`, `data_root()` (honours `$VRPHOTO_HOME`), `resource_path(*parts)`, `is_frozen()`.
* `core.log`: `setup_logging()`, `get_logger(name)`.
* `core.imaging`: `load_image(path, with_alpha=False)`, `exif_hfov(path)`, `save_image(path, img,
  quality=92, xmp=None)`, `save_depth16`, `load_depth16`, `resize_long`, `limit_long`, `to_float`,
  `to_uint8`, `colorize_depth`, `gpano_xmp(full_w, full_h, crop_w, crop_h, left, top)`,
  `is_image`, `is_video`, `IMAGE_EXTS`, `VIDEO_EXTS`.
* `core.fill`: `push_pull_fill(img, known) -> same dtype` (algorithm F; re-exported by
  `geometry.projection.push_pull_fill`). Use this everywhere instead of re-implementing.

## 2. models (owner A1)

```python
# models/registry.py
@dataclass(frozen=True)
class ModelSpec:
    id: str                      # e.g. "da2-small"
    task: str                    # "depth" | "inpaint" | "segment" | "geometry"
    title: str
    repo: str                    # HF repo or "" for direct URL
    revision: str                # pinned commit sha or "main"
    files: tuple[str, ...]       # paths inside repo; first is the .onnx entry point
    url: str | None = None       # direct URL override (e.g. GitHub release); else HF resolve URL
    sha256: dict[str, str] = {}  # per-file; empty -> recorded on first download (warn)
    size_mb: float = 0
    license: str = ""
    noncommercial: bool = False
    dml_ok: bool = True          # False -> force CPU on DirectML (known crash)
    input_size: tuple[int, int] | None = None   # fixed (H, W) if static
    notes: str = ""
MODELS: dict[str, ModelSpec]
def get_spec(model_id: str) -> ModelSpec            # raises ModelUnavailable
def list_specs(task: str | None = None) -> list[ModelSpec]

# models/download.py
def model_dir(model_id: str) -> Path
def is_downloaded(model_id: str) -> bool
def ensure_model(model_id: str, *, progress=None, cancel=None, token: str | None = None) -> Path
    # returns path of the entry-point file. Env override VRPHOTO_MODEL_<ID_UPPER_UNDERSCORE>=<path>
    # (e.g. VRPHOTO_MODEL_DA2_SMALL) bypasses download (used by tests).
    # Resumable (.part + Range), sha256 verify when known, honours cancel, raises DownloadError.
def delete_model(model_id: str) -> None

# models/ort.py
class OrtModel:
    def __init__(self, path: str | Path, device: str | int = "auto", *, fp16: bool = False,
                 dml_ok: bool = True): ...
    provider: str                     # actual provider in use
    def input_names(self) -> list[str]
    def input_shapes(self) -> list[list[int | str | None]]
    def run(self, feeds: dict[str, np.ndarray], output_names: list[str] | None = None) -> dict[str, np.ndarray]
        # thread-safe (per-session lock); on provider failure retries once on CPU
def available_providers() -> list[str]
def load_model(model_id: str, *, device="auto", progress=None, cancel=None) -> OrtModel  # ensure_model + cache

# models/gpu.py
@dataclass
class Device: index: int; name: str; vram_mb: int | None; vendor: str  # "nvidia"|"amd"|"intel"|"other"|"cpu"
def list_devices() -> list[Device]       # DXGI adapters on Windows (+ nvidia-smi VRAM); [] elsewhere
def nvidia_vram_mb() -> int | None
```

Initial registry ids: `da2-small` (default), `da2-base` (NC), `da2-large` (NC), `moge2-vitl`,
`da3-mono-large` (experimental), `lama`, `migan`, `birefnet-lite`, `birefnet-portrait`,
`isnet-general`, `u2net`.

## 3. depth (owner A1)

```python
# depth/base.py
class DepthEstimator(Protocol):
    id: str
    def estimate(self, img: RGB, *, progress=None, cancel=None) -> DepthResult: ...
def get_estimator(name: str = "auto", device: str = "auto") -> DepthEstimator
    # names: "auto" (da2-small if downloadable/available else heuristic), "heuristic",
    # "da2-small", "da2-base", "da2-large", "moge2", "da3"
def list_estimators() -> list[tuple[str, str]]   # (id, title)

# depth/post.py
def normalize(raw: FloatMap, lo_pct=2.0, hi_pct=98.0) -> FloatMap            # -> [0,1]
def guided_filter(src: FloatMap, guide: RGB | FloatMap, r: int = 8, eps: float = 1e-3) -> FloatMap
def refine(disparity: FloatMap, img: RGB, strength: float = 1.0) -> FloatMap  # edge-aware smoothing
def temporal_align(prev: FloatMap | None, cur: FloatMap, ema: float = 0.6) -> FloatMap
def depth_edges(inv_z: FloatMap, thresh: float) -> np.ndarray  # bool

# depth/pano.py (owner A4)
def estimate_equirect(eq: RGB, est: DepthEstimator, *, face: int = 518, overlap_deg: float = 10.0,
                      progress=None, cancel=None) -> DepthResult   # disparity map at eq resolution
```

## 4. segment / inpaint (owner A2)

```python
# segment/base.py
class Segmenter(Protocol):
    id: str
    def segment(self, img: RGB, *, rect: tuple[int,int,int,int] | None = None,
                progress=None, cancel=None) -> FloatMap      # alpha [0,1], 1 = subject
def get_segmenter(name: str = "auto", device="auto") -> Segmenter
    # "auto" (birefnet-lite if available, else grabcut), "birefnet-lite", "birefnet-portrait",
    # "isnet", "u2net", "grabcut"
# segment/matte.py
def clean_matte(alpha: FloatMap, *, threshold=0.5, feather_px=2, fill_holes=True,
                largest_only=True) -> FloatMap

# inpaint/base.py
class Inpainter(Protocol):
    id: str
    def inpaint(self, img: RGB, mask: np.ndarray, *, progress=None, cancel=None) -> RGB
        # mask: bool HxW, True = fill.  Output same size; unmasked pixels unchanged.
def get_inpainter(name: str = "auto", device="auto") -> Inpainter
    # "auto" (lama if available, else migan, else opencv), "lama", "migan", "opencv", "pushpull"
# inpaint/crops.py
def inpaint_by_crops(img, mask, fn: Callable[[RGB, bool-mask], RGB], *, size=512, margin=1.0,
                     feather=4) -> RGB   # per connected component crop -> resize -> fn -> paste
```

## 5. geometry (owners: A3 camera/layers/warp/stereo, A4 projection/ods, A5 mesh/billboard/splat/gltf)

```python
# geometry/camera.py
def unproject(K: Intrinsics, depth_z: FloatMap) -> np.ndarray          # HxWx3 source frame
def project(K: Intrinsics, P: np.ndarray) -> np.ndarray               # ...x2 (u,v); NaN if z<=1e-6
def cv_to_gltf(P: np.ndarray) -> np.ndarray
def gltf_to_cv(P: np.ndarray) -> np.ndarray
def estimate_intrinsics(img_shape, *, hfov_deg: float | None = None, exif_hfov: float | None = None,
                        depth: DepthResult | None = None, default_hfov=65.0) -> Intrinsics
    # priority: explicit hfov_deg > depth.intrinsics (MoGe) > exif > default

# geometry/layers.py
def build_layers(img: RGB, inv_z: FloatMap, K: Intrinsics, *, max_baseline_m: float,
                 inpainter: Inpainter | None, alpha: FloatMap | None = None,
                 mode: Literal["band", "cleanplate", "none"] = "band",
                 progress=None, cancel=None) -> LayeredImage
    # "none": single layer. "band": background band inpainted behind depth edges.
    # "cleanplate": subject (alpha) removed from background entirely (needs alpha).

# geometry/warp.py
def render_view(li: LayeredImage, pose: Pose, dst_K: Intrinsics | None = None, *, iters: int = 6) -> RenderResult
def fill_holes(rr: RenderResult, *, small_frac: float = 0.001) -> RGB
def render_filled(li, pose, dst_K=None) -> RGB         # render_view + fill_holes

# geometry/stereo.py
def auto_convergence(inv_z: FloatMap, pct: float = 35.0) -> float      # metres
def stereo_pair(li: LayeredImage, params: StereoParams, dst_K: Intrinsics | None = None) -> tuple[RGB, RGB]
def compose(left: RGB, right: RGB, layout: str) -> RGB
    # layouts: "sbs" (full width each), "half_sbs", "tb", "half_tb", "anaglyph" (Dubois red-cyan),
    # "anaglyph_gray", "cross" (R|L), "left", "right"
def wiggle_frames(li: LayeredImage, params: StereoParams, n: int = 8) -> list[RGB]

# geometry/projection.py
def equirect_dirs(out_w, out_h, lon_range=(-180.0, 180.0), lat_range=(-90.0, 90.0)) -> np.ndarray  # HxWx3 unit (cv frame)
def persp_to_equirect(img: RGB, K: Intrinsics, out_w: int, out_h: int, *, lon_range=(-90.0, 90.0),
                      yaw_deg=0.0, pitch_deg=0.0) -> tuple[RGB, np.ndarray]   # (image, valid bool mask)
def equirect_to_persp(eq: RGB, K: Intrinsics, *, yaw_deg=0.0, pitch_deg=0.0, lon_range=(-180.0,180.0)) -> RGB
def equirect_to_cubemap(eq: RGB, face: int) -> dict[str, RGB]          # keys: "+x","-x","+y","-y","+z","-z" (cv frame)
def cubemap_to_equirect(faces: dict[str, np.ndarray], out_w: int, out_h: int) -> np.ndarray
def push_pull_fill(img: np.ndarray, known: np.ndarray) -> np.ndarray   # fills ~known, any channels
def vr180_frame(left: RGB, right: RGB, K: Intrinsics, eye_px: int, *, size_mult: float = 1.0,
                fill: str | RGB = "blur", layout: str = "sbs") -> RGB
    # returns side-by-side (2*eye_px x eye_px) or tb equirect 180 image
def pano_from_photo(img: RGB, K: Intrinsics, out_w: int, *, fill: str = "blur") -> tuple[RGB, np.ndarray]  # mono 360 equirect

# geometry/ods.py
def ods_pair(eq: RGB, dist_m: FloatMap, ipd_m: float = 0.064, *, inpainter=None,
             pole_fade=(45.0, 75.0), progress=None, cancel=None) -> tuple[RGB, RGB]

# geometry/mesh.py
def depth_mesh(img: RGB, inv_z: FloatMap, K: Intrinsics, *, alpha: FloatMap | None = None,
               target_faces=300_000, cut_px=2.0, ref_baseline_m=0.1, name="layer") -> Mesh
def layered_meshes(li: LayeredImage, *, target_faces=300_000, cut_px=2.0, ref_baseline_m=0.1,
                   anchor_distance_m: float | None = None, eye_height_m: float = 1.5) -> list[Mesh]
    # glTF frame; image centre placed at (0, eye_height, -anchor) when anchor given
def decimate(mesh: Mesh, target_faces: int) -> Mesh

# geometry/billboard.py
def billboard(img: RGB, alpha: FloatMap, depth: DepthResult, K: Intrinsics, *, height_m=1.7,
              thickness=0.17, back: str = "diffuse") -> list[Mesh]   # feet at y=0, faces +z

# geometry/splat.py
def read_splats(path) -> SplatCloud            # .ply (3DGS), .splat
def write_splats(sc: SplatCloud, path) -> Path # by extension .ply | .splat
def decimate_splats(sc, max_n=750_000) -> SplatCloud
def crop_splats(sc, *, radius_pct=98.0, aabb=None) -> SplatCloud
def normalize_orientation(sc, up: np.ndarray | None = None) -> SplatCloud   # floor at y=0, y up
def splats_from_mesh(mesh: Mesh, *, scale_m: float | None = None) -> SplatCloud

# geometry/gltf.py
def write_glb(meshes: list[Mesh], path, *, texture_format="jpeg", max_texture=4096) -> Path
def read_glb_meshes(path) -> list[Mesh]       # minimal reader for tests / re-export
def write_obj(meshes, path) -> Path           # + .mtl + textures
```

## 6. video (owner A6)

```python
# video/paths.py
@dataclass
class CameraPath: poses: list[Pose]; fps: float
def make_path(kind: str, *, n_frames: int, amplitude_m: float = 0.05, fps: float = 30.0,
              loop: bool = True, keyframes: list[dict] | None = None, focus_m: float | None = None) -> CameraPath
    # kinds: "dolly", "circle", "swing", "zoom", "kenburns", "orbit", "custom"
PATH_KINDS: list[str]

# video/encode.py
PRESETS: dict[str, dict]      # "quest3_hevc", "quest2_hevc", "h264_compat", "preview"
def ffmpeg_exe() -> str
def probe_encoders() -> dict[str, bool]   # cached: libx264, libx265, hevc_nvenc, h264_nvenc, hevc_amf, hevc_qsv
class VideoWriter:   # context manager
    def __init__(self, path, width, height, fps, preset="h264_compat", *, audio_from=None,
                 encoder: str = "auto", bitrate: str | None = None): ...
    def write(self, rgb: RGB) -> None
    def close(self) -> Path

# video/decode.py
@dataclass
class VideoInfo: width: int; height: int; fps: float; n_frames: int; duration: float; has_audio: bool
def probe(path) -> VideoInfo
def iter_frames(path, *, max_long: int | None = None, start: float = 0.0, end: float | None = None) -> Iterator[RGB]

# video/spatial.py
def inject(path, *, coverage: int, stereo: str) -> Path      # coverage 180|360; stereo "mono"|"lr"|"tb"; in place
def read(path) -> dict                                       # {"spherical": bool, "stereo": ..., "coverage": ...}

# video/render.py
def render_path_frames(li: LayeredImage, path: CameraPath, *, out_K: Intrinsics | None = None,
                       stereo: StereoParams | None = None, layout: str = "mono",
                       overscan: float = 0.04, cancel=None) -> Iterator[RGB]
def render_vr180_frames(li, path, *, eye_px: int, size_mult=1.0, fill="blur", cancel=None) -> Iterator[RGB]
```

## 7. webxr (owners: A7 viewer + scene.py, A8 server/cert/net/quest)

```python
# webxr/scene.py
SCHEMA_ID = "vrphoto.scene/1"
def new_scene_id(title: str) -> str
def export_scene(manifest: dict, assets: dict[str, Path | bytes], *, library: Path | None = None,
                 thumbnail: RGB | None = None) -> Path      # writes library/<id>/scene.json + assets
def validate_manifest(manifest: dict) -> None              # raises InputError
def list_scenes(library: Path | None = None) -> list[dict] # [{id,title,mode,thumbnail,url,created}]
def delete_scene(scene_id: str, library: Path | None = None) -> None

# webxr/cert.py
def ensure_cert(cert_dir: Path, ips: list[str] | None = None) -> tuple[Path, Path]  # (cert.pem, key.pem)
# webxr/net.py
def lan_ips() -> list[str]
def primary_ip() -> str
def qr_png(text: str, box_size: int = 8) -> bytes
def firewall_status(port: int) -> str        # "ok" | "blocked" | "unknown"
def add_firewall_rule(ports: list[int]) -> bool   # elevates via ShellExecute runas on Windows
# webxr/server.py
class ViewerServer:
    def __init__(self, library: Path, *, https_port=8443, http_port=8080, viewer_dir: Path | None = None,
                 cert_dir: Path | None = None, bind="0.0.0.0"): ...
    def start(self) -> None; def stop(self) -> None; running: bool
    def urls(self) -> dict[str, str]        # {"https": "https://192.168.1.5:8443/", "local": "http://localhost:8080/"}
# webxr/quest.py
def find_adb() -> str | None
def adb_devices() -> list[str]
def adb_reverse(port: int, serial: str | None = None) -> bool
def adb_push(local: Path, remote_dir: str = "/sdcard/Movies", serial=None) -> bool
def mtp_copy(files: list[Path], subfolder: str = "Movies") -> bool       # Windows Shell COM via PowerShell
def open_url_on_quest(url: str, serial=None) -> bool                     # adb am start
```

Viewer HTTP API: `GET /` gallery; `GET /api/scenes` -> JSON list; `GET /scenes/<id>/...` assets
(Range supported); `GET /view.html?scene=<id>`.

## 8. Scene manifest (`webxr/viewer/scene.schema.json`)

```json
{"schema":"vrphoto.scene/1","id":"beach-01","title":"Beach","created":"2026-10-09T12:00:00Z",
 "thumbnail":"thumb.jpg","mode":"photo3d|stereo|sphere|object|splat","units":"m",
 "photo3d":{"meshes":["bg.glb","fg.glb"],"anchor_distance_m":2.0,"eye_height_m":1.5},
 "stereo":{"media":"x_LR.jpg","kind":"image|video","layout":"sbs|tb|mono","aspect":1.5,
           "screen_width_m":3.0,"distance_m":4.0},
 "sphere":{"media":"x_180_LR.jpg","kind":"image|video","coverage":180,"layout":"mono|sbs|tb",
           "depth":"depth.png","radius_m":20},
 "object":{"asset":"person.glb","format":"glb|ply|splat|spz","height_m":1.7,"pivot":"base",
           "placement":"hit-test|front-floor","distance_m":1.5},
 "splat":{"asset":"scene.splat","format":"ply|splat|spz","up":"y","scale":1.0},
 "env":{"prefer":"ar|vr","background":"passthrough|black|#rrggbb"},
 "notices":["Contains output of a non-commercial model"]}
```
Only the block matching `mode` is required.

## 9. addons (owner A9)

```python
# addons/protocol.py (lead - done)
# addons/manager.py
@dataclass
class AddonManifest: id, title, description, experimental: bool, license: str, noncommercial: bool,
    region_restricted: list[str], min_vram_gb: float, disk_gb: float, python: str, kind: "venv"|"binary",
    steps: list[dict], worker: str | None, tasks: list[str], hf_gated: bool, homepage: str
def list_addons(include_experimental=False) -> list[AddonManifest]
def get_manifest(addon_id) -> AddonManifest
def status(addon_id) -> dict      # {"installed": bool, "version": str|None, "path": str, "size_mb": float}
def install(addon_id, *, progress=None, cancel=None, hf_token=None) -> None
def uninstall(addon_id) -> None
def run(addon_id, task: str, inputs: dict, params: dict, out_dir: Path, *, progress=None,
        cancel=None, timeout: float | None = None) -> dict           # returns worker "result.outputs"
def find_uv() -> str
# addons/env.py
def clean_env(extra: dict | None = None) -> dict[str, str]
def popen_kwargs() -> dict       # creationflags / start_new_session etc.
# addons/capture.py
def capture_to_splat(inputs: list[Path], out_dir: Path, *, video_fps: float = 2.0, max_images=200,
                     steps: int = 15000, max_splats=750_000, progress=None, cancel=None) -> Path
```

## 10. pipeline / cli (owner A10), gui (A11)

* `pipeline/jobs.py` (lead - done): job dataclasses + `job_from_dict` / `job_to_dict`.
* `pipeline/context.py`: `RunContext(settings, estimator(), segmenter(), inpainter(), cache)`.
* `pipeline/run.py`: `run_job(spec, ctx=None, *, progress=None, cancel=None) -> JobResult`.
* `pipeline/naming.py`: `output_name(base, kind, ext) -> str` (Quest tokens), `unique_path(path)`.
* `pipeline/queue.py` (no Qt dependency):
  ```python
  @dataclass
  class JobRecord: job_id: str; spec: JobBase; status: str  # queued|running|done|failed|cancelled
      frac: float = 0.0; msg: str = ""; result: JobResult | None = None; error: str | None = None
      created: float; started: float | None; finished: float | None
  @dataclass
  class JobEvent: job_id: str; type: str   # queued|started|progress|finished|failed|cancelled
      frac: float = 0.0; msg: str = ""; result: JobResult | None = None; error: str | None = None
  class JobQueue:
      def __init__(self, runner: Callable = run_job, *, workers: int = 1, ctx: RunContext | None = None)
      def submit(self, spec: JobBase) -> str
      def cancel(self, job_id: str) -> None;  def cancel_all(self) -> None
      def subscribe(self, cb: Callable[[JobEvent], None]) -> Callable[[], None]   # returns unsubscribe
      def jobs(self) -> list[JobRecord];  def get(self, job_id) -> JobRecord
      def clear_finished(self) -> None;  def shutdown(self, wait: bool = True) -> None
  ```
  Callbacks are invoked from worker threads; the GUI marshals them to the Qt thread.
* `cli.py`: `main(argv) -> int`; subcommands `depth stereo parallax vr180 pano scene object capture
  video serve models addons selftest run` (`run --spec job.json`).
* `selftest.py`: `run_selftest(out: Path | None = None, *, models=False) -> dict` (JSON-able, `ok` key).
* `__main__.py`: no args -> GUI; any args -> CLI; `--selftest` handled by CLI.

## 11. Algorithms (summary; implement exactly)

**A. Depth -> inverse depth.** `d = normalize(raw)`; upsample cubic; guided filter (r≈0.006·W,
eps 1e-3, gray guide). `inv_z = DepthScale(near, far).to_inv_z(d)`, or `1/metric` if metric. Defaults:
photo3d/VR near 1.2 m far 15 m. FOV priority: user > MoGe > EXIF > 65°.

**B. Stereo.** `x_L - x_R = f·B/Z`, `B = ipd·strength`. Flat: convergence by principal-point shift
`cx_L = cx - f·B/(2·Zc)`, `cx_R = cx + f·B/(2·Zc)` so a point at Zc lands at the same pixel in both
views; parallax `u_L - u_R = f·B·(1/Z - 1/Zc)` (> 0 = crossed = in front of the screen); auto Zc = 35th percentile depth; if max |parallax| >
`max_parallax_pct`% of W, scale B down. VR: Zc = ∞, B = ipd (0.063), metric-ish depth.
Eyes: `render_view(li, Pose.translate(∓B/2, 0, 0), K_eye)`.

**C. Layer split.** `S = f·Bmax·inv_z` (px). Edges where 4-neighbour |ΔS| > 1 px. Band
`k = ceil(1.3·f·Bmax·(p99(inv_z) - p1(inv_z))) + 4`, ≤ 15% W. `d_bg = erode(inv_z, ellipse k)` then
box blur k/2. Hallucinate mask `H = (inv_z - d_bg > 1/(f·Bmax)) ∩ dilate(E, k)`, open/close 3 px.
BG layer: rgb = inpaint(img, H) (by crops), inv_z = where(H, min(d_bg, inv_z), inv_z), alpha 1.
Cleanplate: `H = dilate(alpha>0.5, 8 px)`. FG layer: original + inv_z, alpha 1 (or matte).

**D. Backward warp** per layer: `P = unproject(K, 1/inv_z)`; `P' = R.T(P - t)`; `q = project(dst_K, P')`
(NaN where z' < 0.05). Solve `q(p) = u` by fixed point `p ← p + (u - q(p))/s`, s = dst_fx/src_fx,
sampling q with bilinear remap; 6 iters from `p0 = u` and `p0 = 2u - q(u)`, keep converged result with
larger inv_z. Residual r → alpha `clip(1.5 - r, 0, 1)` (tears become transparent). Sample rgb/alpha
at p; composite back→front. Holes (alpha<0.5): Telea if area < 0.1% else push-pull. Video: 4% overscan.

**E. Equirect.** pixel (i,j) → `λ = λ0 + (i+.5)/W·(λ1-λ0)`, `φ = 90° - (j+.5)/H·180°`,
dir (cv frame) `d = (cosφ sinλ, -sinφ, cosφ cosλ)`. VR180 λ∈[-90°,90°]; the photo keeps its true
angular size (`size_mult` enlarges virtual FOV ≤ 150°). Outside: blur (push-pull + gaussian σ=2%W,
darken 40% toward edges), black, or add-on outpaint.

**F. Push-pull.** Pyramid of (rgb·m, m) area-downsampled to 1 px; on the way up fill unknown pixels
with upsampled num/den.

**G. ODS.** ρ = D·cosφ, y = D·sinφ, r = (ipd/2)·w(φ), `w = clip((75-|φ|)/30, 0, 1)`.
Eye e = +1 (left) / -1 (right): `λ' = λ + e·asin(min(r/ρ, .99))`, `φ' = atan2(y, sqrt(max(ρ²-r², ε)))`.
A point straight ahead appears shifted RIGHT in the left eye. Backward-warp with longitude wrap. Output TB (left on top).

**H. Mesh.** grid stride `s = ceil(sqrt(W·H/(target/2)))`; drop triangle if `f·ref_B·(max inv_z -
min inv_z) > cut_px` or any vertex alpha < .5; drop components < 50 faces; BG layer keeps triangles
in dilated H plus a stride×4 backdrop pushed back 1%; optional fast-simplification; UVs by projecting
vertices into the source camera; glTF frame.

**I. Billboard.** distance-transform inflation `infl = κ·mpp·sqrt(max(0, Rloc² - (Rloc - dt)²))`,
`Rloc = grey_dilate(dt)`; κ so median thickness = thickness·height; detail ±3 cm from high-passed depth;
front `Z0 - infl - hp`, back `Z0 + 0.9·infl`; shared silhouette vertices; back texture = push-pull of
an 8 px silhouette band, 10% darker (or mirrored front). Feet at y=0, centred, facing +z.

**J. Splat decimation.** prune opacity < .04 and huge splats; voxel moment-matching merge with
weights `opacity·∏scale`; opacity `1-∏(1-α)`; DC colour only. `.splat` record = 3×f32 pos,
3×f32 scale, RGBA u8, quat u8 (`q·128+128`, w first) = 32 bytes.

**K. Spatial metadata.** After encode: V1 GSpherical XML + V2 st3d/sv3d (equi bounds; 180 →
left = right = 0x40000000). JPEG gets GPano XMP. Filenames carry Quest tokens (`_LR`, `_180_LR`,
`_360_TB`, `_360`).


## 12. Changes after the v0.1 review round (authoritative where they differ from above)

* **core**: `core.fsutil.replace_with_retry(src, dst, *, attempts=20)` and
  `atomic_write_bytes(path, data)` - use for every final rename (Windows antivirus/indexer
  locks are retried). `core.imaging.load_image(path, with_alpha=False, *, max_px=None)`
  (reduced-scale JPEG decode), `to_srgb(im)` (embedded ICC profiles converted to sRGB on load),
  `hfov_from_35mm(f35, w, h)`; `exif_hfov` uses the 43.27 mm diagonal definition and returns the
  horizontal FOV of the displayed image. `setup_logging(..., filename=)`; the CLI logs to
  `logs/cli.log` (`CLI_LOG_FILE`), the GUI to `logs/vrphoto.log`. UI hint constants live in
  `core.errors` (`UI_ADDONS`, `UI_MODELS`, `UI_SETTINGS_MODELS`, `UI_LOG`, `MODE_VIDEO`).
* **FOV**: the 65 deg default applies to the LONG side of the image (portrait photos get 65 deg
  vertical); explicit and EXIF values are horizontal.
* **Stereo (algorithm B)**: `StereoParams.screen_width_m` (default 3.0). Uncrossed parallax is
  limited to `min(max_parallax_pct/100, ipd_m/screen_width_m) * width` so the background never
  requires eye divergence on the viewer's virtual screen; crossed parallax keeps the full
  percentage. The pipeline writes the same `screen_width_m` (`min(3.0, 2.2*aspect)`) into the
  scene manifest; the viewer caps the screen height at 2.2 m.
* **Mesh**: `layered_meshes(target_faces=...)` is the budget for the WHOLE scene (all layers);
  `geometry.mesh.anchor_scale(inv_z, K, anchor_distance_m, *, alpha=None)` returns the scale
  `layered_meshes` applies (the pipeline divides the layer `max_baseline` by it).
* **Video**: `encode.resolve_preset(preset, quest_model=None, *, output=None)`; `output="flat"`
  with "auto" -> `h264_compat`; hardware encoders use constant-quality rate control.
  `encode.ntsc_rate`, `encode.fps_arg`: NTSC rates are exact fractions end to end.
* **Depth post**: `temporal_align(prev, cur, ema=0.6, *, range_pull=0.1)` (symmetric trimmed fit +
  range pull so the depth range never drifts over a clip) and `scene_cut(prev, cur, thresh=0.5) -> bool`.
* **Capture**: `capture_to_splat(..., subject_height_m=0.0, camera_height_m=1.45)` gives the
  splat metric scale (median camera height above the floor = 1.45 m, or the subject height when
  given) and writes `<out_dir>/<name>.capture.json` `{scale, height_m, method, camera_height_m}`.
* **Add-ons**: on Windows every add-on child process is assigned to a kill-on-close Job object
  (`addons.env.assign_to_kill_job`), so COLMAP/Brush/workers die with the app.
* **Server**: serves only the viewer app and the scene library (no output-folder sharing; the
  files route is disabled by default); no CORS headers; the localhost HTTP listener is optional
  (HTTPS still starts if its port is busy). The firewall rule covers the HTTPS port only.
* **Scene library**: assets are hard-linked when on the same volume (else copied).
* **Scene manifest**: `object` block also accepts `up` ("y" | "-y" | "z"), `alt_assets`, and
  `scale` (metres per asset unit for an asset that is already real size: shown 1:1, `height_m`
  only describes it). `sphere` accepts `depth_near_m` / `depth_far_m`. The thumbnail is
  `thumb.jpg` and may be absent (`thumbnail: null` in `/api/scenes`). Asset names must be plain
  file names (no separators, no `..`).
* **SHARP 360 (0.2.0-beta, 0.2.0)** - details in section 13:
  * `pipeline.run`: object method `sharp360` (`ADDON_METHODS["sharp360"] = ("sharp",
    "image_to_splat")`, so a missing SHARP fails fast exactly like `sharp`); constants
    `BACK_SOURCES = ("auto", "shell", "trellis", "images")`, `BACK_RELIEFS = ("auto", "depth",
    "flat")`, `BACK_QUALITIES = ("fast", "balanced", "best")`,
    `SIZE_FROM = ("auto", "height", "photo")`, `OBJECT_ENV`; helpers `_free_app_gpu()` (closes
    the app's ONNX Runtime sessions before **every** add-on: at the start of each
    `_run_addon` and of `_mode_capture` (COLMAP / Brush, which do not go through
    `_run_addon`); later in-process Brush runs must call it too; failures only logged), `_sharp_inputs(run, item, photo) -> (image_path, params)` (shared by `sharp`
    and `sharp360`; plain `sharp` is unchanged byte for byte) and `_save_cutout(...)`.
  * `ObjectJob` fields `back_source="auto"`, `back_quality="balanced"`, `seam_deg=60.0`,
    `size_from="auto"`, `save_extras=False`; from 0.2.0 also `back_images=None`,
    `back_relief="auto"`, `color_match=True`, `back_tries=0`, `back_steps=0`; M4 / M5 add
    `back_prompt=""`, `polish="auto"` (`run.POLISH_MODES`), `polish_steps=5000` and
    `back_source="mvadapter"`; `thickness`, `back` and (from 0.2.0) `seed` also apply to
    `sharp360`.
  * GUI (0.2.0): `FieldMeta.enabled_if_all`, `addon_optional`, `file_filter`;
    `options_meta.field_applies(kind, name, values, *, meta=None)` (transitive: a field under
    a greyed-out sibling is greyed out) and `required_addon_groups(kind, values) -> ((ids,
    optional), ...)` (greyed-out fields ask for nothing; `required_addon_candidates` is the
    flattened wrapper); `util.ask_choice(parent, title, text, buttons, *, default=0,
    escape=None, informative=None) -> int | None`. `main_window._check_addons`: a missing
    required group shows the Yes/No dialog; a missing optional group (TRELLIS or MV-Adapter as
    the back source) offers "Open Add-ons" / "Continue (the built-in back is used)" / "Cancel";
    for 'Brush polish' On without Capture (`addon_choices={"on": ("capture",)}`, optional;
    Automatic asks for nothing) the text and the middle button say the polish is skipped
    (`options_meta.addon_fields(kind, values, ids) -> option names` tells the groups apart).
    'Describe the hidden side' (`back_prompt`, kind `text`) applies to back sources auto /
    mvadapter; 'Brush steps' (`polish_steps`) only when `polish` is not off; the panel checks
    the description's length at 'Create' (`sharp360._back_prompt`).
  * `geometry.splat`: `transform_splats(sc, R, t, s=1.0)` (proper rotations only; rotates
    quaternions and SH), `scale_splats(sc, k)`, `concat_splats(clouds)`, `subset_splats(sc, idx)`,
    `fit_plane_ransac(pts, tol, *, seed=0, iters=256) -> (normal, offset, inlier_frac)` (plane
    `normal . p == offset`, sign of the normal arbitrary).
  * `geometry.billboard.inflation_profile(mask) -> (prof, dt_eff)` (`billboard()` unchanged).
  * `addons.capture`: `brush_args(..., extra=())`, `write_colmap_text(dataset_dir, cams, points=None,
    colors=None) -> Path` (`cams` = `(name, K, R, t[, (w, h)])`, world-to-camera OpenCV),
    `train_splat(dataset, out_dir, *, steps, max_splats, extra_args=(), export_name="splat.ply",
    seed=0, progress=None, cancel=None, timeout=None, log_name="brush.log", log_sink=None, cwd=None,
    label=...) -> Path`, `brush_selftest(progress=None, cancel=None, *, work_dir=None) -> dict`
    (`vrphoto-cli addons brush-test`; result kept in `Settings.brush_test`), `brush_test_verdict`.
    The self-test stops at the first run that works: masks at `--sh-degree` 0, then 1; no masks
    at 0, then 1; then Brush's defaults (no `--sh-degree`, no masks). `sh0` is True / False
    (an SH-0 run failed and SH 1 worked with the same masks setting) / None (only the
    defaults run worked); `sh_option` says whether Brush accepted `--sh-degree`.

## 13. SHARP 360 (object method `sharp360`, 0.2.0)

Design and rationale: `docs/design/SHARP360.md`. 0.2.0-beta shipped the fusion core and the
built-in `shell` back; 0.2.0 adds the `trellis` and `images` back sources, the registration
library, view lifting, the CPU splat renderer, the previews and the seed ('Variation')
option. M4 / M5 (unreleased) add the `mvadapter` add-on and back source (orthographic AI views,
`GenView.kind="ortho"`), the view cache and the Brush polish. SHARP multi-image relief
(`back_relief="sharp"`), YuNet and Qwen/ComfyUI arrive in later milestones behind the same
interfaces.

**Frames and grids.**

* **C** - SHARP's camera frame: OpenCV axes, metres, camera at the origin. Continuous pixel
  coordinates `u = f*X/Z + W/2`, `v = f*Y/Z + H/2` (`f = outs["f_px"]`, `W x H` = SHARP's
  `width x height`); the pixel index is `floor(u)` (= `round(u - 0.5)` of `_subject_splats`).
  `geometry.camera` puts pixel centres at integers: subtract 0.5 before its `project`/`unproject`.
* **SHARP grid** `G x G` (anisotropic over the photo): `G = 768` when `N ~= L * 768**2`
  (L in 1..3, within 2 %), else `round(sqrt(N / 2))` with 2 layers (logged). Used only for the
  front z-buffer.
* **Mould grid** `Hm x Wm` (isotropic, `splat_fusion.MouldGrid`): long side 512 / 768 / 1024
  cells by `back_quality`; cell `k` covers photo pixels `[k*cell_px, (k+1)*cell_px)`, its centre
  ray is `((u - W/2)/f, (v - H/2)/f, 1)`. Distance transforms, normals, the back height field and
  back Gaussians live here.
* **S** - upright subject frame (`Canon`): columns `R = [x^ y^ z^]` in C with `y^ = -up` (down),
  `z^` = horizontal viewing direction, `x^ = y^ x z^`; `P_S = R.T @ (P_C - c)`, `c` on the
  orbit axis at the bounding-box centre height.
* **Azimuth** `phi`: direction of a viewing camera from the subject, 0 = the photo's camera,
  90 = "you walked to your right", 180 = behind. `forward(phi) = cos(phi) z^ - sin(phi) x^`,
  `right(phi) = cos(phi) x^ + sin(phi) z^`, `down = y^`; orthographic view with `m` metres per
  pixel and size `N`: `u = (P_S - a).right/m + N/2`, `v = (P_S - a).y^/m + N/2`, `a` = orbit axis.
* Depth tolerance `tau(z) = max(2 cm, 1 % z)` (`splat_fusion.depth_tol`).
* **Levelled frame** (`views_lift`): the photo's visible front forward-splatted with SHARP's
  front depth into an upright orthographic `S x S` picture (`S = 512`, reference 1024) with
  `m_in` metres per pixel, no camera pitch and no perspective. A point `P = (x, y, z)` in S
  lands in a view at azimuth `phi` at `u = (P . w(phi) - o) / m + S/2`, `v = (y - y0) / m + S/2`
  with `w(phi) = (cos phi, 0, sin phi)`, `o` the view's horizontal offset (`-x0` for the back).
  Every registered view (`GenView.meta["registered"]`) is in this frame. Pixel `(i, j)` spans
  `u in [j, j+1)`; OpenCV samplers get `u - 0.5`.
* **Renderer pixels** (`splat_render`): pixel `(row i, col j)` is sampled at its centre
  `(j + 0.5, i + 0.5)`, so `cx = W/2` is the middle (as frame C; `geometry.camera` differs).
  `orbit_camera` follows the azimuth contract above; perspective `fov_deg` is horizontal.
* **Back / side photo names** (`images` source, `views_lift.views_from_folder`): the chosen
  'Back photo' is phi 180 unless its own name says otherwise; pictures in the same folder are
  used when their whole stem (any case; `_`, `-`, a space or nothing between words) is
  `back` / `behind` / `rear` (180), `right` (90 - "you walked to your right", you see the
  subject's left side), `left` (270), `back_right` (135), `back_left` (225), `front_right`
  (45), `front_left` (315) or `azNNN` (NNN degrees). Other names are ignored. One picture per
  azimuth (the chosen file first, then sorted names). A `views.json` (`{"views": [{"file",
  "phi_deg" or "azimuth_deg"}]}`) in the folder overrides the azimuths of the files it lists.

**Interfaces (all numpy / OpenCV, CPU, `progress=`/`cancel=` on the long ones).**

```python
# geometry/splat_fusion.py
@dataclass(frozen=True) class SharpCam: f, width, height, n_raw=0, grid=768, layers=2
    from_outputs(outs, params, *, n_raw, shape) -> SharpCam   # f_px, else params["hfov_deg"] (65)
    project(P) -> (u, v, z); rays(u, v) -> ...x3 (z = 1); sharp_cell(u, v) -> flat cell | -1
@dataclass(frozen=True) class MouldGrid: Wm, Hm, cell_px
    for_photo(width, height, long_side); centers(); cell_of(u, v); resize(img, *, interpolation=None)
@dataclass class FrontModel: sc (C), cam, grid, matte, ring, D_f, D_near, D_far, D_ext, A_f,
    normals, cos_theta, dt, rgb, cell, theta_deg, rim, stats, src
@dataclass class Canon: R, c, up_C, up_source, height_s, width_s, T_prior, z_axis, pitch_deg,
    cut_top=False, cut_bottom=False, diag={}; to_S(P); to_C(P)
def sharp_front(sc_full, alpha, cam, rgb=None, *, mould_long=768, progress=None, cancel=None) -> FrontModel
def canonical_frame(front, sc_full, alpha, *, thickness=0.11) -> Canon
# geometry/back_mould.py
@dataclass class BackPart: sc (C, generated only), kind="mould"|"cloud", source, ok=True, reason="",
    normals=None, back_z=None, hull=None, views=None, diag=None,     # diag["warning"]: for the user
    spacing_m=None                                    # typical distance between its Gaussians (m)
@dataclass class HullIntervals: z_in, z_out, valid
def inflation_prior(front, canon, *, thickness) -> (Z_prior, T)
def poisson_heightfield(target, mask, z_rim, *, weights=None, lam_rim=10, lam_target=0.01, iters=150,
                        levels=3, anchor=None, lam_anchor=10, init=None, cancel=None) -> FloatMap
def gaussians_from_heightfield(z, mask, front, colors, *, stride=1, opacity=0.95, cancel=None)
    -> (SplatCloud, normals)
def rim_wall(front, z_rim, *, colors_front, colors_back=None, spacing_cells=1.0, cancel=None) -> SplatCloud
def mould_back(front, canon, target, *, colors, z_rim=None, clip=None, stride=1, weights=None,
               source="mould", hull=None, progress=None, cancel=None) -> BackPart
def shell_back(front, canon, *, thickness=0.11, back="diffuse", stride=1, clip=None,
               progress=None, cancel=None) -> BackPart          # the built-in back source
# geometry/splat_merge.py
def seam_weights(cos_theta, seam_deg=60) -> weights   # smoothstep(cos(seam+10), cos(seam-10), cos)
def merge_parts(front, part, *, seam_deg=60, seed=0, budget=750_000, cancel=None, progress=None)
    -> (cloud, is_front, diag)                          # part None / ok=False -> the front alone
def add_underlayer(sc, is_front, front, canon, part, *, thickness=None, cancel=None) -> (sc, is_front)
def feather_colors(sc, is_front, *, radius_m=0.03, cancel=None) -> SplatCloud
def voxel_components(P, voxel, *, cancel=None) -> labels   # 26-connected, -1 for non-finite rows
def remove_floaters(sc, is_front, front, canon, *, hull=None, seam_deg=60, spacing_m=None, cancel=None,
                    progress=None) -> (sc, is_front, diag)   # voxel = max(1.5 cm, 1.5 spacing_m)
def to_life_size(sc, *, mode, height_m, cut=False) -> (sc, height, box, used_mode)   # glTF frame
# geometry/register.py (numpy Nelder-Mead / Umeyama / TPS; no scipy)
def nelder_mead(f, x0, step, *, max_iter=400, tol=1e-7, cancel=None) -> (x, fx)   # non-finite = +inf
def umeyama(src, dst, *, with_scale=True, weights=None) -> (s, R, t)    # dst ~ s R src + t, det R = +1
def point_zbuffer(u, v, z, shape, *, radius=None) -> (zbuf, idx)         # nearest z per pixel, ties -> lowest idx
def silhouette_iou(a, b) -> float                                        # two empty masks -> 0.0
def contour_points(mask, n=64) -> (n, 2)          # by arc length from the topmost point, clockwise on screen
def chamfer_similarity_fit(src_mask, dst_mask, *, init=None, allow_roll=False, max_roll_deg=5.0, cancel=None)
    -> (params {s, tx, ty, roll_deg, cx, cy}, cost)                      # symmetric chamfer, (cx, cy) = pivot
def chamfer_perspective_fit(src_mask, dst_mask, *, max_keystone=0.5, trunc_fracs=(0.10, 0.04), cancel=None)
    -> (H 3x3, cost, cost_identity)   # masks in ONE frame; camera-turn homography (similarity, aspect,
                                      # keystone; no shear); truncated chamfer stages; never worse than I
def outline_gap(a, b, pct=98.0) -> float   # px: pct percentile of boundary-to-boundary distance, worse side
def warp_similarity(img, params, out_shape, *, interpolation=cv2.INTER_LINEAR, border_value=0) -> img
@dataclass class TPS: ctrl, w, a, center, scale; __call__(pts); max_displacement(shape, *, mask=None)
def tps_fit(src_pts, dst_pts, *, lam=1e-3) -> TPS                        # K = r^2 log r
def tps_warp(img, tps, out_shape, *, interpolation=cv2.INTER_LINEAR, border_value=0) -> img
def projective_icp(P, front, *, iters=8, trim=0.3, max_rot_deg=10.0, metric="plane", cancel=None)
    -> (s, R, t, diag)
UP_AXES: dict[str, tuple]; up_rotation(axis) -> R; yaw_rotation(yaw_deg) -> R
    # a cloud's rotation into C is canon.R @ yaw_rotation(yaw) @ up_rotation(up)
def register_cloud_to_front(sc, front, canon, *, progress=None, cancel=None) -> (sc_C | None, diag)
    # diag (JSON-ready): ok, reason, iou, resid_m, scale (cloud units -> m), s0, up, yaw_deg,
    # tilt_deg, transform {s, R, t}, candidates, icp, n_in, n_opaque, seconds;
    # gates IoU >= 0.70, median residual <= 5 cm, s in [0.5, 2] s0
def color_affine_fit(src, dst, *, lam=0.1, iters=5) -> (A, b, ok)        # sRGB pairs, fit in linear RGB, clamped
def apply_color_affine(rgb, A, b) -> rgb
def reinhard_lab(img, mask, ref, ref_mask, *, strength=0.5) -> img       # statistics from the masks only
# geometry/views_lift.py (view-based backs; nominal views now, exact "ortho" views in M4)
VIEW_KINDS = ("ortho", "nominal"); RELIEFS = ("auto", "depth", "flat"); VIEW_NAMES; PHI_NAMES; VIEWS_JSON
@dataclass class GenView: name, phi_deg, rgb (uint8 HxWx3), alpha (float32 HxW), kind="nominal", m=None,
    depth=None, meta={}                                   # phi wrapped into [0, 360); InputError if malformed
@dataclass class OrthoInput: rgba (S), rgba_ref (ref), alpha0 (bool S x S), m_in, size, ref, center_S, meta;
    m_ref (property)
def levelled_ortho_input(rgb, alpha, front, canon, *, size=512, ref=1024, progress=None, cancel=None)
    -> OrthoInput                                          # design 3.4; also TRELLIS's input (rgba_ref)
def views_from_folder(back_path) -> [(phi_deg, Path)]      # the naming rule above; InputError if missing
def load_views(entries, *, max_long=1024, dropped=None) -> [GenView]   # a cut-out PNG keeps its alpha;
    # an unreadable picture is skipped with a plain sentence in `dropped` (InputError when none can be read)
def segment_views(views, segmenter, *, cancel=None, dropped=None) -> [GenView]
    # segmenter (None: skipped) else a flat-background key; dropped (a sentence in `dropped`) when the
    # matte is empty ("no subject"), too small ("ortho": < 2 % cover; "nominal": < 64 px across or
    # < 400 px area - "take it from closer or crop it") or covers > 97 %
def hull_intervals(front, canon, ortho, views, *, rule="vote", samples=96, cancel=None) -> HullIntervals
def view_based_back(front, canon, ortho, views, *, depth_fn=None, relief="auto", color_match=True,
                    stride=1, seed=0, fill_holes=True, source="images", progress=None, cancel=None)
    -> BackPart   # kind "mould"; views = the registered (and rejected) views; ok=False + plain reason
                  # when there is no back view, its outline does not match (IoU < 0.75 after the
                  # similarity + perspective fit) or it shows a different pose (outline_gap, 98th
                  # percentile, > 6 % of the subject's size after that fit)
def views_from_cameras(cameras_json_path_or_dict, *, base_dir=None) -> [GenView]   # M4: kind "ortho", by phi;
    # phi = phi_deg, else from c2w relative to the azimuth-0 view (c2w_opencv / c2w_opengl; world axes not
    # assumed), else azimuth_deg; m None (= the levelled input's framing, m_in * S / width); InputError for a
    # non-orthographic set, a view tilted > 2 deg, a picture or camera file that cannot be read
def save_views_folder(views, folder, *, ortho) -> Path   # M4: editable "images"-style folder (see below)
def qc_cost(diag | BackPart | None) -> float             # V11 cost of a view-based back; inf when unusable
def choose_view_set(results) -> int | None               # index of the lowest qc_cost (first on a tie)
def contact_sheet(ortho, views, *, tile=256, labels=True) -> uint8 RGB   # the _views.png preview
# geometry/splat_render.py (CPU EWA splatting, numpy only; off the critical path)
@dataclass class SplatCamera: R, center, width, height, kind="persp"|"ortho", fx, fy, m, cx=None, cy=None
    pinhole(f, width, height, *, R=None, center=None); forward; to_camera(P); project(P) -> (u, v, z)
def orbit_camera(target, phi_deg, *, up=(0,1,0), front=(0,0,1), distance, width, height, kind="ortho",
                 m=None, fov_deg=None, elevation_deg=0.0) -> SplatCamera
@dataclass class SplatRender: rgb (uint8), alpha, depth, depth_median (float32), diag
def prepare_splats(sc, *, cancel=None) -> PreparedSplats   # camera-independent part, shared by views
def render_splats(sc | prepared, cam, *, background=(0,0,0), max_radius_px=64, band_rows=None,
                  progress=None, cancel=None) -> SplatRender   # identical images for any band size
def turntable_sheet(sc, *, n=8, size=384, cols=4, background=(32,32,32), labels=True, progress=None,
                    cancel=None) -> uint8 RGB   # upright glTF-frame cloud, phi = 0, 360/n, ...
# pipeline/sharp360.py
QUALITY: dict[str, Quality(mould_long, stride, auto, tries=1, trellis_steps=(12, 12), mv_steps=30,
                           mv_guidance=3.0, relief="depth", fill_holes=True, preview_px=384)]
    # fast: 512, 2, (shell,), relief "flat", no hole fill, 256 px, mv 20 steps | balanced: 768, 1,
    # (mvadapter, trellis, shell) | best: 1024, 1, (mvadapter, trellis, shell), tries 2, steps 25/25,
    # mv 50 steps, 512 px
Opts(..., seed, tries=1, steps=0, color_match=True, relief="auto", back_images=None, back_prompt="",
     polish="auto", polish_steps=5000)
SOURCES = {"shell": _src_shell, "trellis": _src_trellis, "images": _src_images,
           "mvadapter": _src_mvadapter}                                          # fn(ctx, rep) -> BackPart
SOURCE_NAMES / SOURCE_LABELS / SOURCE_ADDONS = {"trellis": "trellis", "mvadapter": "mvadapter"};
GENERATED_VIEW_SOURCES = {"mvadapter"}; MAX_TRIES = 4; MAX_STEPS = 100; MAX_SEED = 2**31 - 1;
PREVIEW_VIEWS = 8; MAX_PROMPT_CHARS = 300; PROMPT_SOURCES = ("auto", "mvadapter");
POLISH_STEPS_RANGE = (1000, 30000); MV_DEFAULT_PROMPT = "a person, full body, high quality, realistic
photo"; MV_STEPS_RANGE = (5, 100)
Ctx(run, item, photo, alpha, sc_full, front, canon, opts, previews={}, levelled=None, polish=None)
    # previews["views"] = (OrthoInput, [GenView]), previews["views_source"]; levelled: the shared
    # OrthoInput; polish: (runs, note) of the Brush polish
def mv_settings(opts) -> (steps, guidance, prompt)   # 'AI steps' (clamped to 5-100) else Quality.mv_steps
def resolve_options(spec, *, installed: Callable[[str], bool]) -> Opts   # InputError before any work;
    # thickness clamped to THICKNESS_RANGE = (0.06, 0.18) (Opts.thickness_asked; the job warns)
def run_sources(ctx, rep, attempts) -> BackPart | None                   # the fallback chain
def mode_sharp360(run, item, photo, alpha, spec, *, placement, height, rep, opts=None) -> None
# pipeline/view_cache.py (M4; design 3.5.3 "Cache")
SUBDIR = "backviews"; DEFAULT_MAX_BYTES = 2_000_000_000; CACHE_VERSION = 1; ENTRY_FILE = "entry.json"
def make_key(input_digest, *, source, params, worker) -> str   # 40 hex: sha256("v1|digest|source|sorted
                                                                # params JSON|worker")
def worker_version(manifest) -> str             # "<manifest version>+<12 hex of the worker script's sha256>"
class ViewCache(root, max_bytes=DEFAULT_MAX_BYTES):
    get(key) -> Path | None                      # complete entry folder; refreshes its LRU time
    put(key, files, *, meta=None) -> Path | None # copies (flat, unique names); atomic; prunes to the cap
    discard(key) -> bool                         # delete a damaged entry (made again next time)
    salvage(folder, keys: {seed: key}) -> {seed: Path}   # finished sets of an interrupted worker run
    prune(max_bytes=None) -> int; clear(); size_bytes(); len(); key in cache
def default_root(settings=None) -> Path          # <settings.cache_dir>/backviews
def get(key, *, root=None); put(key, files, *, root=None, meta=None); salvage(folder, keys, *, root=None)
# pipeline/sharp360_polish.py (M5; design 3.11)
POLISH_MODES = run.POLISH_MODES; DEFAULT_STEPS = 5000; TELEPHOTO_DIST_M = 40.0; ANCHORS; ANCHORS_BEST;
CAP_RANGE = (300_000, 2_500_000); MIN_SPLATS = 1000; MAX_BOX_GROWTH = 1.5; MIN_KEPT_FRAC = 0.1
def polish_allowed(opts, part, settings, *, quality, brush_installed=None) -> (bool, note)
def polish_with_brush(run, item, merged, is_front, front, canon, part, *, steps, seed, quality,
                      progress=None, cancel=None, seam_deg=60.0, photo_rgb=None, alpha=None,
                      frame_center=None, brush_test=None) -> (cloud, is_front, diag)   # only Cancelled raises
def build_frames(merged, is_front, front, canon, part, *, quality="best", seam_deg=60.0, masks=True,
                 photo_rgb=None, alpha=None, frame_center=None, progress=None, cancel=None) -> ([Frame], info)
def write_dataset(folder, frames, init, *, points=20000, seed=0) -> dict
def check_polished(ply, init) -> (cloud | None, info)
def front_copies(trained, front_rows, back_rows, *, search_m=COPY_SEARCH_M, cancel=None) -> bool mask
    # trained Gaussians whose nearest initial row (voxel-hash search, 2.5 mm / 8 mm / 2 cm) is an
    # exact front row within 2 cm rather than an unpolished back row: Brush's copies of the front
def telephoto_camera(canon, phi_deg, m, size, offset_m, y0, *, distance=40.0) -> (K, R, t)
def levelled_center(front, canon) -> (x0, y0)
```

**Options (`resolve_options`, all before any work).** `back_source` in `run.BACK_SOURCES`,
`back_quality`, `size_from`, `back` and `back_relief` in their `run.*` tuples; `seam_deg`
35-80; `seed` 0..2^31-1, `back_tries` 0..4, `back_steps` 0..100 (whole numbers; strings and
whole floats from job files are accepted); `back_tries=0` = the quality's `tries`,
`back_steps=0` = each source's own default (TRELLIS: the quality's `trellis_steps`). With
`back_source="images"`, `back_images` must name an existing picture (or a folder of views)
whose views include one from behind, with a readable header; it is ignored for the other
sources. `auto` = the quality's `Quality.auto` sources whose add-on is installed (`balanced` / `best`:
MV-Adapter, then TRELLIS; `fast`: only the shell), never `images`; an explicit source is tried
even when its add-on is missing (warning, then the shell). `back_prompt` ('Describe the hidden
side') is tidied (white space collapsed) and at most 300 characters when `back_source` is
`auto` or `mvadapter` (`PROMPT_SOURCES`; the GUI greys it out for the others and the job
ignores it there); `polish` is one of `run.POLISH_MODES` (auto | off | on) and `polish_steps`
a whole number 1000-30000. `seed` ('Variation') is the AI sources' seed (attempt `i` uses
`seed + i`) and the seam dither's; the photographed front never depends on it.

**Front guarantee.** A SHARP Gaussian that faces the photo camera at less than `seam_deg - 10`
or lies in the protected outline band (`FrontModel.rim`) is output bit-identical (before the
export transform) and is never recoloured or removed. The only exceptions: "card" Gaussians
slid along their own ray (sizes rescaled, so they look the same from the photo's camera),
removal of the hidden layer / background in `sharp_front`, the opacity < 0.04 prune of the final
decimation, and a front that alone exceeds 80 % of the 750k budget (decimated, logged). The
export's box crop never removes front rows.

**Clean-up and merge details (authoritative where they differ from the design).**

* `sharp_front`: a SHARP cell is "background seen through a gap" when it lies more than
  `max(3 tau, 15 % of the subject's size)` behind its 5x5 median (size = longer side of the
  matte's bounding box at the median front depth; the design's fixed 25 cm floor let a wall
  close behind a small subject through). The fallback to the plain matte filter happens below
  `max(500, 25 % of the SHARP-grid cells inside the matte)` kept Gaussians (1 % of SHARP's
  two-layer whole-photo cloud made small full-body figures fall back). A collapsed card gets
  the covariance `M Sigma M`, `M = k (I - r r^T) + c r r^T` (`k = D_near / z`, `r` the unit
  ray, `c` so the extent along the ray is `1.5 k s2`), re-diagonalised: the photo-view
  footprint is kept for cards tilted off their ray too (for a card on its ray this is the
  design's `s1' = 1.5 s2'`, `s2, s3` times `k`).
* `merge_parts`: for a `"mould"` part, generated Gaussians on the photographed surface whose
  centre cell is in the protected outline band are not dithered (they are the rim wall and
  outermost back cells closing the side; depth-map normals make that band look
  camera-facing). `"cloud"` parts are dithered everywhere.
* `remove_floaters`: the voxel edge is `max(1.5 cm, 1.5 * spacing_m)` with `spacing_m` from the
  back part (`shell_back` / `mould_back`: `stride` mould cells at the back's median depth), so a
  strided ("fast") back of a subject several metres away is not removed as isolated Gaussians;
  "near the largest component" is `max(5 cm, 1 voxel)`.

**Back sources (0.2.0; authoritative where they differ from the design).**

* `trellis` (design 3.5.5). Input: `levelled_ortho_input(...).rgba_ref` (1024 px) saved as
  `trellis_input.png` (the plain cut-out, at most 1024 px, when it cannot be made; logged).
  `Opts.tries` attempts, each one `_run_addon("trellis", "image_to_3d", {"image"},
  {"seed": seed + i, "ss_steps", "slat_steps", "glb": False, "ply": True, "name": "back<i>"})`
  -> `read_splats` -> `register_cloud_to_front`; the attempt with the best score
  `IoU - 0.5 min(1, resid / 10 cm)` among those passing the gates is kept. An `AddonError`
  ends the attempts (an earlier good attempt is kept, with a warning); when no attempt
  produced a cloud the first error is re-raised; when shapes were made but none lined up, the
  reason counts the shapes made and names each attempt that failed ("...; attempt 2 failed
  (CUDA out of memory)"). Registration: orientation search on 6k
  opaque centres, refinement on 10k (three candidates at least 20 deg apart), projective ICP
  with point-to-plane steps and a shrinking pairing window, kept only when it does not raise
  the refinement cost by more than 0.05; the gates use all 20k. Colour: a per-channel gain in
  linear RGB first, then `color_affine_fit` on the gain-corrected pairs (ridge
  `lam * n * 0.02^2`, Huber scale `median|r| / 1.538`), pairs only on reliable mould cells (on
  the photographed surface, facing the camera within `seam - 10`, at least max(3 cells, 1 %)
  inside the outline, the smoothest 30 % of the photo; at least 64), kept only when it lowers
  the median colour error. `back_z` = the largest z of opaque (>= 0.3) Gaussians behind the
  photographed surface per mould cell, push-pull filled across the matte. TRELLIS's
  non-commercial notice is added only when its back is used. `BackPart.diag`: the
  registration diag plus `tries`, `chosen`, `seed`, `steps`, `input`, `color`, `back_cells`;
  `run_sources` records an unusable part's diag in debug.json.
* `images` (design 3.5.4, V1-V11): `views_from_folder(back_images)` -> `load_views` (long side
  <= 1024) -> `segment_views(run.ctx.segmenter(common.segment_model, common.device))` ->
  `levelled_ortho_input(size=512, ref=1024)` -> `view_based_back(depth_fn, relief,
  color_match, stride, seed, fill_holes=Quality.fill_holes, source="images")`. Relief: `flat`
  -> none; `auto` -> none with `fast` (design 1.3), else like `depth`: the app's depth model
  (`common.depth_model`) on the locked back view, larger disparity = nearer the back camera;
  the heuristic estimator counts as no model (a smooth back; with `depth` a note says so).
  The depth model's non-commercial notice is added when its relief shapes the result, the
  segmenter's when it cut out a view. Never chosen by `auto`; a back photo that does not fit
  (outline IoU < 0.75, or a different pose) is one warning and the shell; so is one that
  cannot be read or shows no subject, and the warning names the back photo's own problem.
  A side photo that cannot be read or shows no subject is named in the warning next to the
  back's notes. `BackPart.diag`: `back` (similarity, `perspective` matrix, `iou_similarity`,
  `iou` after the perspective stage, `iou_locked`, `pose_frac`, `tps_max_frac`), `views`,
  `orbit` (with `pairs`), `hull`, `relief`, `colours`, `holes`, `qc`, `files` (azimuth ->
  file name of the pictures in use), `dropped`. Details that differ from the design:
  * Back registration (V2/V3): after the chamfer similarity a **perspective stage**
    (`chamfer_perspective_fit`: a camera-turn homography - similarity, aspect and the two
    keystone terms, no shear - on a chamfer truncated at 10 % then 4 % of the size) takes
    out the keystone of a phone held above the subject's middle (camera 1.6 m high, 2 m away:
    "a different pose, 9 %" before); the similarity, perspective and TPS are one resampling.
    The **pose gate** is not the TPS displacement: the lock pairs each of the 64 outline
    points with the nearest point of the other outline within +-4 % of its length (both
    outlines first turned to their best common start - a raised hand level with the head
    made the two start on different features), and such a pairing barely moves for an arm in
    another place. The gate is the outline gap after the global fit (`outline_gap`, 98th
    percentile, both directions) > 6 % of the subject's long side. Measured on the synthetic
    person: the same pose up to 30 deg camera tilt 0.4-4.0 %, an arm 15 deg or more off
    6.3-20 %, 10 deg off 4-6 % (accepted: the lock bends it).
  * Side / quarter views (V2): the top / bottom row extent gives the scale only when both
    rows are the subject's own; a view whose matte touches its picture's top / bottom border,
    or a photo with `Canon.cut_top` / `cut_bottom`, keeps the other row as anchor and takes the
    scale that minimises the V5 front-edge row spread (a 321-step search; the row extent is
    an upper bound when only the view is cut, a lower bound when only the photo is; measured
    within 2 % for 0-40 % cut). No reliable row: not used.
  * Left / right naming (V5): judged on the 90th percentile of the row misfit; two views
    named as each other's mirror (90 / 270, 135 / 225, 45 / 315) are turned round both or
    neither (summed misfits, ratio 0.5); when only one alone would turn, neither does and a
    note says the naming "could not be confirmed". At most one view per azimuth (the better
    fitting one), and views named for both sides but placed on one are not used. Outlines
    cannot tell the sides of a near-symmetric subject apart (no colour cue yet).
  * Colours are Reinhard LAB on the rim bands with lightness fully transferred and the colour
    channels only shifted (at most 6 LAB units), so a back of another colour keeps it; a side
    view's offset comes from its silhouette edge facing the photo camera; the orbit axis for
    nominal views is the per-row median of (photo front edge + side back edge) / 2; hull
    voting leaves out the back view; with left / right views and no depth model the back
    silhouette's inflation profile shapes each row; the 40 % carving limit also applies
    without side views (against the shell prior).
  * The visual hull of nominal side views cannot reach the design's "volume within 10 % of
    the truth": four views (90 / 270 / 135 / 315) keep the hull's corners between them, about
    1.3 x the body's depth along the photo's rays (median 3 cm beyond the true back); the
    tests bound that (median < 4 cm, < 1.45 x) and that the hull contains the body.
* `mvadapter` (M4; design 3.5.3 / 4.3; authoritative where it differs). Input:
  `levelled_ortho_input(size=512, ref=1024).rgba` (kept in `Ctx.levelled`, reused by a TRELLIS
  fallback) saved as `mvadapter_input.png`. One `_run_addon("mvadapter", "image_to_views",
  {"image"}, {"seeds": [missing seeds], "steps", "guidance", "prompt", "name": "views"})` for the
  seeds `seed + i` (`i < Opts.tries`) not found in the view cache; `steps` / `guidance` /
  `prompt` = `mv_settings(opts)` (20 / 30 / 50 steps by quality unless 'AI steps' is set; 3.0;
  'Describe the hidden side' or `MV_DEFAULT_PROMPT`). Each returned set is stored in the cache
  (key: `_mv_input_digest` - the photo, the matte, SHARP's camera, its cloud only coarsely (size,
  median depth to 1 cm: GPU inference need not be bit-identical), the levelled sizes and
  `_MV_LEVEL_VERSION`, NOT the levelled pixels, which differ slightly between the qualities' mould
  grids (outline IoU 0.999, 0.01 px, frame 0.02 deg apart on the synthetic person; the lateral
  check absorbs that) - source, `{steps, guidance, prompt, seed}`, worker version). The worker
  writes each set as soon as it is made (`cameras.json` last, atomically), so after a failure or a
  cancel the finished sets are salvaged into the cache (else read from the work folder) and, after
  a failure, used in the same run with a note ("MV-Adapter could not make attempt 2 (...); the
  views made before were used."). A cached set is read when it is looked up; one that cannot be
  read is discarded and generated again (one found damaged only while lifting is discarded and
  its attempt fails with "a saved AI view could not be read; it will be made again next time").
  Then per set
  `views_from_cameras(cameras.json)` -> `segment_views(the app's segmenter, else the grey key)`
  -> `view_based_back(source="mvadapter", depth_fn / relief, color_match, stride, seed,
  fill_holes)`; `choose_view_set` keeps the lowest QC cost. No usable set: `ok=False` with
  the set's reason ("the generated views do not match the photo's outline (...)"; several sets:
  "none of its N sets of views fit the photo: ..."), so the chain warns and falls back. An
  add-on error with no set at all is re-raised (the chain's "could not make the back (CUDA out
  of memory)"); with cached or finished sets they are used and a note names the failed attempt.
  `BackPart.diag`: the chosen set's view-based diag plus `tries` (per set: seed, cached, ok,
  qc, reason / error, seconds), `chosen`, `seed`, `steps`, `guidance`, `prompt`, `input`,
  `cache` (`hits`, `root`, `input`: the first 16 hex of the input's digest), `worker` (`offload`,
  `precision`, `worker_version`). The add-on
  notice (`run.addon_notice`; MV-Adapter is not non-commercial, so it adds nothing today) and the
  segmenter / depth-model notices are added when its back is used. Worker warnings reach the job
  as "<title>: <warning>". The contact sheet shows the chosen set (else the first lifted set,
  outlined as not used). **Deviation:** the cache key includes the steps, so another back quality
  reuses the views only when 'AI steps' is set (each quality has its own default steps,
  design 1.3); a seam / thickness / relief / colour-match / polish change always reuses them.
* **Orthographic sets** in `views_lift` (M4; authoritative where they differ from design V2-V5,
  V11): the lateral similarity is fitted for every set and applied when it moves the outline by
  more than 0.5 px and raises the IoU by at least 0.02 (the design fitted only below IoU 0.85,
  which left a 3 % scale or 3 px shift uncorrected); the 0.75 gate is applied after the fit. The
  fit ignores a uniform outline offset (keyed mattes sit 0.3-0.5 px outside the true outline,
  which a plain chamfer reads as a 1-1.5 % larger subject), and view 0's measured offset corrects
  the row edges of V5 / V7. The azimuth-sign net compares the 90th percentile of the row
  deviations (MAD cannot tell the hypotheses apart; consistent with the nominal path); when that
  says "reversed", the design's second test decides (`_sign_render_check`): SHARP's cleaned front
  rendered orthographically (`render_splats`, 192 px, in the levelled framing about each
  hypothesis' own axis) at each quarter view's azimuth (the +-90 views without quarter views) -
  not at +-40 as in the design - as named and mirrored, scored `IoU + 0.5 SSIM + 0.5 (1 - MAE /
  0.25)` against the colour-matched view (the colour term is an addition: `IoU + 0.5 SSIM` alone
  separated side views drawn 18 deg past their azimuth by only 0.03; with it 0.36). Every view -
  also those dropped by the invariant rows - is turned round with a warning only when the mirrored
  hypothesis scores 0.05 more per view; otherwise the named sides are kept with a warning ("...
  SHARP's front does not confirm that their left and right were swapped ..."). The render test
  runs only when the rows propose a flip (about 1 s per 150k front Gaussians). Colour: a per-channel linear gain, then the Huber affine, each
  kept only when it lowers the median error (no Reinhard fallback: views stay unchanged). A cut-off
  view or photo keeps the measured row ratio as a bound (a 9 % too large view filling its picture
  is still dropped); quarter views must line up with the solved axis; nominal pictures in an
  exact set are not used (warning); the row MAD gate is `max(3 cm, 2 % H)`. QC adds `1 - IoU`
  of view 0 (`lateral`), 0.5 per dropped view (`dropped`) and `sign` (0.5 for a set turned round,
  0.25 when the rows said "reversed" but the render test did not confirm it), so an unflipped
  try wins; the `yaw` / `face` terms are 0 (side views drawn turned by up to ~15 deg are not
  detected: a silent 2-8 cm axis error on the synthetic person). Hull voting (V6, both kinds of
  set): a left / right view whose mirror partner (`phi + 180`) is not used has 2 votes (the two
  carry the same orthographic silhouette; alone with 1 vote it stopped carving behind the body:
  hull median 8.7 instead of 3.9 cm beyond the true back). Dropped views keep their reason
  warnings, worded after the sign decision. `input_fill` / `ortho_half_extent` are stored, not
  used.
* **Saved views folder** (`save_views_folder`, written as `<base>_sharp360_views/` with
  `save_extras` after a generated source, also when its set was rejected): one PNG per azimuth
  named by the folder rule (`front.png` for 0, `back.png`, `right.png`, `left.png`,
  `front_right.png`, `front_left.png`, `azNNN.png`) composited over 0.5 grey, and `views.json`:
  `{"format": "vrphoto-views", "version": 1, "projection", "frame": {"size", "m", "center_S"},
  "views": [{"file", "phi_deg", "kind", "width", "height", "m", "center_S", "color_match",
  "used", "reason", "source_file"}]}`; of two views at one azimuth the one the back was built with
  is saved (the other is logged). Picked as the 'Back photo' (`back.png`), `views_from_folder`
  + `load_views` restore the exact cameras (a picture whose size changed becomes nominal).
* **Worker `image_to_views`** (`addons/workers/mvadapter_worker.py`, manifest `mvadapter`,
  task list `["image_to_views"]`): inputs `image` (square RGBA, 512 px, framed by the app; never
  re-cropped, composited on 0.5 grey; another square size is resized with a warning, a
  non-square picture is refused); params `steps` (30, clamped to 5-100), `guidance` (3.0,
  1-10), `seeds` (list, at most 8; else `seed`), `prompt` / `negative_prompt` (500 characters),
  `offload` (auto | model | sequential | none), `name`; outputs `{"sets": [{"seed", "views": [6
  PNG paths], "cameras": path}], "frame": "opencv", "warnings", "input", "offload", "precision",
  "steps", "guidance", "azimuths", "worker_version"}`, files `<out>/<name>_<seed>/view_0.png ...
  view_5.png` + `cameras.json`. **`cameras.json`:** `{"projection": "orthographic",
  "ortho_half_extent": 0.55, "distance": 1.8, "width": 512, "height": 512, "input_fill": 0.9,
  "input_fill_measured", "pixel_center": 0.5, "convention": "c2w_opencv", "frame": "opencv",
  "seed", "worker_version", "views": [{"file", "azimuth_deg", "phi_deg", "elevation_deg",
  "c2w"}]}` (plus plain-text `world` / `azimuth_convention` / `phi_convention` notes). Frame:
  the input view's OpenCV frame (x right in the input picture, y down, z away from the input
  camera, origin on the orbit axis at the centre of every view); view 0 is `R = I` at
  `(0, 0, -1.8)`; `phi_deg = atan2(C_x, -C_z)` of each camera centre (= MV-Adapter's azimuth,
  the app's convention: 90 = camera on the input's +x side). Views share the input's pixel grid,
  so metres per pixel = `m_in`. Runtime (deviations from design 4.3): `mvadapter.utils` is never
  imported (its package pulls in nvdiffrast, triton, trimesh, ...; the camera function is ported
  and `utils/geometry.py` loaded by path - no stubs); DDPM inside MV-Adapter's ShiftSNR wrapper;
  fp32 VAE with cast inputs / outputs; the NaN / black retry upcasts the UNet, the text and the
  condition encoders and is placed for fp32 at the free VRAM then (all on the GPU from 12 GB, model
  offload from 10 GB, else sequential - fp32 weights plus the reference pass's cached attention
  inputs do not fit 8 GB with model offload), never less memory-saving than before; the OOM ladder
  frees memory and reloads with sequential offload, then a plain error; seeds use a CPU
  generator; each set is written as soon as it is made (`cameras.json` last, atomically); the
  worker refuses an adapter state dict that fits the code below 99 %; the MV-Adapter source
  files it imports only have to exist (`mvadapter/__init__.py` is empty at the pinned commit) and
  an import failure is a reinstall hint. Manifest: MV-Adapter pinned at
  `4277e00...` (LICENSE Apache-2.0, so `noncommercial = false`), SD 2.1 base from the
  `Manojb/stable-diffusion-2-1-base` clone (the stabilityai repo was removed), adapter file
  0.71 GB, no peft (LoRA loading only); `disk_gb = 14` (the torch 2.8 + diffusers venv of about
  9.5 GB, as `sharp` / `sdxl360`, plus 3.5 GB of weights).
* **Brush polish** (M5; design 3.11; authoritative where it differs). `_polish_plan` (after the
  back source) asks `polish_allowed`: `off` never; `auto` only with `best`, a back with
  registered views (MV-Adapter, your back photo) and a passed self-test (`Settings.brush_test`);
  `on` whenever the back has views and Brush is installed (then a warning when the self-test
  has not passed). It is skipped quietly for `auto`; with `on` the job warns "Brush polish was
  skipped (<note>)." `_stitch` runs it after merge / under-layer / feathering and before the
  floater filter (design 4.2); `polish_with_brush` first drops Brush's copies of the front rows
  (`front_copies`: `init.ply` holds them too, and in the protected outline band, where a mould
  back's surface rows are kept, they survived next to the exact rows and drew the outline twice -
  462 of 836 front rows on the test scene, outline-band alpha error 0.057 -> 0.105), re-merges the
  rest as the back's kind ("mould": the rim wall survives), puts back the EXACT kept front rows
  (bit-identical, in order, first) and runs feathering and the floater filter itself, so the job's floater pass is skipped
  after a successful polish (`debug.json` `floaters` = the polish's). A failed polish is one
  warning ("Brush polish did not work (...); the unpolished result was kept.") and the normal
  floater pass. Details beyond the design: the self-test result is honoured (SH 1 when 0 failed,
  no `--sh-degree` when Brush did not accept it, no masked pictures when masks could not be mixed
  with RGBA); retries: an option Brush rejects is dropped, out of memory -> 768 px and the cap x
  0.6, then SH 1, then (no passed test) no masked pictures; at most 6 runs; a run may take
  900 s + 0.6 s per step; a timeout, a missing Brush or a cancel stops at once. Views within
  22.5 deg of the front are skipped, 22.5-67.5 deg are masked RGB, the rest RGBA; anchor pixels
  are trained as the front only on the photographed surface within the seam, at least 2 mould
  cells inside the outline and with render alpha >= 0.95; `best` adds yaw +-18 deg at pitch
  -8 deg and pitch +8 deg. The export is also rejected when fewer than 10 % as many back
  Gaussians survive the re-merge as the unpolished back had (a cloud moved out of the photo's
  frame). NaNs are read from the raw PLY. A failed Brush run's log is copied to
  `<base>_sharp360_polish.log`. `debug.json` `polish`: `asked`, `run`, `note`, `steps` and the
  polish diag (frames, dataset, attempts, check, `front_copies`, merge, floaters, ok / reason /
  warning).
* **Front untouched** (design 5.2), tested for the shell, TRELLIS and back-photo backs: seen
  from the photo's camera with `render_splats`, the fused cloud equals the cleaned front with
  the generated part showing only through it (`front + (1 - front alpha) * generated`; SHARP's
  front is not fully opaque) within a mean of 2/255 inside the matte eroded by 2 px. The
  reference is a copy of the cleaned front taken when `sharp_front` returns (before any back
  source runs), and the front model's own cloud must still equal it bit for bit; fewer than
  2 % of the opaque generated Gaussians may lie on the photographed surface (inside the
  matte, > 3 cells from its outline, nearer than the front depth + 1 cm) - a sparse
  TRELLIS front half coloured like the photo hardly shows in a render.

**Previews (design 1.5 / 3.13).** Written after the splat; a failure is one warning ("... could
not be made (...); the 3D result is not affected."), a cancel propagates; not written by the
plain-SHARP fallback.

* `<base>_sharp360_preview.jpg`: `turntable_sheet(final splat, n=PREVIEW_VIEWS=8,
  size=Quality.preview_px, cols=4)` - views at 0, 45, ..., 315 deg (0 = the photo's side, 90 =
  from the subject's left), 256 / 384 / 512 px each for fast / balanced / best.
* `<base>_sharp360_views.png`: `contact_sheet(ortho, views, tile=256)` whenever the `images`
  source made its levelled frame or MV-Adapter made views - also when the back was rejected (the
  pictures are outlined as not used). Output label "SHARP 360° back photos as lined up (PNG)" /
  "SHARP 360° AI views as lined up (PNG)".
* Progress ranges: SHARP 0.10-0.25, front 0.25-0.30, back 0.30-0.72, stitch 0.72-0.88,
  export 0.88-0.92, previews 0.92-0.97 ("Rendering a turntable preview"), scene 0.97-1.0.
  When the Brush polish runs: stitch 0.72-0.82, "Polishing with Brush" 0.82-0.92 (design
  1.4), export 0.92-0.94, previews 0.94-0.97.

**Fallbacks.** SHARP is the only required add-on (a SHARP failure fails the job as for `sharp`).
Back sources are tried in `Opts.sources` order (`auto` -> the quality's installed sources, then
`shell`; an explicit source -> it, then `shell`); a missing, failing or unusable source is one
plain warning naming what is used instead. `except Cancelled: raise` always comes first
(`Cancelled` is a `VRPhotoError`). If even the shell fails, or stitching / exporting the back
fails, the cleaned front is exported alone with a warning; if the front cannot be cleaned at
all, plain SHARP's matte-filtered result is saved (no `scale`, fitted to `height_m` by the
viewer). Once SHARP has succeeded only a cancel fails the job.

**Export (design 3.12).** `run._splats_cv_to_gltf` -> `normalize_orientation(up=cv_to_gltf(canon.up_C))`
(floor = P0.5, x/z centred on the median) -> `to_life_size` (`h = P99.8(y | opacity > 0.5)`;
`height`: `k = height_m / h`; `photo`: `k = 1`; `auto`: height unless the matte touches the top
or bottom border over > 15 % of the subject width, then photo with a warning) ->
box crop of the generated rows (front rows are kept even outside the box) ->
`decimate_splats(750k)` -> `write_splats`.

**Outputs.** `<base>_cutout.png`, `<base>_sharp360.splat` (feet at y = 0, front facing +z,
life-size), `<base>_sharp360_preview.jpg`, `<base>_sharp360_views.png` (only after a view-based
source: `images`, `mvadapter`); with `save_extras` also `<base>_sharp360.ply`,
`_sharp360_front.splat` (the cleaned SHARP front alone), `_sharp360_back.splat` (the generated
part alone; both in the main splat's frame and size), `_sharp360_debug.json` and, after
MV-Adapter, the views folder `_sharp360_views/`; `_sharp360_polish.log` when a Brush polish
failed. The debug data (options, camera, front stats,
frame, source attempts, merge / floater counts, sizes, previews, timings, warnings) is always written to
the job's work folder and summarised in one log line. None of these names ends in a Quest
player token. Scene block (always `scale: 1.0`):
`{"asset": "object.splat", "format": "splat", "height_m": h, "scale": 1.0, "pivot": "base",
"placement": ...}`.
