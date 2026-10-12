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
* **SHARP 360 (0.2.0-beta)** - details in section 13:
  * `pipeline.run`: object method `sharp360` (`ADDON_METHODS["sharp360"] = ("sharp",
    "image_to_splat")`, so a missing SHARP fails fast exactly like `sharp`); constants
    `BACK_SOURCES = ("auto", "shell")`, `BACK_QUALITIES = ("fast", "balanced", "best")`,
    `SIZE_FROM = ("auto", "height", "photo")`, `OBJECT_ENV`; helpers `_free_app_gpu()` (closes
    the app's ONNX Runtime sessions before **every** add-on: at the start of each
    `_run_addon` and of `_mode_capture` (COLMAP / Brush, which do not go through
    `_run_addon`); later in-process Brush runs must call it too; failures only logged), `_sharp_inputs(run, item, photo) -> (image_path, params)` (shared by `sharp`
    and `sharp360`; plain `sharp` is unchanged byte for byte) and `_save_cutout(...)`.
  * `ObjectJob` fields `back_source="auto"`, `back_quality="balanced"`, `seam_deg=60.0`,
    `size_from="auto"`, `save_extras=False`; `thickness` and `back` also apply to `sharp360`.
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

## 13. SHARP 360 (object method `sharp360`, 0.2.0-beta)

Design and rationale: `docs/design/SHARP360.md`. This version ships the fusion core and the
built-in `shell` back; TRELLIS, back photos, MV-Adapter, Qwen/ComfyUI, the CPU renderer,
previews, Brush polish and the seed option arrive in later milestones behind the same
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
# pipeline/sharp360.py
QUALITY: dict[str, Quality(mould_long, stride, auto)]; Opts; SOURCES: dict[str, fn(ctx, rep) -> BackPart]
def resolve_options(spec, *, installed: Callable[[str], bool]) -> Opts   # InputError before any work;
    # thickness clamped to THICKNESS_RANGE = (0.06, 0.18) (Opts.thickness_asked; the job warns)
def run_sources(ctx, rep, attempts) -> BackPart | None                   # the fallback chain
def mode_sharp360(run, item, photo, alpha, spec, *, placement, height, rep, opts=None) -> None
```

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
life-size); with `save_extras` also `<base>_sharp360.ply`, `_sharp360_front.splat` (the cleaned
SHARP front alone), `_sharp360_back.splat` (the generated part alone; both in the main splat's
frame and size) and `_sharp360_debug.json`. The debug data (options, camera, front stats,
frame, source attempts, merge / floater counts, sizes, timings, warnings) is always written to
the job's work folder and summarised in one log line. None of these names ends in a Quest
player token. Scene block (always `scale: 1.0`):
`{"asset": "object.splat", "format": "splat", "height_m": h, "scale": 1.0, "pivot": "base",
"placement": ...}`.
