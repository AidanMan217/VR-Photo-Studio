# Changelog

All notable changes to VR Photo Studio are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

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
  USB (adb reverse): gallery, Enter VR / Enter AR, meshes, splats, VR180/360 and 3D photos.
- Quest-friendly output names (`_LR`, `_TB`, `_180_LR`, `_360_TB`, `_360`), spatial video metadata
  and GPano photo metadata; HEVC/H.264 presets for Quest 3 and Quest 2 (NVENC when available).
- Add-on manager that installs heavy PyTorch/CUDA tools into private uv environments: Capture to 3D
  (COLMAP + Brush), TRELLIS, AI outpainting for VR180/360 (SDXL), and the experimental SHARP,
  TRELLIS.2 and Hunyuan3D-2.
- Windows installer (per-user, optional firewall rule, keeps or removes user data on request),
  portable zip, command-line tool `vrphoto-cli` with `--selftest`.
- User guide, Quest 3 test checklist and developer documentation.

[0.1.0]: https://github.com/AidanMan217/VR-Photo-Studio/releases/tag/v0.1.0
