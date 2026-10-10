# Third-party notices

VR Photo Studio itself is released under the MIT License (see `LICENSE`). It is built on, ships
with, or downloads the third-party software and AI models listed below. Each keeps its own
license. This file is included in the installer, next to the program (`THIRD_PARTY_NOTICES.txt`)
and in the source repository.

**Non-commercial content is marked with ⚠ NC.** If you use a model or add-on marked NC, the
pictures, videos and 3D files it produces may only be used for **personal, non-commercial**
purposes under that model's license. VR Photo Studio asks you to accept these licenses before
it downloads such a model or add-on, and scenes made with them carry a notice in the viewer.

---

## 1. Software included in the Windows program

| Component | License | Notes / source |
|---|---|---|
| Python 3.12 | PSF License 2.0 | https://www.python.org/ |
| PyInstaller bootloader | GPL-2.0-or-later **with bootloader exception** | The exception allows distributing the bundled app under any license. https://pyinstaller.org/ |
| Qt 6 (via PySide6 / Shiboken6) | **LGPL-3.0** | See section 2. https://www.qt.io/ , https://wiki.qt.io/Qt_for_Python |
| FFmpeg (via imageio-ffmpeg) | **GPL-3.0** build | See section 3. https://ffmpeg.org/ , https://github.com/imageio/imageio-ffmpeg (BSD-2-Clause wrapper) |
| ONNX Runtime (onnxruntime-directml) | MIT | https://github.com/microsoft/onnxruntime - its own notices are bundled as `_internal/onnxruntime/ThirdPartyNotices.txt` |
| DirectML (`DirectML.dll`) | Microsoft DirectML redistributable license | Shipped inside the onnxruntime-directml package. https://github.com/microsoft/DirectML |
| OpenCV (opencv-python-headless) | Apache-2.0 | https://opencv.org/ - the wheel also contains an LGPL-2.1 FFmpeg build (`opencv_videoio_ffmpeg*.dll`) used only by OpenCV's video I/O |
| NumPy | BSD-3-Clause | https://numpy.org/ - bundles OpenBLAS (BSD-3-Clause) and the GCC runtime (GPL-3.0 with GCC Runtime Library Exception) |
| Pillow | MIT-CMU (HPND) | https://python-pillow.org/ - bundles libjpeg-turbo, libpng, zlib, libtiff, libwebp, FreeType, etc. under their permissive licenses |
| pillow-heif | BSD-3-Clause | https://github.com/bigcat88/pillow_heif - see section 4 for the libraries it contains |
| libheif | **LGPL-3.0** | https://github.com/strukturag/libheif (inside pillow-heif) |
| libde265 | **LGPL-3.0** | https://github.com/strukturag/libde265 (inside pillow-heif) |
| x265 | **GPL-2.0-or-later** | https://www.videolan.org/developers/x265.html (inside pillow-heif) |
| MinGW-w64 runtime (libgcc, libstdc++, libwinpthread) | GPL-3.0 with GCC Runtime Library Exception; MIT/ZPL (winpthreads) | inside pillow-heif |
| cryptography | Apache-2.0 OR BSD-3-Clause | https://github.com/pyca/cryptography - bundles OpenSSL (Apache-2.0) |
| cffi | MIT | https://github.com/python-cffi/cffi |
| qrcode | BSD-3-Clause | https://github.com/lincolnloop/python-qrcode |
| uv (`_internal/bin/uv.exe`) | MIT OR Apache-2.0 | https://github.com/astral-sh/uv - used to install add-ons |

### Web viewer (served from your PC to the Quest browser)

| Component | License | Source |
|---|---|---|
| three.js | MIT | https://github.com/mrdoob/three.js |
| Spark (Gaussian splat renderer for three.js) | MIT | https://github.com/sparkjsdev/spark |
| WebXR Input Profiles (controller models) | MIT | https://github.com/immersive-web/webxr-input-profiles |
| Google Spatial Media (360/180 video metadata) | Apache-2.0 | https://github.com/google/spatial-media |

The license files of the vendored web libraries are kept next to them inside
`_internal/vrphoto/webxr/viewer/vendor/` (and `_internal/vrphoto/third_party/`).

---

## 2. Qt / PySide6 (LGPL-3.0)

The user interface uses Qt 6 through PySide6 and Shiboken6, licensed under the GNU Lesser General
Public License v3 (https://www.gnu.org/licenses/lgpl-3.0.html). Qt is used **unmodified** and is
**dynamically linked**: the Qt and PySide6 libraries are separate files in the program folder
(`_internal/PySide6/`). You may replace them with your own compatible build of the same Qt / PySide6
version (6.11.x); VR Photo Studio will use the replaced files. The complete corresponding source code
of Qt and PySide6 is available from https://download.qt.io/official_releases/ and
https://code.qt.io/. If you cannot obtain it there, open an issue in the VR Photo Studio repository and
we will provide it.

## 3. FFmpeg (GPL-3.0)

Video encoding and decoding uses a separate FFmpeg program (`ffmpeg-*.exe` in
`_internal/imageio_ffmpeg/binaries/`) that VR Photo Studio starts as an external process; it is
not linked into VR Photo Studio. This build of FFmpeg is licensed under the GNU General Public
License v3 (https://www.gnu.org/licenses/gpl-3.0.html) because it includes GPL components such as
x264 and x265. Run the file with `-version` to see its exact version and build configuration.
The FFmpeg source code is available at https://ffmpeg.org/download.html; the build is the one
distributed with imageio-ffmpeg (https://github.com/imageio/imageio-ffmpeg). You may replace the
file with another FFmpeg build (or set the environment variable `IMAGEIO_FFMPEG_EXE`).
FFmpeg is a trademark of Fabrice Bellard.

## 4. HEIC/HEIF support (pillow-heif)

Opening iPhone/Android HEIC photos uses pillow-heif, which contains libheif and libde265
(LGPL-3.0, dynamically loaded DLLs that can be replaced) and the x265 encoder (GPL-2.0-or-later).
Because x265 is GPL-licensed, the Windows *binary* distribution of VR Photo Studio as a whole is
also subject to the GPL for that component. VR Photo Studio's own source code is MIT-licensed (a
GPL-compatible license) and is published in full at https://github.com/AidanMan217/VR-Photo-Studio.
Sources: https://github.com/strukturag/libheif , https://github.com/strukturag/libde265 ,
https://bitbucket.org/multicoreware/x265_git .

---

## 5. AI models downloaded on first use (not included in the installer)

| Model | Used for | License |
|---|---|---|
| Depth Anything V2 Small | depth (default) | Apache-2.0 |
| Depth Anything V2 Base | depth (higher quality) | ⚠ NC - CC-BY-NC-4.0 |
| Depth Anything V2 Large | depth (best quality) | ⚠ NC - CC-BY-NC-4.0 |
| MoGe-2 (ViT-L) | metric depth + field of view | MIT |
| Depth Anything 3 Mono Large (experimental) | depth | Apache-2.0 |
| LaMa | "look behind" inpainting | Apache-2.0 |
| MI-GAN | fast inpainting | MIT |
| BiRefNet Lite / Portrait | cutting out people and objects | MIT |
| IS-Net (DIS) | cutting out objects | Apache-2.0 |
| U2-Net | cutting out objects | Apache-2.0 |

Model files come from Hugging Face / GitHub and are stored in `%LOCALAPPDATA%\VRPhotoStudio\models`.
Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2 ·
MoGe: https://github.com/microsoft/MoGe · LaMa: https://github.com/advimman/lama ·
MI-GAN: https://github.com/Picsart-AI-Research/MI-GAN · BiRefNet: https://github.com/ZhengPeng7/BiRefNet ·
IS-Net/DIS: https://github.com/xuebinqin/DIS · U2-Net: https://github.com/xuebinqin/U-2-Net

## 6. Optional add-ons (installed only when you choose to)

Add-ons are downloaded into `%LOCALAPPDATA%\VRPhotoStudio\addons` (Python packages from PyPI and
model weights from Hugging Face/GitHub). They are not part of the VR Photo Studio download.

| Add-on | What it uses | License |
|---|---|---|
| Capture to 3D (photos/video → walk-around 3D) | COLMAP (official Windows release, CUDA build, with the libraries it bundles) | BSD-3-Clause - https://github.com/colmap/colmap |
| | Brush (Gaussian splat training) | Apache-2.0 - https://github.com/ArthurBrussee/brush |
| TRELLIS (single photo → 3D model) | Microsoft TRELLIS code + TRELLIS-image-large weights (installed from the stable-projectorz fork) | MIT - https://github.com/microsoft/TRELLIS |
| | DINOv2 image encoder (Meta) | Apache-2.0 - https://github.com/facebookresearch/dinov2 |
| | nvdiffrast (NVIDIA) | ⚠ NC - NVIDIA Source Code License (non-commercial) - https://github.com/NVlabs/nvdiffrast |
| | diff-gaussian-rasterization (Inria) | ⚠ NC - Gaussian-Splatting License (research / non-commercial) - https://github.com/graphdeco-inria/diff-gaussian-rasterization |
| AI outpainting for VR180 / 360 | Stable Diffusion XL inpainting 0.1 | CreativeML Open RAIL++-M (use-based restrictions) - https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1 |
| | SDXL VAE fp16 fix | MIT - https://huggingface.co/madebyollin/sdxl-vae-fp16-fix |
| SHARP (experimental; single photo → 3D Gaussian splat) | Apple ml-sharp code | Apple Sample Code License - https://github.com/apple/ml-sharp |
| | SHARP model weights | ⚠ NC - Apple Machine Learning Research Model License (non-commercial research only) |
| TRELLIS.2 (experimental) | Microsoft TRELLIS.2 code + weights | MIT - https://github.com/microsoft/TRELLIS.2 |
| | DINOv3 image encoder (Meta; gated download, needs a Hugging Face token) | DINOv3 License - https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m |
| | nvdiffrast / nvdiffrec (NVIDIA) | ⚠ NC - NVIDIA Source Code License (non-commercial) |
| | BiRefNet (background removal) | MIT - https://github.com/ZhengPeng7/BiRefNet |
| Hunyuan3D-2 (experimental) | Tencent Hunyuan3D-2 / Hunyuan3D-2mini | ⚠ Tencent Hunyuan 3D 2.0 Community License - **not licensed for use in the European Union, the United Kingdom and South Korea**; products with more than 1 million monthly users need a separate license from Tencent; plus an acceptable-use policy - https://github.com/Tencent-Hunyuan/Hunyuan3D-2 |
| all PyTorch add-ons | PyTorch, torchvision, xFormers, diffusers, transformers and other PyPI packages | BSD-3-Clause / Apache-2.0 / MIT - see each package |
| | NVIDIA CUDA runtime libraries (pip wheels) | NVIDIA CUDA EULA |

The add-on manager shows each add-on's license before installing it; read it there for the
authoritative terms.

---

## 7. Full license texts

* MIT: https://opensource.org/license/mit
* BSD-2-Clause / BSD-3-Clause: https://opensource.org/license/bsd-3-clause
* Apache-2.0: https://www.apache.org/licenses/LICENSE-2.0
* LGPL-3.0: https://www.gnu.org/licenses/lgpl-3.0.html (builds on GPL-3.0)
* GPL-2.0 / GPL-3.0: https://www.gnu.org/licenses/
* PSF License: https://docs.python.org/3/license.html
* HPND / MIT-CMU (Pillow): https://github.com/python-pillow/Pillow/blob/main/LICENSE
* CC-BY-NC-4.0: https://creativecommons.org/licenses/by-nc/4.0/legalcode
* CreativeML Open RAIL++-M: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md

The license files of the bundled Python packages are copied into the program folder at
`_internal/licenses/<package>/`, next to the full GPL-2.0, GPL-3.0, LGPL-2.1, LGPL-3.0 and
Apache-2.0 texts (`_internal/licenses/*.txt`).
