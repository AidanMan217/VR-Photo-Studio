# VR Photo Studio

**Turn your ordinary photos and videos into VR for Meta Quest 3.** Make 3D photos you can see
with real depth, "living photo" videos where the camera gently floats through the scene, VR180
and 360 pictures, interactive 3D scenes you can lean into and look behind things, and people or
objects you can walk around in your own room using the Quest's passthrough (mixed reality).

Everything runs on your Windows PC. The AI models run locally on your graphics card - your
photos are never uploaded anywhere.

> **Status:** version 0.1 (first release). Please report problems - see
> [Getting help](#getting-help).

---

## What it can make

| Mode | What you get | Watch it with | Notes |
|---|---|---|---|
| **3D Photo** | A stereo (3D) picture: side-by-side, top-bottom or red/cyan anaglyph | Quest media players, the built-in live viewer, 3D TVs, anaglyph glasses | Fast; works with any photo |
| **Living Photo** | A short looping video where a virtual camera moves through the photo (flat, 3D or VR180) | Any video player; Quest players for 3D/VR180 | Flat videos are H.264 - great for sharing on phones too |
| **VR180** | An immersive 180° 3D photo (or short video) in front of you | Quest players, live viewer | The photo keeps its real size; edges can be filled by AI |
| **360** | A full sphere around you - true 3D from a 360-camera photo, or a sphere built around a normal photo | Quest players, live viewer | 360-camera photos become stereo 3D |
| **3D Scene** (6DoF) | A walk-in 3D photo: move your head and lean to look around foreground objects | Live viewer in the Quest Browser | "Look behind" uses AI to paint what was hidden |
| **Person / Object** | A cut-out person or object standing on your floor that you can walk around | Live viewer, **passthrough AR** on Quest 3 | Billboard (instant) or full 3D with add-ons |
| **Capture** | A full 3D "Gaussian splat" from 40-120 photos or a slow video walking around something | Live viewer (VR or AR) | Needs the Capture add-on |
| **3D Video** | Turns a normal video into a 3D (side-by-side) or VR180 video | Quest players | Takes a while: every frame is processed |
| **Depth Map** | Grey-scale depth images (for other 3D tools) | Any image viewer | 16-bit PNG + colour preview |

Optional **add-ons** (big downloads, most need an NVIDIA GPU) add AI photo-to-3D (TRELLIS, plus the
experimental SHARP, TRELLIS.2 and Hunyuan3D), AI back and side views (experimental MV-Adapter),
photo-capture-to-3D (COLMAP + Brush) and AI fill for the rest of a 360°/180° sphere (SDXL).

**New (experimental): SHARP 360°** keeps SHARP's photo-real front of a person, adds a back and makes
them life-size, so you can walk all the way round them in passthrough AR. The back is a smooth built-in
shape, painted from six matching AI views by MV-Adapter, made by TRELLIS (if you have those add-ons), or
taken from your own photo of the person from behind; an optional Brush polish blends a back made from
views into the photo front, and a turntable preview picture shows the result from all sides.

---

## Quick start

1. **Download** `VRPhotoStudio-Setup-<version>.exe` from the
   [Releases page](https://github.com/AidanMan217/VR-Photo-Studio/releases) and run it
   (see [the SmartScreen note](#windows-protected-your-pc) if Windows warns you).
2. **Start VR Photo Studio** from the Start menu.
3. **Drop a photo** into the window, choose a mode (start with **3D Photo** or **3D Scene**) and
   click the big **Create** button. The first time, the app downloads a small AI model (about 100 MB).
4. **See it on your Quest:**
   * *Live (no copying):* click **Send to Quest** in the toolbar, then type the address it shows into
     the Quest Browser, accept the security warning once, and press **Enter VR** (or **Enter AR**).
   * *As files:* connect the Quest with a USB cable, allow file access in the headset, and copy the
     result into the Quest's **Movies** folder.

The [User Guide](docs/USER_GUIDE.md) explains every mode and option step by step.

---

## System requirements

| | Minimum | Recommended |
|---|---|---|
| Windows | Windows 10 (64-bit) | Windows 11 |
| Graphics card | Any DirectX 12 GPU (NVIDIA, AMD or Intel) - or none (CPU only, slow) | NVIDIA RTX with 8-12 GB+ VRAM |
| Memory | 8 GB RAM | 16 GB (32 GB for the big add-ons) |
| Disk | 2 GB for the app and default models | 20-60 GB if you install add-ons |
| Headset | Meta Quest 2 / Pro / 3 / 3S | Quest 3 (colour passthrough for AR) |
| Network | PC and Quest on the same home network | 5 GHz / 6 GHz Wi-Fi, PC on a network cable |

The add-ons that turn a single photo into a full 3D model (TRELLIS, SHARP, TRELLIS.2, Hunyuan3D) and
the AI outpainting add-on need an **NVIDIA GPU**; an RTX card with 12 GB of VRAM is a good fit.
Everything else - including the Capture add-on - also works on AMD and Intel GPUs.

---

## Download and install

* **Installer (recommended):** `VRPhotoStudio-Setup-<version>.exe`. It installs just for you
  (into `%LOCALAPPDATA%\Programs\VR Photo Studio`), so no administrator password is needed. If you
  pick *Install for all users*, Windows asks for permission, and the installer can then also allow
  the Quest through the Windows Firewall for you.
* **Portable zip:** `VRPhotoStudio-<version>-win64.zip`. Unzip it anywhere and run
  `VRPhotoStudio.exe`.
* `SHA256SUMS.txt` lets you check that your download is complete and unmodified (optional).

### "Windows protected your PC"

VR Photo Studio is free, open source and **not code-signed** (a signing certificate costs
hundreds of dollars a year), so Microsoft Defender SmartScreen shows a blue warning the first
time you run the installer:

1. Click **More info**.
2. Click **Run anyway**.

You only need to do this once per downloaded version. Only download VR Photo Studio from this
project's GitHub Releases page.

### Where your files go

| What | Where |
|---|---|
| Pictures, videos and 3D files you create | `Pictures\VR Photo Studio` (changeable in Settings) |
| Downloaded AI models, add-ons, settings, logs | `%LOCALAPPDATA%\VRPhotoStudio` |

Uninstalling removes the program and **asks** whether to delete the models and add-ons folder
(it can be many GB). Your creations are never deleted.

---

## Documentation

* [User Guide](docs/USER_GUIDE.md) - every mode and setting explained, getting files onto the
  Quest, the live viewer, add-ons, how to capture people and objects, troubleshooting.
* [Quest 3 test checklist](docs/QUEST_CHECKLIST.md) - a step-by-step check of everything on your
  PC and headset (handy for reporting problems).
* [Architecture](docs/ARCHITECTURE.md) - for developers: how the code is organised, running from
  source, tests, releases.
* [Changelog](CHANGELOG.md) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [License](LICENSE)

---

## Troubleshooting (short version)

| Problem | Try this |
|---|---|
| The Quest Browser cannot open the address | PC and Quest must be on the **same** network (not a guest network). In Windows set the network to **Private**. Click **Allow through firewall** in the *Send to Quest* window (Windows asks for permission once). Turn off VPNs. |
| "Your connection is not private" in the Quest Browser | Expected - click **Advanced**, then **Proceed**. The certificate was made by your own PC. |
| Files don't show up on the PC when the Quest is plugged in | Put the headset on and answer **Allow** to "Allow access to data". Try another USB-C cable (some only charge). |
| A video stutters on the Quest | Set the *Video format* option to *Meta Quest 2* or *Compatible (H.264)*, or use a smaller width. |
| It's very slow | In **Settings**, check that *Device* shows your graphics card (not "CPU"); update your graphics driver. |
| Something else | The log file is `%LOCALAPPDATA%\VRPhotoStudio\logs\vrphoto.log`. |

More in the [User Guide troubleshooting section](docs/USER_GUIDE.md#troubleshooting).

## Getting help

Open an issue at <https://github.com/AidanMan217/VR-Photo-Studio/issues> and attach the log file
and, if you can, the results of the [test checklist](docs/QUEST_CHECKLIST.md). Running
`vrphoto-cli.exe --selftest` (in the install folder) prints a quick health report you can paste.

---

## For developers

```powershell
git clone https://github.com/AidanMan217/VR-Photo-Studio
cd VR-Photo-Studio
uv sync --python 3.12 --extra dev   # .venv with the app (editable) + test/build tools
uv run vrphoto                      # start the GUI
uv run vrphoto-cli --selftest       # health check
uv run pytest -m "not models"       # run the tests
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the code layout, tests, packaging
(`packaging/build_windows.ps1`) and the release process.

## License

VR Photo Studio is released under the [MIT License](LICENSE). It includes and downloads
third-party software and AI models with their own licenses - some AI models and add-ons are for
**non-commercial use only**; the app marks them and asks before downloading them. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
