# VR Photo Studio - User Guide

This guide explains everything VR Photo Studio can do, in plain words, with recommended settings.
You do not need to be technical. If you are in a hurry, read [Quick start](#2-quick-start) and
[Seeing your creations on the Quest](#6-getting-files-onto-your-quest).

**Contents**

1. [The ideas in two minutes](#1-the-ideas-in-two-minutes)
2. [Quick start](#2-quick-start)
3. [The basic workflow](#3-the-basic-workflow)
4. [The modes](#4-the-modes)
   [3D Photo](#41-3d-photo-stereo) ·
   [Living Photo](#42-living-photo-parallax-video) ·
   [VR180](#43-vr180) ·
   [360](#44-360) ·
   [3D Scene](#45-3d-scene-6dof) ·
   [Person / Object](#46-person--object-walk-around-ar) ·
   [Capture](#47-capture-many-photos--walk-around-3d) ·
   [3D Video](#48-3d-video) ·
   [Depth Map](#49-depth-map)
5. [Settings shared by all modes](#5-settings-shared-by-all-modes)
6. [Getting files onto your Quest](#6-getting-files-onto-your-quest)
7. [The live viewer (Send to Quest)](#7-the-live-viewer-send-to-quest)
8. [Add-ons](#8-add-ons)
9. [How to capture a person or object](#9-how-to-capture-a-person-or-object-for-capture-mode)
10. [Tips for great results](#10-tips-for-great-results)
11. [Troubleshooting](#troubleshooting)
12. [Privacy and licenses](#12-privacy-and-licenses)

---

## 1. The ideas in two minutes

* **Depth map** - an AI looks at your photo and guesses how far away every pixel is (near =
  bright, far = dark). Everything else in the app is built from this.
* **3D / stereo** - your two eyes see the world from slightly different positions. The app
  renders the photo twice, once for each eye, so your brain sees real depth.
* **Parallax** - when you move your head, near things move more than far things. The *Living
  Photo* and *3D Scene* modes use this to make a photo feel alive.
* **Look behind (AI inpainting)** - when the viewpoint moves, you would see "behind" a person or
  object, where the photo has no information. A second AI paints a believable background there,
  so you don't see holes or stretched edges.
* **VR180 / 360** - pictures that surround you. VR180 fills the half-sphere in front of you, 360
  the whole sphere. Both can be 3D (stereo).
* **6DoF** ("six degrees of freedom") - you can not only turn your head but also move it: lean
  forward, sideways, crouch. The *3D Scene*, *Person / Object* and *Capture* modes are 6DoF.
* **Passthrough / AR** - on Quest 3 you can see your real room through the headset cameras.
  People and objects you make can stand on your real floor.
* **Gaussian splat** - a modern kind of 3D model made of millions of tiny soft coloured blobs.
  It looks almost photographic. *Capture* mode makes these.
* **Live viewer** - a small web page your PC sends to the Quest Browser. It shows 3D scenes,
  objects, splats, VR180 and 360 pictures without copying files.

---

## 2. Quick start

1. Install and start VR Photo Studio (see the [README](../README.md#download-and-install)).
2. On first start the app checks your graphics card. Open **Settings** (toolbar) and check that
   *Device* shows your GPU and that *Headset* is set to **Meta Quest 3 / 3S** (or your headset).
3. Drag a photo into the window (or click **Add photos…**).
4. Under *What do you want to make?* pick a mode - **3D Scene** is the most impressive, **3D Photo**
   the fastest.
5. Click the big **Create** button (it names the mode, for example *Create 3D Scene*). The first
   time, the app downloads the AI models it needs (the default depth model is about 100 MB; "look
   behind" adds about 200 MB). Later runs start immediately.
6. Click **Send to Quest** and follow [the live viewer steps](#7-the-live-viewer-send-to-quest), or
   copy the file to the Quest ([section 6](#6-getting-files-onto-your-quest)).

---

## 3. The basic workflow

* **Add photos:** drag & drop files or folders, or use **Add photos…** / **Add folder…**. JPEG, PNG,
  WebP, TIFF, BMP and iPhone/Android **HEIC** photos work. Videos (MP4, MOV, MKV, ...) are used by
  *3D Video* and *Capture*.
* **Options:** every mode has a **Preset** menu at the top of its options (*Default*, and depending
  on the mode *Comfortable*, *Strong 3D* or *Max quality*) and a **Reset** button. Rarely needed
  options are in the folded **Advanced** part, and the AI model choices, output folder and depth
  range are in the folded **Depth, AI models and output** part ([section 5](#5-settings-shared-by-all-modes)).
* **Several photos at once:** select many photos and press **Create** - each becomes its own job
  in the **queue**. You can keep working while the queue runs, **Cancel** a job, or **Clear
  finished** ones. When a job is done, **Open folder** shows the files and **View in headset** opens
  it in the live viewer.
* **Preview:** the depth preview shows what the AI understood (bright = near). If the depth looks
  wrong, try another depth model (see [section 5](#5-settings-shared-by-all-modes)).
* **Results:** files go to `Pictures\VR Photo Studio` (change it in Settings). File names end with
  tokens like `_LR` or `_180_LR` so Quest players show them correctly
  ([section 6](#file-name-endings-and-which-players-understand-them)).
* **Viewer library:** scenes, objects and VR pictures are also added to the live viewer's
  gallery automatically (option *Add to viewer library*), ready for **Send to Quest**.

---

## 4. The modes

Each mode lists its options with what they do and a recommended value. Options marked
*(Advanced)* are in the folded **Advanced** part of the options. The defaults are good for most
photos - change one thing at a time.

### 4.1 3D Photo (stereo)

A normal-looking photo that pops out in 3D. Fastest mode; works with any photo.

| Option | What it does | Recommended |
|---|---|---|
| Layout | *Side-by-side* (two full images next to each other, best for Quest), *Half side-by-side* (for 3D TVs), *Top-bottom* / *Half top-bottom*, *Anaglyph* red/cyan glasses (colour or grey), *Cross-eyed* (free viewing without glasses) | Side-by-side for Quest |
| 3D strength | Multiplies the depth. Lower = softer and more comfortable, higher = more "pop". | 1.0 (0.6-1.4) |
| Look behind | *On*: the AI paints behind edges (cleaner outlines). *Off*: faster. | On |
| Also save a wiggle GIF | Also saves an animated "wiggle" GIF that shows 3D on any screen. | Optional |
| Eye distance (IPD) *(Advanced)* | Distance between the two virtual eyes. 63 mm is an average adult. | 63 mm |
| Screen distance *(Advanced)* | Which distance appears exactly at the "screen". *Auto* puts the main subject there. | Auto |
| Max pop-out *(Advanced)* | Comfort limit for how far things may stick out (as % of the image width). The app reduces the depth automatically if needed. | 3 % |
| Swap eyes *(Advanced)* | Use if the 3D looks "inside out" in some player. | Off |
| Image format *(Advanced)* | JPEG (smaller) or PNG (lossless, big). | JPEG |

**Output:** `name_LR.jpg` (side-by-side) or `name_TB.jpg`, `name_anaglyph.jpg`, ...

### 4.2 Living Photo (parallax video)

A short looping video where a virtual camera moves gently through your photo, like the
"3D photos" on social media. Can also be saved as a 3D or VR180 video.

| Option | What it does | Recommended |
|---|---|---|
| Camera movement | *Circle* (gentle float - the classic), *Swing* (side to side), *Dolly* (forward and back), *Zoom* (push in with depth), *Ken Burns* (pan and zoom), *Orbit* (arc around the subject), *Custom key frames* (advanced users) | Circle or Swing |
| Duration | Length of the video. | 6 s |
| Movement | How far the camera moves. More = more dramatic, but more stretching at edges. | 6 cm (3-8 cm) |
| Output | *Flat video* (phones/PC), *Side-by-side 3D video* (Quest players, 3D TVs), *VR180 3D video* (immersive) | Flat or VR180 |
| Seamless loop | Makes the movement return to the start for a seamless loop. | On |
| Look behind | AI fill behind edges (see above). | On |
| Frames per second *(Advanced)* | Smoothness: 24 (cinematic), 30, or 60 (smoothest, bigger files). | 30 |
| Width *(Advanced)* | Video width in pixels. *Auto* = same as the photo (limited by your headset). | Auto |
| Video format *(Advanced)* | *Automatic* picks the best format for your headset; *Compatible (H.264)* plays everywhere; *Quick preview* is fast and small. | Automatic |
| Eye distance (IPD) *(Advanced)* | For the 3D and VR180 outputs. | 63 mm |

**Output:** `name_parallax.mp4`, `name_parallax_LR.mp4` or `name_parallax_180_LR.mp4`.

### 4.3 VR180

Puts your photo into an immersive 180° 3D picture. The photo keeps its true size (a phone photo
covers roughly 65-75° of view, like a big window in front of you); the rest of the half-sphere is
filled.

| Option | What it does | Recommended |
|---|---|---|
| Enlarge photo | Makes the photo cover more of your view (1.0-2.0, at most 150° wide). Above 1 it feels bigger than life. | 1.0-1.3 |
| Fill the edges | *Soft blur* (a soft, darkened extension of the photo), *Black*, *AI outpaint* (paints new surroundings; needs the AI outpainting add-on) | Soft blur |
| Look behind | AI fill behind edges. | On |
| Also save as video | Also writes an N-second still video, for players that cannot open VR180 photos. *Off* = no video. | 5 s if your player needs it |
| Resolution per eye *(Advanced)* | Pixels per eye. *Auto* = from your headset (Quest 3: 3840, Quest 2: 2880). | Auto |
| Eye distance (IPD) *(Advanced)* | See 3D Photo. | 63 mm |
| Image format *(Advanced)* | JPEG or PNG. | JPEG |

**Output:** `name_180_LR.jpg` (side-by-side, with VR180 metadata) and optionally
`name_180_LR.mp4`.

### 4.4 360

Two different things, chosen automatically from the photo's shape:

* **A 360° photo from a 360 camera** (2:1 shape, e.g. Insta360, Ricoh Theta): becomes a **3D 360**
  (top-bottom stereo) - you look around a full sphere with depth.
* **A normal photo:** becomes a 360 sphere with your photo in front and the rest filled.

| Option | What it does | Recommended |
|---|---|---|
| Input is a 360° photo | *Auto-detect* (2:1 photos); choose *Yes* or *No* if detection is wrong. | Auto-detect |
| 3D (stereo) | For 360-camera photos: make it 3D (top-bottom). | On |
| Fill the surroundings | For normal photos: *Soft blur*, *Black* or *AI outpaint* (add-on). | Soft blur |
| Also save as video | Writes an N-second video as well. | Optional |
| Width *(Advanced)* | Width of the result. *Auto* = from your headset. | Auto |
| Eye distance (IPD) *(Advanced)* | 64 mm. Lower it if things very close to the camera feel strange. | 64 mm |

**Output:** `name_360_TB.jpg` (3D) or `name_360.jpg` (2D), with 360 metadata.

> Tip: 3D from a single 360 photo is strongest for things 1-10 m away. Things directly above and
> below you are kept flat on purpose (that's more comfortable).

### 4.5 3D Scene (6DoF)

The "wow" mode: your photo becomes a small 3D world in the live viewer. You can move your head,
lean in, and look around the person or object in front - the AI paints what was hidden behind.

| Option | What it does | Recommended |
|---|---|---|
| Detail | How detailed the 3D mesh is, in thousands of triangles. More = sharper edges but slower to load. | 300 (150-500) |
| Look behind | *Band* (AI paints a strip behind every edge), *Clean plate* (removes the main subject from the background completely, so you can look all the way around it - best for one clear subject such as a portrait), *Off* (fastest) | Band; Clean plate for one clear subject |
| Distance | How far in front of you the photo is placed. | 2 m |
| Movement range | How far you can move your head before you see the edges of what was prepared. Bigger needs more AI painting. | 40 cm |
| Also save GLB file | Saves a `.glb` 3D file you can open in Blender, Windows 3D Viewer, etc. | On |
| Also save OBJ file *(Advanced)* | Older 3D format (with textures). | Off |

**Output:** a scene in the viewer library (+ a `.glb` file). Open it with **Send to Quest**.

### 4.6 Person / Object (walk-around AR)

Cuts out a person, pet or object from a photo and makes something you can place on your real
floor (Quest 3 passthrough) and walk around.

| Option | What it does | Recommended |
|---|---|---|
| Method | *Billboard* (instant, no add-on: a cut-out with a rounded 3D shape - looks great from the front and the sides, the back is approximated), *TRELLIS* (add-on: AI builds a full 3D model including the back), *SHARP* (experimental add-on: fast AI splat), *TRELLIS.2* / *Hunyuan3D* (experimental add-ons) | Billboard to start; TRELLIS for full 3D |
| Real-world height | Height so it appears life-size: a person's height, or the object's size. | 1.70 m for adults |
| Thickness | (Billboard) how deep the shape is, as a fraction of the height. | 0.17 |
| Also save splat | Also saves a splat version for the viewer. | On |
| Placement | *Point at the floor* (you aim at your floor or table and press the trigger), or *In front of me* (appears on the floor 1.5 m ahead). | Point at the floor |
| Back side *(Advanced)* | (Billboard) *Soft colours from the outline* (subtle), or *Mirror of the front*. | Soft colours |
| Variation *(Advanced)* | For AI methods: change the number to get a different version. | 0 |

**Good photos:** the whole person/object visible (feet included), not cut off at the edges,
clearly separated from the background, even light.

**Output:** an object in the viewer library (+ `.glb` / `.ply` / `.splat` files).

### 4.7 Capture (many photos → walk-around 3D)

The most realistic 3D: take 40-120 photos (or a slow video) walking around a person or object,
and the Capture add-on builds a photo-real Gaussian splat. See
[section 9](#9-how-to-capture-a-person-or-object-for-capture-mode) for how to take the photos.
Add all photos (or the video) - they become **one** job.

| Option | What it does | Recommended |
|---|---|---|
| Frames per second (video) | How many frames are taken from each second of video. | 2 (3 for fast walking) |
| Max photos | Upper limit; more photos = slower. | 200 |
| Training steps | More = sharper but slower. ~7,000 is a quick test, 30,000 is best quality. | 15,000 |
| Place as object (AR) | *On*: centred and placed on your floor for AR. *Off*: keep the whole surroundings as a scene for VR. | On for people/objects |
| Max splats *(Advanced)* | Limits the size so it runs smoothly in the Quest Browser. | 750,000 |
| Keep *(Advanced)* | Removes stray floating bits far from the centre (keeps this % of points). | 98 % |

Expect 10-40 minutes on an RTX GPU with 12 GB, depending on photos and steps.

### 4.8 3D Video

Converts a normal video into a 3D (side-by-side) or VR180 video, frame by frame.

| Option | What it does | Recommended |
|---|---|---|
| Output | *Side-by-side 3D*, *Half side-by-side* (3D TVs), *Top-bottom 3D*, *VR180* (immersive) | Side-by-side 3D |
| 3D strength | As in 3D Photo. | 1.0 |
| Steadiness | Smooths depth between frames to avoid shimmering. Higher = steadier, but slower to react to cuts. | 0.6 |
| Keep audio | Copies the sound. | On |
| Eye distance (IPD) *(Advanced)* | As in 3D Photo. | 63 mm |
| Max pop-out *(Advanced)* | Comfort limit (slightly lower than photos because motion adds strain). | 2.5 % |
| Processing size *(Advanced)* | Long side in pixels used for depth. Bigger = finer but slower. | 1920 |
| Start at / End at *(Advanced)* | Convert only part of the video (seconds). | - |
| Video format *(Advanced)* | See Living Photo. | Automatic |

A one-minute 1080p clip takes roughly 5-20 minutes on an RTX GPU. Try a 10-second piece first.

**Output:** `name_LR.mp4`, `name_TB.mp4` or `name_180_LR.mp4` (with spatial metadata).

### 4.9 Depth Map

Saves the AI depth map: a 16-bit PNG (for Blender, displacement, other 3D tools) and a colourful
preview. Options: *Save colour preview*, *Save 16-bit PNG*.

---

## 5. Settings shared by all modes

### In every mode: "Depth, AI models and output"

Fold this part open below a mode's options. The values start from your **Settings** and apply to
the jobs you create next. Options marked *(More)* are in its folded **More** part.

| Option | What it does | Recommended |
|---|---|---|
| Depth model | Which AI estimates depth: *Auto* (Depth Anything V2 Small - fast, good, free for any use), *V2 Base/Large* (sharper, slower, **non-commercial**), *MoGe-2* (real-world distances and lens angle - great for 3D Scenes), *Depth Anything 3* (experimental), *Simple* (no AI, last resort) | Auto; MoGe-2 for 3D Scene/VR |
| Look-behind model | AI that paints hidden areas: *Auto* (LaMa, best), *MI-GAN* (faster), *OpenCV* (no AI) | Auto |
| Cut-out model | AI that separates people/objects: *Auto* (BiRefNet Lite), *BiRefNet Portrait* (people, hair), *IS-Net*, *U2-Net*, *GrabCut* (no AI) | Auto; Portrait for people |
| Output folder | Where the results of this job go. | Empty (your Settings folder) |
| Add to viewer library | Also makes the result available in the live viewer. | On |
| Processing device *(More)* | *Auto* uses your best GPU; pick a specific GPU or *CPU*. | Auto |
| Field of view *(More)* | The camera's angle of view. *Auto* = from MoGe-2, then the photo's EXIF data, then 65°. Set it if you know it (e.g. 0.5× ultra-wide ≈ 110°). | Auto |
| Nearest distance / Farthest distance *(More)* | Rough distance of the nearest and farthest things in the photo when the depth model doesn't know real distances. | 1.2 m / 15 m |
| Depth smoothing *(More)* | Edge-aware smoothing of the depth (0-2). Lower keeps fine detail, higher removes noise. | 1.0 |
| Max processing size *(More)* | Photos larger than this (long side, pixels) are reduced while processing. Lower it if you run out of memory. | 4096 |
| Output name *(More)* | Base name of the files (default: the photo's name). | Empty |
| Headset *(More)* | *Meta Quest 3 / 3S*, *Meta Quest 2* or *PC VR / other*: picks resolutions and video formats your headset plays smoothly. | Your headset |

### The Settings window (toolbar → Settings)

| Setting | What it does | Recommended |
|---|---|---|
| Device | The graphics card the AI runs on (or CPU). *Use half precision* makes it faster and use less GPU memory. | Your GPU |
| Max processing size | Default for the option above. | 4096 |
| Depth / Look-behind / Cut-out model | Defaults for the options above. | Auto |
| Hugging Face token | Only needed for some add-ons whose models require a free Hugging Face account. | Empty |
| Output folder | Where results go (default `Pictures\VR Photo Studio`). | - |
| Headset | Your headset (see above). | Your headset |
| Video encoder | *Automatic* uses your graphics card's video encoder (NVIDIA NVENC, AMD AMF or Intel Quick Sync) when it works, else the slower *Software* encoder. | Automatic |
| Wi-Fi (https) port / USB (http) port | Network ports of the live viewer. Only change them if another program uses them. | 8443 / 8080 |
| Show experimental features | Shows experimental models, modes and add-ons. | Off |
| Theme | *Same as Windows*, *Dark* or *Light*. | Same as Windows |

### AI models (downloaded on first use)

Open **AI models** in the toolbar to download models in advance or delete them.

| Model | Size | Used for | License |
|---|---|---|---|
| Depth Anything V2 Small | ~100 MB | depth (default) | Apache-2.0 |
| Depth Anything V2 Base | ~390 MB | depth | **non-commercial** |
| Depth Anything V2 Large | ~1.3 GB | depth | **non-commercial** |
| MoGe-2 ViT-L | ~1.3 GB | depth + distances + field of view | MIT |
| Depth Anything 3 (experimental) | ~1.4 GB | depth | Apache-2.0 |
| LaMa | ~210 MB | look behind | Apache-2.0 |
| MI-GAN | ~30 MB | look behind (fast) | MIT |
| BiRefNet Lite / Portrait | ~220 MB / ~970 MB | cut-outs | MIT |
| IS-Net, U2-Net | ~180 MB each | cut-outs | Apache-2.0 |

Models are stored in `%LOCALAPPDATA%\VRPhotoStudio\models`; to free space you can delete them
(they are downloaded again when needed). Downloads resume if interrupted. If a model is marked
**non-commercial**, the app asks you to accept that before downloading; results made with it may
only be used privately.

---

## 6. Getting files onto your Quest

### Copy with a USB cable

1. Connect the Quest to the PC with a USB-C cable that supports data (some cables only charge).
2. Put the headset on. A message **"Allow access to data?"** appears - choose **Allow**. (Missed it?
   Open the notifications in the headset menu.)
3. On the PC open **File Explorer → This PC → Quest 3 → Internal shared storage**.
4. Copy your files into the **Movies** folder (photos also work there; you can make sub-folders).
5. In the headset, open your player (below) and look in its local files / Movies section.

Shortcut: in **Send to Quest → USB cable**, the **Copy files to Quest** button copies the results
of the selected job into the Quest's Movies folder for you (steps 1-2 are still needed).

The headset must stay awake while copying. If **Quest 3** does not appear in File Explorer, unplug,
plug in again and answer the prompt again.

### File name endings (and which players understand them)

VR Photo Studio adds an ending to every file name so players know how to show it. **Don't remove
these endings when you rename files.**

| Ending | What it is | Example |
|---|---|---|
| `_LR` | 3D side-by-side (left eye on the left), shown on a big flat screen | `beach_LR.jpg`, `beach_parallax_LR.mp4` |
| `_TB` | 3D top-bottom (left eye on top), flat screen | `beach_TB.mp4` |
| `_180_LR` | VR180 3D, side-by-side | `beach_180_LR.jpg`, `beach_180_LR.mp4` |
| `_360_TB` | 360° 3D, top-bottom | `garden_360_TB.jpg` |
| `_360` | 360° 2D (mono) | `garden_360.jpg` |

| Player (Quest) | Price | What it plays | Notes |
|---|---|---|---|
| **Meta Quest TV / Files** (built-in media viewer) | free | videos and photos, flat, 3D, 180, 360 | Reads the endings above for videos. Meta renames these apps now and then - look for the built-in *Files* / *Media* / *Meta Quest TV* app. |
| **Skybox VR Player** | free + paid | videos **and photos** in all formats | Reads the endings; you can also change the format in its menu. Very reliable for 3D photos. |
| **DeoVR** | free | videos (and photos) in all formats | Reads `_LR`, `_TB`, `_180_LR`, `_360_TB`. Great for VR180 video. |
| **HereSphere** | paid | videos in all formats | Reads the endings and video metadata; fine-tuning of projection and stereo. Best image quality for VR180. |

If a player shows two pictures next to each other (or a squashed picture), it didn't detect the
format: open the player's format/3D menu and pick **Side-by-side 3D** / **180° SBS** /
**360° top-bottom** to match the ending.

**Photos vs videos:** some players only open videos. For VR180 and 360, the option *Also save as
video* writes a short still video that plays everywhere. The [live viewer](#7-the-live-viewer-send-to-quest)
shows every kind of picture without copying anything.

**Video formats:** the *Automatic* video format uses HEVC (H.265) sized for your headset, which the
Quest plays smoothly. If a video stutters or does not play, set *Video format* to *Meta Quest 2* or
*Compatible (H.264)*, or use a smaller width.

---

## 7. The live viewer (Send to Quest)

The live viewer is a web page that your PC sends to the **Quest Browser** over your home
network. It shows 3D Scenes, Persons/Objects, Captures, VR180/360 pictures and 3D photos, in VR or
in passthrough AR.

### Over Wi-Fi (normal way)

1. Make sure the PC and the Quest are on the **same network** (the same router; not a "guest"
   network). The PC can be on a cable.
2. In VR Photo Studio click **Send to Quest**. The live viewer starts and the window shows an
   address like `https://192.168.1.23:8443/` and a QR code.
   * The first time, **Windows Firewall** may ask whether VR Photo Studio may communicate on
     networks: tick **Private networks** and click **Allow access**.
   * The **Firewall** line at the bottom of the window tells you whether Windows lets the Quest in.
     If it says the port is blocked, click **Allow through firewall** (Windows asks for permission
     once). This also fixes things if you clicked *Cancel* on the Windows question earlier.
   * **Open on this PC** opens the same page in your PC's browser - a quick test that it works.
3. Put on the Quest, open the **Browser**, tap the address bar and type the address exactly as
   shown, including `https://` and `:8443`. (If your Quest offers to open QR codes through the
   cameras, you can scan the QR code instead.)
4. The browser warns **"Your connection is not private"**. This is expected. Tap **Advanced**, then
   **Proceed to 192.168.1.23 (unsafe)**.
   *Why is this safe?* VR in a web page requires an encrypted (https) connection. Your PC made
   its own certificate for this; it isn't signed by a public authority, so the browser can't
   check it. On your home network that's fine. You'll see the warning again only if your PC's
   address changes.
5. The **gallery** shows your scenes, newest first. Tap one, then press **Enter VR** (fully
   immersive) or **Enter AR (passthrough)** (see your room - best for Persons/Objects and
   Captures). Allow the permission prompts for VR/passthrough.
6. Tip: **bookmark** the gallery page in the Quest Browser.

The viewer keeps running when you close the *Send to Quest* window (the status bar shows
*Viewer: ...*); use **Stop viewer** in that window to switch it off. The **Your scenes** tab lists
everything in the viewer library, with the address of each scene and a **Delete** button.

**Controls inside the viewer** (the **?** button in the page shows them too):

| Quest controllers | What it does |
|---|---|
| Left stick | Walk (or simply walk and lean - your real movement is used) |
| Right stick | Turn in steps |
| Right trigger | Hold, aim at the floor, release to jump there (where available); in AR: place an object |
| Grip | Grab objects; both grips together resize them; stick left/right rotates |
| Y or B | Menu: recenter, next scene, passthrough, exit |
| Hand tracking | Pinch to click; turn your left palm towards you and tap the blue dot for the menu |

Leave VR with the Meta button. On the PC, drag to look around, scroll to zoom, and use
<kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> to move.

### Over a USB cable (adb mode)

Useful if your network blocks devices from seeing each other, if you have no Wi-Fi, or to avoid
the certificate warning. It needs a one-time setup:

1. **Developer Mode:** create a free Meta developer account/organization at
   <https://developers.meta.com/horizon/>, then in the **Meta Horizon** phone app go to
   *Devices → your headset → Headset settings → Developer mode* and switch it on. Restart the headset.
2. **USB tools (adb):** in VR Photo Studio open **Send to Quest → USB cable** and click
   **Install USB tools** (downloads Google's *Android SDK Platform-Tools*, about 7 MB). If you already
   have adb from *Meta Quest Developer Hub*, SideQuest or the Android SDK, the app finds it by itself.
3. Connect the Quest by USB. In the headset, answer **Allow USB debugging?** with **Always allow
   from this computer**.
4. Click **Set up USB connection**. The app connects the Quest's `http://localhost:8080/` to your PC
   and opens the page in the Quest Browser for you.
5. If the page does not open by itself, open the Quest Browser and type `http://localhost:8080/` -
   no certificate warning.

Do step 4 again after unplugging the cable or restarting the headset.

---

## 8. Add-ons

Add-ons are optional, large AI tools that the app installs for you into a private folder
(`%LOCALAPPDATA%\VRPhotoStudio\addons`). They do not change anything else on your PC. Open
**Add-ons** (toolbar or *Tools* menu), pick one, read its license and click **Install**.
Installing downloads PyTorch, NVIDIA CUDA libraries and the AI model (several GB) and can take
5-30 minutes. **Uninstall** removes it again and frees the space.

| Add-on | What it gives you | GPU (minimum) | Disk (approx.) | License |
|---|---|---|---|---|
| **Capture to 3D** (COLMAP + Brush) | Photos or a video walking around something → photo-real 3D splat ([Capture mode](#47-capture-many-photos--walk-around-3d)) | Any DirectX 12 / Vulkan GPU, 4 GB+ (an NVIDIA GPU speeds up the photo matching) | ~2 GB | BSD-3 + Apache-2.0 - any use |
| **TRELLIS** (Microsoft) | One photo → full 3D model with an AI-invented back side (*Person / Object*, method *TRELLIS*) | NVIDIA, 8 GB+ (12 GB recommended) | ~20 GB | MIT, but uses NVIDIA nvdiffrast and Inria diff-gaussian-rasterization: **non-commercial** |
| **AI outpainting for VR180 / 360** (Stable Diffusion XL) | AI-painted surroundings for VR180/360 (*Fill: AI outpaint*) | NVIDIA, 8 GB+ | ~17 GB | CreativeML Open RAIL++-M (use restrictions) |
| **SHARP** (Apple, experimental) | One photo → 3D Gaussian splat in seconds (*Person / Object*, method *SHARP*) | NVIDIA, 6 GB+ | ~12 GB | Apple research model license: **non-commercial** |
| **TRELLIS.2** (experimental) | Newer, more detailed photo-to-3D | NVIDIA, 12 GB+ (16-24 GB recommended) | ~36 GB | MIT + Meta DINOv3 license (needs a Hugging Face token) + **non-commercial** NVIDIA parts |
| **Hunyuan3D-2** (experimental) | Alternative photo-to-3D by Tencent | NVIDIA, 6 GB+ (more for textures) | ~16 GB | Tencent community license - **not licensed in the EU, the UK and South Korea** |

The numbers are guides; the Add-ons window shows each add-on's exact requirements and checks your
GPU. Experimental add-ons appear when *Show experimental add-ons* (in the Add-ons window) or *Show
experimental features* (Settings) is on, and may not always work.

* **Non-commercial** means: fine for your own photos and fun, not for paid work or products.
  Scenes made with these add-ons show a small notice in the viewer.
* **Hugging Face token:** a few models require a free account at <https://huggingface.co>. Log in,
  open the model page linked in the Add-ons window, accept its terms, create a *Read* token under
  *Settings → Access Tokens*, and paste it into VR Photo Studio's Settings.
* Keep your **NVIDIA driver up to date** (NVIDIA App / GeForce Experience) - the add-ons use CUDA 12.

---

## 9. How to capture a person or object (for Capture mode)

Capture builds 3D from many views, so the photos must overlap and the subject must not move.

**Photos (best quality):**

1. **Light:** bright, even light - outdoors in the shade or on an overcast day is ideal. Avoid hard
   sunlight, strong shadows, and flash.
2. **Subject:** must stay **completely still** (people: a relaxed pose they can hold for about two
   minutes, eyes open, arms slightly away from the body). Pets rarely work.
3. **Walk in a circle** around the subject at a constant distance (about 1.5-3 m for a person),
   taking a photo every step - about every 10-15°, so **each photo overlaps the previous one by
   ~70 %**.
4. Do **two rounds at two heights:** one at chest/eye height, one lower (waist or knee height,
   looking slightly up). Optionally a few from above for the head/top.
5. **40-120 photos** in total. Keep the whole subject in frame, fill most of the picture, keep the
   same zoom, and avoid blur (hold still while pressing the shutter).
6. **Avoid:** mirrors, glass, shiny or transparent things, plain white walls with no texture, people
   walking through the background.

**Video (easier):** walk slowly around the subject once or twice (two heights if you can) in
**30-60 seconds**, 1080p or 4K, keeping the phone steady and the subject centred. Don't use
"cinematic"/portrait blur modes. VR Photo Studio picks the sharp frames automatically.

Then choose **Capture**, add all the photos (or the video) as one job, and click **Create**.

---

## 10. Tips for great results

* **Best photos for 3D:** a clear subject with some distance to the background (people in front of
  a landscape, a street, a room). Flat walls and close-ups of a single surface have little depth.
* **Sharp, well-lit photos** give much better depth. Phone portrait-mode blur confuses the AI a bit -
  use the normal photo if you have both.
* **Comfort first:** if 3D feels straining, lower *3D strength* or *Max pop-out*, or pick the
  *Comfortable* preset.
* **Edges look stretched:** keep *Look behind* on, reduce *Movement* (Living Photo) or *Movement range*
  (3D Scene), or try the MoGe-2 depth model.
* **People with hair:** use the *BiRefNet Portrait* cut-out model.
* **Life-size people in AR:** set *Real-world height* to the person's real height.
* **Quest 3 room setup:** for AR placement on the floor, the Quest needs a room/space setup
  (*Settings → Physical space → Space setup*).

---

## Troubleshooting

### Installing and starting

* **"Windows protected your PC" (SmartScreen):** click **More info → Run anyway** (the app isn't
  code-signed). See the [README](../README.md#windows-protected-your-pc).
* **Antivirus removed a file:** some antivirus tools wrongly flag new, unsigned programs. Restore the
  file, add the VR Photo Studio folder as an exception, and please report it.
* **The app does not start:** run `vrphoto-cli.exe --selftest` from the install folder
  (`%LOCALAPPDATA%\Programs\VR Photo Studio`) in a Command Prompt and look at the report; also see the
  log `%LOCALAPPDATA%\VRPhotoStudio\logs\vrphoto.log`.

### Speed and memory

* **Very slow:** Settings → *Device* should show your GPU, not CPU. Update the graphics driver.
  On laptops, plug in and set Windows *Settings → Display → Graphics* to "High performance" for
  VRPhotoStudio.exe.
* **"Out of memory":** lower *Max processing size* (e.g. 2048), close games/other GPU apps, use a smaller
  model (V2 Small, MI-GAN), or for videos lower *Processing size*.

### Downloads

* **Model download fails:** check your internet connection; some company/school networks, VPNs or
  security software block `huggingface.co` or `github.com`. Just retry - downloads resume. Make sure the
  disk has space.

### 3D quality

* **3D looks "inside out" / hurts:** turn on *Swap eyes* or pick the right format in the player.
* **Flat-looking result:** raise *3D strength* a little, try MoGe-2, or use a photo with more depth.
* **Holes or smeared edges when moving:** keep *Look behind* on, reduce movement.

### Files on the Quest

* **The Quest doesn't appear in File Explorer:** put the headset on and choose **Allow** for data access;
  try another cable/port; the headset must be awake.
* **Video shows two images side by side:** the file name ending is missing or the player didn't detect
  it - pick the format manually in the player.
* **Video stutters / doesn't play:** set *Video format* to *Meta Quest 2* or *Compatible (H.264)*, or use
  a smaller width.

### Live viewer

* **The page doesn't load in the Quest Browser:**
  1. Same network? (Guest networks and some mesh "isolation" modes block devices from seeing each other.)
  2. In Windows, the network must be **Private**: *Settings → Network & internet → Wi-Fi (or Ethernet) →
     your network → Network profile type → Private*.
  3. **Firewall:** click **Allow through firewall** in the *Send to Quest* window and then **Check again**.
     (Or: *Windows Security → Firewall & network protection → Allow an app through firewall* and tick
     *Private* for VR Photo Studio, or reinstall choosing *Install for all users* and tick *Allow Quest
     headset to connect*.) The ports are TCP 8443 and 8080. If you once clicked *Cancel* on the Windows
     Firewall question, Windows blocks the app until you do one of these.
  4. Turn off VPN software on the PC.
  5. Test on the PC: click **Open on this PC**, or open the same `https://...:8443/` address in your PC's
     browser.
  6. Still stuck? Use [USB (adb) mode](#over-a-usb-cable-adb-mode).
* **"Enter VR" is missing or greyed out:** use the `https://` address (or `http://localhost:8080/` in USB
  mode), accept the certificate warning, and update the Quest system software.
* **Certificate warning every time:** your PC's network address changes. Ask your router to always give the
  PC the same address ("DHCP reservation"), or use USB mode.
* **"Enter AR" not available / objects float:** allow passthrough permissions when asked; do the Quest
  space setup so the floor is known; check the object's *Real-world height* setting.

### Add-ons

* **Install fails:** free up disk space (see the table in [section 8](#8-add-ons)), update the NVIDIA driver,
  check the internet connection and retry. The error message and the log
  (`%LOCALAPPDATA%\VRPhotoStudio\logs`) say what went wrong.
* **"CUDA out of memory":** close other GPU programs; use lower quality settings; some add-ons need more VRAM
  than your GPU has.

### Starting fresh

* Delete `%LOCALAPPDATA%\VRPhotoStudio\settings.json` to reset all settings.
* Deleting the whole `%LOCALAPPDATA%\VRPhotoStudio` folder removes models, add-ons and settings (they are
  downloaded again when needed). Your creations in `Pictures\VR Photo Studio` are not affected.

When asking for help, attach `vrphoto.log` and the output of `vrphoto-cli.exe --selftest`.

---

## 12. Privacy and licenses

* All processing happens **on your PC**. Photos are never uploaded. The app only connects to the
  internet to download models and add-ons (Hugging Face, GitHub, PyPI).
* The live viewer is only reachable on your home network while it is running (*Send to Quest →
  Start viewer* / *Stop viewer*).
* VR Photo Studio is MIT-licensed. Some models and add-ons are **non-commercial only** - they are marked
  in the app and listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
