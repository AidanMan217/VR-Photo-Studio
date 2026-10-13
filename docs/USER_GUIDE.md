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
* **Results:** files go to your output folder - **File → Open output folder** opens it (default
  `Pictures\VR Photo Studio`; change it in Settings). File names end with
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
| Width *(Advanced)* | Video width in pixels. *Auto* = same as the photo (limited by your headset; flat videos at most 1920 px on the long side so phones and PCs play them). | Auto |
| Video format *(Advanced)* | *Automatic* saves flat videos as H.264 (plays on phones and PCs) and 3D/VR180 videos as HEVC sized for your headset; *Compatible (H.264)* plays everywhere; *Quick preview* is fast and small. | Automatic |
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
| Detail | How detailed the 3D scene is: about this many thousand triangles in total (all layers together). More = sharper edges but slower to load. | 300 (150-500) |
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
| Method | *Billboard* (instant, no add-on: a cut-out with a rounded 3D shape - looks great from the front and the sides, the back is approximated), *TRELLIS* (add-on: AI builds a full 3D model including the back), *SHARP* (experimental add-on: fast AI splat, front only), *SHARP 360°* (experimental, uses the SHARP add-on: SHARP's photo-real front plus a back, see [below](#sharp-360-experimental)), *TRELLIS.2* / *Hunyuan3D* (experimental add-ons) | Billboard to start; TRELLIS for full 3D |
| Real-world height | Height so it appears life-size: a person's height, or the object's size. | 1.70 m for adults |
| Thickness | (Billboard and SHARP 360°) how deep the shape is, as a fraction of the height. | 0.11 |
| Also save splat | Also saves a splat version for the viewer. | On |
| Placement | *Point at the floor* (you aim at your floor or table and press the trigger), or *In front of me* (appears on the floor 1.5 m ahead). | Point at the floor |
| Back side *(Advanced)* | (Billboard and SHARP 360°) *Soft colours from the outline* (subtle), or *Mirror of the front* (careful: a person gets their face on the back of their head). | Soft colours |
| Variation *(Advanced)* | For TRELLIS, TRELLIS.2, Hunyuan3D and SHARP 360°: change the number to get a different version. With SHARP 360° it changes the AI-made back (MV-Adapter or TRELLIS) and the pattern where the photo blends into the back; the photographed front stays the same. (Plain SHARP always gives the same result for the same photo.) | 0 |
| Paint texture *(Advanced)* | (Hunyuan3D) also paints colours onto the model; without it you get a plain grey shape. Slower, needs more graphics memory, and downloads the painting models (about 8 GB) on the first textured run. Needs the add-on's optional texture-painting part (see [section 8](#8-add-ons)). | Off; on for colour |

**Good photos:** the whole person/object visible (feet included), not cut off at the edges,
clearly separated from the background, even light.

**Output:** an object in the viewer library (+ `.glb` / `.ply` / `.splat` files).

#### SHARP 360° (experimental)

Plain *SHARP* turns your photo into a photo-real splat, but only of what the camera saw: walk
to the side and the person ends in smears, and from behind there is nothing. *SHARP 360°* keeps
SHARP's front exactly as it is, cleans off the smears at the edges and the bits of wall SHARP
puts behind the person, **adds a back**, and makes the result life-size, so in the Quest the
person stands on your floor and you can walk all the way round.

It needs only the **SHARP** add-on (see [section 8](#8-add-ons)); turn on *Show experimental
features* in Settings to see the method. The back can come from four places (*Back made by*),
plus two experimental ones described further down (*Qwen photo editor* and *Your own ComfyUI*):

* **Built-in shape** (no download): the outline "inflated" into a smooth, rounded back in the
  colours from around the person's outline - like the person from behind in soft focus.
* **MV-Adapter AI views** (the experimental MV-Adapter add-on, about 14 GB): an AI paints the
  person from six directions - the front, both sides, straight from behind and two in-between -
  and the app builds the back and sides from those pictures. Because the six pictures are made
  with exact, matching cameras, the app can check them against your photo and line them up
  precisely (see *The MV-Adapter back* below).
* **TRELLIS** (the TRELLIS add-on, which you may already have for the *TRELLIS* method): an AI
  builds the whole person in 3D from the photo; the app lines that shape up exactly behind
  SHARP's front and uses only its back half. Real volume (hair, shoulders, a hood, a bag), but
  a soft, flat-lit texture. If you pick TRELLIS and it is not installed, the app asks whether to
  open the Add-ons window or to continue with the built-in back.
* **Your own back photo** (no download): a photo of the same person from behind - or a back
  view you made with any AI photo tool - bent onto the outline of your front photo. Optional
  side photos make the sides more accurate. This can look the most real.
* **Qwen photo editor** (experimental add-on, about 30 GB, **slow**): a photo-editing AI turns
  the person round; the most photo-like AI backs, at several minutes per view on a GTX 1080 (see
  *The Qwen photo-editor back* below).
* **Your own ComfyUI** (expert, no download): your own ComfyUI server runs your own image-editing
  workflow (see *Your own ComfyUI* below).

| Option | What it does | Recommended |
|---|---|---|
| Back made by | *Automatic* uses MV-Adapter when its add-on is installed, else TRELLIS when that is installed, else the built-in shape (with *Fast* back quality always the built-in shape; it never picks your own back photo, the Qwen photo editor or your own ComfyUI). *Built-in shape*, *MV-Adapter AI views*, *TRELLIS*, *Your own back photo* or - with *Show experimental features* - *Qwen photo editor* and *Your own ComfyUI* choose one. If the chosen back cannot be made, the next one is used and a note says why. | Automatic |
| Back photo | (Your own back photo) the picture of the back; side photos in the same folder are found by their names (see *Taking the back and side photos* below). | - |
| Back quality | *Fast* builds a coarser back, never runs an AI back by itself and makes small previews; *Balanced*; *Best* builds the finest back surface, gives MV-Adapter and TRELLIS more steps and two attempts, and (with *Brush polish* on *Automatic*) polishes a back made from views (slower). The photographed front is the same with all three. | Balanced |
| Thickness / Back side | As for Billboard: how deep the built-in back is (and a back photo's, unless side photos show it), and its colours. SHARP 360° keeps *Thickness* within 0.06-0.18, because its back is life-size (the presets set *Thickness* for Billboard; a note says when it was changed). | 0.11, soft colours |
| Back detail *(Advanced)* | (A back made from pictures: your own back photo, MV-Adapter, the Qwen photo editor, your own ComfyUI) *From the picture* lets the depth model add folds, hair and clothing relief seen in the back picture; *Sharp* runs SHARP once more on the lined-up back picture and uses its 3D surface - usually the crispest relief, about half a minute more on a GTX 1080 (if SHARP cannot run, the depth model is used and a note says so); *Smooth* keeps the rounded shape and only takes the colours. *Automatic* uses SHARP with *Best*, the depth model with *Balanced* (smooth when only the built-in *Heuristic* depth model is available) and a smooth back with *Fast*. | Automatic |
| Match colours to the photo *(Advanced)* | Adjusts the brightness and colours of the AI back (MV-Adapter, TRELLIS) or your back photo to the front photo, so the join does not show as a change of light. Turn it off if your back photo already has the right colours. | On |
| AI attempts / AI steps *(Advanced)* | (MV-Adapter, TRELLIS, the Qwen photo editor, your own ComfyUI) how many backs the AI makes - the best-fitting one is kept - and how many refining steps each takes. *Automatic*: 1 attempt (2 for *Best*); TRELLIS 12 steps (25 for *Best*), MV-Adapter 20 / 30 / 50 steps for *Fast* / *Balanced* / *Best*, the Qwen photo editor 4 steps (what its Lightning LoRA is made for; up to 40). Your own ComfyUI workflow keeps its own steps. Each attempt takes a few minutes (the Qwen photo editor: several minutes per back view). | Automatic |
| Describe the hidden side *(Advanced)* | (MV-Adapter, the Qwen photo editor, your own ComfyUI) a few words about what the photo cannot show, for example *long brown ponytail, black backpack*. Empty: a neutral description. See *The MV-Adapter back* below. | Empty |
| ComfyUI address / ComfyUI workflow *(Advanced)* | (Your own ComfyUI) the address of your ComfyUI server and your workflow file in ComfyUI's API format; see *Your own ComfyUI* below. | http://127.0.0.1:8188 |
| Brush polish *(Advanced)* | (A back made from views: MV-Adapter or your own back photo) Brush - the trainer of the Capture add-on - trains the finished splat against all the pictures at once, so the views agree with each other and the join softens; SHARP's photographed front is put back exactly afterwards. *Automatic*: only with *Best* back quality and after a passed Brush test (see below); *On*: whenever possible (a note says when it is not); *Off*. If the polish fails, the unpolished result is kept. | Automatic |
| Brush steps *(Advanced)* | How long Brush trains (1000-30000). | 5000 |
| Keep the photo up to *(Advanced)* | Where SHARP's photo ends and the made-up back begins: the photo is kept wherever the body faces the camera within this angle and blended into the back over the next 10-20°. Higher keeps more of the sharp photo on the sides but can show stretched edges. | 60° |
| Size from *(Advanced)* | *Real-world height*: exactly the height you entered. *Measured by SHARP*: SHARP's own estimate from the photo (handy for half-body photos). *Automatic*: your height, unless the subject is cut off by the edge of the photo - then SHARP's measurement (with a note). | Automatic |
| Also save parts and debug data *(Advanced)* | Also saves `<name>_sharp360.ply`, the cleaned photo front and the back as separate splats (`_sharp360_front.splat`, `_sharp360_back.splat`, handy for comparing), `_sharp360_debug.json` (useful when reporting a problem) and, after MV-Adapter, the AI views as an editable folder `_sharp360_views` (see below). | Off |

**Taking the back and side photos** (for *Your own back photo*):

* Same person, same clothes and **the same pose** - arms, legs and head as in the front photo.
  A back in a clearly different pose is not used (you get a note and the built-in back).
* Stand **straight behind** the person at about the same distance and camera height as the
  front photo (chest height), with the whole person in the picture (feet included). A plain,
  even background helps the cut-out; a cut-out PNG with a transparent background is used as it is.
* The back photo can have any name. **Side photos** go into the same folder and are found by
  their name (any picture type, upper or lower case): `right` - you walked to **your** right
  around the person, so you see **their left** side; `left`; `back_right` (half way between
  `right` and the back); `back_left`; `front_right` (half way between the front and `right`);
  `front_left`. For example `back.jpg`, `right.jpg`, `left.jpg`. Pictures with other names are
  ignored. Left and right swapped by mistake are often noticed and turned round (with a note);
  when the outlines cannot tell, a note says the naming could not be confirmed.
* An AI-made back view works too (for example a photo editor that can turn the person round),
  as long as the pose and the outline match the front photo.

**The MV-Adapter back** (the experimental MV-Adapter add-on, see [section 8](#8-add-ons)):

* **What happens:** the app turns your photo into a straight-on, level picture of the person
  (it takes out the camera's tilt and perspective) and gives it to MV-Adapter, which paints the
  person from 0°, 45°, 90°, 180°, 270° and 315°. The app checks that the AI's front view matches
  your photo's outline, that every view shows the person at the same height, and that the back
  view's outline is the mirror of the front's; a view that does not fit is left out, and a set
  that does not fit at all is not used (you get a note and the next back - TRELLIS or the
  built-in shape). With two attempts (*Best*), the set that fits best is kept.
* **What to expect:** views that agree all the way round (including straight from behind), with
  plausible clothing and hair outlines - better than the built-in shape or TRELLIS for that. But
  the AI paints softer than your photo (about 3-4 mm per pixel for a 1.7 m person), and it was
  trained on objects, not people: a face or hands seen from the side may not quite match the
  person ("identity drift"), and details it cannot see are invented.
* **Describe the hidden side:** without a description, MV-Adapter is told only *a person, full
  body, high quality, realistic photo*. If you know what the back looks like, say so in a few
  words - *long brown ponytail*, *black backpack*, *hood up*, *grey hoodie, short hair* - about the
  person as a whole, not the camera or the background. Short is better (at most 300 characters).
  Changing it (or *Variation*, *AI steps* or *Back quality*'s steps) makes new views.
* **Time on a GeForce GTX 1080** (estimates, not yet measured): *Balanced* about 6-10 minutes in
  all; *Best* with two attempts and the Brush polish about 15-30 minutes. The first run after
  installing also loads the models from disk.
* **Made once, re-used:** the AI views are kept (up to 2 GB, the oldest are removed first), so
  running the same photo again with another *Keep the photo up to*, *Back detail*, *Match
  colours* or *Brush polish* reuses them and takes only a minute or two. Another *Back quality*
  reuses them too when *AI steps* is set; on *Automatic* each quality paints with its own number
  of steps, which makes new views (a second attempt for *Best* adds one more set). MV-Adapter
  saves each attempt as soon as it is finished: if the second attempt runs out of memory, the
  first one is used; after a cancel, the finished attempts are not painted again next time.

**The Qwen photo-editor back** (experimental; the *Qwen photo editor* add-on, see
[section 8](#8-add-ons); *Show experimental features* must be on):

* **What it is:** Qwen-Image-Edit-2511, a photo-editing AI, with a "camera angle" add-on that
  turns the person round, run by a private copy of ComfyUI that the add-on installs and starts by
  itself (you do not need to know ComfyUI). It gets the plain cut-out of the person on light grey
  and makes **back views** first (one per *AI attempt*); the app keeps the one that fits your
  photo's outline best and **never uses a back view that shows a face** (the AI sometimes paints
  the front onto the back of the head; a small face detector checks every back view). Then it
  makes the **side views** (left and right; with *Best* also the two back quarters) with the
  chosen back as a second picture, so the sides match it. *Fast* makes the back view only. The
  views are lined up like your own back photos.
* **What to expect:** the most photo-like AI backs - real-looking fabric, hair and folds - but
  each view is a separate picture, so the pose can drift a little (a back that does not fit is
  not used: you get a note and the built-in back).
* **Time, honestly:** it is **slow on a GeForce GTX 1080** - the card has no fast half-precision
  maths, so every view takes **several minutes** (estimate), and a whole person **15-60 minutes**
  (more with *Best* or more *AI attempts*). The first run also starts ComfyUI and loads about
  20 GB of models (a few minutes). The default is 4 *AI steps*, which its Lightning LoRA is made
  for; without that LoRA (its download is optional) every view takes 40 steps - about 20 times
  longer - and a note says so. When a run is slow, a note says how long it took.
* **What it needs:** about **30 GB** of disk space; an NVIDIA graphics card with 6 GB or more;
  at least **16 GB of system memory (RAM)** - **32 GB recommended**. With less than 16 GB the add-on
  refuses to run (you get a note and the built-in back); with 16-24 GB it runs without ComfyUI's
  memory cache (slower). Close games and browsers with many tabs first.
* **Made once, re-used:** like MV-Adapter's, the views are kept in the view cache, so running
  the same photo again with another *Keep the photo up to*, *Back quality* or *Brush polish* skips
  the minutes of painting; after a failure or a cancel the finished views are kept too.
* If a view runs out of memory, the add-on restarts ComfyUI with less graphics memory, then makes
  the pictures a bit smaller, then skips the view (with a note).

**Your own ComfyUI** (expert; *Show experimental features* must be on): if you run
[ComfyUI](https://github.com/comfyanonymous/ComfyUI) yourself, the app can send the cut-out to
your server and use **your own** image-editing workflow (FLUX Kontext, Qwen with other LoRAs, ...)
to make the back and side views. Nothing is installed; your server, models and licences are your
own.

1. In ComfyUI, build a workflow that turns a person round from one picture and saves the result
   with a *Save Image* node. Check it works by hand.
2. Save it with **Workflow → Export (API)** - the normal *Save* format does not work (the app says
   so).
3. Open the exported `.json` in a text editor and write these **placeholders** where the values go
   (with the double braces, inside the quotes):
   * `{{IMAGE}}` (**required**) - the image name of the *Load Image* node: the app uploads the
     cut-out person on light grey and puts its name here;
   * `{{PROMPT}}` - the text: the wanted view (for example *the same person seen from directly
     behind (back view), full body, same pose, same clothes, plain light grey background*) followed
     by *Describe the hidden side*;
   * `{{SEED}}` - the seed (*Variation*; + 1 for each further attempt);
   * `{{AZIMUTH}}` - the view's angle in degrees: 180 = from behind, 90 = from the person's left
     ("you walked to your right"), 270 = from their right, 135 / 225 the back quarters (handy for
     angle LoRAs or a file name prefix);
   * `{{WIDTH}}`, `{{HEIGHT}}` - the picture size (the cut-out's shape, long side 768).
   A number placeholder alone in its value (`"seed": "{{SEED}}"`) becomes a number. Other
   `{{...}}` names are refused.
4. In VR Photo Studio choose *Back made by: Your own ComfyUI*, enter the server's address under
   *ComfyUI address* (`http://127.0.0.1:8188` when ComfyUI runs on this PC) and pick the file under
   *ComfyUI workflow*.

The workflow runs once per view (the back views, then the sides; references are not passed, so
the sides are made from the cut-out alone) and the last *Save Image* picture of each run is used;
the app chooses the back and lines the views up exactly as for the Qwen photo editor. If ComfyUI
is not running, rejects the workflow (a missing node or model) or fails, you get a note with
ComfyUI's own message and the built-in back. A cancel stops the running prompt. The app frees its
own AI models first, in case ComfyUI shares the graphics card.

**Brush polish and the Brush test:** the polish needs the **Capture to 3D** add-on (it brings
Brush). Brush's behaviour on older graphics cards (GeForce 10-series) is not known in advance, so
test it once: open **Add-ons**, select *Capture to 3D* and click **Test Brush on this PC** (or run
`vrphoto-cli addons brush-test`). The test takes a few minutes and is remembered. *Automatic*
polishes only after a passed test; *On* tries anyway and keeps the unpolished result if Brush
fails (Brush's log is then saved as `<name>_sharp360_polish.log`).

**Fixing an AI view and running again** (the saved views folder): with *Also save parts and debug
data* on, an MV-Adapter run also saves the AI views as a folder `<name>_sharp360_views` with
`back.png`, `right.png`, `left.png`, `front_right.png`, `front_left.png`, `front.png` and a
`views.json` that remembers their exact cameras. Edit a picture in any photo editor (keep its size
and the grey background; paint out a wrong detail, fix a colour), then run the same front photo
again with *Back made by: Your own back photo* and pick that folder's `back.png` as the *Back
photo*: the app uses your edited views with their exact cameras.

**Preview pictures** (saved next to the splat, so you can check the result before putting the
headset on):

* `<name>_sharp360_preview.jpg` - the finished splat from 8 directions around the person (0° is
  the photo's side, 90° from the person's left, 180° from behind).
* `<name>_sharp360_views.png` - only with *MV-Adapter AI views*, *Your own back photo*, the *Qwen
  photo editor* or *Your own ComfyUI*: the levelled front photo and the AI views (or your back and
  side photos) as the app lined them up; back views that were not chosen (for example because they
  show a face) are shown too.
  Green is the outline the app expected, blue the person found in the picture; a picture outlined
  in red was not used (the note says why).

**What to expect:**

* **From the photo's angle** it looks like plain SHARP - the same photo-real front, whatever
  makes the back.
* **From the side** you see where the photo ends and the back begins; the sides of the body are
  softer than the front, and the join can show as a slight change in sharpness or colour.
* **From behind:**
  * *Built-in shape*: a closed, smooth, "mannequin" back in the colours around the outline: no
    clothing folds, hair detail, hood or backpack. Never see-through, always the right size.
  * *MV-Adapter*: clothing and hair that follow the person's outline from every side, softer than
    the photo and invented by the AI - faces seen from the side may drift. Colours are matched to
    your photo; *Brush polish* blends the views further.
  * *TRELLIS*: the real volume of hair, shoulders and a bag, but soft and evenly lit, and invented
    by the AI - it can get details wrong. Its colours are matched to your photo.
  * *Your own back photo*: the most real back - your photo's colours and detail on a shape that
    follows the outline (side photos make the sides follow them too). Areas no photo shows (for
    example between the legs) take neighbouring colours. If the app finds a face on your back
    photo, a note says so (it is used anyway: check that it really shows the person from behind).
  * *Qwen photo editor* / *Your own ComfyUI*: the most photo-like AI backs, but each view is
    painted separately, so details can disagree between the back and the sides.
* **Time on a GeForce GTX 1080** (estimates): built-in shape 1-2 minutes in all, most of it
  loading SHARP; your own back photo 1-3 minutes; TRELLIS 4-8 minutes (*Best*: two longer
  attempts, about 10-20 minutes); MV-Adapter 6-10 minutes (*Best* with two attempts and the Brush
  polish: 15-30 minutes); the Qwen photo editor 15-60 minutes (several minutes per view). *Back
  detail: Sharp* adds about half a minute per back. The turntable preview adds a few seconds to
  about half a minute.
* **Size:** the splat is saved life-size (feet on the floor), so the viewer shows it 1:1. With
  *Size from: Automatic* and feet or head cut off by the photo edge, SHARP's own estimate is
  used and a note says so.

**Limits (honest):** one photo cannot tell what the back really looks like: the built-in back,
MV-Adapter and TRELLIS are plausible guesses, and only your own back photo shows the real back. Loose hair, open
arms held away from the body, bags and anything that sticks out sideways come out flattened with
the built-in back; a seated person or an object with holes (a chair) looks worse than a standing
person. A camera that looks down or up steeply (more than about 30°) gives a less accurate back
(you get a note). SHARP's model is **non-commercial**, so every SHARP 360° result is for personal
use only (TRELLIS is non-commercial too; your own photos are your responsibility).

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
| Subject height *(Advanced)* | Real height of the captured person or object, so it appears life-size. *Automatic* works out the size from how high you held the camera (about chest height, 1.45 m). | Automatic; the real height if the size looks wrong |
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

**Output:** `name_LR.mp4`, `name_TB.mp4`, or `name_180_LR.mp4` (VR180, with spatial metadata so
players show it around you).

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
| Depth model | Which AI estimates depth: *Auto* (Depth Anything V2 Small - fast, good, free for any use), *V2 Base/Large* (sharper, slower, **non-commercial**), *MoGe-2* (real-world distances and lens angle - great for 3D Scenes), *Depth Anything 3* (experimental), *Heuristic* (no AI model, instant - last resort) | Auto; MoGe-2 for 3D Scene/VR |
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

**Video formats:** the *Automatic* video format saves 3D and VR180 videos as HEVC (H.265) sized for
your headset, which the Quest plays smoothly, and flat Living Photo videos as H.264 (at most 1920 px),
which plays on phones and PCs too. If a video stutters or does not play, set *Video format* to
*Meta Quest 2* or *Compatible (H.264)*, or use a smaller width.

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

If another program on your PC already uses port 8080, the *USB cable* tab tells you: the app then
uses another free port for USB (the tab shows the address to use), or USB mode is unavailable. Wi-Fi
works either way. For a fixed address choose another *USB (http) port* in **Settings** (for example
8081).

---

## 8. Add-ons

Add-ons are optional, large AI tools that the app installs for you into a private folder
(`%LOCALAPPDATA%\VRPhotoStudio\addons`). They do not change anything else on your PC. Open
**Add-ons** (toolbar or *Tools* menu), pick one, read its license and click **Install**.
Installing downloads PyTorch, NVIDIA CUDA libraries and the AI model (several GB) and can take
5-30 minutes. **Uninstall** removes it again and frees the space.

| Add-on | What it gives you | GPU (minimum) | Disk (approx.) | License |
|---|---|---|---|---|
| **Capture to 3D** (COLMAP + Brush) | Photos or a video walking around something → photo-real 3D splat ([Capture mode](#47-capture-many-photos--walk-around-3d)); also the *Brush polish* of *SHARP 360°* | Any DirectX 12 / Vulkan GPU, 4 GB+ (an NVIDIA GPU speeds up the photo matching) | ~2 GB | BSD-3 + Apache-2.0 - any use |
| **TRELLIS** (Microsoft) | One photo → full 3D model with an AI-invented back side (*Person / Object*, method *TRELLIS*; also an AI back of *SHARP 360°*) | NVIDIA, 8 GB+ (12 GB recommended) | ~20 GB | MIT, but uses NVIDIA nvdiffrast and Inria diff-gaussian-rasterization: **non-commercial** |
| **AI outpainting for VR180 / 360** (Stable Diffusion XL) | AI-painted surroundings for VR180/360 (*Fill: AI outpaint*) | NVIDIA, 8 GB+ | ~17 GB | CreativeML Open RAIL++-M (use restrictions) |
| **SHARP** (Apple, experimental) | One photo → 3D Gaussian splat in seconds (*Person / Object*, methods *SHARP* and *SHARP 360°*) | NVIDIA, 6 GB+ | ~12 GB | Apple research model license: **non-commercial** |
| **TRELLIS.2** (experimental) | Newer, more detailed photo-to-3D | NVIDIA, 12 GB+ (16-24 GB recommended) | ~36 GB | MIT + Meta DINOv3 license (needs a Hugging Face token) + **non-commercial** NVIDIA parts |
| **Hunyuan3D-2** (experimental) | Alternative photo-to-3D by Tencent; a grey shape, or coloured with *Paint texture* | NVIDIA, 6 GB+ (more for textures) | ~16 GB | Tencent community license - **not licensed in the EU, the UK and South Korea** |
| **MV-Adapter** (experimental) | AI back and side views for *SHARP 360°* (*Back made by: MV-Adapter AI views*): six matching views of the person, including straight from behind | NVIDIA, 6 GB+ (runs on a GTX 1080's 8 GB with part of the model in system memory) | ~14 GB | Apache-2.0 (MV-Adapter) + CreativeML Open RAIL++-M (Stable Diffusion 2.1: use restrictions) |
| **Qwen photo editor** (experimental, slow) | Photo-like AI back and side views for *SHARP 360°* (*Back made by: Qwen photo editor*): Qwen-Image-Edit-2511 with a camera-angle LoRA, run by a private ComfyUI; several minutes per view on a GTX 1080 | NVIDIA, 6 GB+; **16 GB of system memory (32 GB recommended)** | ~30 GB | Apache-2.0 (Qwen models, LoRAs, ComfyUI-GGUF); ComfyUI GPL-3.0 (runs as a separate program) |

The numbers are guides; the Add-ons window shows each add-on's exact requirements and checks your
GPU. Experimental add-ons appear when *Show experimental add-ons* (in the Add-ons window) or *Show
experimental features* (Settings) is on, and may not always work.

* **Non-commercial** means: fine for your own photos and fun, not for paid work or products.
  Scenes made with these add-ons show a small notice in the viewer.
* **Hugging Face token:** a few models require a free account at <https://huggingface.co>. Log in,
  open the model page linked in the Add-ons window, accept its terms, create a *Read* token under
  *Settings → Access Tokens*, and paste it into VR Photo Studio's Settings.
* **Hunyuan3D colours:** the *Paint texture* option (Person / Object, under *Advanced*) needs the add-on's optional
  texture-painting part, which **Install** can only build when the NVIDIA CUDA Toolkit and the Visual
  Studio C++ build tools are on your PC (otherwise Install reports that the optional texture-painting
  step failed, and you get the grey shape). The painting models (about 8 GB) download on the first
  textured run.
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
  log `%LOCALAPPDATA%\VRPhotoStudio\logs\vrphoto.log` (the command-line tool writes `cli.log` in the
  same folder).

### Speed and memory

* **Very slow:** Settings → *Device* should show your GPU, not CPU. Update the graphics driver.
  On laptops, plug in and set Windows *Settings → Display → Graphics* to "High performance" for
  VRPhotoStudio.exe.
* **"Out of memory":** lower *Max processing size* (e.g. 2048), close games/other GPU apps, use a smaller
  model (V2 Small, MI-GAN), or for videos lower *Processing size*. On 8 GB cards the *Max quality* cut-out
  (BiRefNet portrait) is large; the default preset uses the smaller model and is usually just as good.

### Downloads

* **Model download fails:** check your internet connection; some company/school networks, VPNs or
  security software block `huggingface.co` or `github.com`. Just retry - downloads resume. Make sure the
  disk has space.

### 3D quality

* **3D looks "inside out" / hurts:** turn on *Swap eyes* or pick the right format in the player.
* **Flat-looking result:** raise *3D strength* a little, try MoGe-2, or use a photo with more depth.
* **Holes or smeared edges when moving:** keep *Look behind* on, reduce movement.
* **SHARP 360°: the back looks flat or too thick:** change *Thickness* (0.08-0.15 suits most
  people). **The join at the sides shows:** try *Keep the photo up to* 50-55°, and keep *Match
  colours to the photo* on. **The person has the wrong size:** check *Real-world height*, or set
  *Size from* to *Real-world height* when the note says SHARP's estimate was used. Look at
  `<name>_sharp360_preview.jpg` first: it shows the result from all sides.
* **SHARP 360°: "Your back photo could not make a usable back":** the note says why. *Does not
  match the photo's outline* or *shows a different pose*: take the back photo again from straight behind with the
  same pose (arms!), distance and height as the front photo; `<name>_sharp360_views.png` shows
  how it was lined up. A phone held at eye level and tilted down is fine; a very steep angle
  (looking down from high above) is not. *No subject was found*: use a plainer background or a
  cut-out PNG. *The subject is too small in the picture*: take it from closer, or crop the
  picture around the person (they should be at least a few hundred pixels tall). *Could not be
  read*: the file is damaged or only half copied - save or copy it again. A back that looks
  bumpy or dented: set *Back detail* to *Smooth*.
* **SHARP 360°: "TRELLIS could not make a usable back":** *its shape did not line up with the
  photo* - try another *Variation*, *AI attempts* 2 or *Best* back quality; photos with the whole
  person, clearly cut out, work best. *CUDA out of memory* - close games, browsers and other GPU
  programs (the built-in back is used meanwhile). TRELLIS's back colours look off: switch *Match
  colours to the photo* off or on.
* **SHARP 360°: the bag or side detail is on the wrong side:** your side photos are probably
  named the wrong way round - `right` means *you* walked to your right (you see the person's
  left side). When the outlines show it clearly both pictures are turned round (with a note);
  when they do not, a note says the naming *could not be confirmed* - rename the two files.
  **A side photo "cannot be lined up":** it (or the front photo) cuts the person off at both
  the top and the bottom; a picture cut off at one end only (feet missing) is fine.

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
     headset to connect*.) The port is TCP 8443 (the USB port 8080 is only used inside this PC and needs no
     firewall rule). If you once clicked *Cancel* on the Windows Firewall question, Windows blocks the app
     until you do one of these.
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

* **Install fails:** free up disk space (see the table in [section 8](#8-add-ons)), update the NVIDIA driver
  and retry - installs resume where they stopped. The error message and the add-on's `install.log`
  (*Add-ons* window → *Open folder*) say what went wrong.
* **"untrusted mount point" / os error 448:** close VR Photo Studio and start it again from the Start menu
  (not from the installer's last page), then retry the install.
* **"This GPU is not supported by the installed PyTorch build":** the add-on's PyTorch version has no code
  for your graphics card generation. Update to the latest VR Photo Studio and reinstall the add-on.
* **GeForce GTX 10-series (e.g. GTX 1080, 8 GB):** TRELLIS, MV-Adapter and Hunyuan3D shape generation are supported but
  slow (several minutes per object, also as the back of *SHARP 360°*) and close to the memory limit - close games, browsers with many tabs
  and other GPU programs first. Texture painting (Hunyuan3D) and TRELLIS.2 need more memory than 8 GB.
* **"CUDA out of memory":** close other GPU programs; use lower quality settings; some add-ons need more VRAM
  than your GPU has.
* **SHARP 360° with MV-Adapter:** on an 8 GB card MV-Adapter keeps part of its model in system memory (slower,
  but it fits); if it still runs out of memory the job notes it and uses TRELLIS or the built-in back. A
  note like "the generated views do not match the photo's outline" means the AI drew the person in another
  shape or place - try another *Variation*, a short *Describe the hidden side*, or *Best* (two attempts).
  If *Brush polish* never runs, run **Test Brush on this PC** (Add-ons window, *Capture to 3D*) first.
* **SHARP 360° with the Qwen photo editor:** "needs at least 16 GB" - the PC has too little system memory
  (RAM) for it; use *Your own back photo* or the built-in back. "a face was found on all back views" - the
  AI painted the front onto the back of the head; try another *Variation*, more *AI attempts*, or describe
  the back (*Describe the hidden side*, e.g. "back of the head, long hair"). It is slow on GTX 10-series
  cards by nature (several minutes per view); *AI steps* 4 halves the time.
* **SHARP 360° with your own ComfyUI:** "ComfyUI is not running at ..." - start ComfyUI and check the
  address; "saved in ComfyUI's normal format" - export the workflow with *Export (API)*; "ComfyUI rejected
  the workflow" - a node or model of the workflow is missing in that ComfyUI (the note names it).

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
  Start viewer* / *Stop viewer*). It shares only the viewer page and the scenes in its library (the
  *Your scenes* tab) - not your output folder or any other files on your PC.
* VR Photo Studio is MIT-licensed. Some models and add-ons are **non-commercial only** - they are marked
  in the app and listed in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
