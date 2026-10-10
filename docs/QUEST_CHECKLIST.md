# VR Photo Studio - PC + Quest 3 test checklist

This checklist walks through every important feature on **your** PC and Quest 3. It takes about
1-2 hours without add-ons (add-ons add download time). Tick each box, write short notes, and send
the filled-in [report](#report-template) back (copy this file, or paste the report section into an
issue).

**Result codes:** ✅ works · ⚠️ works with problems (describe) · ❌ fails (describe) · ⏭ skipped

**Before you start**

* [ ] PC and Quest 3 on the same home network (PC may be on a cable).
* [ ] Quest 3 charged, room/space setup done (*Settings → Physical space → Space setup*).
* [ ] A USB-C cable that transfers data.
* [ ] Test pictures in a folder, for example:
  * **P1** a photo of a person standing, full body visible (phone photo, JPEG)
  * **P2** a landscape or street photo with clear foreground and background
  * **P3** an iPhone/Android **HEIC** photo (any)
  * **P4** a 360° photo from a 360 camera (2:1 shape), if you have one
  * **V1** a 10-20 s phone video (1080p)
  * **C1** (optional, for the Capture add-on) 40-120 photos or a 30-60 s video walking around a
    still person or object - see [User Guide section 9](USER_GUIDE.md#9-how-to-capture-a-person-or-object-for-capture-mode)

---

## A. Install

| ID | Steps | Expected | Result | Notes |
|---|---|---|---|---|
| A1 | Download `VRPhotoStudio-Setup-<version>.exe` from GitHub Releases and run it | SmartScreen warning appears; **More info → Run anyway** starts the installer | | |
| A2 | Choose **Install for me only** (default). Click through (license page shows MIT + third-party notices) | Installs without asking for an administrator password; into `%LOCALAPPDATA%\Programs\VR Photo Studio` | | |
| A3 | Finish with "Launch VR Photo Studio" ticked | The app window opens within ~15 s | | |
| A4 | Start menu: search "VR Photo Studio" | Shortcut exists and starts the app | | |
| A5 | (Optional) Run the installer again and choose **Install for all users**; tick **Allow Quest headset to connect** | Windows asks for permission; install completes; a firewall rule named "VR Photo Studio" for TCP port **8443** exists (*Windows Defender Firewall → Advanced settings → Inbound Rules*) | | |
| A6 | Open a Command Prompt in the install folder and run `vrphoto-cli.exe --selftest` | Prints a health report ending in OK | | paste the output in the report |

## B. First start

| ID | Steps | Expected | Result | Notes |
|---|---|---|---|---|
| B1 | Open **Settings** (toolbar) | *Device* lists your NVIDIA GPU; *Headset* can be set to **Meta Quest 3 / 3S** | | GPU name: |
| B2 | Drag **P2** into the window | The photo appears; a preview/thumbnail is shown | | |
| B3 | Mode **Depth Map**, click **Create Depth Map** | First time: model download progress (~100 MB), then a depth preview (near = bright); **File → Open output folder** shows the files (default `Pictures\VR Photo Studio`) | | time: |
| B4 | Drag **P3** (HEIC) in and make a depth map | HEIC opens correctly (not rotated, correct colours) | | |
| B5 | Close and reopen the app | Settings are kept; no new model download | | |

## C. Modes on the PC

For each, use the default settings first. Write the processing time.

| ID | Steps | Expected | Result | Notes / time |
|---|---|---|---|---|
| C1 | **3D Photo** with P2, layout side-by-side | `..._LR.jpg` created; two slightly different images side by side | | |
| C2 | **3D Photo** with P2, layout *Anaglyph* (if you have red/cyan glasses) | Red/cyan image with visible depth through the glasses | | |
| C3 | **Living Photo** with P2, camera movement *Circle*, output *Flat video* | `..._parallax.mp4` (H.264, at most 1920 px wide) plays on the PC when double-clicked (Windows' own player); smooth loop, no big smears | | |
| C4 | **Living Photo** with P2, output *VR180 3D video* | `..._parallax_180_LR.mp4` created | | |
| C5 | **VR180** with P1 | `..._180_LR.jpg` created; edges filled with a soft blur | | |
| C6 | **360** with P4 (if available) | `..._360_TB.jpg` created (two 360 images on top of each other) | | |
| C7 | **360** with P2 (normal photo) | `..._360.jpg` created | | |
| C8 | **3D Scene** with P2 | Finishes; the scene appears in the viewer library; `.glb` file created | | |
| C9 | **3D Scene** with P1, *Look behind: Clean plate* | Finishes | | |
| C10 | **Person / Object** with P1, method *Billboard*, *Real-world height* = person's height | Finishes; object in the viewer library | | |
| C11 | **3D Video** with V1, output *Side-by-side 3D* | `..._LR.mp4` with sound; plays on the PC | | |
| C12 | Queue: select 3 photos, mode 3D Photo, **Create**; click **Cancel** on the 2nd job while it runs | Jobs 1 and 3 finish, job 2 shows "cancelled"; the app stays responsive | | |
| C13 | Open the log: `%LOCALAPPDATA%\VRPhotoStudio\logs\vrphoto.log` | Exists; no "Traceback" errors for the steps above | | |

## D. Files on the Quest (USB)

| ID | Steps | Expected | Result | Notes |
|---|---|---|---|---|
| D1 | Connect the Quest by USB, put it on, answer **Allow** to "Allow access to data" | **Quest 3** appears in File Explorer | | |
| D2 | Copy the files from C1, C3, C4, C5, C6, C11 into **Quest 3 → Internal shared storage → Movies** | Copy succeeds | | |
| D3 | In the Quest's built-in media app (Files / Meta Quest TV), open the C11 `_LR.mp4` | Plays as **3D** on a flat screen (not two pictures side by side) | | |
| D4 | Built-in app: open the C4 `_180_LR.mp4` | Plays as **VR180 3D** around you | | |
| D5 | Built-in app: open the C5 `_180_LR.jpg` and C6 `_360_TB.jpg` | Shown as 180 / 360 (note if the app cannot show 3D photos) | | |
| D6 | (If installed) Skybox / DeoVR / HereSphere: open C1 `_LR.jpg`, C4, C6 | Formats detected automatically | | which player: |
| D7 | C3 flat parallax video in any player | Plays smoothly | | |
| D8 | (Optional) Select the C1 job in the queue, open **Send to Quest → USB cable** and click **Copy files to Quest** | The file appears in the Quest's Movies folder | | |

## E. Live viewer over Wi-Fi

| ID | Steps | Expected | Result | Notes |
|---|---|---|---|---|
| E1 | Click **Send to Quest** | The viewer starts; address `https://<PC address>:8443/` and a QR code are shown; Windows may ask about the firewall → allow on **Private** networks. The *Firewall* line says the port is open (if not: click **Allow through firewall**) | | address: |
| E2 | Click **Open on this PC** | The gallery opens in your PC's browser (no warning - it uses the local `http://localhost:8080/` address). If you type the `https://` address into the PC's browser instead, accept the certificate warning first | | |
| E3 | In the **Quest Browser** type the address | "Your connection is not private" → **Advanced → Proceed** → gallery with thumbnails | | |
| E4 | Open the C8 **3D Scene**, press **Enter VR** | The photo appears as 3D in front of you; leaning left/right/forward shows depth; background behind the foreground is filled (no big holes) | | comfort 1-5: |
| E5 | Open the C10 **Person / Object** (billboard), press **Enter AR (passthrough)** | Your room is visible; the person can be placed on the floor at life size (aim at the floor, pull the trigger); you can walk around it | | |
| E6 | Open C5 (VR180) and C6 (360) in the viewer | Shown around you in 3D | | |
| E7 | Open C1 (3D photo) in the viewer | Shown as a 3D screen/window | | |
| E8 | Leave VR, go back to the gallery, open another scene | Works without reloading the page | | |
| E9 | In the app click **Stop viewer** (Send to Quest window), reload the page in the Quest Browser | Page no longer loads (server stopped) | | |
| E10 | Performance: in E4/E5, does it feel smooth (no stutter)? | Smooth | | |
| E11 | Start the viewer again; in the queue select the C8 job and click **View in headset** | The window shows the address of that scene | | |

## F. Live viewer over USB (optional, needs Developer Mode + adb)

| ID | Steps | Expected | Result | Notes |
|---|---|---|---|---|
| F1 | Enable Developer Mode ([User Guide](USER_GUIDE.md#over-a-usb-cable-adb-mode)); in **Send to Quest → USB cable** click **Install USB tools**; connect USB; in the headset allow USB debugging | The *USB cable* tab reports the Quest as connected (click **Check again** if needed) | | |
| F2 | Click **Set up USB connection** | The page opens in the Quest Browser at `http://localhost:8080/` | | |
| F3 | In the Quest Browser open `http://localhost:8080/` | Gallery appears **without** a certificate warning; Enter VR works | | |

## G. Add-ons (optional; NVIDIA GPU)

| ID | Steps | Expected | Result | Notes / time / disk |
|---|---|---|---|---|
| G1 | Open **Add-ons**, tick *Show experimental add-ons* | List shows Capture to 3D, TRELLIS, AI outpainting (SDXL) and the experimental SHARP, TRELLIS.2, Hunyuan3D-2, with GPU/disk needs and licenses; non-commercial ones are marked | | |
| G2 | Install **Capture to 3D** | Install finishes; status "installed" | | |
| G3 | **Capture** with C1 | A splat object appears in the viewer library; in the Quest (E-steps) it looks photo-real and can be walked around in AR | | |
| G4 | Install **TRELLIS** (accept the non-commercial notice) | Install finishes | | |
| G5 | **Person / Object** with P1, method *TRELLIS* | A full 3D model including the back; viewable in AR | | |
| G6 | Install **SHARP** (experimental), run **Person / Object** with P1, method *SHARP* | Finishes; splat viewable in VR/AR | | |
| G7 | Install **AI outpainting for VR180 / 360**, run **VR180** with P2 and *Fill the edges: AI outpaint* | Edges filled with plausible scenery | | |
| G8 | Uninstall one add-on | Disk space is freed; status "not installed" | | |

## H. Uninstall

| ID | Steps | Expected | Result | Notes |
|---|---|---|---|---|
| H1 | *Settings → Apps → VR Photo Studio → Uninstall* | Program removed; you are **asked** whether to delete models/add-ons/settings (answer **No** the first time) | | |
| H2 | Check `%LOCALAPPDATA%\VRPhotoStudio` and your output folder (default `Pictures\VR Photo Studio`) | Both still exist (your creations are never deleted) | | |
| H3 | Reinstall | Works; models are not downloaded again | | |

---

## Report template

Copy this block into a GitHub issue or a message and fill it in.

```text
VR Photo Studio test report
===========================
App version:            (Help/About, or installer file name)
Windows version:        (Settings > System > About, e.g. Windows 11 24H2)
GPU and driver:         (e.g. RTX 4070 12 GB, driver 581.xx)
RAM:                    
Quest model / version:  (Quest 3, Settings > System > Software update)
Network:                (PC on cable/Wi-Fi; router model if known)

Results (ID: code - notes):
A1:   A2:   A3:   A4:   A5:   A6:
B1:   B2:   B3:   B4:   B5:
C1:   C2:   C3:   C4:   C5:   C6:   C7:   C8:   C9:   C10:   C11:   C12:   C13:
D1:   D2:   D3:   D4:   D5:   D6:   D7:   D8:
E1:   E2:   E3:   E4:   E5:   E6:   E7:   E8:   E9:   E10:   E11:
F1:   F2:   F3:
G1:   G2:   G3:   G4:   G5:   G6:   G7:   G8:
H1:   H2:   H3:

Processing times (C-steps):

What looked best / worst:

Output of "vrphoto-cli.exe --selftest":

Attach: %LOCALAPPDATA%\VRPhotoStudio\logs\vrphoto.log  (+ screenshots/photos through the headset if something looks wrong)
```
