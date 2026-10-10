"""How every job option is presented in the GUI: labels, plain-language tooltips, widget
kinds, ranges and whether an option is "advanced" (collapsed by default).

The option panels are built generically from the job dataclasses in
:mod:`vrphoto.pipeline.jobs` plus the tables in this module:

* :data:`JOB_META` - ``kind -> field name -> FieldMeta`` for every job-specific field.
* :data:`COMMON_META` - ``field name -> FieldMeta`` for :class:`CommonOptions`.
* :data:`MODES` - the "What do you want to make?" cards (title, description, inputs).
* :data:`PRESETS` - "Comfortable" / "Strong 3D" / "Max quality" per mode.

A unit test fails if a dataclass field has no metadata, so adding a job option means
adding a row here.  This module has no Qt dependency.

Numeric conventions: ``minimum``/``maximum``/``step`` are in DISPLAY units; the stored
(dataclass) value is ``display / scale`` (e.g. ``ipd_m`` is stored in metres and shown in
millimetres with ``scale=1000``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Widget kinds understood by :mod:`vrphoto.gui.panels.options`.
KINDS = ("slider", "spin", "combo", "check", "file", "folder", "text", "json")

#: Fields of :class:`JobBase` that are not options (set from the input list / Common section).
STRUCTURAL_FIELDS = ("inputs", "common")


@dataclass(frozen=True)
class FieldMeta:
    """Presentation of one dataclass field.

    Attributes:
        label: short label shown next to the widget.
        tooltip: plain-language explanation (shown on hover and as a "?" hint).
        kind: one of :data:`KINDS`.
        minimum / maximum / step / decimals: numeric range in display units.
        scale: display value = stored value * scale.
        unit: suffix shown after numbers ("mm", "%", "m", ...).
        choices: ``(value, human label)`` pairs for combos.
        choices_from: name of a dynamic choice provider ("depth_models", "inpaint_models",
            "segment_models", "devices") - see ``panels.options.dynamic_choices``.
        experimental_choices: values only offered when experimental features are shown.
        advanced: collapsed under the "Advanced" section.
        none_label: for optional (``X | None``) fields, the text representing ``None``.
        zero_label: text shown instead of the minimum value (e.g. "Off" for 0 seconds).
        placeholder: grey hint text for text/path fields.
        enabled_if: ``(sibling field, allowed values)`` - widget is greyed out otherwise.
        only_modes: Common fields only - job kinds where the field is shown (None = all).
        addon_choices: ``value -> add-on id candidates`` for choices that need an add-on.
    """

    label: str
    tooltip: str
    kind: str
    minimum: float | None = None
    maximum: float | None = None
    step: float = 1.0
    decimals: int = 0
    scale: float = 1.0
    unit: str = ""
    choices: tuple[tuple[Any, str], ...] = ()
    choices_from: str | None = None
    experimental_choices: frozenset = frozenset()
    advanced: bool = False
    none_label: str | None = None
    zero_label: str | None = None
    placeholder: str = ""
    enabled_if: tuple[str, tuple[Any, ...]] | None = None
    only_modes: frozenset | None = None
    addon_choices: dict[Any, tuple[str, ...]] = field(default_factory=dict)

    def __hash__(self) -> int:  # dict field is unhashable; labels are unique enough
        return hash((self.label, self.kind))

    def choice_values(self) -> list[Any]:
        return [v for v, _ in self.choices]


# --------------------------------------------------------------------------- shared rows
def _ipd(default_mm: int = 63, advanced: bool = True, enabled_if=None) -> FieldMeta:
    return FieldMeta(
        "Eye distance (IPD)",
        "Distance between the two virtual eyes. 63 mm is an average adult; "
        f"{default_mm} mm is recommended here. Smaller values make the scene feel bigger "
        "and the 3D softer.",
        "slider",
        minimum=50,
        maximum=75,
        step=0.5,
        decimals=1,
        scale=1000,
        unit="mm",
        advanced=advanced,
        enabled_if=enabled_if,
    )


_LOOK_BEHIND = FieldMeta(
    "Look behind",
    "When the viewpoint moves you would see behind people and objects, where the photo has "
    "no information. 'On' lets the AI paint a believable background there (cleaner "
    "outlines, a little slower). 'Off' is faster but edges may look stretched.",
    "combo",
    choices=(("band", "On - AI paints behind edges"), ("none", "Off (faster)")),
)

_IMAGE_FORMAT = FieldMeta(
    "Image format",
    "JPEG files are small and open everywhere. PNG is lossless but much bigger.",
    "combo",
    choices=(("jpg", "JPEG (smaller)"), ("png", "PNG (lossless)")),
    advanced=True,
)

_ENCODER_PRESET = FieldMeta(
    "Video format",
    "'Automatic' picks the right format for where the video is watched: flat videos are saved "
    "as H.264, which plays on phones and PCs; 3D and VR180 videos use HEVC sized for your "
    "headset (Quest 3 or Quest 2). Choose 'Compatible (H.264)' if a video does not play "
    "somewhere, or 'Quick preview' for a small, fast test file.",
    "combo",
    choices=(
        ("auto", "Automatic (recommended)"),
        ("quest3_hevc", "Meta Quest 3 (HEVC, high quality)"),
        ("quest2_hevc", "Meta Quest 2 (HEVC)"),
        ("h264_compat", "Compatible (H.264, plays everywhere)"),
        ("preview", "Quick preview (small, fast)"),
    ),
    advanced=True,
)


def _as_video(advanced: bool = False) -> FieldMeta:
    return FieldMeta(
        "Also save as video",
        "Also writes a short still video of this length, for headset players that cannot open "
        "VR photos. 'Off' saves only the photo.",
        "slider",
        minimum=0,
        maximum=30,
        step=1,
        unit="s",
        zero_label="Off",
        advanced=advanced,
    )


_FILL_CHOICES = (
    ("blur", "Soft blur (extends the photo)"),
    ("black", "Black"),
    ("outpaint", "AI outpaint (needs the 360 outpaint add-on)"),
)
_OUTPAINT_ADDONS = ("sdxl360",)

_STRENGTH = FieldMeta(
    "3D strength",
    "Multiplies the depth effect. Lower is softer and more comfortable, higher gives more "
    "'pop'. 1.0 is natural; 0.6-1.4 works for most photos.",
    "slider",
    minimum=0.2,
    maximum=2.0,
    step=0.05,
    decimals=2,
    unit="×",
)


def _max_parallax(default_hint: str) -> FieldMeta:
    return FieldMeta(
        "Max pop-out",
        "Comfort limit for how far things may stick out of (or sink into) the screen, as a "
        f"percentage of the image width. The app reduces the depth automatically if needed. "
        f"Recommended: {default_hint}.",
        "slider",
        minimum=0.5,
        maximum=8.0,
        step=0.1,
        decimals=1,
        unit="%",
        advanced=True,
    )


# --------------------------------------------------------------------------- Common options
COMMON_META: dict[str, FieldMeta] = {
    "depth_model": FieldMeta(
        "Depth model",
        "Which AI guesses how far away everything in the photo is - everything else is built "
        "from this. 'Automatic' (Depth Anything V2 Small) is fast and free for any use. "
        "MoGe-2 knows real distances and the lens angle (great for 3D Scenes and VR). "
        "V2 Base/Large are sharper but for non-commercial use only. Models download on first use.",
        "combo",
        choices_from="depth_models",
        experimental_choices=frozenset({"da3"}),
    ),
    "inpaint_model": FieldMeta(
        "Look-behind model",
        "The AI that paints hidden areas (behind people and objects) when the viewpoint moves. "
        "'Automatic' uses LaMa (best quality); MI-GAN is faster; OpenCV needs no download.",
        "combo",
        choices_from="inpaint_models",
    ),
    "segment_model": FieldMeta(
        "Cut-out model",
        "The AI that separates a person or object from the background. 'Automatic' uses "
        "BiRefNet Lite; 'BiRefNet Portrait' is best for people and hair; GrabCut needs no download.",
        "combo",
        choices_from="segment_models",
        only_modes=frozenset({"object", "scene"}),
    ),
    "device": FieldMeta(
        "Processing device",
        "Which graphics card runs the AI. 'Automatic' uses your fastest GPU. Choose 'CPU only' "
        "if the GPU causes problems (much slower).",
        "combo",
        choices_from="devices",
        advanced=True,
    ),
    "hfov_deg": FieldMeta(
        "Field of view",
        "The camera's horizontal angle of view. 'Auto' takes it from MoGe-2, then from the "
        "photo's EXIF data, else 65°. Set it if you know it (phone main camera ≈ 70°, "
        "0.5× ultra-wide ≈ 110°).",
        "spin",
        minimum=20,
        maximum=170,
        step=1,
        decimals=0,
        unit="°",
        none_label="Auto",
        advanced=True,
    ),
    "near_m": FieldMeta(
        "Nearest distance",
        "Rough distance of the nearest thing in the photo, used when the depth model does not "
        "know real distances. Smaller values give stronger depth up close.",
        "spin",
        minimum=0.1,
        maximum=50,
        step=0.1,
        decimals=1,
        unit="m",
        advanced=True,
    ),
    "far_m": FieldMeta(
        "Farthest distance",
        "Rough distance of the farthest thing in the photo (the background), used when the "
        "depth model does not know real distances.",
        "spin",
        minimum=1,
        maximum=1000,
        step=1,
        decimals=0,
        unit="m",
        advanced=True,
    ),
    "depth_refine": FieldMeta(
        "Depth smoothing",
        "Edge-aware smoothing of the depth map. Lower keeps fine detail, higher removes noise "
        "and blotches. 1.0 suits most photos.",
        "slider",
        minimum=0,
        maximum=2,
        step=0.1,
        decimals=1,
        advanced=True,
    ),
    "max_process_px": FieldMeta(
        "Max processing size",
        "Photos larger than this (long side, in pixels) are reduced while processing. Lower "
        "it (e.g. 2048) if you run out of memory; higher keeps more detail but is slower.",
        "spin",
        minimum=512,
        maximum=16384,
        step=256,
        unit="px",
        advanced=True,
    ),
    "output_dir": FieldMeta(
        "Output folder",
        "Where the results are saved. Leave empty to use the default folder from Settings "
        "(Pictures\\VR Photo Studio).",
        "folder",
        placeholder="Default folder (see Settings)",
    ),
    "name": FieldMeta(
        "Output name",
        "Base name of the result files. Leave empty to use each photo's own name. Endings like "
        "_LR or _180_LR are added automatically so headset players recognise the format.",
        "text",
        placeholder="Same as the photo",
        advanced=True,
    ),
    "export_scene": FieldMeta(
        "Add to viewer library",
        "Also make the result available in the live viewer, so you can see it in your Quest "
        "with 'Send to Quest' without copying files.",
        "check",
        only_modes=frozenset({"stereo", "parallax", "vr180", "pano", "scene", "object", "capture", "video"}),
    ),
    "quest_model": FieldMeta(
        "Headset",
        "Picks resolutions and video formats your headset plays smoothly.",
        "combo",
        choices=(("quest3", "Meta Quest 3 / 3S"), ("quest2", "Meta Quest 2"), ("pcvr", "PC VR / other")),
        advanced=True,
    ),
}


# --------------------------------------------------------------------------- per job type
JOB_META: dict[str, dict[str, FieldMeta]] = {
    "depth": {
        "save_color": FieldMeta(
            "Save colour preview",
            "Also save a colourful picture of the depth map (warm = near, cool = far).",
            "check",
        ),
        "save_16bit": FieldMeta(
            "Save 16-bit PNG",
            "Save the precise depth map as a 16-bit grey PNG (white = near) for Blender, "
            "displacement maps and other 3D tools.",
            "check",
        ),
    },
    "stereo": {
        "layout": FieldMeta(
            "Layout",
            "How the two eye images are arranged. Side-by-side gives the best quality on the Quest. "
            "Half side-by-side is for 3D TVs; anaglyph is for red/cyan glasses; cross-eyed is for "
            "free viewing without glasses.",
            "combo",
            choices=(
                ("sbs", "Side-by-side (best for Quest)"),
                ("half_sbs", "Half side-by-side (3D TVs)"),
                ("tb", "Top-bottom"),
                ("half_tb", "Half top-bottom"),
                ("anaglyph", "Anaglyph red/cyan glasses (colour)"),
                ("anaglyph_gray", "Anaglyph red/cyan glasses (grey)"),
                ("cross", "Cross-eyed (no glasses)"),
            ),
        ),
        "ipd_m": _ipd(63),
        "strength": _STRENGTH,
        "convergence_m": FieldMeta(
            "Screen distance",
            "Which distance appears exactly at the 'screen'; nearer things pop out, farther "
            "things sink in. 'Auto' puts the main subject there.",
            "spin",
            minimum=0.3,
            maximum=50,
            step=0.1,
            decimals=1,
            unit="m",
            none_label="Auto",
            advanced=True,
        ),
        "max_parallax_pct": _max_parallax("3 %"),
        "look_behind": _LOOK_BEHIND,
        "swap_eyes": FieldMeta(
            "Swap eyes",
            "Swap the left and right images. Use this if the 3D looks 'inside out' in some player.",
            "check",
            advanced=True,
        ),
        "image_format": _IMAGE_FORMAT,
        "wiggle_gif": FieldMeta(
            "Also save a wiggle GIF",
            "Also save an animated GIF that rocks between the two eye views - it shows the 3D "
            "effect on any screen, no glasses needed.",
            "check",
        ),
    },
    "parallax": {
        "path": FieldMeta(
            "Camera movement",
            "How the virtual camera moves through your photo. Circle (a gentle float) and Swing "
            "(side to side) are classics; Dolly moves forward and back; Zoom pushes in; Ken Burns "
            "pans and zooms; Orbit arcs around the subject.",
            "combo",
            choices=(
                ("circle", "Circle (gentle float)"),
                ("swing", "Swing (side to side)"),
                ("dolly", "Dolly (forward and back)"),
                ("zoom", "Zoom (push in)"),
                ("kenburns", "Ken Burns (pan and zoom)"),
                ("orbit", "Orbit (arc around the subject)"),
                ("custom", "Custom key frames (advanced)"),
            ),
        ),
        "keyframes": FieldMeta(
            "Custom key frames",
            "Only for the 'Custom' movement: a JSON list of key frames like "
            '[{"t": 0, "x": -0.05}, {"t": 0.5, "x": 0.05, "yaw": 3}]. t runs from 0 to 1 over the '
            "video; x/y/z are offsets in metres; yaw/pitch/roll are angles in degrees.",
            "json",
            advanced=True,
            enabled_if=("path", ("custom",)),
            placeholder='[{"t": 0, "x": -0.05}, {"t": 0.5, "x": 0.05}]',
        ),
        "duration_s": FieldMeta(
            "Duration",
            "Length of the video in seconds.",
            "slider",
            minimum=2,
            maximum=30,
            step=0.5,
            decimals=1,
            unit="s",
        ),
        "fps": FieldMeta(
            "Frames per second",
            "Smoothness of the motion. 30 is standard; 60 is smoother but takes twice as long.",
            "combo",
            choices=((24.0, "24 (cinematic)"), (30.0, "30 (recommended)"), (60.0, "60 (smoothest)")),
            advanced=True,
        ),
        "amplitude_m": FieldMeta(
            "Movement",
            "How far the camera moves. More is more dramatic but stretches edges more. 3-8 cm works best.",
            "slider",
            minimum=1,
            maximum=15,
            step=0.5,
            decimals=1,
            scale=100,
            unit="cm",
        ),
        "output": FieldMeta(
            "Output",
            "Flat: a normal video for phones and PCs. Side-by-side 3D: for Quest video players and "
            "3D TVs. VR180: an immersive 3D video that surrounds you.",
            "combo",
            choices=(
                ("flat", "Flat video (phone / PC)"),
                ("sbs", "Side-by-side 3D video"),
                ("vr180", "VR180 3D video (immersive)"),
            ),
        ),
        "width": FieldMeta(
            "Width",
            "Video width in pixels. 'Auto' uses the photo's size, limited to what your headset plays "
            "smoothly; flat videos are at most 1920 px on the long side so phones and PCs play them.",
            "spin",
            minimum=480,
            maximum=8192,
            step=16,
            unit="px",
            none_label="Auto",
            advanced=True,
        ),
        "loop": FieldMeta(
            "Seamless loop",
            "The camera returns to its start so the video loops without a jump.",
            "check",
        ),
        "look_behind": _LOOK_BEHIND,
        "encoder_preset": _ENCODER_PRESET,
        "ipd_m": _ipd(63, enabled_if=("output", ("sbs", "vr180"))),
    },
    "vr180": {
        "eye_px": FieldMeta(
            "Resolution per eye",
            "Pixels per eye. 'Auto' uses your headset (Quest 3: 3840, Quest 2: 2880).",
            "spin",
            minimum=1024,
            maximum=8192,
            step=64,
            unit="px",
            none_label="Auto",
            advanced=True,
        ),
        "size_mult": FieldMeta(
            "Enlarge photo",
            "Makes the photo cover more of your view (up to 150°). 1.0 keeps its true size - like a "
            "big window in front of you; above 1 feels bigger than life.",
            "slider",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            decimals=2,
            unit="×",
        ),
        "fill": FieldMeta(
            "Fill the edges",
            "What fills the rest of the half-sphere around your photo: a soft, darkened extension "
            "of the photo, black, or new surroundings painted by AI (needs an add-on).",
            "combo",
            choices=_FILL_CHOICES,
            addon_choices={"outpaint": _OUTPAINT_ADDONS},
        ),
        "ipd_m": _ipd(63),
        "look_behind": _LOOK_BEHIND,
        "as_video_s": _as_video(),
        "image_format": _IMAGE_FORMAT,
    },
    "pano": {
        "input_is_360": FieldMeta(
            "Input is a 360° photo",
            "Photos from 360 cameras (2:1 shape, e.g. Insta360, Ricoh Theta) become 3D 360 pictures. "
            "Normal photos become a 360 sphere with the photo in front. 'Auto' decides from the shape.",
            "combo",
            choices=(
                (None, "Auto-detect (2:1 photos)"),
                (True, "Yes - from a 360 camera"),
                (False, "No - a normal photo"),
            ),
        ),
        "stereo": FieldMeta(
            "3D (stereo)",
            "For 360-camera photos: make it 3D (top-bottom stereo) so you see depth all around.",
            "check",
        ),
        "width": FieldMeta(
            "Width",
            "Width of the 360 picture in pixels. 'Auto' uses your headset's best size.",
            "spin",
            minimum=2048,
            maximum=16384,
            step=256,
            unit="px",
            none_label="Auto",
            advanced=True,
        ),
        "fill": FieldMeta(
            "Fill the surroundings",
            "For normal photos: what fills the rest of the sphere - soft blur, black, or AI outpaint (add-on).",
            "combo",
            choices=_FILL_CHOICES,
            addon_choices={"outpaint": _OUTPAINT_ADDONS},
        ),
        "ipd_m": FieldMeta(
            "Eye distance (IPD)",
            "Distance between the two virtual eyes. 64 mm is recommended for 360; lower it if very "
            "close things feel strange.",
            "slider",
            minimum=50,
            maximum=75,
            step=0.5,
            decimals=1,
            scale=1000,
            unit="mm",
            advanced=True,
        ),
        "as_video_s": _as_video(),
    },
    "scene": {
        "target_faces": FieldMeta(
            "Detail",
            "How detailed the 3D scene is: about this many thousand triangles in total (all "
            "layers together). More gives sharper edges but loads more slowly in the headset. "
            "150-500 k is a good range.",
            "slider",
            minimum=50,
            maximum=1000,
            step=10,
            scale=0.001,
            unit="k",
        ),
        "look_behind": FieldMeta(
            "Look behind",
            "Band: the AI paints a strip behind every edge. Clean plate: removes the main subject "
            "from the background completely so you can look all the way around it - best for one "
            "clear subject such as a portrait. Off: fastest.",
            "combo",
            choices=(
                ("band", "Band - paints behind every edge"),
                ("cleanplate", "Clean plate - best for one clear subject"),
                ("none", "Off (fastest)"),
            ),
        ),
        "anchor_distance_m": FieldMeta(
            "Distance",
            "How far in front of you the photo is placed in the viewer.",
            "slider",
            minimum=0.5,
            maximum=10,
            step=0.1,
            decimals=1,
            unit="m",
        ),
        "max_baseline_m": FieldMeta(
            "Movement range",
            "How far you can move your head before you reach the edges of what was prepared. "
            "Bigger needs more AI painting.",
            "slider",
            minimum=5,
            maximum=100,
            step=5,
            scale=100,
            unit="cm",
        ),
        "export_glb": FieldMeta(
            "Also save GLB file",
            "Save a .glb 3D file you can open in Blender, Windows 3D Viewer and many other apps.",
            "check",
        ),
        "export_obj": FieldMeta(
            "Also save OBJ file",
            "Save the older .obj 3D format (with textures) as well.",
            "check",
            advanced=True,
        ),
    },
    "object": {
        "method": FieldMeta(
            "Method",
            "Billboard is instant and needs no add-on: a cut-out with a rounded 3D shape that looks "
            "great from the front and sides. TRELLIS builds a full 3D model including the back "
            "(add-on, NVIDIA GPU). SHARP, TRELLIS.2 and Hunyuan3D are experimental add-ons "
            "(shown when experimental features are on in Settings).",
            "combo",
            choices=(
                ("billboard", "Billboard (instant, no add-on)"),
                ("trellis", "TRELLIS - full 3D model (add-on)"),
                ("sharp", "SHARP - fast AI splat (experimental add-on)"),
                ("trellis2", "TRELLIS.2 (experimental add-on)"),
                ("hunyuan", "Hunyuan3D (experimental add-on)"),
            ),
            experimental_choices=frozenset({"sharp", "trellis2", "hunyuan"}),
            addon_choices={
                "trellis": ("trellis",),
                "sharp": ("sharp",),
                "trellis2": ("trellis2",),
                "hunyuan": ("hunyuan",),
            },
        ),
        "height_m": FieldMeta(
            "Real-world height",
            "How tall the person or object is in real life, so it appears life-size in your room.",
            "slider",
            minimum=0.1,
            maximum=3.0,
            step=0.01,
            decimals=2,
            unit="m",
        ),
        "thickness": FieldMeta(
            "Thickness",
            "Billboard only: how deep the rounded shape is, as a fraction of the height.",
            "slider",
            minimum=0.05,
            maximum=0.5,
            step=0.01,
            decimals=2,
            enabled_if=("method", ("billboard",)),
        ),
        "back": FieldMeta(
            "Back side",
            "Billboard only: what the back looks like - soft colours taken from the outline "
            "(subtle), or a mirror image of the front.",
            "combo",
            choices=(("diffuse", "Soft colours from the outline"), ("mirror", "Mirror of the front")),
            advanced=True,
            enabled_if=("method", ("billboard",)),
        ),
        "export_splat": FieldMeta(
            "Also save splat",
            "Also save a Gaussian-splat version for the viewer.",
            "check",
        ),
        "placement": FieldMeta(
            "Placement",
            "Point at the floor: you aim at your floor or a table and press the trigger. In front "
            "of me: it appears on the floor about 1.5 m ahead.",
            "combo",
            choices=(("hit-test", "Point at the floor"), ("front-floor", "In front of me")),
        ),
        "seed": FieldMeta(
            "Variation",
            "TRELLIS, TRELLIS.2 and Hunyuan3D only: change this number to get a different version "
            "of the 3D model. (SHARP always gives the same result for the same photo.)",
            "spin",
            minimum=0,
            maximum=999999,
            step=1,
            advanced=True,
            # SHARP is deterministic and takes no seed
            enabled_if=("method", ("trellis", "trellis2", "hunyuan")),
        ),
        "hunyuan_texture": FieldMeta(
            "Paint texture",
            "Hunyuan3D only: also paint colours onto the 3D model; without it the model is a plain "
            "grey shape. Slower and needs more graphics memory. The painting models (about 8 GB) "
            "are downloaded on the first textured run, and painting needs the add-on's optional "
            "texture-painting part (see the Add-ons window) - without it you get the grey shape.",
            "check",
            # advanced: Hunyuan3D is an experimental (normally hidden) method, so in the basic
            # form this would be a checkbox that is almost always greyed out
            advanced=True,
            enabled_if=("method", ("hunyuan",)),
        ),
    },
    "capture": {
        "video_fps": FieldMeta(
            "Frames per second (video)",
            "For videos: how many frames are taken from each second. 2 is good; use 3 if you walked fast.",
            "slider",
            minimum=0.5,
            maximum=10,
            step=0.5,
            decimals=1,
        ),
        "max_images": FieldMeta(
            "Max photos",
            "Upper limit on the number of photos used; more photos take longer.",
            "spin",
            minimum=10,
            maximum=2000,
            step=10,
        ),
        "steps": FieldMeta(
            "Training steps",
            "How long the 3D model is trained. About 7,000 is a quick test, 15,000 is good and "
            "30,000 is best quality.",
            "slider",
            minimum=1000,
            maximum=50000,
            step=1000,
        ),
        "max_splats": FieldMeta(
            "Max splats",
            "Limits the size of the result so it runs smoothly in the Quest Browser.",
            "spin",
            minimum=50000,
            maximum=5000000,
            step=50000,
            advanced=True,
        ),
        "crop_pct": FieldMeta(
            "Keep",
            "Removes stray floating bits far from the centre; keeps this percentage of the points.",
            "slider",
            minimum=80,
            maximum=100,
            step=0.5,
            decimals=1,
            unit="%",
            advanced=True,
        ),
        "subject_height_m": FieldMeta(
            "Subject height",
            "Real height of the captured person or object, so it appears life-size. 'Automatic' "
            "works out the size from how high you held the camera (about chest height, 1.45 m).",
            "spin",
            minimum=0,
            maximum=5,
            step=0.05,
            decimals=2,
            unit="m",
            zero_label="Automatic",
            advanced=True,
        ),
        "as_object": FieldMeta(
            "Place as object (AR)",
            "On: centred and placed on your floor, for people and objects in passthrough AR. "
            "Off: keep the whole surroundings as a scene for VR.",
            "check",
        ),
    },
    "video": {
        "output": FieldMeta(
            "Output",
            "Side-by-side gives the best quality in Quest players; half side-by-side is for 3D TVs; "
            "VR180 makes an immersive video.",
            "combo",
            choices=(
                ("sbs", "Side-by-side 3D"),
                ("half_sbs", "Half side-by-side (3D TVs)"),
                ("tb", "Top-bottom 3D"),
                ("vr180", "VR180 (immersive)"),
            ),
        ),
        "ipd_m": _ipd(63),
        "strength": _STRENGTH,
        "max_parallax_pct": _max_parallax("2.5 % (motion adds strain)"),
        "temporal_smooth": FieldMeta(
            "Steadiness",
            "Smooths the depth between frames to avoid shimmering. Higher is steadier but reacts "
            "more slowly to scene cuts.",
            "slider",
            minimum=0,
            maximum=0.95,
            step=0.05,
            decimals=2,
        ),
        "max_long": FieldMeta(
            "Processing size",
            "Long side in pixels used while converting. Bigger is finer but slower.",
            "spin",
            minimum=480,
            maximum=7680,
            step=64,
            unit="px",
            advanced=True,
        ),
        "start_s": FieldMeta(
            "Start at",
            "Convert only part of the video: start this many seconds in.",
            "spin",
            minimum=0,
            maximum=86400,
            step=0.5,
            decimals=1,
            unit="s",
            advanced=True,
        ),
        "end_s": FieldMeta(
            "End at",
            "Stop converting at this time (seconds). 'End of video' converts to the end.",
            "spin",
            minimum=0.5,
            maximum=86400,
            step=0.5,
            decimals=1,
            unit="s",
            none_label="End of video",
            advanced=True,
        ),
        "keep_audio": FieldMeta("Keep audio", "Copy the sound from the original video.", "check"),
        "encoder_preset": _ENCODER_PRESET,
    },
}


# --------------------------------------------------------------------------- modes (cards)
@dataclass(frozen=True)
class ModeInfo:
    """A "What do you want to make?" card."""

    kind: str
    title: str
    tagline: str  # a few words shown on the card
    description: str  # one or two sentences shown above the options
    accepts: frozenset  # subset of {"image", "video"}
    batch: bool = False  # True: one job with ALL inputs (capture); False: one job per input
    needs_addon: tuple[str, ...] = ()  # add-on id candidates required for every job
    min_inputs: int = 1


MODES: tuple[ModeInfo, ...] = (
    ModeInfo(
        "stereo",
        "3D Photo",
        "Pops out in 3D",
        "A photo that pops out in 3D - for Quest players, 3D TVs or red/cyan glasses. Fastest mode.",
        frozenset({"image"}),
    ),
    ModeInfo(
        "parallax",
        "Living Photo",
        "Looping parallax video",
        "A short looping video where the camera floats gently through your photo.",
        frozenset({"image"}),
    ),
    ModeInfo(
        "vr180",
        "VR180",
        "Immersive 3D picture",
        "An immersive 3D picture that fills the view in front of you.",
        frozenset({"image"}),
    ),
    ModeInfo(
        "pano",
        "360",
        "All-around sphere",
        "3D from a 360-camera photo, or a 360 sphere around a normal photo.",
        frozenset({"image"}),
    ),
    ModeInfo(
        "scene",
        "3D Scene",
        "Walk-around photo (6DoF)",
        "A walk-around 3D photo for the live viewer: lean in and look behind things (6DoF).",
        frozenset({"image"}),
    ),
    ModeInfo(
        "object",
        "Person / Object",
        "Life-size in your room",
        "Cut out a person or object and place it life-size in your room (passthrough AR).",
        frozenset({"image"}),
    ),
    ModeInfo(
        "capture",
        "Capture",
        "Many photos → 3D splat",
        "Many photos or a video walking around something become a photo-real 3D splat (add-on).",
        frozenset({"image", "video"}),
        batch=True,
        needs_addon=("capture",),
    ),
    ModeInfo(
        "video",
        "3D Video",
        "Video → 3D or VR180",
        "Convert a normal video into a side-by-side 3D or VR180 video, frame by frame.",
        frozenset({"video"}),
    ),
    ModeInfo(
        "depth",
        "Depth Map",
        "Depth for 3D tools",
        "Save the AI depth map (16-bit PNG + colour preview) for Blender and other 3D tools.",
        frozenset({"image"}),
    ),
)

MODE_BY_KIND: dict[str, ModeInfo] = {m.kind: m for m in MODES}


# --------------------------------------------------------------------------- presets
#: Preset names in display order.
PRESET_NAMES: tuple[str, ...] = ("Comfortable", "Strong 3D", "Max quality")

#: ``kind -> preset name -> {field: value}``; ``"common.<field>"`` keys target CommonOptions.
#: Presets are applied on top of the dataclass defaults, so they are deterministic.
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "stereo": {
        "Comfortable": {"strength": 0.75, "max_parallax_pct": 2.0},
        "Strong 3D": {"strength": 1.4, "max_parallax_pct": 4.5},
        "Max quality": {
            "image_format": "png",
            "look_behind": "band",
            "common.inpaint_model": "lama",
            "common.depth_refine": 1.0,
        },
    },
    "parallax": {
        "Comfortable": {"path": "circle", "amplitude_m": 0.04, "duration_s": 8.0},
        "Strong 3D": {"path": "swing", "amplitude_m": 0.09},
        # no encoder_preset: 'auto' already uses HEVC for the headset outputs and keeps the
        # default flat (phone / PC) video playable everywhere
        "Max quality": {
            "fps": 60.0,
            "look_behind": "band",
            "common.inpaint_model": "lama",
        },
    },
    "vr180": {
        "Comfortable": {"size_mult": 1.0, "ipd_m": 0.058},
        "Strong 3D": {"size_mult": 1.3, "ipd_m": 0.068},
        "Max quality": {
            "image_format": "png",
            "look_behind": "band",
            "common.depth_model": "moge2",
            "common.inpaint_model": "lama",
        },
    },
    "pano": {
        "Comfortable": {"ipd_m": 0.056},
        "Strong 3D": {"ipd_m": 0.07},
        "Max quality": {"stereo": True, "common.inpaint_model": "lama"},
    },
    "scene": {
        "Comfortable": {"max_baseline_m": 0.25, "target_faces": 200_000},
        "Strong 3D": {"max_baseline_m": 0.6, "look_behind": "band"},
        "Max quality": {
            "target_faces": 600_000,
            "common.depth_model": "moge2",
            "common.inpaint_model": "lama",
        },
    },
    "object": {
        "Comfortable": {"thickness": 0.08},
        "Strong 3D": {"thickness": 0.24},
        "Max quality": {"common.segment_model": "birefnet-portrait", "export_splat": True},
    },
    "capture": {
        "Comfortable": {"steps": 7000, "max_splats": 500_000},
        "Strong 3D": {"steps": 15000},
        "Max quality": {"steps": 30000, "max_images": 300, "max_splats": 1_000_000},
    },
    "video": {
        "Comfortable": {"strength": 0.75, "max_parallax_pct": 1.8, "temporal_smooth": 0.75},
        "Strong 3D": {"strength": 1.3, "max_parallax_pct": 3.5},
        "Max quality": {"max_long": 2560, "temporal_smooth": 0.6, "encoder_preset": "quest3_hevc"},
    },
    "depth": {
        "Comfortable": {},
        "Strong 3D": {"common.depth_refine": 0.5},
        "Max quality": {"common.depth_model": "moge2", "common.depth_refine": 1.2},
    },
}


def field_meta(kind: str, name: str) -> FieldMeta:
    """Metadata of job field ``name`` of ``kind`` (``common.<x>`` for Common fields)."""
    if name.startswith("common."):
        return COMMON_META[name.split(".", 1)[1]]
    return JOB_META[kind][name]


def required_addon_candidates(kind: str, values: dict[str, Any]) -> tuple[str, ...]:
    """Add-on id candidates a job with these option ``values`` needs (empty if none)."""
    mode = MODE_BY_KIND.get(kind)
    cands: list[str] = list(mode.needs_addon) if mode else []
    for name, meta in JOB_META.get(kind, {}).items():
        if meta.addon_choices and values.get(name) in meta.addon_choices:
            cands.extend(meta.addon_choices[values[name]])
    return tuple(dict.fromkeys(cands))


__all__ = [
    "FieldMeta",
    "ModeInfo",
    "KINDS",
    "STRUCTURAL_FIELDS",
    "COMMON_META",
    "JOB_META",
    "MODES",
    "MODE_BY_KIND",
    "PRESETS",
    "PRESET_NAMES",
    "field_meta",
    "required_addon_candidates",
]
