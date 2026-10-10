"""vrphoto.video: camera paths, frame rendering, ffmpeg encode/decode and spatial metadata.

Submodules (see docs/CONTRACTS.md section 6):

* :mod:`.paths` - :class:`CameraPath`, :func:`make_path`, :data:`PATH_KINDS`
* :mod:`.render` - :func:`render_path_frames`, :func:`render_vr180_frames`
* :mod:`.encode` - :data:`PRESETS`, :func:`ffmpeg_exe`, :func:`probe_encoders`, :class:`VideoWriter`
* :mod:`.decode` - :class:`VideoInfo`, :func:`probe`, :func:`iter_frames`
* :mod:`.spatial` - :func:`inject`, :func:`read` (360/180 + stereo MP4 metadata)

Importing this package is cheap: no subprocess is started and the geometry modules
are only imported when frames are rendered.
"""

from .decode import VideoInfo, iter_frames, probe
from .encode import PRESETS, VideoWriter, ffmpeg_exe, probe_encoders
from .paths import PATH_KINDS, CameraPath, make_path
from .render import render_path_frames, render_vr180_frames

__all__ = [
    "CameraPath",
    "make_path",
    "PATH_KINDS",
    "PRESETS",
    "ffmpeg_exe",
    "probe_encoders",
    "VideoWriter",
    "VideoInfo",
    "probe",
    "iter_frames",
    "render_path_frames",
    "render_vr180_frames",
]
