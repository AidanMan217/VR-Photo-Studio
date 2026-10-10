"""Exception hierarchy.  All user-facing failures derive from :class:`VRPhotoError`."""

from __future__ import annotations

# Where things are in the GUI, for error hints (keep in sync with gui/main_window.py).
# Add-ons and AI models are their own windows (toolbar buttons / Tools menu), not
# sections of the Settings dialog.
UI_ADDONS = "Add-ons window (toolbar > Add-ons)"
UI_MODELS = "AI models window (toolbar > AI models)"
UI_SETTINGS_MODELS = "Settings > AI models"  # default model picks and the Hugging Face token
UI_LOG = "Help > Log"
MODE_VIDEO = "3D Video"


class VRPhotoError(Exception):
    """Base class.  ``str(e)`` must be a human-readable message suitable for the GUI."""

    def __init__(self, message: str, *, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


class Cancelled(VRPhotoError):
    """Raised by CancelToken.check() when the user cancels an operation."""


class ModelUnavailable(VRPhotoError):
    """A model could not be found / downloaded / loaded."""


class DownloadError(ModelUnavailable):
    """Network download failed or checksum mismatch."""


class InferenceError(VRPhotoError):
    """A model failed at run time (all providers)."""


class AddonError(VRPhotoError):
    """Add-on install or execution failure."""


class AddonNotInstalled(AddonError):
    pass


class EncodeError(VRPhotoError):
    """Video encode/decode failure (ffmpeg)."""


class InputError(VRPhotoError):
    """Bad user input (unreadable image, unsupported format, invalid option)."""
