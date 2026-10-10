"""Exception hierarchy.  All user-facing failures derive from :class:`VRPhotoError`."""

from __future__ import annotations


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
