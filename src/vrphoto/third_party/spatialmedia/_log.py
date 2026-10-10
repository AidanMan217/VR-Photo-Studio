"""Logging shim for the vendored spatial-media library (added by vrphoto).

Upstream reports problems with ``print()``.  Every vendored module imports
``log_print as print`` from here so those messages go to the application log.
"""

import logging

_logger = logging.getLogger("vrphoto.third_party.spatialmedia")


def log_print(*args, **kwargs):
    """Drop-in replacement for ``print`` that logs instead of writing to stdout."""
    msg = kwargs.get("sep", " ").join(str(a) for a in args)
    low = msg.lstrip().lower()
    if low.startswith(("error", "warning", "unsupported", "unknown")):
        _logger.warning(msg)
    else:
        _logger.debug(msg)
