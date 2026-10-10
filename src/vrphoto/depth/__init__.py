"""vrphoto.depth: monocular depth estimation and depth-map post-processing.

``get_estimator(name)`` returns an estimator whose ``estimate(img)`` yields a
:class:`~vrphoto.core.types.DepthResult` (disparity in [0, 1], 1 = near) at the input
resolution.  Importing this package is cheap: models load on first use.
"""

from .base import DepthEstimator, get_estimator, list_estimators

__all__ = ["DepthEstimator", "get_estimator", "list_estimators"]
