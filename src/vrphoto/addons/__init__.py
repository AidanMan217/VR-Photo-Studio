"""Optional heavy add-ons (TRELLIS, TRELLIS.2, Hunyuan3D, SHARP, SDXL outpainting, capture).

* :mod:`.manager`  - list / install / uninstall / run add-ons (public API, CONTRACTS.md section 9)
* :mod:`.manifest` - TOML manifest schema and validation (``manifests/*.toml``)
* :mod:`.installer`, :mod:`.fetch` - install steps (uv venvs, pip, archives, Hugging Face)
* :mod:`.runner`, :mod:`.env` - subprocess plumbing (clean environment, cancel, protocol)
* :mod:`.capture` - photos/video -> COLMAP -> Brush -> Gaussian splat
* :mod:`.protocol` - JSON-lines protocol shared with ``workers/*.py`` (run in add-on venvs)

Nothing here imports torch / onnxruntime / PySide6; workers run in separate processes.
"""
