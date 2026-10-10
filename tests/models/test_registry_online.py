"""Online checks of the model registry (CI 'models' job only: VRPHOTO_REAL_MODELS=1).

Every file the app may download must exist upstream (catches renamed / guessed file names
such as the TODO entries for the community exports) and, when the registry pins a size,
the server must report that size.  Only the first byte of each file is fetched.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request

import pytest

from vrphoto.models.download import USER_AGENT, _build_opener, variant_files
from vrphoto.models.registry import MODELS, ModelSpec, file_url

pytestmark = [
    pytest.mark.models,
    pytest.mark.network,
    pytest.mark.skipif(os.environ.get("VRPHOTO_REAL_MODELS") != "1", reason="set VRPHOTO_REAL_MODELS=1"),
]


def _probe(url: str, token: str | None) -> tuple[int, int | None]:
    """GET the first byte only (HEAD is refused by some signed CDN URLs); returns
    ``(status, total size or None)``."""
    headers = {"User-Agent": USER_AGENT, "Range": "bytes=0-0", "Accept-Encoding": "identity"}
    if token and "huggingface.co" in url:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with _build_opener().open(req, timeout=30) as resp:  # drops the token on cross-host redirects
            total = (resp.headers.get("Content-Range") or "").rpartition("/")[2]
            if not total.isdigit():
                total = resp.headers.get("Content-Length") or ""
            return resp.status, int(total) if total.isdigit() else None
    except urllib.error.HTTPError as e:
        return e.code, None


def _all_files(spec: ModelSpec) -> list[str]:
    files = list(variant_files(spec, "default"))
    if spec.fp16_file:
        files += [f for f in variant_files(spec, "fp16") if f not in files]
    return files


@pytest.mark.parametrize("model_id", sorted(MODELS))
def test_every_registry_file_exists_upstream(model_id):
    spec = MODELS[model_id]
    token = os.environ.get("HF_TOKEN")
    for f in _all_files(spec):
        status, size = _probe(file_url(spec, f), token)
        assert status < 400, f"{model_id}: {f} -> HTTP {status}"
        want = spec.sizes.get(f)
        if want is not None and size is not None:
            assert size == want, f"{model_id}: {f} is {size} bytes upstream, lock says {want}"
