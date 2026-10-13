"""YuNet face check (segment/faces.py): API with fake detectors, the head-region logic, graceful
0.0 without the model, and - when the real model is on this PC - detection on a drawn bust."""

from __future__ import annotations

import os
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from vrphoto.core.errors import Cancelled, DownloadError, InputError
from vrphoto.segment import _runtime as rt
from vrphoto.segment import faces

#: A real YuNet file for the last test (the other tests run as if the model were missing)
REAL_OVERRIDE = os.environ.get("VRPHOTO_MODEL_YUNET", "").strip()


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.delenv("VRPHOTO_MODEL_YUNET", raising=False)
    faces.reset()
    yield
    faces.reset()


class FakeDetector:
    """Returns fixed rows (in the pixels of the picture it is given) and records its calls."""

    def __init__(self, rows=()):
        self.rows = np.asarray(rows, np.float32).reshape(len(rows), -1) if len(rows) else np.zeros((0, 15))
        self.calls: list[tuple[np.ndarray, float]] = []

    def detect(self, bgr, score_threshold):
        self.calls.append((bgr.copy(), score_threshold))
        return self.rows


def _row(x, y, w, h, score, *, landmarks=True, profile=False):
    """A 15-value YuNet row; frontal landmarks unless ``profile`` (nose beside both eyes)."""
    if not landmarks:
        return [x, y, w, h, score]
    rex, lex, ey = x + 0.3 * w, x + 0.7 * w, y + 0.4 * h
    nx = x + 0.95 * w if profile else x + 0.5 * w
    return [x, y, w, h, rex, ey, lex, ey, nx, y + 0.6 * h, x + 0.35 * w, y + 0.8 * h, x + 0.65 * w, y + 0.8 * h,
            score]


def _person_alpha(H=400, W=200):
    """Matte of an upright subject: rows 100-399, columns 50-149 (head region rows 100-190)."""
    a = np.zeros((H, W), np.float32)
    a[100:400, 50:150] = 1.0
    return a


# ------------------------------------------------------------------ find_faces / detect_faces
def test_find_faces_scales_back_and_filters():
    img = np.zeros((960, 1280, 3), np.uint8)
    img[..., 0] = 255  # red in RGB
    det = FakeDetector([_row(100, 50, 40, 40, 0.92), _row(10, 10, 20, 20, 0.7), _row(300, 60, 30, 30, 0.95)])
    out = faces.find_faces(img, detector=det)
    assert out == [pytest.approx((600.0, 120.0, 60.0, 60.0, 0.95)), pytest.approx((200.0, 100.0, 80.0, 80.0, 0.92))]
    bgr, thr = det.calls[0]
    assert bgr.shape == (480, 640, 3) and bgr.dtype == np.uint8 and thr == pytest.approx(0.8)
    assert bgr[0, 0, 2] == 255 and bgr[0, 0, 0] == 0  # handed over as BGR
    raw = faces.detect_faces(img, detector=det, score_threshold=0.5)
    assert raw.shape == (3, 15) and list(raw[:, -1]) == pytest.approx([0.95, 0.92, 0.7])
    assert raw[1, 8] == pytest.approx(2 * (100 + 0.5 * 40))  # landmarks scaled too
    # small pictures are not enlarged; max_long=0 keeps the size
    det2 = FakeDetector([_row(1, 2, 3, 4, 0.9)])
    assert faces.find_faces(img[:100, :120], detector=det2) == [pytest.approx((1.0, 2.0, 3.0, 4.0, 0.9))]
    faces.find_faces(img, detector=det2, max_long=0)
    assert det2.calls[-1][0].shape == (960, 1280, 3)


@pytest.mark.parametrize(
    "img",
    [
        np.full((40, 60), 0.5, np.float32),  # grey float
        np.full((40, 60, 4), 128, np.uint8),  # RGBA
        np.full((40, 60, 3), 128 * 257, np.uint16),
        np.full((40, 60, 3), 128.0, np.float64),  # float on 0-255
    ],
    ids=["grey-float", "rgba", "uint16", "float255"],
)
def test_input_conversion(img):
    det = FakeDetector([_row(1, 1, 5, 5, 0.9, landmarks=False)])
    assert faces.find_faces(img, detector=det) == [pytest.approx((1.0, 1.0, 5.0, 5.0, 0.9))]
    bgr = det.calls[0][0]
    assert bgr.shape == (40, 60, 3) and bgr.dtype == np.uint8 and abs(int(bgr[0, 0, 0]) - 128) <= 1


def test_bad_inputs_and_tiny_pictures():
    det = FakeDetector([_row(1, 1, 5, 5, 0.9)])
    with pytest.raises(InputError):
        faces.find_faces(np.zeros((4, 4, 2), np.uint8), detector=det)
    assert faces.find_faces(np.zeros((8, 200, 3), np.uint8), detector=det) == []  # too small to hold a face
    assert not det.calls
    with pytest.raises(InputError, match="values per face"):
        faces.find_faces(np.zeros((40, 40, 3), np.uint8), detector=FakeDetector([[1, 2, 3]]))


def test_detector_failure_is_graceful_but_cancel_propagates():
    class Broken:
        def detect(self, bgr, thr):
            raise cv2.error("boom")

    class Cancelling:
        def detect(self, bgr, thr):
            raise Cancelled("stop")

    img = np.zeros((64, 64, 3), np.uint8)
    assert faces.find_faces(img, detector=Broken()) == []
    assert faces.frontal_face_score(img, detector=Broken()) == 0.0
    with pytest.raises(Cancelled):
        faces.find_faces(img, detector=Cancelling())


# ------------------------------------------------------------------ head region / frontal score
def test_head_region():
    assert faces.head_region(_person_alpha()) == pytest.approx((50.0, 100.0, 150.0, 190.0))
    assert faces.head_region(_person_alpha(), head_frac=1.0) == pytest.approx((50.0, 100.0, 150.0, 400.0))
    assert faces.head_region(_person_alpha()[..., None] * 0.4) is None  # nothing at alpha >= 0.5
    assert faces.head_region(np.zeros((10, 10))) is None


def test_is_frontal():
    assert faces.is_frontal(_row(0, 0, 100, 100, 0.9))
    assert not faces.is_frontal(_row(0, 0, 100, 100, 0.9, profile=True))
    assert faces.is_frontal(_row(0, 0, 100, 100, 0.9, landmarks=False))  # no landmarks: counted
    narrow = _row(0, 0, 100, 100, 0.9)
    narrow[4], narrow[6] = 48.0, 52.0  # eyes 4 px apart on a 100 px face: a profile
    assert not faces.is_frontal(narrow)
    upside = _row(0, 0, 100, 100, 0.9)
    upside[9] = 10.0  # nose above the eyes
    assert not faces.is_frontal(upside)


def test_frontal_face_score_head_region_logic():
    img = np.zeros((400, 200, 3), np.uint8)
    a = _person_alpha()
    head = _row(80, 110, 40, 40, 0.93)  # centre (100, 130): in the head region, on the matte
    belly = _row(80, 250, 40, 40, 0.97)  # a face printed on a T-shirt: below the head region
    beside = _row(150, 110, 40, 40, 0.96)  # centre (170, 130): right of the subject
    off_matte = _row(52, 100, 40, 40, 0.99)  # centre in the region, box mostly off the matte
    assert faces.frontal_face_score(img, a, detector=FakeDetector([head])) == pytest.approx(0.93)
    assert faces.frontal_face_score(img, a, detector=FakeDetector([belly, beside])) == 0.0
    assert faces.frontal_face_score(img, a, detector=FakeDetector([belly, head])) == pytest.approx(0.93)
    a2 = a.copy()
    a2[100:140, 50:75] = 0.0
    off_matte = _row(40, 105, 40, 40, 0.99)  # centre (60, 125) in the region but over the cut-away corner
    assert faces.frontal_face_score(img, a2, detector=FakeDetector([off_matte])) == 0.0
    assert faces.frontal_face_score(img, a, detector=FakeDetector([off_matte])) == pytest.approx(0.99)
    # without a matte every face counts; a larger head fraction reaches the belly
    assert faces.frontal_face_score(img, detector=FakeDetector([belly, head])) == pytest.approx(0.97)
    assert faces.frontal_face_score(img, a, head_frac=0.7, detector=FakeDetector([belly])) == pytest.approx(0.97)
    # profiles are ignored unless frontal_only=False
    prof = _row(80, 110, 40, 40, 0.9, profile=True)
    assert faces.frontal_face_score(img, a, detector=FakeDetector([prof])) == 0.0
    assert faces.frontal_face_score(img, a, frontal_only=False, detector=FakeDetector([prof])) == pytest.approx(0.9)
    # min_score: weak detections are not reported (the detector is asked with that threshold too)
    det = FakeDetector([_row(80, 110, 40, 40, 0.6)])
    assert faces.frontal_face_score(img, a, detector=det) == pytest.approx(0.6)
    assert det.calls[0][1] == pytest.approx(faces.MIN_SCORE)
    assert faces.frontal_face_score(img, a, min_score=0.7, detector=det) == 0.0
    # empty matte: no head, no score
    assert faces.frontal_face_score(img, np.zeros_like(a), detector=FakeDetector([head])) == 0.0


def test_frontal_face_score_matte_formats():
    img = np.zeros((400, 200, 3), np.uint8)
    head = _row(80, 110, 40, 40, 0.93)
    small = cv2.resize(_person_alpha(), (100, 200), interpolation=cv2.INTER_NEAREST)  # half size: resized
    assert faces.frontal_face_score(img, small, detector=FakeDetector([head])) == pytest.approx(0.93)
    u8 = (_person_alpha() * 255).astype(np.uint8)  # 0-255 matte
    assert faces.frontal_face_score(img, u8, detector=FakeDetector([head])) == pytest.approx(0.93)
    assert faces.frontal_face_score(img, u8[..., None], detector=FakeDetector([head])) == pytest.approx(0.93)
    # a big picture is detected shrunk, but the head region is judged in the caller's pixels
    big = np.zeros((1600, 800, 3), np.uint8)
    a_big = cv2.resize(_person_alpha(), (800, 1600), interpolation=cv2.INTER_NEAREST)
    det = FakeDetector([_row(128, 176, 64, 64, 0.91)])  # x 2.5 = (320, 440, 160, 160) at full size
    assert faces.frontal_face_score(big, a_big, detector=det) == pytest.approx(0.91)
    assert det.calls[0][0].shape == (640, 320, 3)


# ------------------------------------------------------------------ model availability
def test_no_model_means_no_faces_and_score_zero(monkeypatch):
    from vrphoto.models import download as dl

    calls = []

    def offline(model_id, **kw):
        calls.append(model_id)
        raise DownloadError("No internet connection")

    monkeypatch.setattr(dl, "ensure_model", offline)
    img = np.zeros((64, 64, 3), np.uint8)
    assert not faces.available()
    assert faces.find_faces(img) == []
    assert faces.frontal_face_score(img, np.ones((64, 64))) == 0.0
    assert calls == ["yunet"]  # the failed download is remembered: not retried at once
    assert rt.recent_download_failure("yunet")
    assert faces.get_detector() is None and calls == ["yunet"]
    assert faces.find_faces(img, download=False) == [] and calls == ["yunet"]


def test_download_cancel_propagates(monkeypatch):
    from vrphoto.models import download as dl

    def cancelled(model_id, **kw):
        raise Cancelled("Cancelled")

    monkeypatch.setattr(dl, "ensure_model", cancelled)
    with pytest.raises(Cancelled):
        faces.frontal_face_score(np.zeros((64, 64, 3), np.uint8))


def test_without_facedetectoryn_the_check_is_skipped(monkeypatch):
    monkeypatch.delattr(cv2, "FaceDetectorYN", raising=False)
    assert not faces.supported() and not faces.available()
    assert faces.get_detector() is None
    assert faces.frontal_face_score(np.zeros((64, 64, 3), np.uint8)) == 0.0


def test_unloadable_model_file_is_skipped(monkeypatch, tmp_path):
    bad = tmp_path / "yunet.onnx"
    bad.write_bytes(b"not an onnx file")
    monkeypatch.setenv("VRPHOTO_MODEL_YUNET", str(bad))
    assert faces.available()  # the file is there ...
    assert faces.get_detector() is None  # ... but OpenCV cannot load it
    assert faces.frontal_face_score(np.zeros((64, 64, 3), np.uint8)) == 0.0
    assert faces.get_detector() is None  # remembered, not loaded again


def test_downloaded_model_path_and_shared_detector(monkeypatch, tmp_path):
    from vrphoto.models import download as dl

    made = []

    class FakeCv2Det:
        def __init__(self, *args):
            made.append(args)
            self.size = None

        def setScoreThreshold(self, t):
            self.thr = t

        def setInputSize(self, size):
            self.size = size

        def detect(self, img):
            assert img.shape[1::-1] == self.size
            return 1, np.array([[10, 10, 20, 20] + [0] * 10 + [self.thr + 0.1]], np.float32)

    monkeypatch.setattr(cv2, "FaceDetectorYN", types.SimpleNamespace(create=lambda *a: FakeCv2Det(*a)))
    model = tmp_path / "face.onnx"
    model.write_bytes(b"x")
    got = []
    monkeypatch.setattr(dl, "ensure_model", lambda model_id, cancel=None: got.append((model_id, cancel)) or model)
    token = object()
    det = faces.get_detector(cancel=token)
    assert isinstance(det, faces.YuNet) and det.path == model and got == [("yunet", token)]
    assert faces.get_detector() is det and len(made) == 1  # one shared detector
    assert made[0][0] == str(model)
    out = faces.find_faces(np.zeros((50, 70, 3), np.uint8), score_threshold=0.6)
    assert out == [pytest.approx((10.0, 10.0, 20.0, 20.0, 0.7))]


def test_yunet_falls_back_to_its_fixed_input_size(monkeypatch, tmp_path):
    sizes = []

    class FixedOnly:
        def setScoreThreshold(self, t):
            pass

        def setInputSize(self, size):
            self.size = size

        def detect(self, img):
            sizes.append(self.size)
            if self.size != (faces.FIXED_SIZE, faces.FIXED_SIZE):
                raise cv2.error("Shape mismatch: the graph wants 640 x 640")
            assert img.shape == (faces.FIXED_SIZE, faces.FIXED_SIZE, 3)
            assert int(img[-1, -1, 0]) == 127  # padded bottom / right: coordinates unchanged
            return 1, np.array([[5, 6, 7, 8] + [0] * 10 + [0.9]], np.float32)

    monkeypatch.setattr(cv2, "FaceDetectorYN", types.SimpleNamespace(create=lambda *a: FixedOnly()))
    det = faces.YuNet(tmp_path / "m.onnx")
    rows = det.detect(np.zeros((300, 200, 3), np.uint8), 0.8)
    assert rows.shape == (1, 15) and rows[0, 0] == 5
    det.detect(np.zeros((100, 120, 3), np.uint8), 0.8)
    assert sizes == [(200, 300), (640, 640), (640, 640)]  # tried once, then always the fixed size
    big = det.detect(np.zeros((1280, 200, 3), np.uint8), 0.8)  # shrunk by 2 to fit, coordinates back
    assert sizes[-1] == (640, 640) and big[0, :4].tolist() == [10, 12, 14, 16] and big[0, -1] == pytest.approx(0.9)


def test_registry_entry():
    from vrphoto.models.registry import file_url, get_spec

    spec = get_spec(faces.MODEL_ID)
    assert spec.license == "MIT" and spec.pinned and spec.size_mb < 1
    assert file_url(spec, spec.files[0]) == spec.url and spec.revision in spec.url


# ------------------------------------------------------------------ the real model (when available)
def _drawn_bust(face: bool, size: int = 512, head_px: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """RGB picture + matte of a cartoon bust on grey: a frontal face, or the back of the head."""
    img = np.full((size, size, 3), 200, np.uint8)
    alpha = np.zeros((size, size), np.float32)
    c, hr = size // 2, head_px // 2
    hy = int(size * 0.12) + int(hr * 1.3)
    body = np.array([[c - int(hr * 2.2), size - 1], [c - int(hr * 1.9), hy + int(hr * 1.9)],
                     [c + int(hr * 1.9), hy + int(hr * 1.9)], [c + int(hr * 2.2), size - 1]], np.int32)
    for target, colour in ((img, (60, 70, 140)), (alpha, 1.0)):
        cv2.fillPoly(target, [body], colour)
        cv2.rectangle(target, (c - hr // 3, hy + hr), (c + hr // 3, hy + 2 * hr), (225, 180, 150), -1)
        cv2.ellipse(target, (c, hy - int(hr * 0.15)), (int(hr * 1.25), int(hr * 1.45)), 0, 0, 360,
                    (40, 30, 30) if target is img else 1.0, -1, cv2.LINE_AA)
    alpha = np.clip(alpha, 0.0, 1.0)
    if face:
        s = 2 * hr
        cv2.ellipse(img, (c, hy), (int(hr * 0.95), int(hr * 1.2)), 0, 0, 360, (225, 180, 150), -1, cv2.LINE_AA)
        ey, ex = int(hy - s * 0.06), int(s * 0.18)
        for sx in (-1, 1):
            cv2.ellipse(img, (c + sx * ex, ey), (int(s * 0.09), int(s * 0.045)), 0, 0, 360, (255, 255, 255), -1,
                        cv2.LINE_AA)
            cv2.circle(img, (c + sx * ex, ey), max(1, int(s * 0.036)), (20, 30, 40), -1, cv2.LINE_AA)
            cv2.line(img, (c + sx * ex - int(s * 0.1), ey - int(s * 0.1)),
                     (c + sx * ex + int(s * 0.1), ey - int(s * 0.11)), (50, 40, 40), max(1, s // 40), cv2.LINE_AA)
        nose = np.array([[c, ey + s * 0.04], [c - s * 0.06, hy + s * 0.16], [c + s * 0.06, hy + s * 0.16]], np.int32)
        cv2.polylines(img, [nose], False, (190, 140, 110), max(1, s // 50), cv2.LINE_AA)
        cv2.ellipse(img, (c, int(hy + s * 0.32)), (int(s * 0.14), int(s * 0.05)), 0, 0, 180, (180, 80, 80),
                    max(2, s // 30), cv2.LINE_AA)
    return cv2.GaussianBlur(img, (0, 0), 0.8), alpha


def _why_no_yunet() -> str:
    """The download / load error ``faces.get_detector`` logged and swallowed, for the failure message."""
    from vrphoto.models import download as dl

    try:
        faces.YuNet(dl.ensure_model(faces.MODEL_ID))
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return "no error on a second try (see the log)"


@pytest.fixture
def real_yunet(monkeypatch):
    """The real model: ``VRPHOTO_MODEL_YUNET`` (a local file), else downloaded in the CI models
    job (``VRPHOTO_REAL_MODELS=1``) into its shared cache.  Skipped only when neither is set:
    once a real model is asked for, a model that cannot be downloaded or loaded FAILS the test
    (the app's face check would silently find nothing on every user's PC)."""
    override = REAL_OVERRIDE
    if override:
        if not Path(override).is_file():
            pytest.fail(f"VRPHOTO_MODEL_YUNET points at {override}, which is not a file")
        monkeypatch.setenv("VRPHOTO_MODEL_YUNET", override)
    elif os.environ.get("VRPHOTO_REAL_MODELS") == "1":
        home = Path(os.environ.get("VRPHOTO_REAL_MODELS_HOME", Path.home() / ".cache" / "vrphoto-test-models"))
        home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("VRPHOTO_HOME", str(home))
        from vrphoto.core import config

        config.set_settings(None)
    else:
        pytest.skip("set VRPHOTO_MODEL_YUNET to the YuNet ONNX file, or VRPHOTO_REAL_MODELS=1")
    if not faces.supported():
        pytest.fail(f"this OpenCV ({cv2.__version__}) has no FaceDetectorYN, so the app has no face check")
    faces.reset()
    det = faces.get_detector()
    if det is None:
        pytest.fail(f"the YuNet model could not be loaded or downloaded: {_why_no_yunet()}")
    yield det
    from vrphoto.core import config

    config.set_settings(None)


@pytest.mark.models
def test_real_yunet_flags_a_face_on_the_back_view(real_yunet):
    img, alpha = _drawn_bust(face=True)
    found = faces.find_faces(img)
    assert len(found) == 1 and found[0][4] > faces.REJECT_SCORE
    x, y, w, h, _ = found[0]
    region = faces.head_region(alpha)
    assert region[0] <= x + w / 2 < region[2] and region[1] <= y + h / 2 < region[3]
    assert faces.frontal_face_score(img, alpha) > faces.REJECT_SCORE
    back, back_alpha = _drawn_bust(face=False)
    assert faces.find_faces(back) == []
    assert faces.frontal_face_score(back, back_alpha) == 0.0
    # the same face, but in a big picture (shrunk to 640 px for the detector)
    big = cv2.resize(img, (1536, 1536), interpolation=cv2.INTER_CUBIC)
    assert faces.frontal_face_score(big, cv2.resize(alpha, (1536, 1536))) > faces.REJECT_SCORE
