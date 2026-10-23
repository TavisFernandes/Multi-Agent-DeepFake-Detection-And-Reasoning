"""
Face detection / cropping (face_crop.py) and the no-face decision path.
Run: python -m pytest test_face_crop.py
Face tests use the first fusion_val RealVideo-RealAudio clip from the local FakeAVCeleb copy (via the
manifest) and skip when the dataset is not on disk. (00025_fake.mp4 is a colour-bar test pattern, not a face.)
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import face_crop
from main import _compute_verdict

_MANIFEST = Path(__file__).resolve().parent / "training" / "leakage_controlled_split_manifest.json"


@pytest.fixture(scope="module")
def face():
    clip = Path(json.loads(_MANIFEST.read_text())["clips"]["fusion_val"]["RealVideo-RealAudio"][0]["full_path"])
    if not clip.exists():
        pytest.skip("FakeAVCeleb dataset not on disk")
    cap = cv2.VideoCapture(str(clip))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
    ok, frame = cap.read()
    cap.release()
    assert ok
    return frame


def place(face, W, H, frac, cx=0.5):
    """Put the face clip at `frac` of the canvas height on a plain grey background."""
    out = np.full((H, W, 3), 110, np.uint8)
    s = int(H * frac)
    x, y = int(W * cx - s / 2), (H - s) // 2
    out[y:y + s, x:x + s] = cv2.resize(face, (s, s))
    return out, (x, y, s)


def test_native_frame_is_used_whole(face):
    box, note = face_crop.detect(face)
    assert box is not None, note
    assert face_crop.crop(face, box).shape == face.shape  # already training-framed: no resampling


def test_small_face_is_cropped_with_training_framing(face):
    img, _ = place(face, 1280, 720, 0.25)
    box, note = face_crop.detect(img)
    assert box is not None, note
    c = face_crop.crop(img, box)
    assert c.shape[0] == c.shape[1] < 720
    # Inside the crop the face must be framed like a training clip: size within the measured training range
    # (p5-p95 0.672-0.747, plus detector box jitter across scales), centre (0.50, 0.367).
    fb, _ = face_crop.detect(c)
    side = c.shape[0]
    assert 0.65 < max(fb[2], fb[3]) / side < 0.77
    assert abs((fb[0] + fb[2] / 2) / side - 0.5) < 0.04
    assert abs((fb[1] + fb[3] / 2) / side - face_crop.FACE_CY) < 0.04


def test_largest_of_several_faces_is_used(face):
    img, (x, y, s) = place(face, 1280, 720, 0.45, cx=0.3)
    small = int(720 * 0.2)
    img[300:300 + small, 950:950 + small] = cv2.resize(face, (small, small))
    box, note = face_crop.detect(img)
    assert box is not None and "2 faces found" in note, note
    assert x <= box[0] + box[2] / 2 <= x + s  # the big one


@pytest.mark.parametrize("make, reasons", [
    # 43 px tall: below MIN_FACE_PX, and often below YuNet's reach after the 640 px detection downscale.
    (lambda f: place(f, 1280, 720, 0.06)[0], ("too small", "No face detected")),
    (lambda f: np.full((720, 1280, 3), 110, np.uint8), ("No face detected",)),
    # Clip spans x 460-820: cutting at 640/680 leaves a half/two-thirds face that YuNet still "finds".
    (lambda f: place(f, 1280, 720, 0.5)[0][:, :640], ("partly outside",)),
    (lambda f: place(f, 1280, 720, 0.5)[0][:, :680], ("partly outside",)),
])
def test_unusable_faces_are_rejected(face, make, reasons):
    box, note = face_crop.detect(make(face))
    assert box is None and any(r in note for r in reasons), note


def test_agents_score_a_real_face(face, tmp_path):
    import spatial_agent, temporal_agent
    if spatial_agent._model is None:
        spatial_agent.load_model()
    p = tmp_path / "face.png"
    img, _ = place(face, 1280, 720, 0.3)  # a normal-looking upload: face 30% of a 720p frame
    cv2.imwrite(str(p), img)
    r = spatial_agent.predict(str(p))
    assert r["status"] == "trained" and isinstance(r["is_fake"], bool)
    assert r["is_fake"] == (r["confidence"] >= 0.5) and r["finding"].startswith("Face detected")
    clip = json.loads(_MANIFEST.read_text())["clips"]["fusion_val"]["RealVideo-RealAudio"][0]["full_path"]
    t = temporal_agent.run_temporal_agent(clip)
    assert t["status"] == "trained" and isinstance(t["is_fake"], bool)


def test_spatial_agent_reports_no_face_without_a_score(tmp_path):
    import spatial_agent
    if spatial_agent._model is None:
        spatial_agent.load_model()
    p = tmp_path / "blank.png"
    cv2.imwrite(str(p), np.full((480, 640, 3), 128, np.uint8))
    r = spatial_agent.predict(str(p))
    assert r["status"] == "no_face" and r["is_fake"] is None and r["confidence"] == 0.0


NO_FACE = [
    {"agent": "Spatial Analysis", "status": "no_face", "is_fake": None, "confidence": 0.0},
    {"agent": "Temporal Analysis", "status": "no_face", "is_fake": None, "confidence": 0.0},
]
REAL_AUDIO = {"agent": "Audio Analysis", "status": "trained", "is_fake": False, "confidence": 0.02, "has_audio": True, "duration": 5.0}
FAKE_AUDIO = {**REAL_AUDIO, "is_fake": True, "confidence": 0.97}


@pytest.mark.parametrize("mode", ["corrected_fixed", "ppo", "legacy"])
def test_missing_face_never_yields_authentic(mode):
    *_, verdict, contributing, unavailable, details = _compute_verdict(NO_FACE + [REAL_AUDIO], "video", return_details=True, mode_override=mode)
    assert verdict.startswith("UNCERTAIN") and details["is_abstention"] is True
    assert contributing == ["Audio Analysis"]
    # Strong audio evidence still decides: the guard only blocks AUTHENTIC.
    assert _compute_verdict(NO_FACE + [FAKE_AUDIO], "video", mode_override=mode)[1] == "DEEPFAKE"


def test_image_without_face_is_unverified():
    no_audio = {"agent": "Audio Analysis", "status": "no_audio", "is_fake": None, "confidence": 0}
    assert _compute_verdict(NO_FACE + [no_audio], "image")[1] == "UNVERIFIED MODALITY"
