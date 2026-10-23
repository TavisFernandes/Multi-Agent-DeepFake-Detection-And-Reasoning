"""
face_crop.py

Face detection + cropping shared by the spatial and temporal agents, so an upload is framed the way
FakeAVCeleb's training clips are (VoxCeleb2-style face tracks) before it is resized for the models.

Detector: OpenCV YuNet (cv2.FaceDetectorYN, OpenCV >= 4.8), model file
models/face_detection_yunet_2023mar.onnx (232,589 bytes, sha256 8f2383e4...2552fa4, MIT, from opencv_zoo).

Training framing, measured with YuNet on 240 detector_train middle frames: the face box side is 0.708 of the
frame side (p5-p95 0.67-0.75) and its centre sits at x = 0.50, y = 0.367 of the frame. crop() reproduces that.

Rules (deterministic):
- several faces  -> the largest box (ties: highest score) is used; the note says how many were found
- score < MIN_SCORE, box side < MIN_FACE_PX, > MAX_OUTSIDE of the box outside the frame, or a box touching the
  frame edge with a shape outside ASPECT_RANGE (a cut-off face) -> no usable face
- no usable face -> detect() returns (None, reason); callers must report the visual modality as unavailable
"""

import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
FACE_TO_CROP = 0.708   # face box side / crop side in training frames
FACE_CY = 0.367        # face centre height / crop side in training frames
SNAP_TO_FRAME = 0.9    # crop >= 90% of the short side (face/frame >= 0.64, below training p5 0.67) -> no crop
MIN_SCORE = 0.7
MIN_FACE_PX = 64
MAX_OUTSIDE = 0.25
ASPECT_RANGE = (0.65, 1.4)  # box w/h on 240 training faces spans 0.667-1.349
_DETECT_MAX_SIDE = 640  # detect on a downscaled copy; boxes are scaled back

Box = Tuple[float, float, float, float]  # x, y, w, h in original pixels

_detector = None
_lock = threading.Lock()  # one detector instance; agents run in executor threads


def detect(bgr: np.ndarray) -> Tuple[Optional[Box], str]:
    """Return (box, note) for the face to analyse, or (None, reason) when there is no usable face."""
    global _detector
    h, w = bgr.shape[:2]
    scale = min(1.0, _DETECT_MAX_SIDE / max(h, w))
    img = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    with _lock:
        if _detector is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Face detector model missing: {MODEL_PATH}")
            # Low internal threshold so weak detections are reported as "too uncertain", not as "no face".
            _detector = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (320, 320), 0.5, 0.3, 50)
        _detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = _detector.detect(np.ascontiguousarray(img))
    if faces is None or len(faces) == 0:
        return None, "No face detected"

    best = max(faces, key=lambda f: (f[2] * f[3], f[-1]))
    x, y, bw, bh = (float(v) / scale for v in best[:4])
    score = float(best[-1])
    if score < MIN_SCORE:
        return None, f"Face detection too uncertain (score {score:.2f})"
    if max(bw, bh) < MIN_FACE_PX:
        return None, f"Face too small to analyse ({max(bw, bh):.0f} px)"
    inside = max(0.0, min(x + bw, w) - max(x, 0.0)) * max(0.0, min(y + bh, h) - max(y, 0.0))
    # YuNet clips boxes to the image, so a cut-off face shows up as an edge-touching box with an odd shape.
    at_edge = x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2
    if 1.0 - inside / (bw * bh) > MAX_OUTSIDE or (at_edge and not ASPECT_RANGE[0] <= bw / bh <= ASPECT_RANGE[1]):
        return None, "Face is partly outside the frame"
    many = f"; {len(faces)} faces found, the largest was analysed" if len(faces) > 1 else ""
    return (x, y, bw, bh), f"Face detected (score {score:.2f}{many})"


def crop(img: np.ndarray, box: Box) -> np.ndarray:
    """Square crop around `box` with training framing, kept inside the image (works for BGR or RGB)."""
    x, y, bw, bh = box
    h, w = img.shape[:2]
    side = max(bw, bh) / FACE_TO_CROP
    # Already framed like training: take the whole short side. The models flip on a ~2% rescale, so a native
    # clip must not be cropped to 97% and blown back up (18 real fusion_val clips turned "fake" that way).
    if side >= SNAP_TO_FRAME * min(w, h):
        side = min(w, h)
    # shortcut: a face too close to fill a training-sized crop is capped at the frame, so it appears larger than in training.
    side = min(side, w, h)
    x0 = min(max(x + bw / 2 - side / 2, 0.0), w - side)
    y0 = min(max(y + bh / 2 - FACE_CY * side, 0.0), h - side)
    x0, y0, s = int(round(x0)), int(round(y0)), int(round(side))
    return img[y0:y0 + s, x0:x0 + s]
