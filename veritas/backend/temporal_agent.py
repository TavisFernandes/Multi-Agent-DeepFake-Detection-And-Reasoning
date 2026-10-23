"""
temporal_agent.py

R3D-18 temporal deepfake detector for VERITAS.
Trained on FakeAVCeleb v1.2 (identity-aware split).

Architecture
------------
- torchvision.models.video.r3d_18(weights=None)
- model.fc = nn.Linear(512, 1)
- Output: sigmoid(logit) → P(FAKE in [0, 1])
- Class 0 = REAL VIDEO
- Class 1 = FAKE VIDEO

Checkpoint
----------
Expected at: veritas/backend/models/temporal_agent_lc.pth
Loaded key : checkpoint["model_state_dict"]   (strict=True)

Inference preprocessing (from checkpoint metadata)
---------------------------------------------------
- Sample 16 frames at ~8 FPS from a window centred in the video
- Resize each frame to 112 × 112
- BGR → RGB
- Normalize: mean=[0.43216, 0.39467, 0.37645]
             std =[0.22803, 0.22145, 0.21699]
- Tensor shape: (1, 3, 16, 112, 112)

Decision threshold
------------------
Stored in checkpoint["decision_threshold"] = 0.35 (selected on detector_val).
Checkpoint trained on detector_train only (leakage-controlled split).

IMPORTANT: This agent detects FAKE VIDEO only.
  RealVideo-RealAudio → tends REAL
  RealVideo-FakeAudio → tends REAL  (video is genuine)
  FakeVideo-RealAudio → tends FAKE
  FakeVideo-FakeAudio → tends FAKE
"""

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

import face_crop

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent  # veritas/backend/

_CANDIDATE_PATHS = [
    _HERE / "models" / "temporal_agent_lc.pth",
    _HERE / ".." / "backend" / "models" / "temporal_agent_lc.pth",
    _HERE / ".." / ".." / "backend" / "models" / "temporal_agent_lc.pth",
]
_CHECKPOINT_PATH: Path = next(
    (p for p in _CANDIDATE_PATHS if p.resolve().exists()),
    _CANDIDATE_PATHS[0],
)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Preprocessing constants (loaded from checkpoint; fallbacks listed here)
# Training values from checkpoint["normalization"]:
#   mean=[0.43216, 0.39467, 0.37645]
#   std =[0.22803, 0.22145, 0.21699]
# These differ slightly from the old [0.43, 0.40, 0.37] / [0.15, 0.14, 0.14]
# placeholders. The checkpoint values are authoritative.
# ---------------------------------------------------------------------------
_MEAN = np.array([0.43216, 0.39467, 0.37645], dtype=np.float32)
_STD  = np.array([0.22803, 0.22145, 0.21699], dtype=np.float32)

_FRAME_SIZE   = (112, 112)  # H × W expected by R3D-18
_NUM_FRAMES   = 16
_TARGET_FPS   = 8.0         # frames per second during training sampling
_WINDOW_SECS  = 2.0         # seconds of video to sample

# ---------------------------------------------------------------------------
# Decision threshold (loaded from checkpoint; 0.5 is the absolute fallback)
# ---------------------------------------------------------------------------
_DECISION_THRESHOLD: float = 0.05  # checkpoint["decision_threshold"]

# ---------------------------------------------------------------------------
# Singleton model state
# ---------------------------------------------------------------------------
_model: Optional[nn.Module] = None
_model_loaded: bool = False


# ---------------------------------------------------------------------------
# Model loader — called explicitly at startup from main.py
# ---------------------------------------------------------------------------

def load_model() -> None:
    """
    Build R3D-18 with fc=Linear(512,1) and load the trained checkpoint.

    Reads normalization constants and decision threshold from checkpoint
    metadata so inference always matches training.

    Raises
    ------
    FileNotFoundError   — checkpoint file missing
    KeyError            — checkpoint missing "model_state_dict"
    RuntimeError        — state_dict mismatch (strict=True)
    """
    global _model, _model_loaded, _MEAN, _STD, _DECISION_THRESHOLD
    global _NUM_FRAMES, _TARGET_FPS, _WINDOW_SECS

    checkpoint_path = _CHECKPOINT_PATH.resolve()
    logger.info("Temporal Agent — resolved checkpoint: %s", checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Temporal Agent checkpoint not found at: {checkpoint_path}"
        )

    logger.info("Loading Temporal Agent checkpoint …")
    ckpt = torch.load(str(checkpoint_path), map_location=_DEVICE, weights_only=False)

    # ── Pull metadata from checkpoint ────────────────────────────────────────
    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"'model_state_dict' missing from checkpoint. "
            f"Available keys: {list(ckpt.keys())}"
        )

    # Override preprocessing constants from checkpoint metadata
    norm = ckpt.get("normalization", {})
    if norm.get("mean") and norm.get("std"):
        _MEAN = np.array(norm["mean"], dtype=np.float32)
        _STD  = np.array(norm["std"],  dtype=np.float32)
        logger.info("Normalization loaded from checkpoint: mean=%s  std=%s",
                    _MEAN.tolist(), _STD.tolist())
    else:
        logger.warning("Normalization not in checkpoint — using fallback values.")

    threshold = ckpt.get("decision_threshold")
    if threshold is not None:
        _DECISION_THRESHOLD = float(threshold)
        logger.info("Decision threshold loaded from checkpoint: %.4f", _DECISION_THRESHOLD)
    else:
        logger.warning("decision_threshold not in checkpoint — using fallback 0.5")
        _DECISION_THRESHOLD = 0.5

    if ckpt.get("num_frames"):
        _NUM_FRAMES = int(ckpt["num_frames"])
    if ckpt.get("target_fps"):
        _TARGET_FPS = float(ckpt["target_fps"])
    if ckpt.get("window_seconds"):
        _WINDOW_SECS = float(ckpt["window_seconds"])

    # ── Build model and load weights ─────────────────────────────────────────
    model = r3d_18(weights=None)        # no internet download
    model.fc = nn.Linear(512, 1)

    incompatible = model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"State dict mismatch — "
            f"missing: {incompatible.missing_keys}, "
            f"unexpected: {incompatible.unexpected_keys}"
        )

    model.to(_DEVICE)
    model.eval()

    _model = model
    _model_loaded = True

    logger.info(
        "Temporal Agent loaded successfully. "
        "epoch=%s  device=%s  threshold=%.4f  num_frames=%d  target_fps=%.1f",
        ckpt.get("epoch", "?"), _DEVICE, _DECISION_THRESHOLD, _NUM_FRAMES, _TARGET_FPS,
    )


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _extract_frames(video_path: str) -> tuple[Optional[np.ndarray], Optional[str]]:
    """
    Sample _NUM_FRAMES frames at ~_TARGET_FPS from a window centred in the
    video — the same indices train_temporal_lc.py used — and crop all of them
    with one face box found on the middle frame (training clips are face tracks).

    Returns (np.ndarray (16, 3, 112, 112) float32 normalised, face note), or
    (None, reason) when no usable face was found, or (None, None) when the
    video cannot be read.
    """
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open video: %s", video_path)
        return None, None

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.warning("Video reports 0 frames: %s", video_path)
            return None, None

        step = max(1, int(round(fps / _TARGET_FPS)))
        start = max(0, (total_frames - _NUM_FRAMES * step) // 2)
        # Short clips run past the end; those reads fall back to the last good frame below.
        indices = [start + i * step for i in range(_NUM_FRAMES)]

        frames: list[np.ndarray] = []
        last_good: Optional[np.ndarray] = None

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, bgr = cap.read()
            if ret and bgr is not None:
                last_good = bgr
            else:
                if last_good is None:
                    logger.debug("Frame %d unreadable, no fallback yet.", idx)
                    continue
                bgr = last_good  # repeat last good frame

            frames.append(bgr)

        if not frames:
            logger.warning("No frames could be read from %s", video_path)
            return None, None

        # Pad to exactly _NUM_FRAMES by repeating the last frame
        while len(frames) < _NUM_FRAMES:
            frames.append(frames[-1])

        # shortcut: one box from the middle frame; a face that moves a lot drifts out of it — per-frame tracking if that shows up.
        box, note = face_crop.detect(frames[_NUM_FRAMES // 2])
        if box is None:
            return None, note
        frames = [
            cv2.resize(cv2.cvtColor(face_crop.crop(f, box), cv2.COLOR_BGR2RGB), _FRAME_SIZE, interpolation=cv2.INTER_LINEAR)
            for f in frames[:_NUM_FRAMES]
        ]

        # (16, H, W, 3) → normalise → (16, 3, H, W)
        stack = np.stack(frames).astype(np.float32) / 255.0
        stack = (stack - _MEAN) / _STD                # broadcast over (H, W, 3)
        stack = stack.transpose(0, 3, 1, 2)           # (16, 3, 112, 112)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("Frame decode: %.1f ms  frames=%d  video_fps=%.1f",
                     elapsed_ms, len(frames), fps)
        return stack, note

    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Finding text — grounded in actual model output only
# ---------------------------------------------------------------------------

def _make_finding(fake_prob: float, num_frames: int) -> str:
    if fake_prob >= 0.85:
        return (
            f"Strong temporal inconsistencies detected across {num_frames} sampled frames. "
            f"R3D-18 assigned {fake_prob:.1%} fake probability — high confidence of "
            f"video manipulation."
        )
    if fake_prob >= _DECISION_THRESHOLD:
        return (
            f"Temporal anomalies detected across {num_frames} sampled frames. "
            f"R3D-18 assigned {fake_prob:.1%} fake probability — above the "
            f"{_DECISION_THRESHOLD:.2f} decision threshold."
        )
    return (
        f"No significant temporal inconsistencies across {num_frames} sampled frames. "
        f"R3D-18 assigned {fake_prob:.1%} fake probability — below the "
        f"{_DECISION_THRESHOLD:.2f} decision threshold."
    )


# ---------------------------------------------------------------------------
# Public API — called by main.py orchestrator
# ---------------------------------------------------------------------------

def run_temporal_agent(video_path: str) -> dict:
    """
    Run the Temporal Agent on a video file.

    Parameters
    ----------
    video_path : str
        Path to a local video file (mp4 / mov / avi / mkv / webm / flv).

    Returns
    -------
    dict
        {
            "agent"      : "Temporal Analysis",
            "confidence" : float,   # fake_probability in [0, 1]
            "is_fake"    : bool,    # fake_prob >= decision_threshold
            "finding"    : str,
            "gradcam_b64": None,    # not available for R3D-18 checkpoint
        }
    """
    t_total = time.perf_counter()

    # ── Model not loaded ──────────────────────────────────────────────────────
    if _model is None:
        logger.warning("Temporal Agent not loaded — returning error result.")
        return {
            "agent":       "Temporal Analysis",
            "status":      "unavailable",
            "confidence":  0.0,
            "is_fake":     None,
            "finding":     "Temporal Agent unavailable — checkpoint not loaded.",
            "gradcam_b64": None,
        }

    try:
        # ── 1. Frame extraction ───────────────────────────────────────────────
        t_decode = time.perf_counter()
        frames_np, face_note = _extract_frames(video_path)
        decode_ms = (time.perf_counter() - t_decode) * 1000

        if frames_np is None and face_note:
            return {
                "agent":       "Temporal Analysis",
                "status":      "no_face",
                "confidence":  0.0,
                "is_fake":     None,
                "finding":     f"{face_note}. The temporal model needs a clearly visible face, so no score was produced.",
                "gradcam_b64": None,
            }
        if frames_np is None:
            return {
                "agent":       "Temporal Analysis",
                "status":      "error",
                "confidence":  0.0,
                "is_fake":     None,
                "finding":     "No readable frames — temporal analysis skipped.",
                "gradcam_b64": None,
            }

        # ── 2. Build tensor (1, 3, 16, 112, 112) ─────────────────────────────
        #   frames_np: (16, 3, 112, 112)
        tensor = torch.from_numpy(frames_np)   # (16, 3, 112, 112)
        tensor = tensor.unsqueeze(0)           # (1, 16, 3, 112, 112)
        tensor = tensor.permute(0, 2, 1, 3, 4) # (1, 3, 16, 112, 112)
        tensor = tensor.to(_DEVICE)

        # ── 3. Inference ──────────────────────────────────────────────────────
        t_infer = time.perf_counter()
        _model.eval()
        with torch.no_grad():
            logit = _model(tensor)                         # (1, 1)
            fake_probability: float = torch.sigmoid(logit.squeeze()).item()
        infer_ms = (time.perf_counter() - t_infer) * 1000

        is_fake = fake_probability >= _DECISION_THRESHOLD
        total_ms = (time.perf_counter() - t_total) * 1000

        logger.info(
            "Temporal inference: fake_prob=%.4f  threshold=%.4f  is_fake=%s  "
            "decode=%.0fms  infer=%.0fms  total=%.0fms",
            fake_probability, _DECISION_THRESHOLD, is_fake,
            decode_ms, infer_ms, total_ms,
        )

        return {
            "agent":       "Temporal Analysis",
            "status":      "trained",
            "confidence":  round(fake_probability, 4),
            "is_fake":     is_fake,
            "finding":     _make_finding(fake_probability, _NUM_FRAMES),
            "gradcam_b64": None,
        }

    except Exception as exc:
        logger.error("Temporal Agent error: %s", exc, exc_info=True)
        return {
            "agent":       "Temporal Analysis",
            "status":      "error",
            "confidence":  0.0,
            "is_fake":     None,
            "finding":     f"Temporal analysis failed: {exc}",
            "gradcam_b64": None,
        }


# ---------------------------------------------------------------------------
# Legacy: load at module import for backward-compat with any direct imports
# that don't call load_model() explicitly.  Failures are logged, not raised,
# so the backend continues if the checkpoint is missing.
# ---------------------------------------------------------------------------
try:
    load_model()
except FileNotFoundError as _e:
    logger.warning("Temporal Agent: %s", _e)
except Exception as _e:
    logger.error("Temporal Agent failed to load at import: %s", _e, exc_info=True)
