"""
spatial_agent.py

EfficientNet-B0 spatial deepfake detector.
Trained on FakeAVCeleb dataset.

Checkpoint format
-----------------
torch.save({
    "epoch": ...,
    "state_dict": model.state_dict(),   # ← key is "state_dict"
    "model_name": "efficientnet_b0",
    "num_classes": 2,
    "class_names": {"0": "REAL", "1": "FAKE"},
    "image_size": 224,
    ...
}, checkpoint_path)

Classifier head (must match exactly)
-------------------------------------
EfficientNet-B0 original classifier input: 1280
→ Dropout(p=0.3)
→ Linear(1280, 256)
→ ReLU(inplace=True)
→ Dropout(p=0.2)
→ Linear(256, 2)

Class mapping
-------------
0 → REAL VIDEO
1 → FAKE VIDEO

Inference
---------
logits = model(image_tensor)                # shape (1, 2)
probabilities = softmax(logits, dim=1)
fake_probability = probabilities[:, 1]
real_probability = probabilities[:, 0]
decision: fake_probability >= 0.50 → FAKE
"""

import base64
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent   # veritas/backend/

# Primary path: veritas/backend/models/spatial_agent_lc.pth
# Fallbacks handle the case where __file__ sits at a different depth.
_CANDIDATE_PATHS = [
    _HERE / "models" / "spatial_agent_lc.pth",
    _HERE / ".." / "backend" / "models" / "spatial_agent_lc.pth",
    _HERE / ".." / ".." / "backend" / "models" / "spatial_agent_lc.pth",
]
_CHECKPOINT_PATH: Path = next(
    (p for p in _CANDIDATE_PATHS if p.resolve().exists()),
    _CANDIDATE_PATHS[0],
)

# ---------------------------------------------------------------------------
# Preprocessing — deterministic, matches training evaluation transforms
# ---------------------------------------------------------------------------
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_model: Optional[nn.Module] = None
_model_loaded: bool = False
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model definition helper
# ---------------------------------------------------------------------------
def _build_model() -> nn.Module:
    """
    Construct EfficientNet-B0 with the custom 2-class classifier head.

    Architecture (must match checkpoint exactly):
        EfficientNet-B0 backbone (features)
        ↓
        classifier:
            [0] Dropout(p=0.3)
            [1] Linear(1280 → 256)
            [2] ReLU(inplace=True)
            [3] Dropout(p=0.2)
            [4] Linear(256 → 2)
    """
    model = efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(1280, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(256, 2),
    )
    return model


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_model() -> None:
    """
    Load the trained EfficientNet-B0 checkpoint into the module singleton.

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    KeyError
        If the checkpoint dict does not contain "state_dict".
    RuntimeError
        If state_dict loading fails (missing/unexpected keys with strict=True).
    """
    global _model, _model_loaded

    checkpoint_path = _CHECKPOINT_PATH.resolve()
    logger.info("Spatial Agent — resolved checkpoint path: %s", checkpoint_path)

    # 1. Validate file existence — fail loudly
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Spatial Agent checkpoint not found at: {checkpoint_path}\n"
            f"Expected file: veritas/backend/models/spatial_agent_lc.pth"
        )

    try:
        # 2. Load checkpoint to target device
        logger.info("Loading checkpoint to device=%s …", _device)
        ckpt = torch.load(str(checkpoint_path), map_location=_device, weights_only=False)

        # 3. Verify "state_dict" key exists
        if "state_dict" not in ckpt:
            available = list(ckpt.keys())
            raise KeyError(
                f"Checkpoint missing 'state_dict'. Available keys: {available}"
            )

        # 4. Build the model with the correct head
        model = _build_model()

        # 5. Load weights — strict=True ensures zero missing / zero unexpected keys
        incompatible = model.load_state_dict(ckpt["state_dict"], strict=True)
        # strict=True raises RuntimeError on mismatch; this line is only reached
        # on full success.
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"State dict mismatch — missing: {incompatible.missing_keys}, "
                f"unexpected: {incompatible.unexpected_keys}"
            )

        # 6. Move to device and put in eval mode
        model.to(_device)
        model.eval()

        _model = model
        _model_loaded = True

        logger.info(
            "Spatial Agent loaded successfully. "
            "epoch=%s | device=%s | checkpoint=%s",
            ckpt.get("epoch", "?"),
            _device,
            checkpoint_path,
        )

    except (FileNotFoundError, KeyError):
        _model_loaded = False
        raise  # re-raise so caller sees the real error
    except Exception as exc:
        print(f"Exception loading spatial agent: {exc}")
        _model_loaded = False
        logger.error("Failed to load Spatial Agent: %s", exc, exc_info=True)
        raise  # never silently fall back


# ---------------------------------------------------------------------------
# GradCAM helper — kept for future use, NOT called during inference.
# The trained checkpoint does not include GradCAM artefacts; gradcam_b64
# is always returned as None in predict().
# ---------------------------------------------------------------------------
def _compute_gradcam(
    image_path: str, tensor: torch.Tensor, logits: torch.Tensor
) -> Optional[str]:
    """
    Generate GradCAM heatmap for the spatial agent prediction.

    NOTE: This function is NOT called from predict() because the current
    trained checkpoint does not include GradCAM-compatible metadata.
    It is preserved here for future integration once proper explainability
    support is added.

    Returns base64-encoded PNG string or None on failure.
    """
    try:
        if _model is None:
            return None

        activations: dict = {}
        gradients: dict = {}

        def forward_hook(module, input, output):
            activations["feat"] = output.detach()

        def backward_hook(module, grad_in, grad_out):
            if grad_out and grad_out[0] is not None:
                gradients["feat"] = grad_out[0].detach()

        target_layer = _model.features[-1]
        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            grad_tensor = tensor.detach().clone().requires_grad_(True)
            output = _model(grad_tensor)
            # Use fake class (index 1) for backward
            score = output[:, 1]
            _model.zero_grad()
            score.backward()

            if "feat" not in gradients or "feat" not in activations:
                return None

            grads = gradients["feat"].squeeze(0)   # (C, H, W)
            acts = activations["feat"].squeeze(0)  # (C, H, W)
            weights = grads.mean(dim=(1, 2))       # (C,)

            cam = torch.zeros(acts.shape[1:], dtype=torch.float32, device=acts.device)
            for i, w in enumerate(weights):
                cam += w * acts[i]
            cam = torch.relu(cam)

            cam_min, cam_max = cam.min(), cam.max()
            cam = cam - cam_min
            if cam_max > cam_min:
                cam = cam / (cam_max - cam_min)

            cam_np = (cam.detach().cpu().numpy() * 255).astype(np.uint8)
            orig = cv2.imread(image_path)
            if orig is None:
                return None

            h, w = orig.shape[:2]
            cam_resized = cv2.resize(cam_np, (w, h))
            heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)

            ok, buffer = cv2.imencode(".png", overlay)
            if not ok:
                return None
            return base64.b64encode(buffer).decode("utf-8")

        finally:
            fwd_handle.remove()
            bwd_handle.remove()

    except Exception as e:
        logger.warning("GradCAM failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Finding text
# ---------------------------------------------------------------------------
def _make_finding(fake_prob: float, num_frames: Optional[int] = None) -> str:
    """
    Return a human-readable description of the model's inference result.
    All statements are grounded in what the model actually computes.
    """
    frame_note = f" ({num_frames} frame(s) evaluated)" if num_frames is not None else ""
    if fake_prob >= 0.85:
        return (
            f"Strong visual inconsistencies detected{frame_note}. "
            f"EfficientNet-B0 assigned {fake_prob:.1%} fake probability, "
            f"indicating high confidence of AI-generated or manipulated imagery."
        )
    if fake_prob >= 0.65:
        return (
            f"Moderate visual anomalies identified{frame_note}. "
            f"EfficientNet-B0 assigned {fake_prob:.1%} fake probability, "
            f"suggesting likely digital manipulation."
        )
    if fake_prob >= 0.50:
        return (
            f"Mild visual inconsistencies detected{frame_note}. "
            f"EfficientNet-B0 assigned {fake_prob:.1%} fake probability — "
            f"above the 0.50 decision threshold, though the signal is weak."
        )
    return (
        f"No significant visual inconsistencies detected{frame_note}. "
        f"EfficientNet-B0 assigned {fake_prob:.1%} fake probability — "
        f"below the 0.50 decision threshold, consistent with authentic media."
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def predict(image_path: str) -> dict:
    """
    Run the EfficientNet-B0 spatial deepfake detector on a single image.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to an image file.  For video inputs the
        caller (main.py) should extract frames first and call this function
        per-frame, then aggregate — or pass a representative single frame.

    Returns
    -------
    dict with keys:
        agent        : "Spatial Analysis"
        color        : "cyan"
        confidence   : float  — fake_probability in [0, 1]
        is_fake      : bool   — fake_probability >= 0.50
        finding      : str    — human-readable model-grounded description
        gradcam_b64  : None   — GradCAM not available for this checkpoint
    """
    if _model is None:
        # Hard failure — no silent fallback
        raise RuntimeError(
            "Spatial Agent model is not loaded. "
            "Call spatial_agent.load_model() before predict()."
        )

    try:
        img = Image.open(image_path).convert("RGB")
        tensor = _TRANSFORM(img).unsqueeze(0).to(_device)  # (1, 3, 224, 224)

        with torch.no_grad():
            logits = _model(tensor)                                    # (1, 2)
            probabilities = torch.softmax(logits, dim=1)               # (1, 2)
            fake_probability = probabilities[:, 1].item()              # class 1 = FAKE
            real_probability = probabilities[:, 0].item()              # class 0 = REAL

        logger.debug(
            "Spatial inference: real=%.4f  fake=%.4f  decision=%s",
            real_probability,
            fake_probability,
            "FAKE" if fake_probability >= 0.50 else "REAL",
        )

        return {
            "agent":       "Spatial Analysis",
            "color":       "cyan",
            "confidence":  round(fake_probability, 4),
            "is_fake":     fake_probability >= 0.50,
            "finding":     _make_finding(fake_probability),
            "gradcam_b64": None,   # GradCAM not available for this checkpoint
        }

    except Exception as exc:
        logger.error("Spatial Agent inference error: %s", exc, exc_info=True)
        raise  # no silent fallback

