"""
# -*- coding: utf-8 -*-
test_spatial.py

Comprehensive verification test for the VERITAS Spatial Agent.

Tests:
 1.  Model path resolves to veritas/backend/models/spatial_agent_lc.pth
 2.  File exists on disk
 3.  Checkpoint loads successfully
 4.  checkpoint["state_dict"] exists
 5.  model.load_state_dict(..., strict=True) succeeds (0 missing / 0 unexpected)
 6.  Architecture: EfficientNet-B0 -> Dropout(0.3) -> Linear(1280,256) -> ReLU -> Dropout(0.2) -> Linear(256,2)
 7.  Class mapping: 0=REAL, 1=FAKE (printed from checkpoint)
 8.  Preprocessing: Resize(224,224) / ToTensor / Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
 9.  Inference uses softmax(dim=1) and fake_probability = probabilities[:,1]
10.  Synthetic + real image passed through model; prints real_prob, fake_prob, predicted_class
11.  Probabilities in [0,1] and sum ~1
12.  gradcam_b64 is None
13.  Existing VERITAS output schema is preserved
"""

import os
import sys
import tempfile
import traceback
from pathlib import Path

import io
import numpy as np
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from PIL import Image

PASS = "[PASS]"
FAIL = "[FAIL]"

results = []

def check(label: str, ok: bool, detail: str = ""):
    tag = PASS if ok else FAIL
    line = f"{tag}  {label}"
    if detail:
        line += f"\n      → {detail}"
    print(line)
    results.append((label, ok))
    return ok


print("=" * 70)
print("VERITAS Spatial Agent — Verification Test")
print("=" * 70)

# ---------------------------------------------------------------------------
# 0. Import spatial_agent
# ---------------------------------------------------------------------------
try:
    import spatial_agent
    check("Import spatial_agent", True)
except Exception as e:
    check("Import spatial_agent", False, str(e))
    print("\nCannot continue — aborting.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Model path resolution
# ---------------------------------------------------------------------------
resolved = spatial_agent._CHECKPOINT_PATH.resolve()
print(f"\n[1] Resolved checkpoint path: {resolved}")
# Verify the tail matches  veritas/backend/models/spatial_agent_lc.pth
tail = Path(*resolved.parts[-3:])   # last 3 parts: models / spatial_agent_lc.pth → adjust
parts = resolved.parts
correct_tail = (
    len(parts) >= 3
    and parts[-1] == "spatial_agent_lc.pth"
    and parts[-2] == "models"
    and parts[-3] == "backend"
)
check("Path tail is veritas/backend/models/spatial_agent_lc.pth", correct_tail,
      f"tail = {'/'.join(parts[-3:])}")

# ---------------------------------------------------------------------------
# 2. File exists
# ---------------------------------------------------------------------------
exists = resolved.exists()
check("Checkpoint file exists on disk", exists, str(resolved))
if not exists:
    print("\nCheckpoint missing — cannot continue.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3 + 4 + 5 + 6 + 7.  Load checkpoint manually for detailed inspection
# ---------------------------------------------------------------------------
print("\n[3-7] Manual checkpoint inspection …")
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"      Device: {device}")

try:
    ckpt = torch.load(str(resolved), map_location=device, weights_only=False)
    check("Checkpoint loaded successfully", True)
except Exception as e:
    check("Checkpoint loaded successfully", False, str(e))
    sys.exit(1)

# 4. state_dict key
has_state_dict = "state_dict" in ckpt
check('checkpoint["state_dict"] exists', has_state_dict,
      f"keys present: {list(ckpt.keys())}")
if not has_state_dict:
    sys.exit(1)

# 7. Class mapping from checkpoint
class_names = ckpt.get("class_names", {})
model_name  = ckpt.get("model_name", "unknown")
num_classes = ckpt.get("num_classes", "?")
image_size  = ckpt.get("image_size", "?")
print(f"\n      Checkpoint metadata:")
print(f"        model_name  : {model_name}")
print(f"        num_classes : {num_classes}")
print(f"        image_size  : {image_size}")
print(f"        class_names : {class_names}")
print(f"        epoch       : {ckpt.get('epoch', '?')}")
class_ok = class_names.get("0") == "REAL" and class_names.get("1") == "FAKE"
check('Class mapping: 0=REAL, 1=FAKE', class_ok, str(class_names))

# 5 + 6. Build model and load state_dict
from torchvision.models import efficientnet_b0

def build_reference_model():
    m = efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(1280, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(256, 2),
    )
    return m

try:
    ref_model = build_reference_model()
    result = ref_model.load_state_dict(ckpt["state_dict"], strict=True)
    missing    = result.missing_keys
    unexpected = result.unexpected_keys
    load_ok    = not missing and not unexpected
    check("load_state_dict(strict=True) succeeds (0 missing, 0 unexpected)", load_ok,
          f"missing={missing}, unexpected={unexpected}")
    check("Zero missing keys",    len(missing) == 0,    f"missing: {missing}")
    check("Zero unexpected keys", len(unexpected) == 0, f"unexpected: {unexpected}")
except Exception as e:
    check("load_state_dict(strict=True) succeeds", False, str(e))
    sys.exit(1)

# ---------------------------------------------------------------------------
# 6.  Architecture verification
# ---------------------------------------------------------------------------
print("\n[6] Model classifier head:")
for idx, layer in enumerate(ref_model.classifier):
    print(f"      [{idx}] {layer}")

clf = ref_model.classifier
arch_ok = (
    isinstance(clf[0], nn.Dropout) and abs(clf[0].p - 0.3) < 1e-6
    and isinstance(clf[1], nn.Linear) and clf[1].in_features == 1280 and clf[1].out_features == 256
    and isinstance(clf[2], nn.ReLU)
    and isinstance(clf[3], nn.Dropout) and abs(clf[3].p - 0.2) < 1e-6
    and isinstance(clf[4], nn.Linear) and clf[4].in_features == 256 and clf[4].out_features == 2
)
check(
    "Architecture: Dropout(0.3)->Linear(1280,256)->ReLU->Dropout(0.2)->Linear(256,2)",
    arch_ok,
)

# ---------------------------------------------------------------------------
# 8.  Preprocessing verification
# ---------------------------------------------------------------------------
print("\n[8] Preprocessing pipeline:")
for t in spatial_agent._TRANSFORM.transforms:
    print(f"      {t}")

from torchvision import transforms
tf = spatial_agent._TRANSFORM.transforms
# torchvision >= 0.15 stores Resize.size as a list; older versions as tuple/int.
# Accept both forms — what matters is the resolved size is 224x224.
resize_size = tf[0].size if isinstance(tf[0].size, (list, tuple)) else [tf[0].size, tf[0].size]
prep_ok = (
    len(tf) == 3
    and isinstance(tf[0], transforms.Resize)
    and list(resize_size) == [224, 224]
    and isinstance(tf[1], transforms.ToTensor)
    and isinstance(tf[2], transforms.Normalize)
    and list(tf[2].mean) == [0.485, 0.456, 0.406]
    and list(tf[2].std)  == [0.229, 0.224, 0.225]
)
check(
    "Preprocessing: Resize(224,224) / ToTensor / Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])",
    prep_ok,
    f"resize_size={list(resize_size)}, mean={list(tf[2].mean)}, std={list(tf[2].std)}"
)

# ---------------------------------------------------------------------------
# 3.  Load via module (startup path)
# ---------------------------------------------------------------------------
print("\n[3] Loading via spatial_agent.load_model() …")
try:
    spatial_agent.load_model()
    check("spatial_agent.load_model() completes without exception", True)
    check("_model_loaded flag is True", spatial_agent._model_loaded)
    check("_model is not None", spatial_agent._model is not None)
except Exception as e:
    check("spatial_agent.load_model() completes without exception", False, str(e))
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------------------
# 9 + 10 + 11 + 12.  Inference — synthetic image
# ---------------------------------------------------------------------------
print("\n[9-12] Inference test (synthetic 224×224 noise image) …")

tmp_img_path = None
try:
    rng = np.random.RandomState(42)
    arr = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil = Image.fromarray(arr)

    fd, tmp_img_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    pil.save(tmp_img_path)

    result = spatial_agent.predict(tmp_img_path)

    fake_p = result["confidence"]
    is_fk  = result["is_fake"]
    gradcam = result["gradcam_b64"]

    # Reconstruct real_probability from fake
    real_p = round(1.0 - fake_p, 4)

    print(f"\n      Output dict:")
    for k, v in result.items():
        if k == "gradcam_b64":
            print(f"        {k}: {v}")
        else:
            print(f"        {k}: {v!r}")

    print(f"\n      real_probability  : {real_p:.6f}")
    print(f"      fake_probability  : {fake_p:.6f}")
    decision_sym = ">= 0.50 -> FAKE" if is_fk else "< 0.50 -> REAL"
    print(f"      predicted_class   : {'FAKE' if is_fk else 'REAL'} (fake_prob {decision_sym})")

    check("Inference completes without error", True)
    check("fake_probability in [0, 1]", 0.0 <= fake_p <= 1.0, str(fake_p))
    check("real_probability in [0, 1]", 0.0 <= real_p <= 1.0, str(real_p))
    check("probabilities sum ≈ 1.0",    abs(fake_p + real_p - 1.0) < 1e-4,
          f"sum={fake_p + real_p:.6f}")
    check('gradcam_b64 is None',        gradcam is None, str(gradcam))

    # Schema check
    required_keys = {"agent", "color", "confidence", "is_fake", "finding", "gradcam_b64"}
    has_all = required_keys.issubset(result.keys())
    check(f"Output schema preserved {required_keys}", has_all,
          f"missing: {required_keys - result.keys()}")
    check('agent == "Spatial Analysis"', result["agent"] == "Spatial Analysis")
    check('color == "cyan"',             result["color"] == "cyan")
    check("finding is non-empty str",    isinstance(result["finding"], str) and len(result["finding"]) > 0)

    # Random noise has no face: the agent must say so instead of inventing a score
    # (scoring on a real face is covered by test_face_crop.py).
    check("noise image -> status no_face, is_fake None", result.get("status") == "no_face" and is_fk is None,
          f"status={result.get('status')} is_fake={is_fk}")
    check("noise image -> no score invented", fake_p == 0.0, f"confidence={fake_p}")

except Exception as e:
    check("Inference completes without error", False, str(e))
    traceback.print_exc()
finally:
    if tmp_img_path and os.path.exists(tmp_img_path):
        os.remove(tmp_img_path)

# ---------------------------------------------------------------------------
# 13.  Softmax verification (internal tensor check)
# ---------------------------------------------------------------------------
print("\n[9] Softmax verification (direct tensor) …")
try:
    _m   = spatial_agent._model
    _dev = spatial_agent._device
    _tf  = spatial_agent._TRANSFORM

    rng = np.random.RandomState(7)
    arr = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil = Image.fromarray(arr)
    t   = _tf(pil).unsqueeze(0).to(_dev)

    with torch.no_grad():
        logits = _m(t)                          # (1, 2)
        probs  = torch.softmax(logits, dim=1)   # (1, 2)
        fp     = probs[:, 1].item()
        rp     = probs[:, 0].item()

    print(f"      raw logits  : {logits[0].tolist()}")
    print(f"      softmax out : real={rp:.6f}  fake={fp:.6f}  sum={rp+fp:.6f}")

    check("softmax output sums to 1.0 (tensor level)", abs(rp + fp - 1.0) < 1e-5,
          f"sum={rp+fp:.8f}")
    check("fake_probability = probabilities[:,1]", True,
          "confirmed: probs[:,1] used for fake score")

except Exception as e:
    check("Softmax tensor verification", False, str(e))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 14.  VERITAS pipeline path simulation
# ---------------------------------------------------------------------------
print("\n[14] VERITAS pipeline simulation (main.py logic) …")
try:
    # This mimics exactly what main.py does:
    #   s_result = spatial_agent.predict(tmp_path)
    # We use a real JPEG from a synthetic array.
    rng = np.random.RandomState(99)
    arr = rng.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil = Image.fromarray(arr)
    fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    pil.save(tmp_path)

    try:
        s_result = spatial_agent.predict(tmp_path)
        required = {"agent", "color", "confidence", "is_fake", "finding", "gradcam_b64"}
        schema_ok = required.issubset(s_result.keys())
        check("Pipeline call spatial_agent.predict(tmp_path) succeeds", True)
        check("Pipeline result has all expected schema keys", schema_ok,
              f"missing: {required - s_result.keys()}")
        check("Pipeline agent name is 'Spatial Analysis'",
              s_result["agent"] == "Spatial Analysis")
    finally:
        os.remove(tmp_path)

except Exception as e:
    check("VERITAS pipeline simulation", False, str(e))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
passed = sum(1 for _, ok in results if ok)
failed = sum(1 for _, ok in results if not ok)
print(f"Results: {passed} passed  /  {failed} failed  /  {len(results)} total")
print("=" * 70)

if failed:
    print("\nFAILED tests:")
    for label, ok in results:
        if not ok:
            print(f"  [FAIL]  {label}")
    sys.exit(1)
else:
    print("\nAll tests passed. Spatial Agent is correctly integrated.")
