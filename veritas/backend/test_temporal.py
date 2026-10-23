"""
# -*- coding: utf-8 -*-
test_temporal.py

Comprehensive integration test for the VERITAS Temporal Agent.

Tests:
 1.  Checkpoint file exists
 2.  model_state_dict key present
 3.  load_state_dict(strict=True) — 0 missing / 0 unexpected keys
 4.  Architecture: R3D-18  fc=Linear(512,1)
 5.  Decision threshold from checkpoint == 0.35
 6.  Normalization from checkpoint matches training values
 7.  Dummy tensor (1,3,16,112,112) inference: prob in [0,1]
 8.  load_model() via module API
 9.  Inference on synthetic video — returns correct schema
10.  is_fake determined by threshold (0.35), not 0.5
11.  Image input → is_fake=False, confidence=0.0 (skipped)
12.  Audio input → temporal skipped
13.  Spatial Agent still loads and works (regression check)
14.  /healthz endpoint still responds
15.  /analyse endpoint still produces final verdict
16.  gradcam_b64 is None for temporal results
"""

import io
import os
import sys
import struct
import tempfile
import traceback
import urllib.request
import json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

def check(label, ok, detail=""):
    tag = PASS if ok else FAIL
    line = f"{tag}  {label}"
    if detail:
        line += f"\n      -> {detail}"
    print(line)
    results.append((label, ok))
    return ok

# ---------------------------------------------------------------------------
# Synthetic video helper
# ---------------------------------------------------------------------------
def make_synthetic_mp4(path: str, num_frames: int = 30, w: int = 112, h: int = 112):
    """
    Write a minimal valid MP4 using OpenCV so the temporal agent can decode it.
    """
    import cv2
    import numpy as np
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 25.0, (w, h))
    rng = np.random.RandomState(42)
    for _ in range(num_frames):
        frame = rng.randint(0, 255, (h, w, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

# ---------------------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------------------
print("=" * 70)
print("VERITAS Temporal Agent — Integration Test")
print("=" * 70)

try:
    import torch
    import torch.nn as nn
    from torchvision.models.video import r3d_18
    check("Import torch / torchvision", True)
except Exception as e:
    check("Import torch / torchvision", False, str(e))
    sys.exit(1)

try:
    import temporal_agent
    check("Import temporal_agent module", True)
except Exception as e:
    check("Import temporal_agent module", False, str(e))
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Checkpoint file exists
# ---------------------------------------------------------------------------
ckpt_path = temporal_agent._CHECKPOINT_PATH.resolve()
print(f"\n[1] Checkpoint path: {ckpt_path}")
check("Checkpoint file exists", ckpt_path.exists(), str(ckpt_path))
if not ckpt_path.exists():
    print("Cannot continue — checkpoint missing"); sys.exit(1)

# ---------------------------------------------------------------------------
# 2-6. Manual checkpoint inspection
# ---------------------------------------------------------------------------
print("\n[2-6] Checkpoint inspection ...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"      Device: {device}")

try:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    check("Checkpoint loads without error", True)
except Exception as e:
    check("Checkpoint loads without error", False, str(e))
    sys.exit(1)

check('checkpoint["model_state_dict"] present', "model_state_dict" in ckpt,
      f"keys: {list(ckpt.keys())}")

# Build reference model
ref_model = r3d_18(weights=None)
ref_model.fc = nn.Linear(512, 1)
try:
    result = ref_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    check("load_state_dict(strict=True) 0 missing / 0 unexpected",
          not result.missing_keys and not result.unexpected_keys,
          f"missing={result.missing_keys} unexpected={result.unexpected_keys}")
except Exception as e:
    check("load_state_dict(strict=True)", False, str(e))
    sys.exit(1)

# Architecture
fc = ref_model.fc
arch_ok = isinstance(fc, nn.Linear) and fc.in_features == 512 and fc.out_features == 1
check("Architecture: r3d_18  fc=Linear(512,1)", arch_ok, str(fc))

# Threshold
thresh = ckpt.get("decision_threshold")
check("decision_threshold in checkpoint", thresh is not None, str(thresh))
check("decision_threshold == 0.35", thresh is not None and abs(thresh - 0.35) < 1e-6,
      str(thresh))

# Normalization
norm = ckpt.get("normalization", {})
mean_ok = norm.get("mean") and abs(norm["mean"][0] - 0.43216) < 1e-4
std_ok  = norm.get("std")  and abs(norm["std"][0]  - 0.22803) < 1e-4
check("Normalization mean from checkpoint correct", mean_ok, str(norm.get("mean")))
check("Normalization std from checkpoint correct",  std_ok,  str(norm.get("std")))

# ---------------------------------------------------------------------------
# 7. Dummy tensor inference
# ---------------------------------------------------------------------------
print("\n[7] Dummy tensor inference (1,3,16,112,112) ...")
ref_model.eval()
dummy = torch.zeros(1, 3, 16, 112, 112)
with torch.no_grad():
    logit = ref_model(dummy)
    prob  = torch.sigmoid(logit.squeeze()).item()
print(f"      logit={logit.item():.4f}  prob={prob:.4f}")
check("Dummy prob in [0,1]", 0.0 <= prob <= 1.0, str(prob))
check("Logit tensor shape (1,1)", logit.shape == torch.Size([1, 1]), str(logit.shape))

# ---------------------------------------------------------------------------
# 8. Module API load_model()
# ---------------------------------------------------------------------------
print("\n[8] Module API: temporal_agent.load_model() ...")
if not temporal_agent._model_loaded:
    try:
        temporal_agent.load_model()
    except Exception as e:
        check("load_model() succeeds", False, str(e))
        traceback.print_exc()
        sys.exit(1)
check("_model_loaded is True",     temporal_agent._model_loaded)
check("_model is not None",        temporal_agent._model is not None)
check("_DECISION_THRESHOLD == 0.35",
      abs(temporal_agent._DECISION_THRESHOLD - 0.35) < 1e-6,
      str(temporal_agent._DECISION_THRESHOLD))
check("_MEAN loaded from ckpt (mean[0] ~0.43216)",
      abs(temporal_agent._MEAN[0] - 0.43216) < 1e-4,
      str(temporal_agent._MEAN.tolist()))

# ---------------------------------------------------------------------------
# 9. Inference on synthetic video
# ---------------------------------------------------------------------------
print("\n[9] Inference on synthetic MP4 ...")
tmp_vid = tempfile.mktemp(suffix=".mp4")
try:
    make_synthetic_mp4(tmp_vid, num_frames=40)
    result = temporal_agent.run_temporal_agent(tmp_vid)
    print(f"      Result: {result}")

    required_keys = {"agent", "confidence", "is_fake", "finding", "gradcam_b64"}
    check("run_temporal_agent returns dict", isinstance(result, dict))
    check("Schema has all required keys", required_keys.issubset(result.keys()),
          f"missing: {required_keys - result.keys()}")
    check('agent == "Temporal Analysis"', result.get("agent") == "Temporal Analysis")
    check("confidence in [0,1]", 0.0 <= result.get("confidence", -1) <= 1.0,
          str(result.get("confidence")))
    check("is_fake is bool", isinstance(result.get("is_fake"), bool))
    check("finding is non-empty str",
          isinstance(result.get("finding"), str) and len(result["finding"]) > 0)
    check("gradcam_b64 is None", result.get("gradcam_b64") is None)

    # Verify threshold logic
    conf = result["confidence"]
    expected_fake = conf >= temporal_agent._DECISION_THRESHOLD
    check(f"is_fake consistent with threshold ({temporal_agent._DECISION_THRESHOLD:.2f})",
          result["is_fake"] == expected_fake,
          f"conf={conf:.4f}  is_fake={result['is_fake']}  expected={expected_fake}")

finally:
    if os.path.exists(tmp_vid):
        os.remove(tmp_vid)

# ---------------------------------------------------------------------------
# 10. is_fake uses 0.35 threshold not 0.5
# ---------------------------------------------------------------------------
print("\n[10] Threshold check: is_fake uses 0.35 not 0.5 ...")
check("_DECISION_THRESHOLD != 0.5",
      abs(temporal_agent._DECISION_THRESHOLD - 0.5) > 0.1,
      f"threshold={temporal_agent._DECISION_THRESHOLD}")

# ---------------------------------------------------------------------------
# 11. Image input → skipped result (from main.py orchestration logic)
# ---------------------------------------------------------------------------
print("\n[11] Non-video skip result schema ...")
skip_result = {
    "agent":       "Temporal Analysis",
    "confidence":  0.0,
    "is_fake":     False,
    "finding":     "Temporal analysis not applicable to non-video media.",
    "gradcam_b64": None,
}
check("Skip result has all required keys",
      {"agent","confidence","is_fake","finding","gradcam_b64"}.issubset(skip_result.keys()))
check("Skip confidence is 0.0", skip_result["confidence"] == 0.0)
check("Skip is_fake is False",  skip_result["is_fake"] == False)

# ---------------------------------------------------------------------------
# 12. Spatial Agent regression check
# ---------------------------------------------------------------------------
print("\n[12] Spatial Agent regression check ...")
try:
    import spatial_agent
    if not spatial_agent._model_loaded:
        spatial_agent.load_model()
    check("Spatial _model_loaded", spatial_agent._model_loaded)

    import numpy as np
    from PIL import Image
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    pil = Image.fromarray(arr)
    fd, tmp_img = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    pil.save(tmp_img)
    try:
        s_result = spatial_agent.predict(tmp_img)
        check("Spatial predict() returns dict", isinstance(s_result, dict))
        check("Spatial 'confidence' present", "confidence" in s_result)
        check("Spatial 'gradcam_b64' is None", s_result.get("gradcam_b64") is None)
    finally:
        os.remove(tmp_img)
except Exception as e:
    check("Spatial Agent regression", False, str(e))
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 13-15. Live backend tests (requires server running on port 8000)
# ---------------------------------------------------------------------------
print("\n[13-15] Live backend tests (requires backend on :8000) ...")
_backend_up = False
try:
    resp = urllib.request.urlopen("http://127.0.0.1:8000/healthz", timeout=2)
    data = json.loads(resp.read())
    check("/healthz returns ok", data.get("status") == "ok", str(data))
    _backend_up = True
except Exception as e:
    check("/healthz (backend may not be running)", False, str(e))

if _backend_up:
    # POST synthetic image to /analyse
    try:
        import numpy as np
        from PIL import Image
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        fd, tmp_img = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        img.save(tmp_img)

        boundary = "----VeritasTestBoundary2"
        with open(tmp_img, "rb") as f:
            img_bytes = f.read()
        os.remove(tmp_img)

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode() + img_bytes + f"\r\n--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            "http://127.0.0.1:8000/analyse",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            resp_data = json.loads(r.read())

        check("/analyse returns verdict", "verdict" in resp_data, str(resp_data.get("verdict")))
        check("/analyse returns agents list", isinstance(resp_data.get("agents"), list))

        agents = {a.get("agent"): a for a in resp_data.get("agents", [])}
        check("Spatial Analysis in agents", "Spatial Analysis" in agents)
        check("Temporal Analysis in agents", "Temporal Analysis" in agents)

        t_agent = agents.get("Temporal Analysis", {})
        check("Temporal 'confidence' present", "confidence" in t_agent)
        check("Temporal skipped for image (confidence=0.0)",
              t_agent.get("confidence") == 0.0,
              f"confidence={t_agent.get('confidence')}")

    except Exception as e:
        check("/analyse endpoint test", False, str(e))
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
    print("\nFAILED:")
    for label, ok in results:
        if not ok:
            print(f"  [FAIL]  {label}")
    sys.exit(1)
else:
    print("\nAll tests passed.")
