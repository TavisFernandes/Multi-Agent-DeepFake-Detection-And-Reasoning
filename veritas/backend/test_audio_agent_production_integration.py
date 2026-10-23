"""
test_audio_agent_production_integration.py
=========================================
Comprehensive integration test suite for the VERITAS Audio Agent:
1. Safe checkpoint loading with weights_only=True and strict=True.
2. Parameter count and architectural head verification.
3. Decision threshold loading and calibrated value verification (0.50).
4. Preprocessing parity between training pipeline and production agent.
5. End-to-end execution on authentic speech (a real dataset clip) and the no-speech guard (test_audio.wav, a tone).
6. End-to-end execution on deepfake audio (fake video).
7. Standalone model vs Production agent output numerical parity.
8. CPU inference verification and GPU/CPU cross-validation.
9. Error handling for missing files, files without audio track, and corrupt files.
10. Schema and semantic verification of all return fields for VERITAS fusion.
"""

import os
import sys
import tempfile
from pathlib import Path

# Setup paths
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "training"))

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

import audio_agent
import train_audio_kaggle
import train_audio_lc

PASS = 0
FAIL = 0

def ok(label: str):
    global PASS
    PASS += 1
    print(f"  [PASS] {label}")

def fail(label: str, detail: str = ""):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {label} -- {detail}" if detail else f"  [FAIL] {label}")

def check(condition: bool, label: str, detail: str = ""):
    if condition:
        ok(label)
    else:
        fail(label, detail)

print("=" * 70)
print("VERITAS AUDIO AGENT -- PRODUCTION INTEGRATION VERIFICATION")
print("=" * 70)

# ---------------------------------------------------------------------------
# Test Group 1: Safe Checkpoint Loading & State Dict Integrity
# ---------------------------------------------------------------------------
print("\n[Group 1] Safe Production Checkpoint Loading & Architecture")
ckpt_path = audio_agent._CHECKPOINT_PATH.resolve()  # the checkpoint production loads
check(ckpt_path.exists(), f"Production checkpoint exists at {ckpt_path}")

# Load using weights_only=True
ckpt_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
check(isinstance(ckpt_data, dict), "Checkpoint loads successfully with weights_only=True")
check("model_state_dict" in ckpt_data, "Checkpoint contains 'model_state_dict'")
check("decision_threshold" in ckpt_data, "Checkpoint contains 'decision_threshold'")
check(ckpt_data.get("decision_threshold") == 0.50, f"Calibrated threshold is 0.50 (got {ckpt_data.get('decision_threshold')})")

# Verify strict loading into production architecture
model = efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(1280, 1)
missing, unexpected = model.load_state_dict(ckpt_data["model_state_dict"], strict=True)
check(len(missing) == 0, f"Zero missing keys on strict load (got {len(missing)})")
check(len(unexpected) == 0, f"Zero unexpected keys on strict load (got {len(unexpected)})")

total_elements = sum(v.numel() for v in ckpt_data["model_state_dict"].values())
learnable_params = sum(p.numel() for p in model.parameters())
check(total_elements == 4050894, f"Checkpoint state_dict elements match exact expected 4,050,894 (got {total_elements})")
check(learnable_params == 4008829, f"Learnable parameter count matches expected 4,008,829 (got {learnable_params})")

# ---------------------------------------------------------------------------
# Test Group 2: Production Agent Eager Loading & Threshold Alignment
# ---------------------------------------------------------------------------
print("\n[Group 2] Production Agent State & Threshold Alignment")
audio_agent.load_model(str(ckpt_path))
check(audio_agent._model_loaded is True, "audio_agent._model_loaded is True")
check(audio_agent._model is not None, "audio_agent._model is initialized")
check(audio_agent._DECISION_THRESHOLD == 0.50, f"audio_agent._DECISION_THRESHOLD is 0.50 (got {audio_agent._DECISION_THRESHOLD})")

# ---------------------------------------------------------------------------
# Test Group 3: Standalone vs Production Preprocessing & Output Parity
# ---------------------------------------------------------------------------
print("\n[Group 3] Preprocessing and Forward Output Parity")
# Speech from a real dataset clip: test_audio.wav is a pure tone, which the agent refuses to score (Group 6).
real_video = Path(r"D:\VERITAS_TRAINING\data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2\RealVideo-RealAudio\African\men\id00076\00109.mp4")
check(real_video.exists(), f"Real speech source exists: {real_video}")
test_wav = Path(audio_agent._extract_wav(str(real_video)))

# Preprocessing parity
prod_tensor = audio_agent._wav_to_tensor(str(test_wav)).cpu()
# Production loads audio_agent_lc.pth, so parity is against its trainer's preprocessing.
stand_tensor, stand_err = train_audio_lc.wav_to_mel_tensor(str(test_wav)), None

check(prod_tensor is not None and stand_err is None, "Both preprocessors succeeded without error")
check(prod_tensor.shape == (1, 3, 224, 224), f"Production tensor shape is (1, 3, 224, 224) (got {prod_tensor.shape})")
check(stand_tensor.shape == (3, 224, 224), f"Standalone tensor shape is (3, 224, 224) (got {stand_tensor.shape})")

preproc_diff = torch.max(torch.abs(prod_tensor.squeeze(0) - stand_tensor)).item()
check(preproc_diff < 1e-5, f"Preprocessing parity confirmed: max abs diff = {preproc_diff:.2e}")

# Forward pass numerical parity on GPU (if available) or CPU
device = audio_agent._DEVICE
model.to(device).eval()
with torch.no_grad():
    stand_logit = model(prod_tensor.to(device)).squeeze()
    stand_prob = torch.sigmoid(stand_logit).item()

res_prod = audio_agent.run_audio_agent(str(test_wav))
prod_prob = res_prod["fake_probability"]

forward_diff = abs(stand_prob - prod_prob)
check(round(stand_prob, 4) == prod_prob, f"Inference probability parity: standalone rounded={round(stand_prob, 4):.4f} == prod={prod_prob:.4f}")
check(forward_diff <= 5e-5, f"Unrounded vs rounded difference {forward_diff:.2e} <= 5e-5 (rounding quantum)")

# ---------------------------------------------------------------------------
# Test Group 4: End-to-End Real and Fake Media Smoke Tests
# ---------------------------------------------------------------------------
print("\n[Group 4] End-to-End Media Smoke Tests & Semantics")

# 1. Reference authentic speech WAV
check(res_prod["status"] in ("ok", "trained"), f"Real speech WAV status == 'trained' (got {res_prod['status']})")
check(res_prod["is_fake"] is False, f"Real speech WAV correctly classified as REAL (is_fake={res_prod['is_fake']})")
check(res_prod["fake_probability"] < 0.50, f"Real speech WAV fake_prob < 0.50 (got {res_prod['fake_probability']:.6f})")
check(res_prod["confidence"] == res_prod["fake_probability"], f"Confidence matches fake_prob score for fusion ({res_prod['confidence']})")

# 2. Real video from dataset
if real_video.exists():
    res_real = audio_agent.run_audio_agent(str(real_video))
    check(res_real["status"] in ("ok", "trained"), f"Real video {real_video.name} status == 'trained'")
    check(res_real["is_fake"] is False, f"Real video correctly classified as REAL (is_fake={res_real['is_fake']})")
    check(res_real["fake_probability"] < 0.50, f"Real video fake_prob < 0.50 (got {res_real['fake_probability']:.6f})")
    check(res_real["confidence"] == res_real["fake_probability"], f"Real confidence matches fake_prob score ({res_real['confidence']})")
else:
    print(f"  [SKIP] Real video {real_video} not found")

# 3. Fake video from dataset
fake_video = Path(r"D:\VERITAS_TRAINING\data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2\FakeVideo-FakeAudio\African\men\id00076\00109_10_id00476_wavtolip.mp4")
if fake_video.exists():
    res_fake = audio_agent.run_audio_agent(str(fake_video))
    check(res_fake["status"] in ("ok", "trained"), f"Fake video {fake_video.name} status == 'trained'")
    check(res_fake["is_fake"] is True, f"Fake video correctly classified as FAKE (is_fake={res_fake['is_fake']})")
    check(res_fake["fake_probability"] >= 0.50, f"Fake video fake_prob >= 0.50 (got {res_fake['fake_probability']:.6f})")
    check(res_fake["confidence"] == res_fake["fake_probability"], f"Fake confidence matches fake_prob score ({res_fake['confidence']})")
else:
    print(f"  [SKIP] Fake video {fake_video} not found")

# ---------------------------------------------------------------------------
# Test Group 5: CPU Inference Verification
# ---------------------------------------------------------------------------
print("\n[Group 5] CPU Inference Validation & Cross-Device Parity")
orig_device = audio_agent._DEVICE
audio_agent._DEVICE = torch.device("cpu")
audio_agent._model.to(torch.device("cpu"))

res_cpu = audio_agent.run_audio_agent(str(test_wav))
check(res_cpu["status"] in ("ok", "trained"), "CPU inference succeeded with status == 'trained'")
check(res_cpu["is_fake"] is False, "CPU inference correctly classified the real speech WAV as REAL")
cpu_gpu_diff = abs(res_cpu["fake_probability"] - prod_prob)
check(cpu_gpu_diff < 1e-4, f"CPU vs GPU probability diff within numerical tolerance: {cpu_gpu_diff:.2e}")

# Restore GPU
audio_agent._DEVICE = orig_device
audio_agent._model.to(orig_device)

# ---------------------------------------------------------------------------
# Test Group 6: Error Handling for Invalid/Edge-Case Audio
# ---------------------------------------------------------------------------
print("\n[Group 6] Error Handling & Edge Cases")

# 1. Missing file
res_missing = audio_agent.run_audio_agent("non_existent_file_xyz123.mp4")
check(res_missing["status"] in ("no_audio", "error"), f"Missing file returns status='{res_missing['status']}'")
check(res_missing["is_fake"] is None, "Missing file returns is_fake=None (never falsely labeled REAL)")
check(res_missing["confidence"] == 0, "Missing file returns confidence=0")

# 2. Video without audio track
no_audio_file = _HERE / "test_no_audio.mp4"
if no_audio_file.exists():
    res_no_audio = audio_agent.run_audio_agent(str(no_audio_file))
    check(res_no_audio["status"] == "no_audio", f"No-audio video returns status='no_audio' (got '{res_no_audio['status']}')")
    check(res_no_audio["has_audio"] is False, "No-audio video reports has_audio=False")
    check(res_no_audio["is_fake"] is None, "No-audio video returns is_fake=None")
    check(res_no_audio["confidence"] == 0, "No-audio video returns confidence=0")

# 3. Pure tone: not speech, so no score (never AUTHENTIC)
res_tone = audio_agent.run_audio_agent(str(_HERE / "test_audio.wav"))
check(res_tone["status"] == "unavailable" and res_tone["is_fake"] is None and res_tone["confidence"] == 0,
      f"1 kHz tone returns status='unavailable', is_fake=None (got {res_tone['status']}, {res_tone['is_fake']})")
os.remove(test_wav)

# 4. Silent/Empty audio handling
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
    empty_path = tf.name
try:
    res_empty = audio_agent.run_audio_agent(empty_path)
    check(res_empty["is_fake"] is None, "Empty file returns is_fake=None")
    check(res_empty["confidence"] == 0, "Empty file returns confidence=0")
finally:
    if os.path.exists(empty_path):
        os.remove(empty_path)

# ---------------------------------------------------------------------------
# Test Group 7: Public Interface Schema Conformance
# ---------------------------------------------------------------------------
print("\n[Group 7] VERITAS API Schema Conformance")
required_keys = {
    "agent", "status", "has_audio", "codec", "sample_rate", "channels",
    "duration", "is_fake", "confidence", "fake_probability", "decision_threshold", "finding"
}
missing_keys = required_keys - set(res_prod.keys())
check(len(missing_keys) == 0, f"All required keys present in output dictionary (missing: {missing_keys})")
check(res_prod["agent"] == "Audio Analysis", "Agent identifier is 'Audio Analysis'")
check(isinstance(res_prod["finding"], str) and len(res_prod["finding"]) > 10, "Finding is a descriptive non-empty string")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
total = PASS + FAIL
print(f"Results: {PASS} passed  /  {FAIL} failed  /  {total} total")
print("=" * 70)

if FAIL > 0:
    sys.exit(1)
else:
    print("ALL PRODUCTION INTEGRATION VERIFICATION CHECKS PASSED.")
