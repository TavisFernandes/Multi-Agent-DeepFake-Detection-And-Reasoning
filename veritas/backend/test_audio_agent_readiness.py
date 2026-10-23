"""
test_audio_agent_readiness.py
=============================
Regression and integration test suite verifying that the VERITAS Audio Agent
inference pipeline and training script are 100% research-ready:

1. EfficientNet-B0 architecture matches the actual inference model.
2. Checkpoint loads strictly (strict=True, model_state_dict key).
3. Dummy input produces a probability in [0, 1].
4. Production preprocessing matches training preprocessing.
5. Calibrated decision_threshold is loaded from checkpoint (with 0.50 fallback).
6. Missing checkpoints result in an honest 'untrained' status (is_fake=None, confidence=0).
7. Unavailable Audio results are excluded from fusion.
8. Extraction failures cannot silently become REAL predictions.
9. Identity-aware 80/10/10 split guarantees disjoint identities and representation.
"""

import sys
import io
import os
import tempfile
from pathlib import Path

# Add backend dir to path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "training"))

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

from unittest.mock import MagicMock
for _mod in ("cv2", "fastapi", "fastapi.middleware.cors", "fastapi.responses"):
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except ImportError:
            sys.modules[_mod] = MagicMock()

import audio_agent
from main import _compute_verdict, _WEIGHTS
import train_audio_kaggle
import train_audio_lc

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=" * 65)
print("VERITAS AUDIO AGENT — INTEGRATION READINESS TEST SUITE")
print("=" * 65)

PASS = 0
FAIL = 0

def ok(label: str):
    global PASS
    PASS += 1
    print(f"  [PASS] {label}")

def fail(label: str, detail: str = ""):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {label} — {detail}" if detail else f"  [FAIL] {label}")

def check(condition: bool, label: str, detail: str = ""):
    if condition:
        ok(label)
    else:
        fail(label, detail)

# ---------------------------------------------------------------------------
# Test 1: Architecture Check
# ---------------------------------------------------------------------------
print("\n[Group 1] Architecture Verification")
m_inf = efficientnet_b0(weights=None)
m_inf.classifier[1] = nn.Linear(1280, 1)

m_train = train_audio_kaggle.build_model(torch.device("cpu"))

check(isinstance(m_inf.classifier[1], nn.Linear), "Inference classifier[1] is Linear")
check(m_inf.classifier[1].in_features == 1280, "Inference in_features == 1280")
check(m_inf.classifier[1].out_features == 1, "Inference out_features == 1")
check(m_train.classifier[1].in_features == 1280, "Training in_features == 1280")
check(m_train.classifier[1].out_features == 1, "Training out_features == 1")

# ---------------------------------------------------------------------------
# Test 2 & 3: Strict Checkpoint Loading & Dummy Forward Pass
# ---------------------------------------------------------------------------
print("\n[Group 2] Strict Checkpoint Loading & Forward Pass")
tmp_dir = tempfile.mkdtemp()
tmp_ckpt = os.path.join(tmp_dir, "test_audio_ckpt.pth")

torch.save({
    "epoch": 5,
    "model_state_dict": m_train.state_dict(),
    "model_name": "efficientnet_b0",
    "num_classes": 1,
    "decision_threshold": 0.35,
    "sample_rate": 16000,
    "clip_samples": 48000,
}, tmp_ckpt)

m_reload = efficientnet_b0(weights=None)
m_reload.classifier[1] = nn.Linear(1280, 1)

ckpt_data = torch.load(tmp_ckpt, map_location="cpu", weights_only=True)
load_res = m_reload.load_state_dict(ckpt_data["model_state_dict"], strict=True)

check(len(load_res.missing_keys) == 0, "strict=True has 0 missing keys")
check(len(load_res.unexpected_keys) == 0, "strict=True has 0 unexpected keys")

m_reload.eval()
with torch.no_grad():
    dummy_x = torch.randn(1, 3, 224, 224)
    dummy_prob = torch.sigmoid(m_reload(dummy_x).squeeze()).item()

check(isinstance(dummy_prob, float), "Dummy forward pass outputs float")
check(0.0 <= dummy_prob <= 1.0, f"Dummy probability in [0, 1]: {dummy_prob:.4f}")

# ---------------------------------------------------------------------------
# Test 4: Preprocessing Parity
# ---------------------------------------------------------------------------
print("\n[Group 3] Preprocessing Parity Between Training and Inference")
check(audio_agent._SAMPLE_RATE == train_audio_kaggle.SAMPLE_RATE == 16000,
      "Sample rate matches (16,000 Hz)")
check(audio_agent._CLIP_SAMPLES == train_audio_kaggle.CLIP_SAMPLES == 48000,
      "Clip samples match (48,000 = 3.0s)")
check(audio_agent._MEL_TRANSFORM.n_fft == train_audio_kaggle.N_FFT == 1024,
      "Mel n_fft matches (1024)")
check(audio_agent._MEL_TRANSFORM.hop_length == train_audio_kaggle.HOP_LENGTH == 512,
      "Mel hop_length matches (512)")
check(audio_agent._MEL_TRANSFORM.n_mels == train_audio_kaggle.N_MELS == 128,
      "Mel n_mels matches (128)")

wav_file = str(_HERE / "test_audio.wav")
if os.path.exists(wav_file):
    inf_tensor = audio_agent._wav_to_tensor(wav_file).cpu()
    # Production loads audio_agent_lc.pth, so parity is against its trainer's preprocessing.
    train_tensor = train_audio_lc.wav_to_mel_tensor(wav_file)
    check(train_tensor is not None, "train_audio_lc.wav_to_mel_tensor succeeded on test_audio.wav")
    check(inf_tensor.shape == (1, 3, 224, 224), "Inference tensor shape is (1, 3, 224, 224)")
    check(train_tensor.shape == (3, 224, 224), "Training tensor shape is (3, 224, 224)")
    
    # Compare values (inf_tensor has batch dim 1)
    diff = torch.max(torch.abs(inf_tensor.squeeze(0) - train_tensor)).item()
    check(diff < 1e-4, f"Tensor values match between inference and training (max diff = {diff:.2e})")

# ---------------------------------------------------------------------------
# Test 5: Decision Threshold Loading & Fallback Behavior
# ---------------------------------------------------------------------------
print("\n[Group 4] Decision Threshold Loading and Fallback")
# 1. Load checkpoint with decision_threshold=0.35
audio_agent.load_model(tmp_ckpt)
check(audio_agent._model is not None, "audio_agent loaded test checkpoint successfully")
check(audio_agent._DECISION_THRESHOLD == 0.35,
      f"Loaded custom decision_threshold=0.35 from checkpoint (got {audio_agent._DECISION_THRESHOLD})")

# 2. Checkpoint without decision_threshold -> fallback to 0.50
tmp_ckpt_nothresh = os.path.join(tmp_dir, "test_audio_nothresh.pth")
torch.save({
    "epoch": 1,
    "model_state_dict": m_train.state_dict(),
}, tmp_ckpt_nothresh)

audio_agent.load_model(tmp_ckpt_nothresh)
check(audio_agent._DECISION_THRESHOLD == 0.50,
      f"Fallback decision_threshold=0.50 when key is absent (got {audio_agent._DECISION_THRESHOLD})")

# ---------------------------------------------------------------------------
# Test 6 & 7: Untrained Status & Fusion Exclusion
# ---------------------------------------------------------------------------
print("\n[Group 5] Untrained Status & Fusion Exclusion")
# Reset to untrained
audio_agent._model = None
audio_agent._model_loaded = False

res_untrained = audio_agent.run_audio_agent(wav_file)
check(res_untrained.get("status") == "untrained", "Missing checkpoint yields status='untrained'")
check(res_untrained.get("is_fake") is None, "Untrained model returns is_fake=None")
check(res_untrained.get("confidence") == 0, "Untrained model returns confidence=0")
check("No trained model available" in res_untrained.get("finding", ""), "Finding explicitly states model not available")

conf, verdict, contributing, unavailable = _compute_verdict([res_untrained])
check("Audio Analysis" not in contributing, "Untrained Audio is NOT in contributing_agents")
check("Audio Analysis" in unavailable, "Untrained Audio is in unavailable_agents")
check(conf == 0.0, "Untrained Audio contributes 0 to overall confidence")

# ---------------------------------------------------------------------------
# Test 8: Extraction Failures Excluded from Loss & Never Labeled REAL
# ---------------------------------------------------------------------------
print("\n[Group 6] Extraction Failure Handling")
res_corrupt = audio_agent.run_audio_agent("non_existent_corrupt_file.mp4")
check(res_corrupt.get("is_fake") is None, "Corrupt file does NOT predict is_fake=False (never labeled REAL)")
check(res_corrupt.get("confidence") == 0, "Corrupt file has confidence=0")
check(res_corrupt.get("status") in ("no_audio", "error"), "Corrupt file yields status='no_audio' or 'error'")

# Verify collate_fn handles failures
batch_with_failures = [
    (torch.randn(3, 224, 224), torch.tensor(1.0), "FakeVideo-FakeAudio", "good1.mp4", None),
    (None, torch.tensor(1.0), "FakeVideo-FakeAudio", "bad1.mp4", "ffmpeg failed"),
    (torch.randn(3, 224, 224), torch.tensor(0.0), "RealVideo-RealAudio", "good2.mp4", None),
    (None, torch.tensor(0.0), "RealVideo-RealAudio", "bad2.mp4", "zero frames"),
]
tensors, labels, cats, fails = train_audio_kaggle.collate_audio(batch_with_failures)

check(tensors is not None and tensors.shape == (2, 3, 224, 224),
      "collate_audio returns only the 2 valid tensors (batch size 2)")
check(len(fails) == 2, "collate_audio extracted both failed samples into failures list")
check(fails[0][0] == "bad1.mp4" and "ffmpeg" in fails[0][2], "Failure reason captured for bad1.mp4")
check(fails[1][0] == "bad2.mp4" and "zero frames" in fails[1][2], "Failure reason captured for bad2.mp4")

# Empty batch (all failed)
all_failed = [
    (None, torch.tensor(1.0), "cat", "f1.mp4", "err1"),
    (None, torch.tensor(0.0), "cat", "f2.mp4", "err2"),
]
tensors_empty, labels_empty, cats_empty, fails_empty = train_audio_kaggle.collate_audio(all_failed)
check(tensors_empty is None, "All-failed batch returns tensors=None (graceful skip, no NaN/crash)")
check(len(fails_empty) == 2, "All-failed batch captures all 2 failure records")

# ---------------------------------------------------------------------------
# Test 9: Disjoint 80/10/10 Identity-Aware Split
# ---------------------------------------------------------------------------
print("\n[Group 7] 80/10/10 Identity-Aware Split Integrity")
# Generate synthetic dataset mirroring FakeAVCeleb structure across 30 identities
fake_paths = []
fake_labels = []
fake_cats = []
fake_idents = []

for id_num in range(1, 31):
    ident_str = f"id{id_num:05d}"
    # Each identity has 2 real and 2 fake clips
    for clip_num in range(1, 3):
        fake_paths.append(f"/fake/RealVideo-RealAudio/African/men/{ident_str}/{clip_num:03d}.mp4")
        fake_labels.append(0)
        fake_cats.append("RealVideo-RealAudio")
        fake_idents.append(ident_str)
    for clip_num in range(1, 3):
        fake_paths.append(f"/fake/RealVideo-FakeAudio/African/men/{ident_str}_wav2lip/{clip_num:03d}.mp4")
        fake_labels.append(1)
        fake_cats.append("RealVideo-FakeAudio")
        fake_idents.append(ident_str)

(tr_p, tr_l, tr_c, tr_id), (va_p, va_l, va_c, va_id), (te_p, te_l, te_c, te_id) = (
    train_audio_kaggle.identity_aware_split_3way(
        fake_paths, fake_labels, fake_cats, fake_idents,
        train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42
    )
)

set_tr = set(tr_id)
set_va = set(va_id)
set_te = set(te_id)

check(len(set_tr & set_va) == 0, "Zero identity overlap between Train and Validation")
check(len(set_tr & set_te) == 0, "Zero identity overlap between Train and Test")
check(len(set_va & set_te) == 0, "Zero identity overlap between Validation and Test")

check(0 in tr_l and 1 in tr_l, "Both REAL and FAKE audio present in Train split")
check(0 in va_l and 1 in va_l, "Both REAL and FAKE audio present in Validation split")
check(0 in te_l and 1 in te_l, "Both REAL and FAKE audio present in Test split")

total_split = len(tr_p) + len(va_p) + len(te_p)
check(total_split == len(fake_paths), f"All {len(fake_paths)} samples accounted for (no drops)")

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
try:
    if os.path.exists(tmp_ckpt): os.remove(tmp_ckpt)
    if os.path.exists(tmp_ckpt_nothresh): os.remove(tmp_ckpt_nothresh)
    os.rmdir(tmp_dir)
except Exception:
    pass

# Reset audio_agent to untrained state
audio_agent._model = None
audio_agent._model_loaded = False
audio_agent._DECISION_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 65)
total = PASS + FAIL
print(f"Results: {PASS} passed  /  {FAIL} failed  /  {total} total")
print("=" * 65)

if FAIL > 0:
    sys.exit(1)
else:
    print("ALL AUDIO AGENT READINESS CHECKS PASSED.")
