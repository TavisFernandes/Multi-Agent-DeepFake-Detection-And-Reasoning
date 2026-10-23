"""
test_ppo_fusion.py
==================
Unit and integration test suite for the VERITAS PPO Adaptive Fusion policy.

Verifies:
1. 14-D State extraction and semantics preservation across modalities.
2. Feature Normalizer fit and transform invariants.
3. Continuous action mapping, softmax gating constraints, and threshold modulation.
4. Multiclass 4-category taxonomy alignment (Authentic, Audio-Fake, Video-Fake, Dual-Fake).
5. Cross-modal mismatch handling: audio-fake does NOT get swamped by real-video scores.
6. Availability masking and graceful handling of missing modalities (image-only, audio-only, failed agents).
7. Safe checkpoint saving and loading parity.
8. API response contract preservation with backend endpoints.
"""

import io
import math
import os
import sys
import tempfile
from pathlib import Path

# Setup path and encoding
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

from fusion_policy import (
    ACTION_DIM,
    CATEGORIES,
    CATEGORY_TO_IDX,
    DEFAULT_CHECKPOINT_PATH,
    IDX_TO_CATEGORY,
    STATE_DIM,
    ActorCriticNetwork,
    FeatureNormalizer,
    PPOFusionPolicy,
    apply_action_to_state,
    compute_reward,
    compute_verdict_corrected_fixed,
    compute_verdict_legacy,
    extract_state_from_agents,
)

PASSED = 0
FAILED = 0


def check(cond: bool, name: str, detail: str = ""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f" -> {detail}"
        print(msg)


print("=" * 70)
print("VERITAS PPO ADAPTIVE FUSION TEST SUITE")
print("=" * 70)


# ---------------------------------------------------------------------------
# Test Group 1: State Extraction & Semantics
# ---------------------------------------------------------------------------
print("\n[Group 1] State Extraction & Feature Semantics")

# Scenario: Complete 4-agent video with authentic video and fake audio (e.g. 00014_fake.mp4 pattern)
agents_mismatch = [
    {"agent": "Spatial Analysis", "confidence": 0.014, "is_fake": False, "status": "trained"},
    {"agent": "Temporal Analysis", "confidence": 0.001, "is_fake": False, "status": "trained"},
    {"agent": "Audio Analysis", "confidence": 1.000, "is_fake": True, "status": "trained", "has_audio": True, "duration": 4.5},
    {"agent": "Context Verification", "confidence": 0.990, "is_fake": True, "status": "trained"},
]

s_vec, meta = extract_state_from_agents(agents_mismatch, media_type="video")
check(len(s_vec) == STATE_DIM, f"State vector has length {STATE_DIM} (got {len(s_vec)})")
check(meta["spatial_valid"] is True, "Spatial agent marked valid")
check(meta["temporal_valid"] is True, "Temporal agent marked valid")
check(meta["audio_valid"] is True, "Audio agent marked valid")
check(meta["context_valid"] is True, "Context agent marked valid")
check(s_vec[0] == 0.014, f"Spatial prob is 0.014 (got {s_vec[0]})")
check(s_vec[3] == 0.001, f"Temporal prob is 0.001 (got {s_vec[3]})")
check(s_vec[6] == 1.0, f"Audio prob is 1.0 (got {s_vec[6]})")
check(s_vec[9] > 0.90, f"Context mismatch reflects divergence (~0.99, got {s_vec[9]:.4f})")
check(s_vec[12] == 1.0, "has_audio container flag is 1.0")
check(s_vec[13] == 1.0, "has_video container flag is 1.0")


# ---------------------------------------------------------------------------
# Test Group 2: Feature Normalizer
# ---------------------------------------------------------------------------
print("\n[Group 2] Feature Normalizer Invariants")

dummy_train_X = np.random.randn(100, STATE_DIM).astype(np.float32) * 2.0 + 5.0
norm = FeatureNormalizer(STATE_DIM)
norm.fit(dummy_train_X)
check(norm.count == 100, "Normalizer recorded 100 training samples")

norm_X = norm.transform(dummy_train_X)
check(abs(norm_X.mean()) < 0.01, f"Normalized mean is ~0.0 (got {norm_X.mean():.4f})")
check(abs(norm_X.std() - 1.0) < 0.05, f"Normalized std is ~1.0 (got {norm_X.std():.4f})")

# Serialization roundtrip
norm_dict = norm.to_dict()
norm_reloaded = FeatureNormalizer.from_dict(norm_dict)
check(np.allclose(norm.mean, norm_reloaded.mean), "Normalizer mean roundtrip identical")
check(np.allclose(norm.std, norm_reloaded.std), "Normalizer std roundtrip identical")


# ---------------------------------------------------------------------------
# Test Group 3: Action Mapping & Continuous Action Logic
# ---------------------------------------------------------------------------
print("\n[Group 3] Continuous Action Mapping & Constraints")

action = np.array([0.5, -0.5, 1.2, 0.8, 0.0, 0.1], dtype=np.float32)
res = apply_action_to_state(action, s_vec, meta)

check("verdict" in res, "Result contains verdict")
check("overall_confidence" in res, "Result contains overall_confidence")
check("dynamic_weights" in res, "Result contains dynamic_weights")
check("category_probabilities" in res, "Result contains category_probabilities")
check("predicted_category" in res, "Result contains predicted_category")

# Visual weights must sum to 1.0
w_sp = res["dynamic_weights"]["spatial"]
w_te = res["dynamic_weights"]["temporal"]
check(abs((w_sp + w_te) - 1.0) < 1e-4, f"Visual weights sum to 1.0 ({w_sp} + {w_te})")

# Category probabilities must sum to 1.0
cat_probs = res["category_probabilities"]
cat_sum = sum(cat_probs.values())
check(abs(cat_sum - 1.0) < 1e-3, f"Category probabilities sum to 1.0 (got {cat_sum})")

# For the mismatch case (Real video, fake audio), the predicted category MUST be AUDIO_MANIPULATED
check(res["predicted_category"] == "AUDIO_MANIPULATED",
      f"Category correctly identified as AUDIO_MANIPULATED (got {res['predicted_category']})")
check(res["verdict"] == "DEEPFAKE",
      f"Audio-fake media verdict is DEEPFAKE (got {res['verdict']})")
check(res["overall_confidence"] > 0.90,
      f"Overall confidence is high (>90%, got {res['overall_confidence']:.1%})")


# ---------------------------------------------------------------------------
# Test Group 4: Missing Modalities & Masking
# ---------------------------------------------------------------------------
print("\n[Group 4] Missing Modality & Incomplete Evidence Resilience")

# Scenario: Audio only (e.g. podcast or audio clip upload)
audio_only_agents = [
    {"agent": "Spatial Analysis", "confidence": 0.0, "is_fake": None, "status": "not_applicable"},
    {"agent": "Temporal Analysis", "confidence": 0.0, "is_fake": None, "status": "not_applicable"},
    {"agent": "Audio Analysis", "confidence": 0.88, "is_fake": True, "status": "trained", "has_audio": True},
    {"agent": "Context Verification", "confidence": 0.0, "is_fake": None, "status": "not_applicable"},
]
s_ao, meta_ao = extract_state_from_agents(audio_only_agents, media_type="audio")
check(meta_ao["spatial_valid"] is False, "Spatial correctly marked invalid for audio")
check(meta_ao["temporal_valid"] is False, "Temporal correctly marked invalid for audio")
check(meta_ao["audio_valid"] is True, "Audio marked valid for audio input")

res_ao = apply_action_to_state(action, s_ao, meta_ao)
check(res_ao["verdict"] == "DEEPFAKE", f"Audio-only fake produces DEEPFAKE (got {res_ao['verdict']})")
check(res_ao["dynamic_weights"]["spatial"] == 0.0, "Spatial weight masked to 0.0")
check(res_ao["dynamic_weights"]["temporal"] == 0.0, "Temporal weight masked to 0.0")
check("Spatial Analysis" in res_ao["unavailable_agents"], "Spatial in unavailable agents")
check("Audio Analysis" in res_ao["contributing_agents"], "Audio in contributing agents")

# Scenario: Image only (Spatial only)
image_only_agents = [
    {"agent": "Spatial Analysis", "confidence": 0.92, "is_fake": True, "status": "trained"},
    {"agent": "Temporal Analysis", "confidence": 0.0, "is_fake": None, "status": "not_applicable"},
    {"agent": "Audio Analysis", "confidence": 0.0, "is_fake": None, "status": "no_audio", "has_audio": False},
    {"agent": "Context Verification", "confidence": 0.0, "is_fake": None, "status": "not_applicable"},
]
s_io, meta_io = extract_state_from_agents(image_only_agents, media_type="image")
res_io = apply_action_to_state(action, s_io, meta_io)
check(res_io["verdict"] == "DEEPFAKE", f"Image fake produces DEEPFAKE (got {res_io['verdict']})")
check(res_io["dynamic_weights"]["spatial"] == 1.0, "Spatial takes 100% of visual weight")
check("Temporal Analysis" in res_io["unavailable_agents"], "Temporal in unavailable")
check("Audio Analysis" in res_io["unavailable_agents"], "Audio in unavailable")

# Scenario: All agents unavailable
all_missing_agents = [
    {"agent": "Spatial Analysis", "confidence": 0.0, "is_fake": None, "status": "unavailable"},
    {"agent": "Temporal Analysis", "confidence": 0.0, "is_fake": None, "status": "unavailable"},
    {"agent": "Audio Analysis", "confidence": 0.0, "is_fake": None, "status": "unavailable"},
    {"agent": "Context Verification", "confidence": 0.0, "is_fake": None, "status": "unavailable"},
]
s_all_miss, meta_all_miss = extract_state_from_agents(all_missing_agents)
res_all_miss = apply_action_to_state(action, s_all_miss, meta_all_miss)
check(res_all_miss["verdict"] == "UNVERIFIED MODALITY",
      f"All missing yields UNVERIFIED MODALITY (got {res_all_miss['verdict']})")
check(res_all_miss["overall_confidence"] == 0.0, "Confidence is 0.0 when unverified")


# ---------------------------------------------------------------------------
# Test Group 5: Reward Function
# ---------------------------------------------------------------------------
print("\n[Group 5] Objective Reward Calculation")

# Correct True Positive
r_tp = compute_reward(res, ground_truth_category=1)  # AUDIO_MANIPULATED
check(r_tp > 1.0, f"Correct detection of Audio-Fake yields positive reward (>1.0, got {r_tp:.2f})")

# Severe False Negative penalty (missing a deepfake)
res_fake_called_real = dict(res)
res_fake_called_real["verdict"] = "AUTHENTIC"
res_fake_called_real["overall_confidence"] = 0.10
res_fake_called_real["predicted_category"] = "AUTHENTIC"
res_fake_called_real["category_probabilities"] = {"AUTHENTIC": 0.90, "AUDIO_MANIPULATED": 0.05, "VIDEO_MANIPULATED": 0.03, "AUDIO_VIDEO_MANIPULATED": 0.02}

r_fn = compute_reward(res_fake_called_real, ground_truth_category=1)
check(r_fn < -2.5, f"False Negative correctly receives severe penalty (<-2.5, got {r_fn:.2f})")


# ---------------------------------------------------------------------------
# Test Group 6: Network Architecture & Checkpoint Parity
# ---------------------------------------------------------------------------
print("\n[Group 6] Network Architecture & Safe Checkpoint Parity")

net = ActorCriticNetwork(STATE_DIM, ACTION_DIM)
dummy_s = torch.randn(4, STATE_DIM)
dist, val = net(dummy_s)
check(dist.mean.shape == (4, ACTION_DIM), f"Actor action mean shape is (4, {ACTION_DIM})")
check(val.shape == (4,), f"Critic value shape is (4,)")

with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
    tmp_path = Path(tmp.name)

try:
    policy = PPOFusionPolicy()
    policy.network = net
    policy.normalizer = norm
    policy.save_checkpoint(tmp_path, extra_metadata={"test": "ok"})

    loaded_policy = PPOFusionPolicy(checkpoint_path=tmp_path)
    check(loaded_policy.is_loaded is True, "PPOFusionPolicy loads from file successfully")
    check(loaded_policy.checkpoint_metadata.get("test") == "ok", "Metadata preserved")

    # Output parity between original and reloaded
    c1, v1, _, _, _ = policy.fuse_agents(agents_mismatch)
    c2, v2, _, _, _ = loaded_policy.fuse_agents(agents_mismatch)
    check(c1 == c2, f"Numerical confidence parity on reloaded model ({c1} == {c2})")
    check(v1 == v2, f"Verdict parity on reloaded model ('{v1}' == '{v2}')")
finally:
    if tmp_path.exists():
        tmp_path.unlink()


# ---------------------------------------------------------------------------
# Test Group 7: Baseline Comparison (Diagnostic Scenario)
# ---------------------------------------------------------------------------
print("\n[Group 7] Baseline Contrast on Diagnostic Sample (00014_fake pattern)")

leg_conf, leg_verd, _, _, _ = compute_verdict_legacy(agents_mismatch)
cor_conf, cor_verd, _, _, _ = compute_verdict_corrected_fixed(agents_mismatch)

print(f"  Legacy result:          conf={leg_conf:.4f}  verdict='{leg_verd}'")
print(f"  Corrected Fixed result: conf={cor_conf:.4f}  verdict='{cor_verd}'")

check(leg_verd == "AUTHENTIC", "Legacy reproduces the documented flaw (falsely calls audio fake 'AUTHENTIC')")
check(cor_verd == "DEEPFAKE", "Corrected baseline identifies audio fake as 'DEEPFAKE'")
check(cor_conf >= 0.99, f"Corrected baseline confidence is ~100% fake ({cor_conf:.1%})")


print("\n" + "=" * 70)
print(f"Test Results: {PASSED} passed / {FAILED} failed / {PASSED + FAILED} total")
print("=" * 70)

if FAILED > 0:
    sys.exit(1)
