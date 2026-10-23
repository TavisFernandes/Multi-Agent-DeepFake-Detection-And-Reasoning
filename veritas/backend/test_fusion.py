"""
test_fusion.py

Tests legacy dynamic weight normalization and verdict fusion logic.
Verifies that:
1. Untrained modalities (is_fake=None) contribute ZERO weight to the final decision.
2. The overall_confidence is normalized strictly among contributing agents.
3. The correct verdict categories are assigned.

These numeric expectations document the *legacy* linear-average fusion surface.
Production default remains corrected_fixed; tests pin mode_override='legacy'.
"""

import sys, io
from pathlib import Path
import json

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from main import _compute_verdict, _WEIGHTS, _build_summary

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("============================================================")
print("VERIFYING DYNAMIC FUSION (legacy mode)")
print("============================================================")

def run_test(name, agents, expected_conf, expected_verdict, expected_contributing, expected_unavailable):
    print(f"\n[Test] {name}")
    conf, verdict, contributing, unavailable = _compute_verdict(
        agents, mode_override="legacy"
    )
    print(f"  Result: conf={conf} verdict='{verdict}'")
    print(f"  Contributing: {contributing}")
    print(f"  Unavailable: {unavailable}")
    
    assert verdict == expected_verdict, f"Expected verdict {expected_verdict}, got {verdict}"
    assert conf == expected_conf, f"Expected conf {expected_conf}, got {conf}"
    assert set(contributing) == set(expected_contributing), "Contributing agents mismatch"
    assert set(unavailable) == set(expected_unavailable), "Unavailable agents mismatch"
    print("[PASS]")

# Test 1: All agents trained and contributing
all_trained = [
    {"agent": "Spatial Analysis", "confidence": 0.8, "is_fake": True},
    {"agent": "Temporal Analysis", "confidence": 0.9, "is_fake": True},
    {"agent": "Audio Analysis", "confidence": 0.2, "is_fake": False},
    {"agent": "Context Verification", "confidence": 0.7, "is_fake": True},
]
# W: S=0.4, T=0.3, A=0.2, C=0.1. Total = 1.0
# Score = (0.8*0.4) + (0.9*0.3) + (0.2*0.2) + (0.7*0.1) = 0.32 + 0.27 + 0.04 + 0.07 = 0.70
run_test("All Trained", all_trained, 0.70, "DEEPFAKE", 
         ["Spatial Analysis", "Temporal Analysis", "Audio Analysis", "Context Verification"], [])

# Test 2: Audio and Context Mocked/Untrained (The 00025_fake.mp4 scenario)
mocked = [
    {"agent": "Spatial Analysis", "confidence": 0.48, "is_fake": False},
    {"agent": "Temporal Analysis", "confidence": 0.02, "is_fake": False},
    {"agent": "Audio Analysis", "confidence": 0, "is_fake": None},
    {"agent": "Context Verification", "confidence": 0, "is_fake": None},
]
# W: S=0.4, T=0.3. Total = 0.7
# Score = (0.48*0.4) + (0.02*0.3) = 0.192 + 0.006 = 0.198
# Normalized = 0.198 / 0.7 = 0.282857 -> 0.2829
run_test("Partial Mocked", mocked, 0.2829, "AUTHENTIC", 
         ["Spatial Analysis", "Temporal Analysis"], ["Audio Analysis", "Context Verification"])

# Test 3: Uncertain (close to 0.5)
uncertain = [
    {"agent": "Spatial Analysis", "confidence": 0.49, "is_fake": False},
    {"agent": "Temporal Analysis", "confidence": 0.52, "is_fake": True},
    {"agent": "Audio Analysis", "confidence": 0, "is_fake": None},
    {"agent": "Context Verification", "confidence": 0, "is_fake": None},
]
# W = 0.7
# Score = (0.49*0.4) + (0.52*0.3) = 0.196 + 0.156 = 0.352
# Normalized = 0.352 / 0.7 = 0.502857 -> 0.5029
run_test("Uncertain Evidence", uncertain, 0.5029, "UNCERTAIN / INSUFFICIENT EVIDENCE", 
         ["Spatial Analysis", "Temporal Analysis"], ["Audio Analysis", "Context Verification"])

# Test 4: All agents unavailable
all_missing = [
    {"agent": "Spatial Analysis", "confidence": 0, "is_fake": None},
    {"agent": "Temporal Analysis", "confidence": 0, "is_fake": None},
    {"agent": "Audio Analysis", "confidence": 0, "is_fake": None},
    {"agent": "Context Verification", "confidence": 0, "is_fake": None},
]
run_test("All Unavailable", all_missing, 0.0, "UNVERIFIED MODALITY",
         [], ["Spatial Analysis", "Temporal Analysis", "Audio Analysis", "Context Verification"])

# Test 5: IMAGE input — Temporal not applicable (is_fake=None), Spatial=0.631
# Regression for the is_fake=False dilution bug (Temporal stub poisoning fusion).
# Only Spatial contributes; weight 0.40 normalised over 0.40 → score = 0.631 → DEEPFAKE
image_input = [
    {"agent": "Spatial Analysis",     "confidence": 0.631, "is_fake": True},
    {"agent": "Temporal Analysis",    "confidence": 0.0,   "is_fake": None, "status": "not_applicable"},
    {"agent": "Audio Analysis",       "confidence": 0,     "is_fake": None, "status": "no_audio"},
    {"agent": "Context Verification", "confidence": 0,     "is_fake": None, "status": "mocked"},
]
_conf, _verdict, _contrib, _unavail = _compute_verdict(image_input, mode_override="legacy")
print(f"\n[Test] Image input — Spatial-only fusion (regression for is_fake=False dilution bug)")
print(f"  conf={_conf}  verdict='{_verdict}'")
print(f"  Contributing: {_contrib}")
print(f"  Unavailable:  {_unavail}")
assert _conf == 0.631,            f"Expected 0.631, got {_conf} — Temporal 0.0 is still diluting Spatial!"
assert _verdict == "DEEPFAKE",    f"Expected DEEPFAKE, got {_verdict}"
assert "Spatial Analysis"    in _contrib, "Spatial must be in contributing_agents"
assert "Temporal Analysis" not in _contrib, "Temporal must NOT be in contributing_agents for image input"
assert "Temporal Analysis"   in _unavail, "Temporal must be in unavailable_agents for image input"
assert "Audio Analysis"      in _unavail, "Audio must be in unavailable_agents for image input"
assert "Context Verification" in _unavail, "Context must be in unavailable_agents for image input"
print("[PASS]")

# Test 6: AUDIO input — Spatial not applicable (status="not_applicable", is_fake=None)
# Regression test: Spatial must NOT contribute to fusion for audio-only uploads.
# Trained Audio agent reports 0.82 FAKE.
audio_input_trained = [
    {"agent": "Spatial Analysis",     "confidence": 0.0,   "is_fake": None, "status": "not_applicable"},
    {"agent": "Temporal Analysis",    "confidence": 0.0,   "is_fake": None, "status": "not_applicable"},
    {"agent": "Audio Analysis",       "confidence": 0.82,  "is_fake": True, "status": "trained"},
    {"agent": "Context Verification", "confidence": 0,     "is_fake": None, "status": "not_applicable"},
]
_conf, _verdict, _contrib, _unavail = _compute_verdict(audio_input_trained, mode_override="legacy")
print(f"\n[Test] Audio input — Trained audio only (regression for Spatial is_fake=False bug)")
print(f"  conf={_conf}  verdict='{_verdict}'")
print(f"  Contributing: {_contrib}")
print(f"  Unavailable:  {_unavail}")
assert _conf == 0.82,            f"Expected 0.82, got {_conf} — Spatial 0.0 is still diluting Audio!"
assert _verdict == "DEEPFAKE",    f"Expected DEEPFAKE, got {_verdict}"
assert "Spatial Analysis" not in _contrib, "Spatial must NOT be in contributing_agents for audio input"
assert "Spatial Analysis" in _unavail, "Spatial must be in unavailable_agents for audio input"
assert "Temporal Analysis" not in _contrib, "Temporal must NOT be in contributing_agents for audio input"
assert "Audio Analysis" in _contrib, "Audio must be in contributing_agents"
assert "Context Verification" in _unavail, "Context must be in unavailable_agents"
print("[PASS]")

# Test 7: AUDIO input — Untrained Audio (all agents unavailable)
audio_input_untrained = [
    {"agent": "Spatial Analysis",     "confidence": 0.0,   "is_fake": None, "status": "not_applicable"},
    {"agent": "Temporal Analysis",    "confidence": 0.0,   "is_fake": None, "status": "not_applicable"},
    {"agent": "Audio Analysis",       "confidence": 0,     "is_fake": None, "status": "untrained"},
    {"agent": "Context Verification", "confidence": 0,     "is_fake": None, "status": "not_applicable"},
]
_conf, _verdict, _contrib, _unavail = _compute_verdict(audio_input_untrained, mode_override="legacy")
print(f"\n[Test] Audio input — Untrained audio (all unavailable)")
print(f"  conf={_conf}  verdict='{_verdict}'")
print(f"  Contributing: {_contrib}")
print(f"  Unavailable:  {_unavail}")
assert _conf == 0.0, f"Expected 0.0, got {_conf}"
assert _verdict == "UNVERIFIED MODALITY", f"Expected UNVERIFIED MODALITY, got {_verdict}"
assert len(_contrib) == 0, f"No agents should contribute, got {_contrib}"
assert len(_unavail) == 4, f"All 4 agents should be unavailable, got {_unavail}"
print("[PASS]")

# Test 8: Unavailable status exclusion safeguard
# Even if an agent has non-null is_fake, status="not_applicable" prevents it from contributing
safeguard_input = [
    {"agent": "Spatial Analysis",     "confidence": 0.0,   "is_fake": False, "status": "not_applicable"},
    {"agent": "Temporal Analysis",    "confidence": 0.0,   "is_fake": None,  "status": "not_applicable"},
    {"agent": "Audio Analysis",       "confidence": 0.75,  "is_fake": True,  "status": "trained"},
    {"agent": "Context Verification", "confidence": 0,     "is_fake": None,  "status": "not_applicable"},
]
_conf, _verdict, _contrib, _unavail = _compute_verdict(safeguard_input, mode_override="legacy")
print(f"\n[Test] Safeguard exclusion — status='not_applicable' cannot contribute")
assert "Spatial Analysis" not in _contrib, "Spatial with status='not_applicable' must NOT contribute"
assert "Spatial Analysis" in _unavail, "Spatial must be in unavailable_agents"
assert _conf == 0.75, f"Expected 0.75, got {_conf}"
print("[PASS]")

print("\nTesting Summary Strings...")
conf, verdict, contributing, unavailable = _compute_verdict(all_trained, mode_override="legacy")
summary = _build_summary(all_trained, conf, verdict, contributing)
assert "Multi-agent consensus reached" in summary

conf, verdict, contributing, unavailable = _compute_verdict(mocked, mode_override="legacy")
summary = _build_summary(mocked, conf, verdict, contributing)
assert "Multi-agent consensus reached" not in summary
assert "Assessment based on available modalities: Spatial Analysis, Temporal Analysis" in summary

conf, verdict, contributing, unavailable = _compute_verdict(all_missing, mode_override="legacy")
summary = _build_summary(all_missing, conf, verdict, contributing)
assert "No trained agents were available" in summary

# Contract: uncertain summary must not claim authenticity
uncertain_details = {
    "predicted_category": "AUDIO_MANIPULATED",
    "fake_probability": 0.51,
    "leading_category_is_hypothesis": True,
}
unc_summary = _build_summary(
    all_trained, 0.51, "UNCERTAIN / INSUFFICIENT EVIDENCE",
    ["Spatial Analysis", "Temporal Analysis", "Audio Analysis", "Context Verification"],
    uncertain_details,
)
assert "inconclusive" in unc_summary.lower()
assert "not a definitive authentic" in unc_summary.lower()
assert "authentic media" not in unc_summary.lower()

# Production default corrected_fixed still emits contract fields
_c, _v, _co, _u, _d = _compute_verdict(
    all_trained, return_details=True, mode_override="corrected_fixed"
)
assert _d.get("method") == "corrected_fixed"
assert "fake_probability" in _d
assert "decision_thresholds" in _d
assert "verdict_class" in _d
assert "category_probabilities" in _d

print("[PASS] Summary strings generated correctly.")

print("\n============================================================")
print("ALL FUSION TESTS PASSED")
print("============================================================")
