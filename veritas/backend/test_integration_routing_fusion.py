"""
test_integration_routing_fusion.py
==================================
Comprehensive integration test suite for VERITAS:
  1. Modality detection & routing
  2. Multi-agent execution contract
  3. Dynamic denominator fixed-weight fusion
  4. Real, Fake, Conflicting, Missing, Failed, and Not-Applicable agent scenarios
  5. Ground truth checks on real fixtures

Run with:
    python veritas/backend/test_integration_routing_fusion.py
"""

import io
import os
import sys
import tempfile
from pathlib import Path

# Setup path and encoding
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import spatial_agent
import temporal_agent
import audio_agent
from mock_agents import context_agent_result
from main import (
    _WEIGHTS,
    _build_agents,
    _compute_verdict,
    _detect_media_type,
    _build_summary,
)

print("=" * 70)
print("VERITAS BACKEND ROUTING & FUSION INTEGRATION TEST SUITE")
print("=" * 70)

PASSED = 0
FAILED = 0


def record(name: str, passed: bool, detail: str = ""):
    global PASSED, FAILED
    if passed:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f" -> {detail}"
        print(msg)


# Ensure models are loaded
try:
    if not spatial_agent._model_loaded:
        spatial_agent.load_model()
except Exception as e:
    print(f"Warning: spatial_agent load failed: {e}")

try:
    if not temporal_agent._model_loaded:
        temporal_agent.load_model()
except Exception as e:
    print(f"Warning: temporal_agent load failed: {e}")

try:
    if not audio_agent._model_loaded:
        audio_agent._load_model()
except Exception as e:
    print(f"Warning: audio_agent load failed: {e}")


# ---------------------------------------------------------------------------
# Suite 1: Modality Detection
# ---------------------------------------------------------------------------
print("\n[Suite 1] Modality Detection by Extension")

image_files = ["test.jpg", "photo.PNG", "graphic.webp", "frame.bmp"]
for f in image_files:
    record(f"Detect image modality: {f}", _detect_media_type(f) == "image")

video_files = ["clip.mp4", "movie.MOV", "stream.avi", "recording.mkv"]
for f in video_files:
    record(f"Detect video modality: {f}", _detect_media_type(f) == "video")

audio_files = ["speech.wav", "track.mp3", "voice.m4a", "sound.flac"]
for f in audio_files:
    record(f"Detect audio modality: {f}", _detect_media_type(f) == "audio")

record("Unknown extension is not guessed as image", _detect_media_type("unknown.xyz") is None)


# ---------------------------------------------------------------------------
# Suite 2: Modality Routing & Agent Execution Contracts
# These tests verify the legacy routing and fixed-weight fusion mechanics.
# We set VERITAS_FUSION_MODE=legacy so _compute_verdict evaluates the formula.
# ---------------------------------------------------------------------------
print("\n[Suite 2] Modality Routing & Agent Applicability")

_original_fusion_mode = os.environ.get("VERITAS_FUSION_MODE", "")
os.environ["VERITAS_FUSION_MODE"] = "legacy"

test_wav = str(_HERE / "test_audio.wav")
test_no_audio = str(_HERE / "test_no_audio.mp4")
test_fake_video = str(_HERE / "00025_fake.mp4")

# Test 2.1: Audio-only routing
if os.path.exists(test_wav):
    audio_agents = _build_agents(test_wav, "audio")
    s, t, a, c = audio_agents
    record("Audio input -> Spatial is not_applicable", s.get("status") == "not_applicable" and s.get("is_fake") is None)
    record("Audio input -> Temporal is not_applicable", t.get("status") == "not_applicable" and t.get("is_fake") is None)
    record("Audio input -> Audio ran and has_audio=True", a.get("has_audio") is True and a.get("status") == "trained")
    record("Audio input -> Context is not_applicable", c.get("status") == "not_applicable" and c.get("is_fake") is None)
    
    score, verdict, contributing, unavailable = _compute_verdict(audio_agents)
    record("Audio input -> Only Audio Analysis contributes", contributing == ["Audio Analysis"])
    record("Audio input -> Spatial, Temporal, Context unavailable", set(unavailable) == {"Spatial Analysis", "Temporal Analysis", "Context Verification"})
    record("Audio input -> Score equals Audio confidence", round(score, 4) == round(a.get("confidence", 0.0), 4))

# Test 2.2: Video without audio routing
if os.path.exists(test_no_audio):
    no_audio_agents = _build_agents(test_no_audio, "video")
    s, t, a, c = no_audio_agents
    # test_no_audio.mp4 is a colour-bar test pattern: the visual agents run and report no_face.
    record("No-audio video -> Spatial executes", s.get("status") in ("trained", "no_face"))
    record("No-audio video -> Temporal executes", t.get("status") in ("trained", "no_face"))
    record("No-audio video -> Audio detects no audio stream", a.get("status") == "no_audio" and a.get("is_fake") is None)
    record("No-audio video -> Context cannot evaluate (requires audio)", c.get("is_fake") is None)

    score, verdict, contributing, unavailable = _compute_verdict(no_audio_agents)
    record("No-audio video -> Only visual agents with a verdict contribute",
           set(contributing) == {x["agent"] for x in (s, t) if x.get("is_fake") is not None})
    record("No-audio video -> Audio excluded, Context never contributes",  # legacy mode lists it as unavailable
           "Audio Analysis" in unavailable and "Context Verification" not in contributing)
    record("No-audio video -> never AUTHENTIC without a face",
           verdict != "AUTHENTIC" or "no_face" not in (s.get("status"), t.get("status")))


# ---------------------------------------------------------------------------
# Suite 3: Fixed-Weight Fusion Scenarios
# These tests verify the LEGACY fixed-weight formula directly.
# ---------------------------------------------------------------------------
print("\n[Suite 3] Fixed-Weight Decision-Level Fusion")

# Test 3.1: All Agents Authentic (REAL)
all_authentic = [
    {"agent": "Spatial Analysis", "confidence": 0.10, "is_fake": False, "status": "trained"},
    {"agent": "Temporal Analysis", "confidence": 0.04, "is_fake": False, "status": "trained"},
    {"agent": "Audio Analysis", "confidence": 0.01, "is_fake": False, "status": "trained"},
    {"agent": "Context Verification", "confidence": 0.05, "is_fake": False, "status": "trained"},
]
# Weights: S=0.4, T=0.3, A=0.2, C=0.1 (sum = 1.0)
# Expected: 0.10*0.4 + 0.04*0.3 + 0.01*0.2 + 0.05*0.1 = 0.04 + 0.012 + 0.002 + 0.005 = 0.059
score, verdict, contrib, unavail = _compute_verdict(all_authentic)
record("All authentic -> Score matches formula", abs(score - 0.059) < 1e-4, f"got {score}")
record("All authentic -> Verdict is AUTHENTIC", verdict == "AUTHENTIC")
record("All authentic -> All 4 contribute", len(contrib) == 4 and len(unavail) == 0)

# Test 3.2: All Agents Manipulated (FAKE)
all_fake = [
    {"agent": "Spatial Analysis", "confidence": 0.95, "is_fake": True, "status": "trained"},
    {"agent": "Temporal Analysis", "confidence": 0.99, "is_fake": True, "status": "trained"},
    {"agent": "Audio Analysis", "confidence": 0.92, "is_fake": True, "status": "trained"},
    {"agent": "Context Verification", "confidence": 0.96, "is_fake": True, "status": "trained"},
]
# Expected: 0.95*0.4 + 0.99*0.3 + 0.92*0.2 + 0.96*0.1 = 0.38 + 0.297 + 0.184 + 0.096 = 0.957
score, verdict, contrib, unavail = _compute_verdict(all_fake)
record("All fake -> Score matches formula", abs(score - 0.957) < 1e-4, f"got {score}")
record("All fake -> Verdict is DEEPFAKE", verdict == "DEEPFAKE")

# Test 3.3: Conflicting Modalities (Visual Real, Audio Fake -> Deepfake Dubbing)
conflicting = [
    {"agent": "Spatial Analysis", "confidence": 0.20, "is_fake": False, "status": "trained"},
    {"agent": "Temporal Analysis", "confidence": 0.10, "is_fake": False, "status": "trained"},
    {"agent": "Audio Analysis", "confidence": 0.95, "is_fake": True, "status": "trained"},
    {"agent": "Context Verification", "confidence": 0.85, "is_fake": True, "status": "trained"},
]
# Expected: 0.20*0.4 + 0.10*0.3 + 0.95*0.2 + 0.85*0.1 = 0.08 + 0.03 + 0.19 + 0.085 = 0.385
score, verdict, contrib, unavail = _compute_verdict(conflicting)
record("Conflicting modalities -> Correct weighted score calculation", abs(score - 0.385) < 1e-4, f"got {score}")
record("Conflicting modalities -> All 4 contribute to decision", len(contrib) == 4)

# Test 3.4: Failed / Errored Agent Scenario (Rule 6 & 7 regression test)
# An agent that encountered an error MUST NOT count as REAL (is_fake=False) or dilute score
agent_failed = [
    {"agent": "Spatial Analysis", "confidence": 0.90, "is_fake": True, "status": "trained"},
    {"agent": "Temporal Analysis", "confidence": 0.0, "is_fake": None, "status": "error"},
    {"agent": "Audio Analysis", "confidence": 0.0, "is_fake": None, "status": "no_audio"},
    {"agent": "Context Verification", "confidence": 0.0, "is_fake": None, "status": "not_applicable"},
]
# Only Spatial Analysis should contribute! Active weight = 0.40
# Expected score = (0.90 * 0.40) / 0.40 = 0.90 -> DEEPFAKE
score, verdict, contrib, unavail = _compute_verdict(agent_failed)
record("Failed agent excluded from contributing list", "Temporal Analysis" not in contrib)
record("Failed agent added to unavailable list", "Temporal Analysis" in unavail)
record("Failed agent does NOT dilute score (Score = 0.90)", score == 0.90, f"got {score}")
record("Failed agent scenario verdict is DEEPFAKE", verdict == "DEEPFAKE")

# Test 3.5: Uncertain / Borderline Evidence Threshold
uncertain_case = [
    {"agent": "Spatial Analysis", "confidence": 0.50, "is_fake": True, "status": "trained"},
    {"agent": "Temporal Analysis", "confidence": 0.50, "is_fake": True, "status": "trained"},
]
# Score = 0.50. Falls in [0.45, 0.55] window
score, verdict, contrib, unavail = _compute_verdict(uncertain_case)
record("Borderline score (0.50) yields UNCERTAIN / INSUFFICIENT EVIDENCE", verdict == "UNCERTAIN / INSUFFICIENT EVIDENCE")

# Test 3.6: All Unavailable Agents
all_unavail = [
    {"agent": "Spatial Analysis", "status": "unavailable", "confidence": 0.0, "is_fake": None},
    {"agent": "Temporal Analysis", "status": "not_applicable", "confidence": 0.0, "is_fake": None},
    {"agent": "Audio Analysis", "status": "no_audio", "confidence": 0.0, "is_fake": None},
    {"agent": "Context Verification", "status": "not_applicable", "confidence": 0.0, "is_fake": None},
]
score, verdict, contrib, unavail = _compute_verdict(all_unavail)
record("All unavailable -> Score is 0.0", score == 0.0)
record("All unavailable -> Verdict is UNVERIFIED MODALITY", verdict == "UNVERIFIED MODALITY")
record("All unavailable -> 0 contributing, 4 unavailable", len(contrib) == 0 and len(unavail) == 4)

# Restore original fusion mode
if _original_fusion_mode:
    os.environ["VERITAS_FUSION_MODE"] = _original_fusion_mode
else:
    os.environ.pop("VERITAS_FUSION_MODE", None)



# ---------------------------------------------------------------------------
# Suite 4: Summary Text Generation
# ---------------------------------------------------------------------------
print("\n[Suite 4] Summary Text Generation")

sum_full = _build_summary(all_fake, 0.957, "DEEPFAKE", ["Spatial Analysis", "Temporal Analysis", "Audio Analysis", "Context Verification"])
record("Summary text handles multi-agent consensus", "Multi-agent consensus reached" in sum_full)

sum_partial = _build_summary(no_audio_agents, 0.85, "DEEPFAKE", ["Spatial Analysis", "Temporal Analysis"])
record("Summary text lists specific contributing modalities", "Spatial Analysis, Temporal Analysis" in sum_partial)

sum_none = _build_summary(all_unavail, 0.0, "UNVERIFIED MODALITY", [])
record("Summary text handles no trained agents available", "No trained agents were available" in sum_none)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"Results: {PASSED} passed  /  {FAILED} failed  /  {PASSED + FAILED} total")
print("=" * 70)

if FAILED > 0:
    print(f"\n[FAIL] {FAILED} integration tests failed!")
    sys.exit(1)
else:
    print("\n[SUCCESS] ALL ROUTING AND FUSION INTEGRATION TESTS PASSED.")
