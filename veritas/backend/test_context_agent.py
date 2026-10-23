"""
test_context_agent.py

Verifies the cross-modal consistency context agent (mock_agents.context_agent_result)
and its interaction with the fusion logic in main._compute_verdict.
"""
import sys, io
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from mock_agents import context_agent_result
from main import _compute_verdict

print("=" * 60)
print("CONTEXT AGENT — CROSS-MODAL CONSISTENCY")
print("=" * 60)

def spatial(conf, fake): return {"agent": "Spatial Analysis", "confidence": conf, "is_fake": fake}
def temporal(conf, fake): return {"agent": "Temporal Analysis", "confidence": conf, "is_fake": fake}
def audio(conf, fake): return {"agent": "Audio Analysis", "confidence": conf, "is_fake": fake}
def na(agent): return {"agent": agent, "confidence": 0, "is_fake": None, "status": "not_applicable"}

# 1. Legacy no-arg call → unavailable (backwards compatible with old stub)
r = context_agent_result()
assert r["is_fake"] is None, "no-arg call must be unavailable"
print("[PASS] no-arg call → is_fake=None (legacy compatible)")

# 2. Image-only (audio N/A) → context cannot assess → is_fake=None
r = context_agent_result(spatial(0.63, True), na("Temporal Analysis"), na("Audio Analysis"))
assert r["is_fake"] is None, f"image-only must be unavailable, got {r['is_fake']}"
print("[PASS] image-only (no audio) → is_fake=None")

# 3. Audio-only (no visual) → is_fake=None
r = context_agent_result(None, None, audio(0.8, True))
assert r["is_fake"] is None
print("[PASS] audio-only (no visual) → is_fake=None")

# 3b. Audio-only with Spatial and Temporal marked not_applicable → is_fake=None
r = context_agent_result(na("Spatial Analysis"), na("Temporal Analysis"), audio(0.8, True))
assert r["is_fake"] is None, f"audio-only with N/A visual must be unavailable, got {r['is_fake']}"
print("[PASS] audio-only (spatial/temporal N/A) → is_fake=None")

# 4. Agreement, both FAKE → context FAKE, confidence ≈ mean
r = context_agent_result(spatial(0.9, True), temporal(0.8, True), audio(0.9, True))
assert r["is_fake"] is True, r
assert 0.8 <= r["confidence"] <= 0.95, r["confidence"]
print(f"[PASS] both-fake agreement → is_fake=True conf={r['confidence']}")

# 5. Agreement, both REAL → context REAL
r = context_agent_result(spatial(0.05, False), temporal(0.03, False), audio(0.1, False))
assert r["is_fake"] is False, r
assert r["confidence"] < 0.2, r["confidence"]
print(f"[PASS] both-real agreement → is_fake=False conf={r['confidence']}")

# 6. Mismatch: real video, fake audio (RealVideo-FakeAudio) → FLAG manipulation
r = context_agent_result(spatial(0.1, False), temporal(0.05, False), audio(0.95, True))
assert r["is_fake"] is True, f"mismatch must flag fake, got {r}"
assert r["confidence"] > 0.5, r["confidence"]
assert "inconsistency" in r["finding"].lower()
print(f"[PASS] real-video/fake-audio mismatch → is_fake=True conf={r['confidence']}")

# 7. Mismatch: fake video, real audio (FakeVideo-RealAudio) → FLAG manipulation
r = context_agent_result(spatial(0.92, True), temporal(0.88, True), audio(0.08, False))
assert r["is_fake"] is True, r
assert r["confidence"] > 0.5
print(f"[PASS] fake-video/real-audio mismatch → is_fake=True conf={r['confidence']}")

# 8. Fusion integration: context now contributes a 4th weighted vote.
#    Video with all four agents present and agreeing FAKE.
agents = [
    spatial(0.9, True), temporal(0.85, True), audio(0.8, True),
    context_agent_result(spatial(0.9, True), temporal(0.85, True), audio(0.8, True)),
]
conf, verdict, contributing, unavailable = _compute_verdict(agents)
assert verdict == "DEEPFAKE", verdict
assert "Context Verification" in contributing, "context must contribute when both modalities present"
assert unavailable == [], unavailable
print(f"[PASS] fusion: 4-agent DEEPFAKE conf={conf} contributing={len(contributing)}")

# 9. Fusion integration: image input → context unavailable, must NOT dilute.
agents = [
    spatial(0.63, True), na("Temporal Analysis"), na("Audio Analysis"),
    context_agent_result(spatial(0.63, True), na("Temporal Analysis"), na("Audio Analysis")),
]
conf, verdict, contributing, unavailable = _compute_verdict(agents)
assert conf == 0.63, f"image-only should be spatial-only 0.63, got {conf}"
assert verdict == "DEEPFAKE", verdict
assert "Context Verification" in unavailable, "context must be unavailable for image-only"
print(f"[PASS] fusion: image-only spatial-only conf={conf} (context did not dilute)")

# 10. Fusion integration: audio-only input → Spatial, Temporal, Context all unavailable.
# Only Audio Analysis (0.85 FAKE) contributes → conf=0.85, DEEPFAKE.
agents = [
    na("Spatial Analysis"), na("Temporal Analysis"), audio(0.85, True),
    context_agent_result(na("Spatial Analysis"), na("Temporal Analysis"), audio(0.85, True)),
]
conf, verdict, contributing, unavailable = _compute_verdict(agents)
assert conf == 0.85, f"audio-only should be audio-only 0.85, got {conf}"
assert verdict == "DEEPFAKE", verdict
assert "Spatial Analysis" not in contributing, "Spatial must NOT be in contributing for audio-only"
assert "Spatial Analysis" in unavailable, "Spatial must be in unavailable for audio-only"
assert "Temporal Analysis" in unavailable, "Temporal must be in unavailable for audio-only"
assert "Context Verification" in unavailable, "Context must be in unavailable for audio-only"
assert contributing == ["Audio Analysis"], f"Only Audio Analysis should contribute, got {contributing}"
print(f"[PASS] fusion: audio-only audio-only conf={conf} (spatial and context did not dilute)")

print("\n" + "=" * 60)
print("ALL CONTEXT AGENT TESTS PASSED")
print("=" * 60)
