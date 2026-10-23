"""
test_context_agent.py

Verifies the explanation-only context agent (mock_agents.context_agent_result)
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

# 4-7. Both modalities present → explanation only: never votes, confidence is the score gap.
cases = [
    ("both-fake agreement", spatial(0.9, True), temporal(0.8, True), audio(0.9, True), "same way"),
    ("both-real agreement", spatial(0.05, False), temporal(0.03, False), audio(0.1, False), "same way"),
    ("real-video/fake-audio", spatial(0.1, False), temporal(0.05, False), audio(0.95, True), "different directions"),
    ("fake-video/real-audio", spatial(0.92, True), temporal(0.88, True), audio(0.08, False), "different directions"),
]
for name, s_, t_, a_, words in cases:
    r = context_agent_result(s_, t_, a_)
    visual = (0.4 * s_["confidence"] + 0.3 * t_["confidence"]) / 0.7
    assert r["is_fake"] is None, f"{name}: context must never vote, got {r}"
    assert r["status"] == "explanation", r
    assert abs(r["confidence"] - abs(visual - a_["confidence"])) < 1e-3, r["confidence"]
    assert words in r["finding"] and "not separate evidence" in r["finding"], r["finding"]
    assert "strong indicator" not in r["finding"].lower()
    print(f"[PASS] {name} → explanation, gap={r['confidence']}")

# 8. Fusion: the context result never changes the verdict or score, and is never listed as a contributor.
for mode in ("corrected_fixed", "ppo", "legacy"):
    for _, s_, t_, a_, _ in cases:
        base = _compute_verdict([s_, t_, a_], "video", mode_override=mode)
        with_ctx = _compute_verdict([s_, t_, a_, context_agent_result(s_, t_, a_)], "video", mode_override=mode)
        assert with_ctx[:2] == base[:2], f"{mode}: context changed the result {base[:2]} -> {with_ctx[:2]}"
        if mode != "legacy":  # legacy lists every agent it is given; it is a frozen historical baseline
            assert "Context Verification" not in with_ctx[2] + with_ctx[3], with_ctx
print("[PASS] fusion: context never changes verdict/score (corrected_fixed, ppo, legacy)")

# 9. Fusion integration: image input → context not applicable, must NOT dilute.
agents = [
    spatial(0.63, True), na("Temporal Analysis"), na("Audio Analysis"),
    context_agent_result(spatial(0.63, True), na("Temporal Analysis"), na("Audio Analysis")),
]
conf, verdict, contributing, unavailable = _compute_verdict(agents)
assert conf == 0.63, f"image-only should be spatial-only 0.63, got {conf}"
assert verdict == "DEEPFAKE", verdict
assert agents[3]["status"] == "not_applicable"
assert "Context Verification" not in contributing + unavailable
print(f"[PASS] fusion: image-only spatial-only conf={conf} (context did not dilute)")

# 10. Fusion integration: audio-only input → only Audio contributes.
agents = [
    na("Spatial Analysis"), na("Temporal Analysis"), audio(0.85, True),
    context_agent_result(na("Spatial Analysis"), na("Temporal Analysis"), audio(0.85, True)),
]
conf, verdict, contributing, unavailable = _compute_verdict(agents)
assert conf == 0.85, f"audio-only should be audio-only 0.85, got {conf}"
assert verdict == "DEEPFAKE", verdict
assert unavailable == ["Spatial Analysis", "Temporal Analysis"], unavailable
assert contributing == ["Audio Analysis"], f"Only Audio Analysis should contribute, got {contributing}"
print(f"[PASS] fusion: audio-only conf={conf} (spatial and context did not dilute)")

print("\n" + "=" * 60)
print("ALL CONTEXT AGENT TESTS PASSED")
print("=" * 60)
