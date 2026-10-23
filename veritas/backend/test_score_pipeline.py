"""
test_score_pipeline.py
======================
Regression tests verifying that backend scores flow to the API responses and SSE
stream without numeric transformation, and that the percentage conversion happens
exactly once (in the frontend rendering layer, not in the backend or proxy).

Tests cover:
  1.  Backend _compute_verdict  -> correct normalized float in [0, 1]
  2.  /analyse JSON response     -> overall_confidence matches expected formula
  3.  /analyse agent list        -> per-agent confidence values are unmodified floats
  4.  /analyse/stream NDJSON     -> final_verdict overall_confidence matches /analyse
  5.  /analyse/stream agent_result -> per-agent confidence matches /analyse agents
  6.  Percentage conversion rule  -> value is in [0, 1] (conversion happens once, at render)
  7.  Spatial and Temporal parity -> same confidence in both endpoints
  8.  Fused score consistency     -> same overall_confidence in both endpoints
  9.  Unavailable-agent invariant -> Audio/Context contribute 0 to fusion
  10. No pre-multiplied percentage -> assert no value > 1.0 in any confidence field

Run from the veritas/backend directory:
    python -X utf8 test_score_pipeline.py
"""

import sys
import io
import json
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from main import _compute_verdict, _WEIGHTS

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BACKEND = "http://127.0.0.1:8000"
TEST_VIDEO = str(_HERE / "00025_fake.mp4")

PASS = 0
FAIL = 0

def ok(msg: str):
    global PASS
    PASS += 1
    print(f"  [PASS] {msg}")

def fail(msg: str):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {msg}")

def check(condition: bool, label: str, detail: str = ""):
    if condition:
        ok(label)
    else:
        fail(f"{label} — {detail}" if detail else label)

# ---------------------------------------------------------------------------
# Helper: send multipart POST and return parsed JSON
# ---------------------------------------------------------------------------
def post_json(endpoint: str, file_path: str) -> dict:
    boundary = "----VeritasPipelineTest"
    with open(file_path, "rb") as f:
        body_bytes = f.read()
    fname = Path(file_path).name
    ctype = "audio/wav" if Path(file_path).suffix.lower() == ".wav" else "video/mp4"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode()
    body = header + body_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{BACKEND}/{endpoint}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def post_stream(endpoint: str, file_path: str) -> list[dict]:
    """Return all NDJSON records from the stream as a list."""
    boundary = "----VeritasPipelineStreamTest"
    with open(file_path, "rb") as f:
        body_bytes = f.read()
    fname = Path(file_path).name
    ctype = "audio/wav" if Path(file_path).suffix.lower() == ".wav" else "video/mp4"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{fname}"\r\n'
        f"Content-Type: {ctype}\r\n\r\n"
    ).encode()
    body = header + body_bytes + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{BACKEND}/{endpoint}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    records = []
    with urllib.request.urlopen(req, timeout=60) as r:
        for line in r:
            line = line.strip()
            if line:
                records.append(json.loads(line.decode()))
    return records


# ===========================================================================
print("=" * 65)
print("VERITAS Score Pipeline — Regression Tests")
print("=" * 65)
print()

def expected_corrected_fixed(conf_by_agent, contributing):
    """Independent mirror of fusion_policy.compute_verdict_corrected_fixed (noisy-OR of visual and audio)."""
    sp = conf_by_agent["Spatial Analysis"] if "Spatial Analysis" in contributing else None
    te = conf_by_agent["Temporal Analysis"] if "Temporal Analysis" in contributing else None
    au = conf_by_agent["Audio Analysis"] if "Audio Analysis" in contributing else None
    vis = (0.40 * sp + 0.30 * te) / 0.70 if sp is not None and te is not None else (sp if sp is not None else te)
    if vis is not None and au is not None:
        return 1.0 - (1.0 - vis) * (1.0 - au)
    return vis if vis is not None else (au if au is not None else 0.0)


# ---------------------------------------------------------------------------
# Test 1: _compute_verdict produces a float in [0, 1]
# ---------------------------------------------------------------------------
print("[Group 1] _compute_verdict output range")
spatial_conf = 0.486
temporal_conf = 0.0183

agents_mock = [
    {"agent": "Spatial Analysis",    "confidence": spatial_conf, "is_fake": False},
    {"agent": "Temporal Analysis",   "confidence": temporal_conf, "is_fake": False},
    {"agent": "Audio Analysis",      "confidence": 0,           "is_fake": None},
    {"agent": "Context Verification","confidence": 0,           "is_fake": None},
]
conf, verdict, contributing, unavailable = _compute_verdict(agents_mock)

check(0.0 <= conf <= 1.0, "T1: _compute_verdict returns a value in [0, 1]", f"got {conf}")
check(conf < 1.0, "T2: confidence is not pre-multiplied by 100", f"got {conf}")

# Manual recalc
w_s = _WEIGHTS["Spatial Analysis"]
w_t = _WEIGHTS["Temporal Analysis"]
expected_conf = round((spatial_conf * w_s + temporal_conf * w_t) / (w_s + w_t), 4)
check(conf == expected_conf, f"T3: normalized confidence matches formula ({expected_conf})", f"got {conf}")

# ---------------------------------------------------------------------------
# Test 2: /analyse JSON — per-agent values are raw floats, not percentages
# ---------------------------------------------------------------------------
print()
print("[Group 2] /analyse JSON endpoint")

if not Path(TEST_VIDEO).exists():
    print(f"  [SKIP] Test video not found: {TEST_VIDEO}")
else:
    try:
        analyse_data = post_json("analyse", TEST_VIDEO)

        agents_by_name = {a.get("agent") or a.get("agent_name"): a for a in analyse_data.get("agents", [])}
        api_spatial_conf  = agents_by_name.get("Spatial Analysis", {}).get("confidence")
        api_temporal_conf = agents_by_name.get("Temporal Analysis", {}).get("confidence")
        api_audio_conf    = agents_by_name.get("Audio Analysis", {}).get("confidence")
        api_overall       = analyse_data.get("overall_confidence")
        api_contributing  = analyse_data.get("contributing_agents", [])
        api_unavailable   = analyse_data.get("unavailable_agents", [])

        check(api_spatial_conf is not None and 0.0 <= api_spatial_conf <= 1.0,
              "T4: /analyse Spatial confidence in [0,1]", f"got {api_spatial_conf}")
        check(api_temporal_conf is not None and 0.0 <= api_temporal_conf <= 1.0,
              "T5: /analyse Temporal confidence in [0,1]", f"got {api_temporal_conf}")
        api_audio = agents_by_name.get("Audio Analysis", {})
        check(api_audio_conf is not None and 0.0 <= api_audio_conf <= 1.0
              and (api_audio.get("status") != "untrained" or api_audio_conf == 0),
              "T6: /analyse Audio confidence in [0,1] (0 when untrained)", f"got {api_audio_conf}")
        check(api_overall is not None and 0.0 <= api_overall <= 1.0,
              "T7: /analyse overall_confidence in [0,1]", f"got {api_overall}")

        # Verify no percentage-scaled value slipped through
        check(api_spatial_conf < 1.5,
              "T8: /analyse Spatial is not percent-scaled (e.g. not 48.6)", f"got {api_spatial_conf}")
        check(api_overall < 1.5,
              "T9: /analyse overall not percent-scaled", f"got {api_overall}")

        # Cross-check against formula
        api_expected_conf = round(expected_corrected_fixed(
            {n: a.get("confidence") for n, a in agents_by_name.items()}, api_contributing), 4)
        check(abs(api_overall - api_expected_conf) <= 1e-3,  # agent confidences arrive rounded to 4 dp
              f"T10: /analyse overall_confidence matches normalized formula ({api_expected_conf})",
              f"got {api_overall}")

        audio_has_verdict = api_audio.get("is_fake") is not None
        check(("Audio Analysis" in api_unavailable) == (not audio_has_verdict),
              "T11: Audio Analysis unavailable exactly when it has no verdict")
        check(("Audio Analysis" in api_contributing) == audio_has_verdict,
              "T12: Audio Analysis contributes exactly when it has a verdict")
        check("Spatial Analysis" in api_contributing,
              "T13: Spatial Analysis in contributing_agents")
        check("Temporal Analysis" in api_contributing,
              "T14: Temporal Analysis in contributing_agents")

    except Exception as exc:
        fail(f"/analyse request failed: {exc}")

# ---------------------------------------------------------------------------
# Test 3: /analyse/stream NDJSON — values match /analyse exactly
# ---------------------------------------------------------------------------
print()
print("[Group 3] /analyse/stream NDJSON endpoint")

if not Path(TEST_VIDEO).exists():
    print(f"  [SKIP] Test video not found: {TEST_VIDEO}")
else:
    try:
        stream_records = post_stream("analyse/stream", TEST_VIDEO)

        stream_agents = {
            (r.get("agent") or r.get("agent_name")): r
            for r in stream_records if r.get("type") == "agent_result"
        }
        stream_verdict_records = [r for r in stream_records if r.get("type") == "final_verdict"]
        stream_verdict = stream_verdict_records[0] if stream_verdict_records else {}

        s_stream_conf  = stream_agents.get("Spatial Analysis", {}).get("confidence")
        t_stream_conf  = stream_agents.get("Temporal Analysis", {}).get("confidence")
        au_stream_conf = stream_agents.get("Audio Analysis", {}).get("confidence")
        stream_overall = stream_verdict.get("overall_confidence")
        stream_contributing = stream_verdict.get("contributing_agents", [])
        stream_unavailable  = stream_verdict.get("unavailable_agents", [])

        check(s_stream_conf is not None and 0.0 <= s_stream_conf <= 1.0,
              "T15: stream Spatial confidence in [0,1]", f"got {s_stream_conf}")
        check(t_stream_conf is not None and 0.0 <= t_stream_conf <= 1.0,
              "T16: stream Temporal confidence in [0,1]", f"got {t_stream_conf}")
        au_stream = stream_agents.get("Audio Analysis", {})
        check(au_stream_conf is not None and 0.0 <= au_stream_conf <= 1.0
              and (au_stream.get("status") != "untrained" or au_stream_conf == 0),
              "T17: stream Audio confidence in [0,1] (0 when untrained)", f"got {au_stream_conf}")
        check(stream_overall is not None and 0.0 <= stream_overall <= 1.0,
              "T18: stream overall_confidence in [0,1]", f"got {stream_overall}")

        # Parity with /analyse
        check(s_stream_conf == api_spatial_conf,
              f"T19: Spatial confidence consistent across both endpoints ({api_spatial_conf})",
              f"stream={s_stream_conf} analyse={api_spatial_conf}")
        check(t_stream_conf == api_temporal_conf,
              f"T20: Temporal confidence consistent across both endpoints ({api_temporal_conf})",
              f"stream={t_stream_conf} analyse={api_temporal_conf}")
        check(stream_overall == api_overall,
              f"T21: overall_confidence consistent across both endpoints ({api_overall})",
              f"stream={stream_overall} analyse={api_overall}")

        check(("Audio Analysis" in stream_unavailable) == (au_stream.get("is_fake") is None),
              "T22: stream Audio is unavailable exactly when it has no verdict (is_fake None)")
        check("Spatial Analysis" in stream_contributing,
              "T23: stream Spatial Analysis in contributing_agents")

    except Exception as exc:
        fail(f"/analyse/stream request failed: {exc}")

# ---------------------------------------------------------------------------
# Test 4: Frontend percentage-conversion rule
# ---------------------------------------------------------------------------
print()
print("[Group 4] Frontend percentage-conversion contract")

# The frontend renders: (agent.confidence * 100).toFixed(1) + '%'
# This means confidence MUST be in [0,1] — never pre-multiplied.
# We simulate the rendering and verify no double-conversion.

def simulate_frontend_display(raw_confidence: float) -> str:
    """Mirrors the Console.tsx expression: (agent.confidence * 100).toFixed(1) + '%'"""
    return f"{raw_confidence * 100:.1f}%"

check(simulate_frontend_display(0.486)  == "48.6%",
      "T24: Spatial 0.486 renders as 48.6%")
check(simulate_frontend_display(0.0183) == "1.8%",
      "T25: Temporal 0.0183 renders as 1.8%")
check(simulate_frontend_display(0.2856) == "28.6%",
      "T26: Fused 0.2856 renders as 28.6%")

# Confirm that if the backend were to accidentally send a percentage (e.g. 48.6),
# the display would be wrong (4860.0%) — this is the canary.
check(simulate_frontend_display(48.6) != "48.6%",
      "T27: If backend sent 48.6 (percent-scaled) the display would be wrong (canary test)")

# ---------------------------------------------------------------------------
# Test 5: Proxy transparency — no field is mutated by the proxy
# ---------------------------------------------------------------------------
print()
print("[Group 5] Proxy field transparency invariant")

# We already have stream_agents and analyse_data from the live tests.
# Verify the SSE proxy passes is_fake, status, etc., verbatim.
if not Path(TEST_VIDEO).exists():
    print("  [SKIP]")
else:
    try:
        au_stream_is_fake = stream_agents.get("Audio Analysis", {}).get("is_fake")
        au_stream_status  = stream_agents.get("Audio Analysis", {}).get("status")

        # Status-independent: the proxy must pass None through, never coerce it to 0/False.
        check(au_stream_is_fake is None or isinstance(au_stream_is_fake, bool),
              "T28: stream Audio is_fake is None or a real bool (not coerced)", f"got {au_stream_is_fake!r}")
        check(au_stream_status != "untrained" or au_stream_is_fake is None,
              "T29: stream Audio has no verdict when untrained", f"status={au_stream_status} is_fake={au_stream_is_fake!r}")

        ctx_stream = stream_agents.get("Context Verification", {})
        ctx_is_fake = ctx_stream.get("is_fake")
        check(ctx_is_fake is None or isinstance(ctx_is_fake, bool),
              "T30: stream Context is_fake is None or a real bool", f"got {ctx_is_fake!r}")
        check((ctx_stream.get("status") in ("mocked", "not_applicable")) == (ctx_is_fake is None),
              "T31: stream Context has no verdict exactly when mocked/not_applicable", f"got {ctx_stream.get('status')}")
    except Exception as exc:
        fail(f"Proxy transparency check failed: {exc}")

# ---------------------------------------------------------------------------
# Test 6: Audio-only pipeline invariant — Spatial excluded from contributing
# ---------------------------------------------------------------------------
print()
print("[Group 6] Audio-only pipeline invariant")

TEST_AUDIO = str(_HERE / "test_audio.wav")
if not Path(TEST_AUDIO).exists():
    print(f"  [SKIP] Test audio not found: {TEST_AUDIO}")
else:
    try:
        # POST /analyse with audio
        audio_analyse_data = post_json("analyse", TEST_AUDIO)
        audio_agents = {a.get("agent"): a for a in audio_analyse_data.get("agents", [])}
        spatial_audio = audio_agents.get("Spatial Analysis", {})

        check(spatial_audio.get("status") == "not_applicable",
              "T32: /analyse Spatial status == 'not_applicable' for audio")
        check(spatial_audio.get("is_fake") is None,
              "T33: /analyse Spatial is_fake is None for audio")
        check(spatial_audio.get("confidence") == 0.0,
              "T34: /analyse Spatial confidence is 0.0 for audio")
        check("Spatial Analysis" not in audio_analyse_data.get("contributing_agents", []),
              "T35: /analyse Spatial Analysis NOT in contributing_agents for audio")
        check("Spatial Analysis" in audio_analyse_data.get("unavailable_agents", []),
              "T36: /analyse Spatial Analysis IN unavailable_agents for audio")

        # POST /analyse/stream with audio
        stream_audio_records = post_stream("analyse/stream", TEST_AUDIO)
        stream_audio_agents = {
            (r.get("agent") or r.get("agent_name")): r
            for r in stream_audio_records if r.get("type") == "agent_result"
        }
        spatial_stream_audio = stream_audio_agents.get("Spatial Analysis", {})
        stream_audio_verdict = next(r for r in stream_audio_records if r.get("type") == "final_verdict")

        check(spatial_stream_audio.get("status") == "not_applicable",
              "T37: stream Spatial status == 'not_applicable' for audio")
        check(spatial_stream_audio.get("is_fake") is None,
              "T38: stream Spatial is_fake is None for audio")
        check(spatial_stream_audio.get("confidence") == 0.0,
              "T39: stream Spatial confidence is 0.0 for audio")
        check("Spatial Analysis" not in stream_audio_verdict.get("contributing_agents", []),
              "T40: stream Spatial Analysis NOT in contributing_agents for audio")
        check("Spatial Analysis" in stream_audio_verdict.get("unavailable_agents", []),
              "T41: stream Spatial Analysis IN unavailable_agents for audio")
    except Exception as exc:
        fail(f"Audio-only pipeline check failed: {exc}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 65)
total = PASS + FAIL
print(f"Results: {PASS} passed  /  {FAIL} failed  /  {total} total")
print("=" * 65)

if FAIL > 0:
    sys.exit(1)
else:
    print("All score pipeline tests passed.")
