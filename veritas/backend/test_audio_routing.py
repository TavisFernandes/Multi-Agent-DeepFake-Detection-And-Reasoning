"""
test_audio_routing.py

Tests the fix and architectural correction in audio_agent.py.
Verifies:
1. MP4 with MP3 audio
2. MP4 with no audio
3. WAV input
4. untrained Audio output (status="untrained", is_fake=None, confidence=0)
5. fusion ignores untrained Audio
"""

import sys, io, os, json
import urllib.request
import traceback
from pathlib import Path

# Add backend dir to sys.path to import audio_agent directly for unit testing
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import audio_agent
from main import _WEIGHTS, _build_agents, _compute_verdict

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("============================================================")
print("VERIFYING AUDIO ROUTING AND UNTRAINED FUSION")
print("============================================================")

test_files = {
    "mp4_audio": str(_HERE / "00025_fake.mp4"),
    "mp4_no_audio": str(_HERE / "test_no_audio.mp4"),
    "wav_audio": str(_HERE / "test_audio.wav"),
}

for name, path in test_files.items():
    if not os.path.exists(path):
        print(f"[FAIL] Test file {path} not found!")
        sys.exit(1)

# Ensure the model is NOT loaded to test the untrained behavior
audio_agent._model = None

# --- 1. Test probe and run_audio_agent directly ---
print(f"Testing direct run_audio_agent on MP4 with audio ...")
res_mp4 = audio_agent.run_audio_agent(test_files["mp4_audio"])
assert res_mp4["status"] == "untrained", f"Expected untrained, got {res_mp4.get('status')}"
assert res_mp4["has_audio"] is True
assert res_mp4["codec"] == "mp3"
assert res_mp4["sample_rate"] == 44100
assert res_mp4["channels"] == 2
assert res_mp4["is_fake"] is None
assert res_mp4["confidence"] == 0
print("[PASS] MP4 with audio correctly probed and returns untrained output.")

print(f"Testing direct run_audio_agent on MP4 without audio ...")
res_no_audio = audio_agent.run_audio_agent(test_files["mp4_no_audio"])
assert res_no_audio["status"] == "no_audio"
assert res_no_audio["has_audio"] is False
assert res_no_audio["is_fake"] is None
assert res_no_audio["confidence"] == 0
print("[PASS] MP4 without audio correctly handled.")

print(f"Testing direct run_audio_agent on WAV audio ...")
res_wav = audio_agent.run_audio_agent(test_files["wav_audio"])
assert res_wav["status"] == "untrained"
assert res_wav["has_audio"] is True
assert res_wav["codec"] == "pcm_s16le"
assert res_wav["sample_rate"] == 16000
assert res_wav["channels"] == 1
assert res_wav["is_fake"] is None
assert res_wav["confidence"] == 0
print("[PASS] WAV audio correctly probed and handled.")

# Direct test of _build_agents and _compute_verdict on audio-only input
print(f"Testing _build_agents on WAV audio (audio-only modality routing)...")
agents_audio = _build_agents(test_files["wav_audio"], "audio")
spatial_res = agents_audio[0]
temporal_res = agents_audio[1]
audio_res = agents_audio[2]
context_res = agents_audio[3]

assert spatial_res["agent"] == "Spatial Analysis"
assert spatial_res["status"] == "not_applicable", f"Expected not_applicable, got {spatial_res.get('status')}"
assert spatial_res["is_fake"] is None, f"Expected is_fake=None, got {spatial_res.get('is_fake')}"
assert spatial_res["confidence"] == 0.0
assert spatial_res["gradcam_b64"] is None
assert "not applicable to audio" in spatial_res["finding"]
print("[PASS] Spatial agent correctly marked not_applicable with is_fake=None for audio.")

assert temporal_res["status"] == "not_applicable"
assert temporal_res["is_fake"] is None
assert context_res["status"] == "not_applicable"
assert context_res["is_fake"] is None

conf, verdict, contributing, unavailable = _compute_verdict(agents_audio)
assert "Spatial Analysis" not in contributing, "Spatial must NOT contribute for audio-only input"
assert "Spatial Analysis" in unavailable, "Spatial must be in unavailable_agents"
assert "Temporal Analysis" in unavailable
assert "Context Verification" not in contributing + unavailable  # explanation-only, never listed
assert contributing == [], f"No agent should contribute for audio-only with untrained audio, got {contributing}"
assert verdict == "UNVERIFIED MODALITY"
assert conf == 0.0
print("[PASS] Audio-only _build_agents fusion correctly excludes Spatial, Temporal, Context.")


# --- 2. Test Live Backend (Fusion and Streaming) ---
def send_test(file_path, endpoint):
    boundary = "----VeritasTestBoundaryAudio"
    with open(file_path, "rb") as f:
        vid_bytes = f.read()

    filename = Path(file_path).name
    ext = Path(file_path).suffix.lower()
    content_type = "audio/wav" if ext == ".wav" else "video/mp4"

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + vid_bytes + f"\r\n--{boundary}--\r\n".encode()
    
    req = urllib.request.Request(
        f"http://127.0.0.1:8000/{endpoint}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=60)

try:
    print("\nSending POST /analyse with MP4 (with audio) ...")
    with send_test(test_files["mp4_audio"], "analyse") as r:
        resp_data = json.loads(r.read())

    print("[SUCCESS] /analyse returned 200 OK")
    
    audio_tested = False
    actual_overall = resp_data.get("overall_confidence")
    contributing = resp_data.get("contributing_agents", [])
    unavailable = resp_data.get("unavailable_agents", [])
    
    for agent in resp_data.get("agents", []):
        agent_name = agent.get("agent")
        conf = agent.get("confidence")
        is_fake = agent.get("is_fake")
        status = agent.get("status")
        print(f"  - {agent_name}: confidence={conf} is_fake={is_fake} status={status}")

        if agent_name == "Audio Analysis":
            audio_tested = True
            audio_has_verdict = is_fake is not None
            if status == "untrained":
                assert is_fake is None, "Untrained Audio Agent should return is_fake=None"
                assert conf == 0, "Untrained Audio Agent should return confidence=0"

    # Mirror of corrected_fixed: visual = 0.40/0.30 weighted mean, then noisy-OR with audio.
    conf_by_agent = {a.get("agent"): a.get("confidence") for a in resp_data.get("agents", [])}
    sp = conf_by_agent["Spatial Analysis"] if "Spatial Analysis" in contributing else None
    te = conf_by_agent["Temporal Analysis"] if "Temporal Analysis" in contributing else None
    au = conf_by_agent["Audio Analysis"] if "Audio Analysis" in contributing else None
    vis = (0.40 * sp + 0.30 * te) / 0.70 if sp is not None and te is not None else (sp if sp is not None else te)
    if vis is not None and au is not None:
        expected_overall = 1.0 - (1.0 - vis) * (1.0 - au)
    else:
        expected_overall = vis if vis is not None else (au if au is not None else 0.0)
    expected_overall = round(expected_overall, 4)

    print(f"  -> Fusion: Expected Overall: {expected_overall}, Actual: {actual_overall}")
    assert abs(actual_overall - expected_overall) <= 1e-3, f"Fusion mismatch! Expected {expected_overall}, got {actual_overall}"
    assert audio_tested, "Audio Analysis agent was not in the /analyse response!"
    # The server loads the real audio model, so it may be trained; fusion must match either way.
    assert ("Audio Analysis" in contributing) == audio_has_verdict, "Audio contributes iff it has a verdict"
    assert ("Audio Analysis" in unavailable) == (not audio_has_verdict), "Audio unavailable iff it has no verdict"
    print("[PASS] /analyse routed Audio and fused it consistently with its verdict.")

    print("\nSending POST /analyse/stream with MP4 (with audio) ...")
    audio_stream_tested = False
    with send_test(test_files["mp4_audio"], "analyse/stream") as r:
        print("[SUCCESS] /analyse/stream streaming started")
        for line in r:
            if line.strip():
                event = json.loads(line.decode().strip())
                if event.get("type") == "agent_result" and event.get("agent") == "Audio Analysis":
                    audio_stream_tested = True
                    conf = event.get("confidence")
                    is_fake = event.get("is_fake")
                    status = event.get("status")
                    if status == "untrained":
                        assert is_fake is None, "Untrained Audio Agent should return is_fake=None in stream"
                        assert conf == 0, "Untrained Audio Agent should return confidence=0 in stream"

    assert audio_stream_tested, "Audio Analysis event missing from stream"
    print("[PASS] /analyse/stream returned a well-formed Audio result.")

    print("\nSending POST /analyse with WAV (audio-only) ...")
    with send_test(test_files["wav_audio"], "analyse") as r:
        resp_data_wav = json.loads(r.read())

    print("[SUCCESS] /analyse returned 200 OK for WAV")
    actual_overall_wav = resp_data_wav.get("overall_confidence")
    contributing_wav = resp_data_wav.get("contributing_agents", [])
    unavailable_wav = resp_data_wav.get("unavailable_agents", [])

    assert "Spatial Analysis" not in contributing_wav, "Spatial must NOT be in contributing_agents for audio"
    assert "Spatial Analysis" in unavailable_wav, "Spatial must be in unavailable_agents for audio"

    spatial_in_resp = next(a for a in resp_data_wav.get("agents", []) if a.get("agent") == "Spatial Analysis")
    assert spatial_in_resp.get("status") == "not_applicable"
    assert spatial_in_resp.get("is_fake") is None
    assert spatial_in_resp.get("confidence") == 0.0

    print("[PASS] /analyse with WAV correctly routed Spatial as not_applicable (excluded from fusion).")

    print("\nSending POST /analyse/stream with WAV (audio-only) ...")
    spatial_stream_found = False
    with send_test(test_files["wav_audio"], "analyse/stream") as r:
        print("[SUCCESS] /analyse/stream streaming started for WAV")
        for line in r:
            if line.strip():
                event = json.loads(line.decode().strip())
                if event.get("type") == "agent_result" and event.get("agent") == "Spatial Analysis":
                    spatial_stream_found = True
                    assert event.get("status") == "not_applicable", f"Expected not_applicable, got {event.get('status')}"
                    assert event.get("is_fake") is None, f"Expected is_fake=None, got {event.get('is_fake')}"
                    assert event.get("confidence") == 0.0
                elif event.get("type") == "final_verdict":
                    assert "Spatial Analysis" not in event.get("contributing_agents", [])
                    assert "Spatial Analysis" in event.get("unavailable_agents", [])

    assert spatial_stream_found, "Spatial Analysis event missing from WAV stream"
    print("[PASS] /analyse/stream correctly streamed Spatial as not_applicable for WAV.")
    
    print("\nALL AUDIO ROUTING TESTS PASSED")

except Exception as e:
    print(f"\n[FAIL] Test failed: {e}")
    traceback.print_exc()
    sys.exit(1)
