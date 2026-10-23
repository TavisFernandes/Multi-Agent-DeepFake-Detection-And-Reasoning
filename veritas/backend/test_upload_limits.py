"""
Oversized uploads must get a 413 (not a masked 400/500) and must not leave temp files behind.
Run: python -m pytest test_upload_limits.py
"""
import asyncio
import glob
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from main import RejectOversizedRequests, app

SUFFIX = ".veritas-limit-test"
TOO_BIG = b"\0" * (50 * 1024 * 1024 + 1)


@pytest.mark.parametrize("route", ["/analyse", "/analyse/stream"])
def test_oversized_upload_is_rejected_with_413(route):
    pattern = os.path.join(tempfile.gettempdir(), f"*{SUFFIX}")
    before = set(glob.glob(pattern))
    client = TestClient(app)  # no `with`: skip startup model loading, it isn't needed here
    # The hint gets past the 415 type check; the odd suffix still names the temp file so leaks are detectable.
    res = client.post(route, files={"file": (f"big{SUFFIX}", TOO_BIG)}, data={"media_type": "video"})
    assert res.status_code == 413
    assert res.json()["detail"] == "File exceeds 50MB limit."
    assert set(glob.glob(pattern)) == before  # no temp file leaked


@pytest.mark.parametrize("route", ["/analyse", "/analyse/stream"])
def test_unknown_file_type_is_rejected_with_415(route):
    pattern = os.path.join(tempfile.gettempdir(), "*.xyz")
    before = set(glob.glob(pattern))
    res = TestClient(app).post(route, files={"file": ("notes.xyz", b"not media")})
    assert res.status_code == 415
    assert "Unsupported file type '.xyz'" in res.json()["detail"]
    assert set(glob.glob(pattern)) == before  # rejected before anything is written


def test_still_image_renamed_to_video_gets_no_temporal_score(tmp_path):
    import cv2
    import numpy as np
    import temporal_agent

    fake_video = tmp_path / "photo.mp4"
    fake_video.write_bytes(cv2.imencode(".png", np.full((240, 320, 3), 90, np.uint8))[1].tobytes())
    r = temporal_agent.run_temporal_agent(str(fake_video))
    assert r["status"] == "error" and r["is_fake"] is None and r["confidence"] == 0.0


def test_video_with_unchecked_picture_is_never_authentic():
    from main import _compute_verdict

    broken = [
        {"agent": "Spatial Analysis", "status": "error", "is_fake": None, "confidence": 0.0},
        {"agent": "Temporal Analysis", "status": "error", "is_fake": None, "confidence": 0.0},
        {"agent": "Audio Analysis", "status": "trained", "is_fake": False, "confidence": 0.02, "has_audio": True, "duration": 5.0},
    ]
    for mode in ("corrected_fixed", "ppo", "legacy"):
        assert _compute_verdict(broken, "video", mode_override=mode)[1].startswith("UNCERTAIN"), mode
    # An audio upload has no picture to check, so audio alone may still say AUTHENTIC.
    audio_only = [{**a, "status": "not_applicable"} if a["agent"] != "Audio Analysis" else a for a in broken]
    assert _compute_verdict(audio_only, "audio")[1] == "AUTHENTIC"


def test_oversized_content_length_is_rejected_before_the_body_is_read():
    async def app_must_not_run(scope, receive, send):
        raise AssertionError("request reached the app")

    async def receive():
        raise AssertionError("request body was read")

    sent = []

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "headers": [(b"content-length", str(60 * 1024 * 1024).encode())]}
    asyncio.run(RejectOversizedRequests(app_must_not_run)(scope, receive, send))
    assert sent[0]["status"] == 413
    assert b"File exceeds 50MB limit." in sent[1]["body"]


def test_normal_upload_sizes_pass_through():
    reached = []

    async def app_ok(scope, receive, send):
        reached.append(True)

    for headers in ([(b"content-length", b"1024")], []):  # small body, and no Content-Length at all
        asyncio.run(RejectOversizedRequests(app_ok)({"type": "http", "headers": headers}, None, None))
    assert reached == [True, True]
