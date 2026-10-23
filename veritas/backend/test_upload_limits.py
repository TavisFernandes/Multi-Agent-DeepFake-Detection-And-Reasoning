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
    res = client.post(route, files={"file": (f"big{SUFFIX}", TOO_BIG)})
    assert res.status_code == 413
    assert res.json()["detail"] == "File exceeds 50MB limit."
    assert set(glob.glob(pattern)) == before  # no temp file leaked


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
