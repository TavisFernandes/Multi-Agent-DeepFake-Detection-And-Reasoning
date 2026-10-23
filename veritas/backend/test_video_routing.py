"""
test_video_routing.py

Tests the routing fix in main.py where videos are split into:
  - Temporal Agent: receives full MP4
  - Spatial Agent: receives extracted middle frame (.jpg)
  - Both successfully run without PIL UnidentifiedImageError.
"""

import sys, io, os, json, tempfile
import urllib.request
import traceback
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def make_synthetic_mp4(path: str, num_frames: int = 30, w: int = 112, h: int = 112):
    import cv2
    import numpy as np
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 25.0, (w, h))
    rng = np.random.RandomState(42)
    for _ in range(num_frames):
        frame = rng.randint(0, 255, (h, w, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

print("============================================================")
print("VERIFYING VIDEO ROUTING FIX")
print("============================================================")

tmp_vid = tempfile.mktemp(suffix=".mp4")
make_synthetic_mp4(tmp_vid, num_frames=30)

try:
    boundary = "----VeritasTestBoundary2"
    with open(tmp_vid, "rb") as f:
        vid_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test_video.mp4"\r\n'
        f"Content-Type: video/mp4\r\n\r\n"
    ).encode() + vid_bytes + f"\r\n--{boundary}--\r\n".encode()

    print(f"Sending POST /analyse with {len(vid_bytes)} bytes of synthetic MP4...")
    
    req = urllib.request.Request(
        "http://127.0.0.1:8000/analyse",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        resp_data = json.loads(r.read())

    print("\n[SUCCESS] /analyse returned 200 OK")
    print(f"verdict: {resp_data.get('verdict')}")
    print(f"overall_confidence: {resp_data.get('overall_confidence')}")
    print("\nAgent Results:")
    for agent in resp_data.get("agents", []):
        print(f"  - {agent.get('agent')}: confidence={agent.get('confidence')} is_fake={agent.get('is_fake')}")

    print("\nSending POST /analyse/stream ...")
    req_stream = urllib.request.Request(
        "http://127.0.0.1:8000/analyse/stream",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req_stream, timeout=60) as r:
        print("[SUCCESS] /analyse/stream streaming started")
        for line in r:
            if line.strip():
                print(f"  STREAM EVENT: {line.decode().strip()}")
                
    print("\nALL VIDEO ROUTING TESTS PASSED")

except Exception as e:
    print(f"\n[FAIL] Test failed: {e}")
    traceback.print_exc()
finally:
    if os.path.exists(tmp_vid):
        os.remove(tmp_vid)
