"""
eval_temporal_robustness.py
===========================
Scores the production temporal checkpoint (temporal_agent.run_temporal_agent, so the same face crop, frame sampling
and decision threshold) on fusion_val clips rebuilt under the spatial degradation tests. fusion_val is development
data (already used for candidate selection), not an independent estimate. Never touches final_test.

Per-frame conditions (JPEG, downscale, blur, 720p canvas) are applied to every frame and stored losslessly (FFV1);
the H.264 conditions re-encode every frame with libx264. "original" scores the native file.
FP = real video called fake (false DEEPFAKE); FN = fake video called real (fake passed as AUTHENTIC).

Usage (from veritas/backend/training):
  python eval_temporal_robustness.py OUT.json
"""
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import temporal_agent  # noqa: E402
import wav_cache  # noqa: E402
from eval_spatial_robustness import CONDITIONS, MANIFEST, TMP, metrics, to_canvas  # noqa: E402

# name -> (per-frame transform, encoder args)
LOSSLESS = ["-c:v", "ffv1"]
TESTS = {n: (lambda f, fn=CONDITIONS[n]: fn(None, f), LOSSLESS)
         for n in ["jpeg_q30", "jpeg_q60", "down_x0.5", "blur_s1.5", "720p_face25"]}
TESTS["h264_crf32"] = (lambda f: f, ["-c:v", "libx264", "-crf", "32", "-pix_fmt", "yuv420p"])
TESTS["720p_face25_h264_crf28"] = (to_canvas, ["-c:v", "libx264", "-crf", "28", "-pix_fmt", "yuv420p"])


def rebuild(path, fn, enc):
    """Write every frame of `path` through `fn` to a new clip with encoder args `enc`; returns its path."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    while (f := cap.read()[1]) is not None:
        frames.append(fn(f))
    cap.release()
    h, w = frames[0].shape[:2]
    out = TMP / ("t.avi" if enc is LOSSLESS else "t.mp4")
    p = subprocess.Popen([wav_cache._FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", *enc, str(out)], stdin=subprocess.PIPE)
    p.communicate(b"".join(f.tobytes() for f in frames))
    assert int(cv2.VideoCapture(str(out)).get(cv2.CAP_PROP_FRAME_COUNT)) == len(frames), "frame count changed"
    return str(out)


def main(out_path, limit=None):
    clips = [c for cl in json.loads(MANIFEST.read_bytes())["clips"]["fusion_val"].values() for c in cl][:limit]
    y = np.array([c["type"].startswith("FakeVideo") for c in clips]).astype(int)
    temporal_agent.load_model()
    ck = temporal_agent._CHECKPOINT_PATH.resolve()
    thr = temporal_agent._DECISION_THRESHOLD
    res = {"original": [temporal_agent.run_temporal_agent(c["full_path"]) for c in clips]}
    for name, (fn, enc) in TESTS.items():
        res[name] = [temporal_agent.run_temporal_agent(rebuild(c["full_path"], fn, enc)) for c in clips]
    report = {"split": "fusion_val", "split_role": "development (used for candidate selection)",
              "manifest": MANIFEST.name, "n": len(clips), "checkpoint": ck.name, "sha256": wav_cache.file_sha256(ck),
              "decision_threshold": thr, "metrics": {}, "scores": {}}
    print(f"{ck.name} threshold={thr:.4f}")
    for name, rs in res.items():
        p = np.array([r["confidence"] if r.get("status") == "trained" else np.nan for r in rs])
        m = metrics(y, p)
        # metrics() counts at 0.5; production decides at the checkpoint threshold
        k = ~np.isnan(p)
        m["fp_at_threshold"] = int(((p[k] >= thr) & (y[k] == 0)).sum())
        m["fn_at_threshold"] = int(((p[k] < thr) & (y[k] == 1)).sum())
        report["metrics"][name] = m
        report["scores"][name] = dict(zip((c["rel_path"] for c in clips), map(float, p)))
        print(f"  {name:24s} AUC={m['auc']:.4f} real->DEEPFAKE={m['fp_at_threshold']}/{m['n_real']} "
              f"fake->AUTHENTIC={m['fn_at_threshold']}/{m['n_fake']} no_face={m['no_face']}")
    Path(out_path).write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
