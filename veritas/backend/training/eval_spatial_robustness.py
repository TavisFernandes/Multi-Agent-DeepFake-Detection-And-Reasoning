"""
eval_spatial_robustness.py
==========================
Scores spatial checkpoints on fusion_val middle frames (manifest v3; out-of-sample for every detector) under
degradations, through the production path: face_crop.detect -> face_crop.crop -> spatial_agent._TRANSFORM.
Never touches final_test. fusion_val has been used to compare and select candidates, so it is development data:
results here are not an independent estimate of real-world performance.
FP = real video scored >= 0.5 (false DEEPFAKE); FN = fake video scored < 0.5 (fake passed as AUTHENTIC).

Conditions: original; JPEG q30/q60; downscale x0.5; Gaussian blur sigma 1.5; H.264 CRF 32 re-encode;
720p frame with the face at 25% of the height (grey canvas, lossless) and the same encoded with H.264 CRF 28.
The canvas is synthetic (plain background), so it tests framing/rescaling, not real-world backgrounds.

Usage (from veritas/backend/training):
  python eval_spatial_robustness.py OUT.json ../models/spatial_agent_lc.pth [more.pth ...]
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import face_crop  # noqa: E402
import spatial_agent  # noqa: E402
import wav_cache  # noqa: E402
from eval_audio_robustness import auc, ece  # noqa: E402

MANIFEST = Path(__file__).resolve().parent / "leakage_controlled_split_manifest_v3.json"
TMP = Path(tempfile.mkdtemp())


def mid_frame(path):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def h264(path, crf, canvas=False):
    """Middle frame after re-encoding ~1 s around it with libx264 (optionally on the 720p canvas first)."""
    cap = cv2.VideoCapture(path)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n // 2 - 12))
    frames = [f for f in (cap.read()[1] for _ in range(25)) if f is not None]
    cap.release()
    frames = [to_canvas(f) for f in frames] if canvas else frames
    h, w = frames[0].shape[:2]
    out = TMP / "enc.mp4"
    p = subprocess.Popen([wav_cache._FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", str(crf),
                          "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
    p.communicate(b"".join(f.tobytes() for f in frames))
    cap = cv2.VideoCapture(str(out))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(12, len(frames) - 1))
    ok, f = cap.read()
    cap.release()
    return f


def to_canvas(face_bgr, W=1280, H=720, frac=0.25):
    bg = np.full((H, W, 3), 110, np.uint8)
    s = int(H * frac)
    x, y = (W - s) // 2, (H - s) // 2
    bg[y:y + s, x:x + s] = cv2.resize(face_bgr, (s, s), interpolation=cv2.INTER_AREA)
    return bg


def jpeg(f, q):
    return cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


CONDITIONS = {
    "original": lambda c, f: f,
    "jpeg_q30": lambda c, f: jpeg(f, 30),
    "jpeg_q60": lambda c, f: jpeg(f, 60),
    "down_x0.5": lambda c, f: cv2.resize(cv2.resize(f, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA),
                                         (f.shape[1], f.shape[0]), interpolation=cv2.INTER_LINEAR),
    "blur_s1.5": lambda c, f: cv2.GaussianBlur(f, (0, 0), 1.5),
    "h264_crf32": lambda c, f: h264(c["full_path"], 32),
    "720p_face25": lambda c, f: to_canvas(f),
    "720p_face25_h264_crf28": lambda c, f: h264(c["full_path"], 28, canvas=True),
}


def model_input(bgr):
    """Production preprocessing; None when face_crop finds no usable face."""
    box, _ = face_crop.detect(bgr)
    if box is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return spatial_agent._TRANSFORM(Image.fromarray(np.ascontiguousarray(face_crop.crop(rgb, box))))


@torch.no_grad()
def score(ckpt, xs, device):
    model = spatial_agent._build_model().to(device).eval()
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["state_dict"])
    keep = [i for i, x in enumerate(xs) if x is not None]
    p = np.full(len(xs), np.nan)
    for i in range(0, len(keep), 64):
        idx = keep[i:i + 64]
        p[idx] = torch.softmax(model(torch.stack([xs[j] for j in idx]).to(device)), 1)[:, 1].cpu().numpy()
    return p


def metrics(y, p):
    k = ~np.isnan(p)
    y, p = y[k], p[k]
    return {"auc": auc(y, p), "brier": float(np.mean((p - y) ** 2)), "ece": ece(y, p),
            "fp": int(((p >= .5) & (y == 0)).sum()), "fn": int(((p < .5) & (y == 1)).sum()),
            "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()), "no_face": int((~k).sum())}


def main(out_path, ckpts):
    clips = [c for cl in json.loads(MANIFEST.read_bytes())["clips"]["fusion_val"].values() for c in cl]
    y = np.array([c["type"].startswith("FakeVideo") for c in clips]).astype(int)
    frames = [mid_frame(c["full_path"]) for c in clips]
    xs = {name: [model_input(fn(c, f)) for c, f in zip(clips, frames)] for name, fn in CONDITIONS.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"split": "fusion_val", "split_role": "development (used for candidate selection)", "manifest": MANIFEST.name, "n": len(clips), "checkpoints": {}}
    for ck in ckpts:
        res = {name: score(ck, x, device) for name, x in xs.items()}
        report["checkpoints"][Path(ck).name] = {
            "sha256": wav_cache.file_sha256(ck), "metrics": {n: metrics(y, p) for n, p in res.items()},
            "scores": {n: dict(zip((c["rel_path"] for c in clips), map(float, p))) for n, p in res.items()}}
        print(Path(ck).name)
        for n, m in report["checkpoints"][Path(ck).name]["metrics"].items():
            print(f"  {n:24s} AUC={m['auc']:.4f} Brier={m['brier']:.4f} real->DEEPFAKE={m['fp']}/{m['n_real']} "
                  f"fake->AUTHENTIC={m['fn']}/{m['n_fake']} no_face={m['no_face']}")
    Path(out_path).write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
