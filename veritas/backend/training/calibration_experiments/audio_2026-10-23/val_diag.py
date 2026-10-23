"""Read-only robustness/shortcut diagnostics on fusion_val (dev split). Never touches final_test."""
import hashlib, json, os, subprocess, sys, tempfile, wave
from pathlib import Path
import cv2, numpy as np, torch, torch.nn.functional as F

BACKEND = Path(sys.argv[1]); OUT = Path(sys.argv[2]); TMP = Path(tempfile.mkdtemp(dir=OUT.parent))
sys.path.insert(0, str(BACKEND)); os.chdir(BACKEND)
import logging; logging.disable(logging.CRITICAL)
import spatial_agent as S, temporal_agent as T, audio_agent as A
S.load_model()
FF = __import__("imageio_ffmpeg").get_ffmpeg_exe()
SMEAN = np.array([0.485, 0.456, 0.406], np.float32); SSTD = np.array([0.229, 0.224, 0.225], np.float32)

def sp(frames):
    x = torch.stack([torch.from_numpy(((cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255 - SMEAN) / SSTD).transpose(2, 0, 1))
                     for f in (cv2.resize(f, (224, 224)) for f in frames)])
    with torch.no_grad(): return torch.softmax(S._model(x), 1)[:, 1].tolist()

def temporal(frames_bgr):
    st = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (112, 112)) for f in frames_bgr]).astype(np.float32) / 255
    x = torch.from_numpy(((st - T._MEAN) / T._STD).transpose(3, 0, 1, 2)).unsqueeze(0)
    with torch.no_grad(): return torch.sigmoid(T._model(x)).item()

def ff(args): subprocess.run([FF, "-y", "-loglevel", "error", *args], check=True, timeout=60)

def read16(wav):
    with wave.open(str(wav)) as w: return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768

def audio_p(x, where="center"):
    n, L = len(x), A._CLIP_SAMPLES
    if n < L: x = np.tile(x, -(-L // n))[:L]
    else: x = x[{"start": 0, "center": (n - L) // 2, "end": n - L}[where]:][:L]
    m = A._DB_TRANSFORM(A._MEL_TRANSFORM(torch.from_numpy(x.copy()).unsqueeze(0)))
    t = ((F.interpolate(m.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0) - 0.5) / 0.5).repeat(3, 1, 1)
    with torch.no_grad(): return torch.sigmoid(A._model(t.unsqueeze(0))).item()

m = json.load(open("training/leakage_controlled_split_manifest.json"))
seen, clips = set(), []
for t, l in m["clips"]["fusion_val"].items():
    for c in l:
        if c["full_path"] not in seen: seen.add(c["full_path"]); clips.append(c)
rows = []
for i, c in enumerate(clips):
    p = c["full_path"]; r = {k: c[k] for k in ("rel_path", "type", "source", "method", "race", "gender")}
    cap = cv2.VideoCapture(p); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
    r.update(frames=n, fps=fps, w=int(cap.get(3)), h=int(cap.get(4)), size=os.path.getsize(p))
    def grab(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(min(max(idx, 0), n - 1))); ok, f = cap.read(); return f
    five = [grab(int(n * q)) for q in (0.1, 0.3, 0.5, 0.7, 0.9)]
    mid = five[2]; r["mid_md5"] = hashlib.md5(mid.tobytes()).hexdigest()
    r["sp5"] = sp(five)
    _, jb = cv2.imencode(".jpg", mid, [cv2.IMWRITE_JPEG_QUALITY, 30])
    r["sp_jpeg30"] = sp([cv2.imdecode(jb, 1)])[0]
    r["sp_down112"] = sp([cv2.resize(cv2.resize(mid, (112, 112)), (224, 224))])[0]
    step = max(1, round(fps / 8)); span = 16 * step
    for name, s0 in (("start", 0), ("center", max(0, (n - span) // 2)), ("end", max(0, n - span))):
        r[f"te_{name}"] = temporal([grab(s0 + k * step) for k in range(16)])
    cap.release()
    base = TMP / "b.wav"; ff(["-i", p, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(base)])
    x = read16(base); r["dur_audio"] = len(x) / 16000
    for w_ in ("start", "center", "end"): r[f"au_{w_}"] = audio_p(x, w_)
    r["au_gain-6"] = audio_p(x * 0.5); r["au_gain+6"] = audio_p(np.clip(x * 2, -1, 1))
    for name, enc in (("mp3_64k", ["-c:a", "libmp3lame", "-b:a", "64k", "t.mp3"]), ("aac_64k", ["-c:a", "aac", "-b:a", "64k", "t.m4a"]),
                      ("wav44st", ["-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le", "t.wav"])):
        tf = TMP / enc[-1]; ff(["-i", p, "-vn", *enc[:-1], str(tf)])
        ff(["-i", str(tf), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(base)]); r[f"au_{name}"] = audio_p(read16(base))
    rows.append(r)
    if i % 20 == 0: print(i, len(clips), flush=True)
json.dump(rows, open(OUT, "w"))
print("done", len(rows))
