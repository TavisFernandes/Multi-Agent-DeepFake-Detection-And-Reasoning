"""
eval_audio_robustness.py
========================
Scores audio checkpoints on fusion_val (manifest v3; out-of-sample for every detector) under each decoding
format, through the validated wav_cache. Never touches final_test.

Formats: original, mp3_64k, aac_64k, wav44st (wav_cache.VARIANTS), gain -6/+6 dB, white noise at 20 dB SNR.
Per format: AUC, Brier, ECE, real audio called fake (FP) and fake audio called real (FN) at 0.5.

Usage (from veritas/backend/training):
  python eval_audio_robustness.py OUT.json ../models/audio_agent_lc.pth [more.pth ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import wav_cache
from train_audio_lc import CATEGORY_AUDIO_LABELS, MANIFEST_PATH, build_model, mel_from_array

FORMATS = ["original", *wav_cache.VARIANTS, "gain-6", "gain+6", "noise20"]


def read_wav(path):
    import wave
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def inputs(clips):
    """{format: [(3,224,224) tensor per clip]}; derived formats follow val_diag.py (gain clips to [-1, 1])."""
    out = {f: [] for f in FORMATS}
    for i, c in enumerate(clips):
        x = read_wav(wav_cache.get_wav(c["full_path"], c["rel_path"])[0])
        out["original"].append(mel_from_array(x))
        for v in wav_cache.VARIANTS:
            out[v].append(mel_from_array(read_wav(wav_cache.get_wav(c["full_path"], c["rel_path"], variant=v)[0])))
        out["gain-6"].append(mel_from_array(x * 0.5))
        out["gain+6"].append(mel_from_array(np.clip(x * 2, -1, 1)))
        rng = np.random.default_rng(i)  # fixed per clip, identical for every checkpoint
        noise = rng.standard_normal(len(x)).astype(np.float32) * np.sqrt(np.mean(x ** 2) / 100)
        out["noise20"].append(mel_from_array(np.clip(x + noise, -1, 1)))
    return out


def auc(y, p):
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    for v in np.unique(p):  # average ranks of ties
        ranks[p == v] = ranks[p == v].mean()
    n1 = y.sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def ece(y, p, b=10):
    i = np.minimum((p * b).astype(int), b - 1)
    return float(sum(abs(p[i == k].mean() - y[i == k].mean()) * (i == k).mean() for k in range(b) if (i == k).any()))


def metrics(y, p):
    return {"auc": auc(y, p), "brier": float(np.mean((p - y) ** 2)), "ece": ece(y, p),
            "fp": int(((p >= .5) & (y == 0)).sum()), "fn": int(((p < .5) & (y == 1)).sum()),
            "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}


@torch.no_grad()
def score(ckpt_path, x, device):
    model = build_model().to(device).eval()
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
    return torch.cat([torch.sigmoid(model(torch.stack(x[i:i + 64]).to(device)).squeeze(-1)).cpu()
                      for i in range(0, len(x), 64)]).numpy()


def main(out_path, ckpts):
    manifest = json.loads(MANIFEST_PATH.read_bytes())
    clips = [c for cl in manifest["clips"]["fusion_val"].values() for c in cl]
    y = np.array([CATEGORY_AUDIO_LABELS[c["type"]] for c in clips])
    x = inputs(clips)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"split": "fusion_val", "manifest": MANIFEST_PATH.name, "n": len(clips), "checkpoints": {}}
    for ck in ckpts:
        scores = {f: score(ck, x[f], device) for f in FORMATS}
        report["checkpoints"][Path(ck).name] = {
            "sha256": wav_cache.file_sha256(ck),
            "metrics": {f: metrics(y, s) for f, s in scores.items()},
            "scores": {f: dict(zip((c["rel_path"] for c in clips), map(float, s))) for f, s in scores.items()},
        }
        print(Path(ck).name)
        for f, m in report["checkpoints"][Path(ck).name]["metrics"].items():
            print(f"  {f:9s} AUC={m['auc']:.4f} Brier={m['brier']:.4f} ECE={m['ece']:.4f} "
                  f"FP={m['fp']}/{m['n_real']} FN={m['fn']}/{m['n_fake']}")
    Path(out_path).write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
