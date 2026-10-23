"""Audio calibration re-evaluation on collision-free features. Reports only: no calibrator is selected or deployed.
Fit Platt/isotonic on fusion_train audio scores from cache/fusion_features_lc_v4.json (out-of-sample for
audio_agent_lc.pth), evaluate on fusion_val. Per-format rows apply the same calibrators to live re-decodes of the
fusion_val files (../audio_2026-10-23/val_diag.json, decoded directly from each file, no cache).
Run from veritas/backend:  py -3.10 training/calibration_experiments/audio_2026-10-23_corrected/audio_calib.py
"""
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parents[1]
recs = json.loads((TRAINING / "cache" / "fusion_features_lc_v4.json").read_bytes())["records"].values()


def data(split):
    rs = [r for r in recs if r["split"] == split and r["state_vec"][8] > 0]  # state_vec[8]: audio present
    for r in rs:  # state_vec[6] is the raw audio probability
        assert abs(r["state_vec"][6] - next(a for a in r["agents"] if a["agent"] == "Audio Analysis")["raw_prob"]) < 1e-4
    return (np.array([r["state_vec"][6] for r in rs]), np.array([r["type"].endswith("FakeAudio") for r in rs]), rs,
            sum(r["split"] == split for r in recs))


ptr, ytr, _, ntr = data("fusion_train")
pva, yva, rva, nva = data("fusion_val")
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(ptr)[:, None], ytr)
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(ptr, ytr)
cal = {"raw": lambda p: p, "platt": lambda p: platt.predict_proba(logit(p)[:, None])[:, 1], "isotonic": iso.predict}


def ece(y, p, b=10):
    i = np.minimum((p * b).astype(int), b - 1)
    return sum(abs(p[i == k].mean() - y[i == k].mean()) * (i == k).mean() for k in range(b) if (i == k).any())


def counts(q, y):
    return f"real audio called fake {int(((q >= .5) & ~y).sum())}/{int((~y).sum())}  fake audio called real {int(((q < .5) & y).sum())}/{int(y.sum())}"


print(f"fit fusion_train: {len(ytr)} with audio of {ntr} records ({int(ytr.sum())} fake audio, {int((~ytr).sum())} real)")
print(f"eval fusion_val:  {len(yva)} with audio of {nva} records ({int(yva.sum())} fake audio, {int((~yva).sum())} real)")
print(f"Platt: p = sigmoid({platt.coef_[0][0]:.4f}*logit(raw) + {platt.intercept_[0]:.4f}); threshold 0.5 throughout\n")
for name, f in cal.items():
    q = f(pva)
    print(f"  {name:8s} Brier={brier_score_loss(yva, q):.4f} ECE={ece(yva, q):.4f} AUC={roc_auc_score(yva, q):.4f}  {counts(q, yva)}")

V = {r["rel_path"]: r for r in json.loads((HERE.parent / "audio_2026-10-23" / "val_diag.json").read_bytes())}
live = np.array([V[r["rel_path"]]["au_center"] for r in rva])
print(f"\n  feature-cache vs live decode (fusion_val, n={len(rva)}): max |diff| = {np.abs(live - pva).max():.2e}")
print("\n  by decoding format (fusion_val live re-decode; Brier / ECE / counts):")
for var in ("au_center", "au_mp3_64k", "au_aac_64k", "au_wav44st", "au_gain-6", "au_gain+6"):
    s = np.array([V[r["rel_path"]][var] for r in rva])
    for name, f in cal.items():
        q = f(s)
        print(f"   {var[3:]:9s} {name:8s} Brier={brier_score_loss(yva, q):.4f} ECE={ece(yva, q):.4f}  {counts(q, yva)}")
