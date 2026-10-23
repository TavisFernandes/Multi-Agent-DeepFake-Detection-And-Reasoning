"""Fit audio calibration on fusion_train (out-of-sample for audio_agent_lc.pth), evaluate on fusion_val.
Selection rule fixed before looking: lowest val Brier, provided fake-audio-called-real does not increase."""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
B, SP = Path(sys.argv[1]), Path(sys.argv[2])
bad = {x["rel_path"] for g in json.load(open(SP / "dup_groups.json")) for x in g if not x["rel_path"].startswith("RealVideo-RealAudio")}
recs = json.load(open(B / "training/cache/fusion_features_lc_v3.json"))["records"].values()
def data(split):
    rs = [r for r in recs if r["split"] == split and r["rel_path"] not in bad and r["state_vec"][8] > 0]
    return np.array([r["state_vec"][6] for r in rs]), np.array([r["ground_truth_category"] in (1, 3) for r in rs]).astype(int), rs
ptr, ytr, _ = data("fusion_train"); pva, yva, rva = data("fusion_val")
print(f"fit: fusion_train n={len(ytr)} (fake audio {ytr.sum()}), eval: fusion_val n={len(yva)} (fake audio {yva.sum()}); excluded mislabelled copies: {len(bad)}")
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(ptr)[:, None], ytr)
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(ptr, ytr)
cal = {"raw": lambda p: p, "platt": lambda p: platt.predict_proba(logit(p)[:, None])[:, 1], "isotonic": iso.predict}
def ece(y, p, b=10):
    i = np.minimum((p * b).astype(int), b - 1); return sum(abs(p[i == k].mean() - y[i == k].mean()) * (i == k).mean() for k in range(b) if (i == k).any())
print(f"Platt: a={platt.coef_[0][0]:.4f} b={platt.intercept_[0]:.4f}  (p_cal = sigmoid(a*logit(p)+b))")
for name, f in cal.items():
    q = f(pva)
    print(f"  {name:8s} Brier={brier_score_loss(yva, q):.4f} ECE={ece(yva, q):.4f} AUC={roc_auc_score(yva, q):.4f}"
          f"  real audio called fake@0.5={int(((q >= .5) & (yva == 0)).sum())}/{int((yva == 0).sum())}"
          f"  fake audio called real@0.5={int(((q < .5) & (yva == 1)).sum())}/{int(yva.sum())}"
          f"  real-audio score median={np.median(q[yva == 0]):.3f} p90={np.quantile(q[yva == 0], .9):.3f}")
# Codec variants (val_diag.json, same fusion_val clips): do decisions stay stable after calibration?
V = {r["rel_path"]: r for r in json.load(open(SP / "val_diag.json"))}
keep = [r["rel_path"] for r in rva if r["rel_path"] in V]; yk = np.array([r["ground_truth_category"] in (1, 3) for r in rva if r["rel_path"] in V])
print("\n  decoding variants, real audio called fake@0.5 (of %d) / fake audio called real (of %d):" % ((~yk).sum(), yk.sum()))
for var in ("au_center", "au_mp3_64k", "au_aac_64k", "au_wav44st", "au_gain-6", "au_gain+6"):
    s = np.array([V[k][var] for k in keep])
    print("   %-11s" % var[3:], "  ".join(f"{n}: {int(((f(s) >= .5) & ~yk).sum())}/{int(((f(s) < .5) & yk).sum())}" for n, f in cal.items()))
json.dump({"method": "platt", "a": float(platt.coef_[0][0]), "b": float(platt.intercept_[0]), "fit_split": "fusion_train", "n_fit": int(len(ytr))}, open(SP / "audio_platt.json", "w"))
