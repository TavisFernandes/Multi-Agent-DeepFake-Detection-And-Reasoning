"""
train_audio_itw.py
==================
Fine-tune the production audio detector (v2 aug s1) on FakeAVCeleb detector_train + In-the-Wild
(Müller et al. 2022) training speakers, so it recognises voice generators beyond FakeAVCeleb's RTVC.

In-the-Wild is split by SPEAKER (seeded, written to itw_speaker_split_v1.json): train / val / test
speakers never overlap. Epochs are selected on val speakers + FakeAVCeleb detector_val; test speakers
are never read here.

Usage (training venv):
  python training/train_audio_itw.py --itw-root D:/VERITAS_TRAINING/data/in_the_wild/release_in_the_wild \
      --out models/audio_agent_lc_v3_itw_s0.pth
"""
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.amp import GradScaler, autocast
from torch.utils.data import ConcatDataset, DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_audio_lc as base  # noqa: E402  (same mel pipeline, augmentation and model as production)

SPLIT_PATH = Path(__file__).resolve().parent / "itw_speaker_split_v1.json"
INIT_CKPT = base.BACKEND_DIR / "models" / "audio_agent_lc_v2_aug_s1.pth"
N_TEST_SPK, N_VAL_SPK = 12, 6
CAP_TRAIN, CAP_VAL = 300, 100  # clips per (speaker, label): stops Obama/Trump dominating


def speaker_split(rows, seed=0):
    spk = sorted({r["speaker"] for r in rows})
    random.Random(seed).shuffle(spk)
    split = {"test": sorted(spk[:N_TEST_SPK]), "val": sorted(spk[N_TEST_SPK:N_TEST_SPK + N_VAL_SPK]),
             "train": sorted(spk[N_TEST_SPK + N_VAL_SPK:])}
    assert not (set(split["test"]) & set(split["val"]) | set(split["test"]) & set(split["train"])
                | set(split["val"]) & set(split["train"])), "speaker leakage"
    return split


def itw_items(rows, speakers, cap, root, seed=0):
    groups = defaultdict(list)
    for r in rows:
        if r["speaker"] in speakers:
            groups[(r["speaker"], r["label"])].append(r)
    rng, out = random.Random(seed), []
    for key in sorted(groups):
        g = groups[key]
        for r in rng.sample(g, min(cap, len(g))):
            out.append({"path": str(root / r["file"]), "label": int(r["label"] == "spoof")})
    return out


def read_itw(path: str) -> np.ndarray:
    _, x = wavfile.read(path)  # float32 mono 16 kHz
    x = np.clip(x.astype(np.float32), -1, 1)
    return np.round(x * 32767) / 32768.0  # same int16 quantisation inference's ffmpeg step applies


class ITWDataset(Dataset):
    # shortcut: gain/noise/window augmentation only, no codec variants (wav_cache needs a source video); add if MP3-heavy inputs fail
    def __init__(self, items, augment, seed=0):
        self.items, self.augment, self.rng = items, augment, np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it, r, A = self.items[i], self.rng, base.AUGMENT
        x = read_itw(it["path"])
        if len(x) == 0:
            return torch.zeros(3, 224, 224), it["label"], it["path"], 0
        start = None
        if self.augment:
            if r.random() < A["p_gain"]:
                x = np.clip(x * 10 ** (r.uniform(*A["gain_db"]) / 20), -1, 1)
            if r.random() < A["p_noise"]:
                snr = r.uniform(*A["noise_snr_db"])
                x = np.clip(x + r.standard_normal(len(x)).astype(np.float32) * np.sqrt(np.mean(x ** 2) / 10 ** (snr / 10)), -1, 1)
            if len(x) > base.TARGET_SAMPLES:
                start = int(r.integers(0, len(x) - base.TARGET_SAMPLES + 1))
        return base.mel_from_array(x.astype(np.float32), start), it["label"], it["path"], 1


def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for xb, yb, _ in loader:
            if xb is None:
                continue
            with autocast("cuda", enabled=device.type == "cuda"):
                probs += torch.sigmoid(model(xb.to(device)).squeeze(-1)).float().cpu().tolist()
            ys += yb.tolist()
    return np.array(ys), np.array(probs)


def scores(y, p):
    return {"balanced_acc": float(balanced_accuracy_score(y, p >= 0.5)), "auc": float(roc_auc_score(y, p)), "n": int(len(y))}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--itw-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new name (existing checkpoints are kept)")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = list(csv.DictReader(open(args.itw_root / "meta.csv", encoding="utf-8")))
    split = speaker_split(rows)
    SPLIT_PATH.write_text(json.dumps({"dataset": "In-the-Wild (mueller91/In-The-Wild)", "seed": 0, **split}, indent=2))
    itw_train = itw_items(rows, set(split["train"]), CAP_TRAIN, args.itw_root, args.seed)
    itw_val = itw_items(rows, set(split["val"]), CAP_VAL, args.itw_root, args.seed)

    manifest = json.load(open(base.MANIFEST_PATH, encoding="utf-8"))
    favc_train = [c for cl in manifest["clips"]["detector_train"].values() for c in cl]
    favc_val = [c for cl in manifest["clips"]["detector_val"].values() for c in cl]
    base.warm_wav_cache(favc_train + favc_val, [None, *base.AUGMENT["variants"]])

    labels = [it["label"] for it in itw_train] + [base.CATEGORY_AUDIO_LABELS[c["type"]] for c in favc_train]
    pos_weight = torch.tensor([labels.count(0) / max(labels.count(1), 1)], device=device)
    base.logger.info("train: ITW %d + FAVC %d clips | real %d fake %d | val: ITW %d + FAVC %d",
                     len(itw_train), len(favc_train), labels.count(0), labels.count(1), len(itw_val), len(favc_val))

    mk = lambda ds, shuffle: DataLoader(ds, batch_size=32, shuffle=shuffle, num_workers=0,
                                        collate_fn=base.collate_audio_fn, pin_memory=device.type == "cuda")
    train_loader = mk(ConcatDataset([ITWDataset(itw_train, True, args.seed),
                                     base.AugmentedAudioDataset(favc_train, args.seed)]), True)
    val_loaders = {"itw_val": mk(ITWDataset(itw_val, False), False), "favc_val": mk(base.AudioDataset(favc_val), False)}

    model = base.build_model()
    model.load_state_dict(torch.load(INIT_CKPT, map_location="cpu", weights_only=True)["model_state_dict"])
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    for name, loader in val_loaders.items():
        base.logger.info("epoch 0 (init) %s %s", name, scores(*predict(model, loader, device)))

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb, _ in train_loader:
            if xb is None:
                continue
            opt.zero_grad()
            with autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(xb.to(device)).squeeze(-1), yb.to(device))
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        metrics = {name: scores(*predict(model, loader, device)) for name, loader in val_loaders.items()}
        sel = (metrics["itw_val"]["balanced_acc"] + metrics["favc_val"]["balanced_acc"]) / 2
        base.logger.info("epoch %d %s selection=%.4f", epoch, metrics, sel)
        if sel > best:
            best = sel
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(), "model_name": "efficientnet_b0",
                "num_classes": 1, "decision_threshold": 0.5,
                "metrics": {**metrics, "selection": sel},
                "split_provenance": "FAVC detector_train (270 IDs) + In-the-Wild train speakers; see itw_speaker_split_v1.json",
                "training_provenance": {
                    "init": INIT_CKPT.name, "manifest": {"file": base.MANIFEST_PATH.name, "sha256": base.wav_cache.file_sha256(base.MANIFEST_PATH)},
                    "itw_split": {"file": SPLIT_PATH.name, "sha256": base.wav_cache.file_sha256(SPLIT_PATH)},
                    "itw_caps": [CAP_TRAIN, CAP_VAL], "augment": base.AUGMENT, "lr": 1e-4, "seed": args.seed, "epochs": args.epochs,
                    "train_clips": {"itw": len(itw_train), "favc": len(favc_train)}},
            }, args.out)
            base.logger.info(">>> saved %s (selection %.4f)", args.out, sel)


if __name__ == "__main__":
    main()
