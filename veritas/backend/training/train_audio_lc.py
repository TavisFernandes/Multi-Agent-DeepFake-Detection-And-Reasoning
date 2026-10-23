"""
train_audio_lc.py
=================
Leakage-controlled training pipeline for VERITAS Audio Synthesis Detector.
Trains an EfficientNet-B0 audio deepfake detector using ONLY clips from
the leakage-controlled `detector_train` split (manifest v3: 270 identities, 5,034 clips).
Evaluates, sweeps decision thresholds, and selects checkpoints on `detector_val` (30 identities, 106 clips).

Target:
  - RealVideo-RealAudio: 0 (REAL)
  - RealVideo-FakeAudio: 1 (FAKE)
  - FakeVideo-RealAudio: 0 (REAL)
  - FakeVideo-FakeAudio: 1 (FAKE)

Checkpoint written to --out (never overwrites; production loads:
  `veritas/backend/models/audio_agent_lc.pth`). --augment adds codec/gain/noise/window augmentation.
"""

import json
import logging
import os
import random
import sys
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_audio_lc")

BACKEND_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BACKEND_DIR / "training" / "leakage_controlled_split_manifest_v3.json"  # v3: v2 minus 11 repeated rows (dataset_audit/manifest_v3_exclusions.json)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wav_cache  # noqa: E402

SAMPLE_RATE = 16000
CLIP_DURATION = 3.0
TARGET_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION)  # 48,000

CATEGORY_AUDIO_LABELS = {
    "RealVideo-RealAudio": 0,
    "RealVideo-FakeAudio": 1,
    "FakeVideo-RealAudio": 0,
    "FakeVideo-FakeAudio": 1,
}

# MelSpectrogram transform
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=1024,
    win_length=1024,
    hop_length=512,
    n_mels=128,
    power=2.0,
)
amp_to_db = T.AmplitudeToDB()


def extract_or_get_wav(mp4_path: str, rel_path: str) -> Optional[str]:
    """16 kHz mono WAV for this exact source file (validated, collision-free cache: wav_cache.py)."""
    if not os.path.exists(mp4_path):
        return None
    got = wav_cache.get_wav(mp4_path, rel_path)
    return got[0] if got else None


def wav_to_mel_tensor(wav_path: str) -> Optional[torch.Tensor]:
    """Convert WAV file to (3, 224, 224) normalized mel-spectrogram tensor."""
    try:
        with wave.open(wav_path, "rb") as wf:
            n_channels = wf.getnchannels()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)

        audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return mel_from_array(audio_np) if len(audio_np) else None
    except Exception:
        return None


def mel_from_array(audio_np: np.ndarray, start: Optional[int] = None) -> torch.Tensor:
    """float32 16 kHz mono -> (3, 224, 224); `start` picks the 3 s window (default: centre, as inference)."""
    # Pad or center crop to 48,000 samples
    if len(audio_np) < TARGET_SAMPLES:
        audio_np = np.pad(audio_np, (0, TARGET_SAMPLES - len(audio_np)), mode="wrap")
    elif len(audio_np) > TARGET_SAMPLES:
        if start is None:
            start = (len(audio_np) - TARGET_SAMPLES) // 2
        audio_np = audio_np[start : start + TARGET_SAMPLES]

    waveform = torch.from_numpy(np.ascontiguousarray(audio_np)).unsqueeze(0)  # (1, 48000)
    mel = mel_transform(waveform)
    mel_db = amp_to_db(mel)

    mel_4d = mel_db.unsqueeze(0)  # (1, 1, 128, T)
    mel_resized = F.interpolate(
        mel_4d, size=(224, 224), mode="bilinear", align_corners=False
    )  # (1, 1, 224, 224)
    mel_2d = mel_resized.squeeze(0)  # (1, 224, 224)
    mel_norm = (mel_2d - 0.5) / 0.5  # (1, 224, 224)
    return mel_norm.repeat(3, 1, 1)  # (3, 224, 224)


class AudioDataset(Dataset):
    def __init__(self, clip_list: List[Dict]):
        self.clip_list = clip_list
        self._cache: Dict[str, Optional[torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self.clip_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, int]:
        item = self.clip_list[idx]
        path = item["full_path"]
        label = CATEGORY_AUDIO_LABELS[item["type"]]

        if path in self._cache:
            tensor = self._cache[path]
        else:
            wav_path = extract_or_get_wav(path, item["rel_path"])
            if wav_path:
                tensor = wav_to_mel_tensor(wav_path)
            else:
                tensor = None
            self._cache[path] = tensor

        if tensor is None:
            ret_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)
            valid = 0
        else:
            ret_tensor = tensor.clone()
            valid = 1
        return ret_tensor, label, path, valid


AUGMENT = {"p_variant": 0.5, "variants": sorted(wav_cache.VARIANTS), "random_window": True,
           "p_gain": 0.5, "gain_db": [-6, 6], "p_noise": 0.3, "noise_snr_db": [15, 40]}


def _read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


class AugmentedAudioDataset(AudioDataset):
    """Each draw: original or a codec/resample decode (validated wav_cache variants), random 3 s window,
    random gain, optional white noise. Labels are untouched."""

    def __init__(self, clip_list: List[Dict], seed: int):
        super().__init__(clip_list)
        self.rng = np.random.default_rng(seed)  # num_workers=0, so one generator is enough

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, int]:
        item, r = self.clip_list[idx], self.rng
        label = CATEGORY_AUDIO_LABELS[item["type"]]
        variant = r.choice(AUGMENT["variants"]) if r.random() < AUGMENT["p_variant"] else None
        got = wav_cache.get_wav(item["full_path"], item["rel_path"], variant=variant)
        x = _read_wav(got[0]) if got else np.zeros(0, np.float32)
        if len(x) == 0:
            return torch.zeros(3, 224, 224), label, item["full_path"], 0
        if r.random() < AUGMENT["p_gain"]:
            x = np.clip(x * 10 ** (r.uniform(*AUGMENT["gain_db"]) / 20), -1, 1)
        if r.random() < AUGMENT["p_noise"]:
            snr = r.uniform(*AUGMENT["noise_snr_db"])
            x = np.clip(x + r.standard_normal(len(x)).astype(np.float32) * np.sqrt(np.mean(x ** 2) / 10 ** (snr / 10)), -1, 1)
        start = int(r.integers(0, len(x) - TARGET_SAMPLES + 1)) if len(x) > TARGET_SAMPLES else None
        return mel_from_array(x.astype(np.float32), start), label, item["full_path"], 1


def warm_wav_cache(clips: List[Dict], variants: List[Optional[str]]) -> None:
    """Extract/validate every (clip, variant) WAV up front, in parallel (ffmpeg dominates)."""
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(c, v) for c in clips for v in variants]
    with ThreadPoolExecutor(8) as ex:
        missing = sum(r is None for r in ex.map(lambda j: wav_cache.get_wav(j[0]["full_path"], j[0]["rel_path"], variant=j[1]), jobs))
    logger.info("WAV cache warm: %d (clip, variant) pairs, %d without audio", len(jobs), missing)


def collate_audio_fn(batch):
    tensors, labels, paths, valids = zip(*batch)
    valid_indices = [i for i, v in enumerate(valids) if v == 1]
    if not valid_indices:
        return None, None, None
    b_tensors = torch.stack([tensors[i] for i in valid_indices], dim=0)
    b_labels = torch.tensor([labels[i] for i in valid_indices], dtype=torch.float32)
    b_paths = [paths[i] for i in valid_indices]
    return b_tensors, b_labels, b_paths


def build_model() -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(1280, 1)
    return model


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="new checkpoint path; refuses to overwrite")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--augment", action="store_true", help="codec/resample, gain, noise, random window")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new name (existing checkpoints are kept)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logger.info("=" * 70)
    logger.info("VERITAS AUDIO DETECTOR LEAKAGE-CONTROLLED TRAINING")
    logger.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if device.type == "cuda":
        logger.info("Device name: %s", torch.cuda.get_device_name(0))

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_clips = []
    for cat, clist in manifest["clips"]["detector_train"].items():
        train_clips.extend(clist)

    val_clips = []
    for cat, clist in manifest["clips"]["detector_val"].items():
        val_clips.extend(clist)

    logger.info("Loaded detector_train clips: %d across %d identities",
                len(train_clips), len(manifest["identities"]["detector_train"]))
    logger.info("Loaded detector_val clips:   %d across %d identities",
                len(val_clips), len(manifest["identities"]["detector_val"]))

    train_labels = [CATEGORY_AUDIO_LABELS[c["type"]] for c in train_clips]
    n_real = sum(1 for l in train_labels if l == 0)
    n_fake = sum(1 for l in train_labels if l == 1)
    pos_weight = torch.tensor([float(n_real) / max(n_fake, 1)], dtype=torch.float32).to(device)
    logger.info("Audio Class Distribution: Real=%d, Fake=%d | pos_weight=%.4f", n_real, n_fake, pos_weight.item())

    warm_wav_cache(train_clips + val_clips, [None, *(AUGMENT["variants"] if args.augment else [])])
    train_dataset = AugmentedAudioDataset(train_clips, args.seed) if args.augment else AudioDataset(train_clips)
    val_dataset = AudioDataset(val_clips)  # selection on clean detector_val, as before

    batch_size = 32 if device.type == "cuda" else 8
    num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_audio_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_audio_fn,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_f1, best_val_loss = -1.0, float("inf")
    best_threshold = 0.50
    best_metrics = {}
    provenance = {
        "manifest": {"file": MANIFEST_PATH.name, "sha256": wav_cache.file_sha256(MANIFEST_PATH)},
        "wav_preprocessing": wav_cache.PREPROCESSING,
        "augment": AUGMENT if args.augment else None,
        "seed": args.seed,
        "epochs": epochs,
        "train_clips": len(train_clips),
        "val_clips": len(val_clips),
    }

    training_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_samples = 0

        for b_tensors, b_labels, _ in train_loader:
            if b_tensors is None:
                continue
            b_tensors = b_tensors.to(device, non_blocking=True)
            b_labels = b_labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(b_tensors).squeeze(-1)
                loss = criterion(logits, b_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(b_labels)
            n_samples += len(b_labels)

        scheduler.step()
        train_loss = total_loss / max(n_samples, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        val_probs = []
        val_targets = []

        with torch.no_grad():
            for b_tensors, b_labels, _ in val_loader:
                if b_tensors is None:
                    continue
                b_tensors = b_tensors.to(device, non_blocking=True)
                b_labels = b_labels.to(device, non_blocking=True)

                with autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(b_tensors).squeeze(-1)
                    loss = criterion(logits, b_labels)
                    probs = torch.sigmoid(logits)

                val_loss += loss.item() * len(b_labels)
                val_samples += len(b_labels)
                val_probs.extend(probs.cpu().numpy())
                val_targets.extend(b_labels.cpu().numpy())

        val_loss /= max(val_samples, 1)
        val_targets = np.array(val_targets)
        val_probs = np.array(val_probs)

        # Sweep thresholds on validation set
        thresholds = np.linspace(0.1, 0.9, 17)
        best_t_f1 = -1.0
        opt_thresh = 0.50
        for thresh in thresholds:
            preds = (val_probs >= thresh).astype(int)
            f1 = f1_score(val_targets, preds, zero_division=0)
            if f1 > best_t_f1 or (f1 == best_t_f1 and abs(thresh - 0.5) < abs(opt_thresh - 0.5)):  # ties: nearest 0.5
                best_t_f1 = f1
                opt_thresh = float(thresh)

        opt_preds = (val_probs >= opt_thresh).astype(int)
        val_acc = accuracy_score(val_targets, opt_preds)
        val_rec = recall_score(val_targets, opt_preds, zero_division=0)
        val_prec = precision_score(val_targets, opt_preds, zero_division=0)
        try:
            val_auc = roc_auc_score(val_targets, val_probs) if len(set(val_targets)) > 1 else 0.5
        except Exception:
            val_auc = 0.5

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d (%.1fs) | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.2f%% | Val F1: %.4f (thresh=%.2f) | Val Rec: %.2f%% | Val AUC: %.4f",
            epoch, epochs, elapsed, train_loss, val_loss, val_acc * 100, best_t_f1, opt_thresh, val_rec * 100, val_auc
        )

        # F1 saturates on 106 clips (1.0 from epoch 1), so ties go to the lower val loss.
        if (best_t_f1, -val_loss) > (best_val_f1, -best_val_loss):
            best_val_f1, best_val_loss = best_t_f1, val_loss
            best_threshold = opt_thresh
            best_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1": best_t_f1,
                "val_roc_auc": val_auc,
                "decision_threshold": best_threshold,
            }
            # Save checkpoint
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "model_name": "efficientnet_b0",
                "num_classes": 1,
                "decision_threshold": best_threshold,
                "metrics": best_metrics,
                "split_provenance": "detector_train strictly disjoint (270 IDs)",
                "training_provenance": provenance,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, args.out)
            logger.info(">>> New best Audio checkpoint saved to %s (Val F1: %.4f, thresh: %.2f)",
                        args.out, best_t_f1, best_threshold)

    total_time = time.time() - training_start
    logger.info("=" * 70)
    logger.info("Audio Agent training complete in %.1f seconds.", total_time)
    logger.info("Best Validation Metrics: %s", best_metrics)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
