"""
train_temporal_lc.py
====================
Leakage-controlled training pipeline for VERITAS Temporal Manipulation Detector.
Trains an R3D-18 (3D ResNet-18) video deepfake detector using ONLY clips from
the leakage-controlled `detector_train` split (270 identities, 5,056 clips).
Evaluates, sweeps decision thresholds, and selects checkpoints on `detector_val` (30 identities, 108 clips).

Target:
  - RealVideo-RealAudio: 0 (REAL)
  - RealVideo-FakeAudio: 0 (REAL)
  - FakeVideo-RealAudio: 1 (FAKE)
  - FakeVideo-FakeAudio: 1 (FAKE)

Checkpoint saved to:
  `veritas/backend/models/temporal_agent_lc.pth`
"""

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import r3d_18, R3D_18_Weights

try:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_temporal_lc")

BACKEND_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BACKEND_DIR / "training" / "leakage_controlled_split_manifest_v2.json"  # v2 excludes 26 mislabelled copies (dataset_audit/manifest_v2_exclusions.json)
CHECKPOINT_OUT = BACKEND_DIR / "models" / "temporal_agent_lc.pth"

NUM_FRAMES = 16
FRAME_SIZE = 112
TARGET_FPS = 8.0
WINDOW_SECS = 2.0
MEAN = np.array([0.43216, 0.39467, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.21699], dtype=np.float32)

CATEGORY_LABELS = {
    "RealVideo-RealAudio": 0,
    "RealVideo-FakeAudio": 0,
    "FakeVideo-RealAudio": 1,
    "FakeVideo-FakeAudio": 1,
}


TEMPORAL_CACHE_DIR = Path(r"D:\VERITAS_TRAINING\temporal_cache")
TEMPORAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def extract_raw_16_frames(video_path: str) -> Optional[np.ndarray]:
    """Sample 16 frames at 8 FPS, returning uint8 array of shape (16, 112, 112, 3)."""
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < NUM_FRAMES:
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps / TARGET_FPS)))

        span = NUM_FRAMES * step
        start_frame = max(0, (total - span) // 2)
        indices = [start_frame + i * step for i in range(NUM_FRAMES)]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
            ret, bgr = cap.read()
            if not ret or bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_LINEAR)
            frames.append(rgb)

        if len(frames) != NUM_FRAMES:
            return None
        return np.stack(frames, axis=0)  # (16, 112, 112, 3) uint8
    except Exception:
        return None
    finally:
        cap.release()


def get_cached_or_sample_temporal(video_path: str) -> Optional[np.ndarray]:
    """Retrieve 16 uint8 frames from disk cache or extract and cache."""
    stem = Path(video_path).stem
    parent_id = Path(video_path).parent.name
    cache_file = TEMPORAL_CACHE_DIR / f"{parent_id}_{stem}.npy"
    if cache_file.exists():
        try:
            return np.load(str(cache_file))
        except Exception:
            pass

    raw = extract_raw_16_frames(video_path)
    if raw is not None:
        try:
            np.save(str(cache_file), raw)
        except Exception:
            pass
    return raw


class TemporalDataset(Dataset):
    def __init__(self, clip_list: List[Dict], augment: bool = False):
        self.clip_list = clip_list
        self.augment = augment

    def __len__(self) -> int:
        return len(self.clip_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, int]:
        item = self.clip_list[idx]
        path = item["full_path"]
        label = CATEGORY_LABELS[item["type"]]

        raw_frames = get_cached_or_sample_temporal(path)
        if raw_frames is None:
            tensor = torch.zeros(3, NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, dtype=torch.float32)
            valid = 0
        else:
            frames = raw_frames.copy()
            flip = self.augment and random.random() < 0.5
            proc_frames = []
            for f in frames:
                if flip:
                    f = np.ascontiguousarray(f[:, ::-1, :])
                arr = f.astype(np.float32) / 255.0
                arr = (arr - MEAN) / STD
                proc_frames.append(arr.transpose(2, 0, 1))  # (3, H, W)
            tensor_arr = np.stack(proc_frames, axis=1)  # (3, 16, 112, 112)
            tensor = torch.from_numpy(tensor_arr)
            valid = 1
        return tensor, label, path, valid


def collate_temporal_fn(batch):
    tensors, labels, paths, valids = zip(*batch)
    valid_indices = [i for i, v in enumerate(valids) if v == 1]
    if not valid_indices:
        return None, None, None
    b_tensors = torch.stack([tensors[i] for i in valid_indices], dim=0)
    b_labels = torch.tensor([labels[i] for i in valid_indices], dtype=torch.float32)
    b_paths = [paths[i] for i in valid_indices]
    return b_tensors, b_labels, b_paths


def build_model() -> nn.Module:
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    model.fc = nn.Linear(512, 1)
    return model


def main():
    logger.info("=" * 70)
    logger.info("VERITAS TEMPORAL DETECTOR LEAKAGE-CONTROLLED TRAINING")
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

    train_labels = [CATEGORY_LABELS[c["type"]] for c in train_clips]
    n_real = sum(1 for l in train_labels if l == 0)
    n_fake = sum(1 for l in train_labels if l == 1)
    pos_weight = torch.tensor([float(n_real) / max(n_fake, 1)], dtype=torch.float32).to(device)
    logger.info("Training Class Distribution: Real=%d, Fake=%d | pos_weight=%.4f", n_real, n_fake, pos_weight.item())

    train_dataset = TemporalDataset(train_clips, augment=True)
    val_dataset = TemporalDataset(val_clips, augment=False)

    batch_size = 8 if device.type == "cuda" else 2
    num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_temporal_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_temporal_fn,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_f1 = -1.0
    best_threshold = 0.05
    best_metrics = {}
    epochs = 4

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
        thresholds = np.linspace(0.05, 0.95, 19)
        best_t_f1 = -1.0
        opt_thresh = 0.50
        for thresh in thresholds:
            preds = (val_probs >= thresh).astype(int)
            f1 = f1_score(val_targets, preds, zero_division=0)
            if f1 > best_t_f1:
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

        if best_t_f1 > best_val_f1:
            best_val_f1 = best_t_f1
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
                "model_name": "r3d_18",
                "num_classes": 1,
                "class_names": {"0": "REAL", "1": "FAKE"},
                "num_frames": NUM_FRAMES,
                "frame_size": FRAME_SIZE,
                "target_fps": TARGET_FPS,
                "window_seconds": WINDOW_SECS,
                "normalization": {"mean": MEAN.tolist(), "std": STD.tolist()},
                "decision_threshold": best_threshold,
                "metrics": best_metrics,
                "split_provenance": "detector_train strictly disjoint (270 IDs)",
            }
            CHECKPOINT_OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, CHECKPOINT_OUT)
            logger.info(">>> New best Temporal checkpoint saved to %s (Val F1: %.4f, thresh: %.2f)",
                        CHECKPOINT_OUT, best_t_f1, best_threshold)

    total_time = time.time() - training_start
    logger.info("=" * 70)
    logger.info("Temporal Agent training complete in %.1f seconds.", total_time)
    logger.info("Best Validation Metrics: %s", best_metrics)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
