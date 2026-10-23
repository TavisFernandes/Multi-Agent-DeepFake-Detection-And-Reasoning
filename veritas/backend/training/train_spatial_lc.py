"""
train_spatial_lc.py
===================
Leakage-controlled training pipeline for VERITAS Spatial Manipulation Detector.
Trains an EfficientNet-B0 visual deepfake detector using ONLY clips from
the leakage-controlled `detector_train` split (270 identities, 5,056 clips).
Evaluates and selects checkpoints on `detector_val` (30 identities, 108 clips).

Target:
  - RealVideo-RealAudio: 0 (REAL)
  - RealVideo-FakeAudio: 0 (REAL)
  - FakeVideo-RealAudio: 1 (FAKE)
  - FakeVideo-FakeAudio: 1 (FAKE)

Checkpoint saved to:
  `veritas/backend/models/spatial_agent_lc.pth`
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
logger = logging.getLogger("train_spatial_lc")

BACKEND_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BACKEND_DIR / "training" / "leakage_controlled_split_manifest_v2.json"  # v2 excludes 26 mislabelled copies (dataset_audit/manifest_v2_exclusions.json)
CHECKPOINT_OUT = BACKEND_DIR / "models" / "spatial_agent_lc.pth"

IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CATEGORY_LABELS = {
    "RealVideo-RealAudio": 0,
    "RealVideo-FakeAudio": 0,
    "FakeVideo-RealAudio": 1,
    "FakeVideo-FakeAudio": 1,
}


def extract_raw_frame(video_path: str) -> Optional[np.ndarray]:
    """Extract and return middle frame as (224, 224, 3) uint8 RGB array."""
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return None
        idx = frame_count // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        return frame  # (224, 224, 3) uint8
    except Exception:
        return None
    finally:
        cap.release()


class SpatialDataset(Dataset):
    def __init__(self, clip_list: List[Dict], augment: bool = False):
        self.clip_list = clip_list
        self.augment = augment
        self._cache: Dict[str, Optional[np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.clip_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, int]:
        item = self.clip_list[idx]
        path = item["full_path"]
        label = CATEGORY_LABELS[item["type"]]

        if path in self._cache:
            raw_frame = self._cache[path]
        else:
            raw_frame = extract_raw_frame(path)
            self._cache[path] = raw_frame

        if raw_frame is None:
            tensor = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
            valid = 0
        else:
            frame = raw_frame.copy()
            if self.augment and random.random() < 0.5:
                frame = np.ascontiguousarray(frame[:, ::-1, :])
            arr = frame.astype(np.float32) / 255.0
            arr = (arr - MEAN) / STD
            tensor = torch.from_numpy(arr.transpose(2, 0, 1))
            valid = 1
        return tensor, label, path, valid


def collate_fn(batch):
    tensors, labels, paths, valids = zip(*batch)
    valid_indices = [i for i, v in enumerate(valids) if v == 1]
    if not valid_indices:
        return None, None, None
    b_tensors = torch.stack([tensors[i] for i in valid_indices], dim=0)
    b_labels = torch.tensor([labels[i] for i in valid_indices], dtype=torch.long)
    b_paths = [paths[i] for i in valid_indices]
    return b_tensors, b_labels, b_paths


def build_model() -> nn.Module:
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(1280, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2, inplace=False),
        nn.Linear(256, 2),
    )
    return model


def main():
    logger.info("=" * 70)
    logger.info("VERITAS SPATIAL DETECTOR LEAKAGE-CONTROLLED TRAINING")
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
    val_labels = [CATEGORY_LABELS[c["type"]] for c in val_clips]

    n_real = sum(1 for l in train_labels if l == 0)
    n_fake = sum(1 for l in train_labels if l == 1)
    logger.info("Training Class Distribution: Real=%d, Fake=%d (Total=%d)", n_real, n_fake, len(train_labels))
    weight_real = len(train_labels) / (2.0 * max(n_real, 1))
    weight_fake = len(train_labels) / (2.0 * max(n_fake, 1))
    class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float32).to(device)
    logger.info("Class Weights: Real=%.3f, Fake=%.3f", weight_real, weight_fake)

    train_dataset = SpatialDataset(train_clips, augment=True)
    val_dataset = SpatialDataset(val_clips, augment=False)

    batch_size = 32 if device.type == "cuda" else 8
    num_workers = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_f1 = -1.0
    best_metrics = {}
    epochs = 5

    training_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        n_batches = 0
        n_samples = 0

        for b_tensors, b_labels, _ in train_loader:
            if b_tensors is None:
                continue
            b_tensors = b_tensors.to(device, non_blocking=True)
            b_labels = b_labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(b_tensors)
                loss = criterion(logits, b_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * len(b_labels)
            n_samples += len(b_labels)
            n_batches += 1

        scheduler.step()
        train_loss = total_loss / max(n_samples, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        val_preds = []
        val_targets = []
        val_probs = []

        with torch.no_grad():
            for b_tensors, b_labels, _ in val_loader:
                if b_tensors is None:
                    continue
                b_tensors = b_tensors.to(device, non_blocking=True)
                b_labels = b_labels.to(device, non_blocking=True)

                with autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(b_tensors)
                    loss = criterion(logits, b_labels)
                    probs = torch.softmax(logits, dim=1)[:, 1]

                val_loss += loss.item() * len(b_labels)
                val_samples += len(b_labels)
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_targets.extend(b_labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        val_loss /= max(val_samples, 1)
        val_targets = np.array(val_targets)
        val_preds = np.array(val_preds)
        val_probs = np.array(val_probs)

        val_acc = accuracy_score(val_targets, val_preds) if len(val_targets) > 0 else 0.0
        val_rec = recall_score(val_targets, val_preds, zero_division=0) if len(val_targets) > 0 else 0.0
        val_prec = precision_score(val_targets, val_preds, zero_division=0) if len(val_targets) > 0 else 0.0
        val_f1 = f1_score(val_targets, val_preds, zero_division=0) if len(val_targets) > 0 else 0.0
        try:
            val_auc = roc_auc_score(val_targets, val_probs) if len(set(val_targets)) > 1 else 0.5
        except Exception:
            val_auc = 0.5

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d (%.1fs) | Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.2f%% | Val F1: %.4f | Val Rec: %.2f%% | Val AUC: %.4f",
            epoch, epochs, elapsed, train_loss, val_loss, val_acc * 100, val_f1, val_rec * 100, val_auc
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1": val_f1,
                "val_roc_auc": val_auc,
            }
            # Save checkpoint
            payload = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "model_name": "efficientnet_b0",
                "num_classes": 2,
                "class_names": {"0": "REAL", "1": "FAKE"},
                "image_size": IMAGE_SIZE,
                "metrics": best_metrics,
                "class_weights": class_weights.cpu().tolist(),
                "split_provenance": "detector_train strictly disjoint (270 IDs)",
            }
            CHECKPOINT_OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, CHECKPOINT_OUT)
            logger.info(">>> New best Spatial checkpoint saved to %s (Val F1: %.4f)", CHECKPOINT_OUT, val_f1)

    total_time = time.time() - training_start
    logger.info("=" * 70)
    logger.info("Spatial Agent training complete in %.1f seconds.", total_time)
    logger.info("Best Validation Metrics: %s", best_metrics)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
