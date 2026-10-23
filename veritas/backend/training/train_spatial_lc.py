"""
train_spatial_lc.py
===================
Leakage-controlled training pipeline for VERITAS Spatial Manipulation Detector.
Trains an EfficientNet-B0 visual deepfake detector using ONLY clips from
the leakage-controlled `detector_train` split (manifest v3: 270 identities, 5,034 clips).
Evaluates and selects checkpoints on `detector_val` (30 identities, 106 clips).

Target:
  - RealVideo-RealAudio: 0 (REAL)
  - RealVideo-FakeAudio: 0 (REAL)
  - FakeVideo-RealAudio: 1 (FAKE)
  - FakeVideo-FakeAudio: 1 (FAKE)

Checkpoint written to --out (never overwrites; production loads `veritas/backend/models/spatial_agent_lc.pth`).
--augment: several frames per clip, H.264 re-encodes, JPEG, rescale, blur, crop jitter; checkpoint selection by
mean detector_val AUC over clean/JPEG/rescaled/H.264 copies instead of clean F1.
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
MANIFEST_PATH = BACKEND_DIR / "training" / "leakage_controlled_split_manifest_v3.json"  # v3: v2 minus 11 repeated rows (dataset_audit/manifest_v3_exclusions.json)

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


# crop_side < 1: tighter, shifted framing (YuNet box jitter, faces capped at the frame in face_crop.crop).
AUGMENT = {"frames_per_clip": 3, "p_h264": 0.4, "h264_crf": [24, 38], "p_jpeg": 0.4, "jpeg_q": [25, 95],
           "p_rescale": 0.3, "rescale": [0.3, 0.9], "p_blur": 0.25, "blur_sigma": [0.3, 1.8],
           "p_crop": 0.4, "crop_side": [0.8, 1.0], "p_color": 0.3, "p_flip": 0.5}
FRAME_CACHE = Path(r"D:\VERITAS_TRAINING\spatial_frames_v1")


def cached_frames(clip: Dict, k: int, crf: Optional[int] = None) -> Path:
    """.npy of `_frames(...)` (k, 224, 224, 3) uint8, reused only if the sidecar matches source sha and settings."""
    import wav_cache
    npy = FRAME_CACHE / f"{wav_cache.cache_key(clip['rel_path'])}__k{k}{f'__crf{crf}' if crf else ''}.npy"
    want = {"rel_path": clip["rel_path"], "source_sha256": wav_cache.file_sha256(clip["full_path"]),
            "frames_per_clip": k, "crf": crf, "image_size": IMAGE_SIZE, "version": 1}
    try:
        if json.loads(npy.with_suffix(".json").read_text(encoding="utf-8")) == want and npy.exists():
            return npy
    except (OSError, ValueError):
        pass
    FRAME_CACHE.mkdir(parents=True, exist_ok=True)
    fs = _frames(clip["full_path"], k, crf, tmp=npy.with_suffix(".partial.mp4"))
    arr = np.stack(fs) if fs else np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    with open(npy.with_suffix(".partial"), "wb") as f:  # np.save would append .npy to a str path
        np.save(f, arr)
    os.replace(npy.with_suffix(".partial"), npy)
    npy.with_suffix(".json").write_text(json.dumps(want, indent=1), encoding="utf-8")
    return npy


def _frames(video_path: str, k: int, crf: Optional[int] = None, tmp: Optional[Path] = None) -> List[np.ndarray]:
    """k evenly spaced frames as 224x224 RGB; with `crf`, from an H.264 re-encode of the whole clip (written to `tmp`)."""
    import subprocess
    import tempfile
    import wav_cache
    src = video_path
    if crf is not None:
        tmp = tmp or Path(tempfile.mkdtemp()) / "re.mp4"
        subprocess.run([wav_cache._FFMPEG, "-y", "-loglevel", "error", "-i", video_path, "-an", "-c:v", "libx264",
                        "-crf", str(crf), "-pix_fmt", "yuv420p", str(tmp)], check=True, timeout=120)
        src = str(tmp)
    cap = cv2.VideoCapture(src)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for q in np.linspace(0.2, 0.8, k):
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(int(n * q)))
        ok, f = cap.read()
        if ok:
            out.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR))
    cap.release()
    if crf is not None:
        Path(src).unlink(missing_ok=True)
    return out


def degrade(rgb: np.ndarray, r: np.random.Generator) -> np.ndarray:
    """Random JPEG / rescale / blur / crop jitter / colour / flip, applied equally to both classes."""
    a = AUGMENT
    if r.random() < a["p_crop"]:
        s = int(IMAGE_SIZE * r.uniform(*a["crop_side"]))
        x, y = r.integers(0, IMAGE_SIZE - s + 1, 2)
        rgb = cv2.resize(rgb[y:y + s, x:x + s], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    if r.random() < a["p_rescale"]:
        f = r.uniform(*a["rescale"])
        small = cv2.resize(rgb, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        rgb = cv2.resize(small, (IMAGE_SIZE, IMAGE_SIZE), interpolation=int(r.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC])))
    if r.random() < a["p_blur"]:
        rgb = cv2.GaussianBlur(rgb, (0, 0), r.uniform(*a["blur_sigma"]))
    if r.random() < a["p_color"]:
        rgb = np.clip(rgb.astype(np.float32) * r.uniform(0.8, 1.2) + r.uniform(-20, 20), 0, 255).astype(np.uint8)
    if r.random() < a["p_jpeg"]:
        q = int(r.integers(a["jpeg_q"][0], a["jpeg_q"][1] + 1))
        rgb = cv2.imdecode(cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)[:, :, ::-1]
    if r.random() < a["p_flip"]:
        rgb = rgb[:, ::-1]
    return np.ascontiguousarray(rgb)


def to_tensor(rgb: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1))


class AugmentedSpatialDataset(Dataset):
    """Frames (original and one H.264 re-encode per clip, cached on disk by `cached_frames`), degraded on each draw."""

    def __init__(self, clip_list: List[Dict], seed: int):
        from concurrent.futures import ThreadPoolExecutor
        self.clip_list, self.rng = clip_list, np.random.default_rng(seed)
        crfs = np.random.default_rng(seed + 1).integers(AUGMENT["h264_crf"][0], AUGMENT["h264_crf"][1] + 1, len(clip_list))
        k = AUGMENT["frames_per_clip"]
        with ThreadPoolExecutor(8) as ex:
            self.orig = list(ex.map(lambda c: cached_frames(c, k), clip_list))
            self.h264 = list(ex.map(lambda ic: cached_frames(ic[1], k, int(crfs[ic[0]])), enumerate(clip_list)))
        logger.info("Frame cache %s: %d clips x %d frames (+ H.264 CRF %s re-encodes)", FRAME_CACHE, len(clip_list),
                    k, AUGMENT["h264_crf"])

    def __len__(self) -> int:
        return len(self.clip_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, int]:
        item, r = self.clip_list[idx], self.rng
        pool = np.load(self.h264[idx] if r.random() < AUGMENT["p_h264"] else self.orig[idx], mmap_mode="r")
        if not len(pool):
            pool = np.load(self.orig[idx], mmap_mode="r")
        if not len(pool):
            return torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE), CATEGORY_LABELS[item["type"]], item["full_path"], 0
        frame = np.array(pool[int(r.integers(len(pool)))])  # copy one frame out of the memory map
        return to_tensor(degrade(frame, r)), CATEGORY_LABELS[item["type"]], item["full_path"], 1


def robust_val_sets(clip_list: List[Dict]) -> Dict[str, Tuple[torch.Tensor, np.ndarray]]:
    """Deterministic detector_val copies (middle frame): clean, JPEG q40, x0.5 rescale, H.264 CRF 32."""
    labels = np.array([CATEGORY_LABELS[c["type"]] for c in clip_list])
    clean = [extract_raw_frame(c["full_path"]) for c in clip_list]
    h264 = [(_frames(c["full_path"], 1, 32) or [None])[0] for c in clip_list]
    jpeg = lambda f: cv2.imdecode(cv2.imencode(".jpg", f[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 40])[1], cv2.IMREAD_COLOR)[:, :, ::-1]
    half = lambda f: cv2.resize(cv2.resize(f, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA), (IMAGE_SIZE, IMAGE_SIZE))
    sets = {"clean": clean, "jpeg_q40": [jpeg(f) for f in clean], "rescale_x0.5": [half(f) for f in clean], "h264_crf32": h264}
    assert all(f is not None for fs in sets.values() for f in fs), "unreadable detector_val frame"
    return {n: (torch.stack([to_tensor(np.ascontiguousarray(f)) for f in fs]), labels) for n, fs in sets.items()}


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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="new checkpoint path; refuses to overwrite")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--augment", action="store_true", help="degradation augmentation + robust selection")
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; choose a new name (existing checkpoints are kept)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

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

    train_dataset = AugmentedSpatialDataset(train_clips, args.seed) if args.augment else SpatialDataset(train_clips, augment=True)
    val_dataset = SpatialDataset(val_clips, augment=False)
    robust_val = robust_val_sets(val_clips) if args.augment else None

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
    epochs = args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_f1 = -1.0
    best_metrics = {}
    import wav_cache
    provenance = {
        "manifest": {"file": MANIFEST_PATH.name, "sha256": wav_cache.file_sha256(MANIFEST_PATH)},
        "augment": AUGMENT if args.augment else {"p_flip": 0.5},
        "selection": "mean detector_val AUC over clean/jpeg_q40/rescale_x0.5/h264_crf32" if args.augment
                     else "detector_val F1 (clean)",
        "seed": args.seed, "epochs": epochs, "train_clips": len(train_clips), "val_clips": len(val_clips),
    }

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

        selection_score = val_f1
        if robust_val is not None:
            robust_auc = {}
            with torch.no_grad():
                for name, (x, y) in robust_val.items():
                    p = torch.cat([torch.softmax(model(x[i:i + 64].to(device)), 1)[:, 1].float().cpu()
                                   for i in range(0, len(x), 64)]).numpy()
                    robust_auc[name] = float(roc_auc_score(y, p))
            selection_score = float(np.mean(list(robust_auc.values())))
            logger.info("Robust detector_val AUC: %s | mean %.4f",
                        {k: round(v, 4) for k, v in robust_auc.items()}, selection_score)

        if selection_score > best_val_f1:
            best_val_f1 = selection_score
            best_metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_f1": val_f1,
                "val_roc_auc": val_auc,
                "selection_score": selection_score,
                "robust_val_auc": robust_auc if robust_val is not None else None,
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
                "training_provenance": provenance,
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, args.out)
            logger.info(">>> New best Spatial checkpoint saved to %s (selection score: %.4f)", args.out, selection_score)

    total_time = time.time() - training_start
    logger.info("=" * 70)
    logger.info("Spatial Agent training complete in %.1f seconds.", total_time)
    logger.info("Best Validation Metrics: %s", best_metrics)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
