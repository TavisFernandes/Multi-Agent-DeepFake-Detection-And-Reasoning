"""
train_ppo_v3.py
===============
Mathematically sound PPO v3 Adaptive Fusion Policy training pipeline.
Uses truly out-of-sample feature representations generated from leakage-controlled
base detectors:
  - Feature Cache: veritas/backend/training/cache/fusion_features_lc_v3.json
  - Train Set:     fusion_train (100 identities, 802 clips)
  - Val Set:       fusion_val (50 identities, 220 clips)

Architecture & Training Invariants:
-----------------------------------
1. 14-dimensional state representation.
2. 5-dimensional continuous action space (visual weights, audio trust, mismatch gain, adaptive threshold).
3. FeatureNormalizer fitted strictly on fusion_train.
4. Single-step contextual bandit (gamma=0, advantage = R - V).
5. Ground-truth rewards computed using verified training labels.
6. Checkpoint selection strictly driven by fusion_val performance.
7. Saves to: veritas/backend/models/ppo_fusion_policy_v3.pth.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

_HERE = Path(__file__).resolve().parent
_BACKEND_DIR = _HERE.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fusion_policy import (
    ACTION_DIM,
    CATEGORIES,
    CATEGORY_TO_IDX,
    IDX_TO_CATEGORY,
    STATE_DIM,
    ActorCriticNetwork,
    FeatureNormalizer,
    apply_action_to_state,
    compute_reward,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_ppo_v3")

FEATURES_CACHE_PATH = _HERE / "cache" / "fusion_features_lc_v3.json"
MANIFEST_PATH = _HERE / "leakage_controlled_split_manifest_v2.json"
CHECKPOINT_OUT = _BACKEND_DIR / "models" / "ppo_fusion_policy_v3.pth"


class PPOTrainerV3:
    def __init__(
        self,
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]],
        normalizer: FeatureNormalizer,
        lr: float = 3e-4,
        clip_epsilon: float = 0.20,
        c_val: float = 0.50,
        c_ent: float = 0.01,
        device: Optional[torch.device] = None,
        seed: int = 42,
        epochs: int = 20,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_records = train_records
        self.val_records = val_records
        self.normalizer = normalizer

        self.clip_epsilon = clip_epsilon
        self.c_val = c_val
        self.c_ent = c_ent

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.network = ActorCriticNetwork(STATE_DIM, ACTION_DIM).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

    def collect_rollout(self, batch_size: int = 512) -> Dict[str, torch.Tensor]:
        self.network.eval()
        sampled = random.choices(self.train_records, k=batch_size)

        states_raw = np.array([r["state_vec"] for r in sampled], dtype=np.float32)
        states_norm = self.normalizer.transform(states_raw)
        gt_cats = [r["ground_truth_category"] for r in sampled]

        states_t = torch.tensor(states_norm, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            dist, values = self.network(states_t)
            actions_t = dist.sample()
            log_probs_t = dist.log_prob(actions_t).sum(dim=-1)

        actions_np = actions_t.cpu().numpy()
        rewards = []

        for i in range(batch_size):
            action = actions_np[i]
            state_raw = states_raw[i]
            metadata = sampled[i]["metadata"]
            gt_cat = gt_cats[i]

            fusion_output = apply_action_to_state(action, state_raw, metadata)
            reward = compute_reward(fusion_output, gt_cat)
            rewards.append(reward)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        # 1-step contextual bandit formulation: target value is observed reward
        advantages = rewards_t - values
        # Standardize advantages
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "states": states_t,
            "actions": actions_t,
            "old_log_probs": log_probs_t,
            "returns": rewards_t,
            "advantages": advantages,
            "mean_reward": float(np.mean(rewards)),
        }

    def update_policy(
        self,
        rollout: Dict[str, torch.Tensor],
        ppo_epochs: int = 4,
        minibatch_size: int = 128,
    ) -> Dict[str, float]:
        self.network.train()

        states = rollout["states"]
        actions = rollout["actions"]
        old_log_probs = rollout["old_log_probs"]
        returns = rollout["returns"]
        advantages = rollout["advantages"]

        n_samples = states.size(0)
        policy_losses = []
        value_losses = []
        entropies = []

        for _ in range(ppo_epochs):
            indices = np.random.permutation(n_samples)
            for start_idx in range(0, n_samples, minibatch_size):
                end_idx = min(start_idx + minibatch_size, n_samples)
                mb_idx = indices[start_idx:end_idx]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                dist, values = self.network(mb_states)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_returns)
                total_loss = policy_loss + self.c_val * value_loss - self.c_ent * entropy

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.item())

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
        }

    @torch.no_grad()
    def evaluate_val(self) -> Dict[str, Any]:
        self.network.eval()

        states_raw = np.array([r["state_vec"] for r in self.val_records], dtype=np.float32)
        states_norm = self.normalizer.transform(states_raw)
        states_t = torch.tensor(states_norm, dtype=torch.float32, device=self.device)

        dist = self.network.get_action_dist(states_t)
        # Deterministic inference uses actor mean
        actions = dist.mean.cpu().numpy()

        correct_binary = 0
        tp = tn = fp = fn = 0
        correct_cat = 0
        total = len(self.val_records)
        cat_correct = [0, 0, 0, 0]
        cat_total = [0, 0, 0, 0]

        for i, rec in enumerate(self.val_records):
            action = actions[i]
            state_raw = states_raw[i]
            metadata = rec["metadata"]
            gt_cat = rec["ground_truth_category"]
            gt_is_fake = 0 if gt_cat == 0 else 1

            fusion_output = apply_action_to_state(action, state_raw, metadata)
            pred_fake = 1 if fusion_output["verdict"] == "DEEPFAKE" else 0
            pred_cat_name = fusion_output["predicted_category"]
            pred_cat_idx = CATEGORY_TO_IDX.get(pred_cat_name, 0)

            if pred_fake == gt_is_fake:
                correct_binary += 1
            if gt_is_fake == 1 and pred_fake == 1:
                tp += 1
            elif gt_is_fake == 0 and pred_fake == 0:
                tn += 1
            elif gt_is_fake == 0 and pred_fake == 1:
                fp += 1
            elif gt_is_fake == 1 and pred_fake == 0:
                fn += 1

            cat_total[gt_cat] += 1
            if pred_cat_idx == gt_cat:
                correct_cat += 1
                cat_correct[gt_cat] += 1

        acc = correct_binary / max(total, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        spec = tn / max(tn + fp, 1)
        bal_acc = 0.5 * (rec + spec)
        cat_acc = correct_cat / max(total, 1)

        return {
            "val_accuracy": acc,
            "val_balanced_acc": bal_acc,
            "val_f1": f1,
            "val_precision": prec,
            "val_recall": rec,
            "val_specificity": spec,
            "val_category_acc": cat_acc,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "per_category_accuracy": {
                CATEGORIES[k]: cat_correct[k] / max(cat_total[k], 1)
                for k in range(4)
            },
        }


def eligible_records(records: Dict[str, Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    """Cache records of `split` that the versioned manifest still lists (the cache was built from v1)."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    eligible = {c["full_path"] for clist in manifest["clips"][split].values() for c in clist}
    return [r for r in records.values() if r["split"] == split and r["full_path"] in eligible]


def run_ppo_training(seed: int = 42, epochs: int = 20, out: Path = CHECKPOINT_OUT):
    logger.info("=" * 80)
    logger.info("VERITAS PPO v3 ADAPTIVE FUSION TRAINING")
    logger.info("=" * 80)

    assert FEATURES_CACHE_PATH.exists(), f"Feature cache missing: {FEATURES_CACHE_PATH}"
    with open(FEATURES_CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    records = cache["records"]
    train_records = eligible_records(records, "fusion_train")
    val_records = eligible_records(records, "fusion_val")

    logger.info("Loaded fusion_train records: %d", len(train_records))
    logger.info("Loaded fusion_val records:   %d", len(val_records))

    # Fit FeatureNormalizer strictly on fusion_train
    train_states = np.array([r["state_vec"] for r in train_records], dtype=np.float32)
    normalizer = FeatureNormalizer(STATE_DIM)
    normalizer.fit(train_states)
    logger.info("FeatureNormalizer fitted strictly on fusion_train (N=%d).", len(train_states))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device for PPO training: %s", device)

    trainer = PPOTrainerV3(
        train_records=train_records,
        val_records=val_records,
        normalizer=normalizer,
        lr=3e-4,
        clip_epsilon=0.20,
        c_val=0.50,
        c_ent=0.01,
        device=device,
        seed=seed,
        epochs=epochs,
    )

    # Select on balanced accuracy: val is ~77% fake, so F1 rewards flagging everything.
    best_val_score = -1.0
    best_metrics = {}

    start_time = time.time()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        rollout = trainer.collect_rollout(batch_size=512)
        up_metrics = trainer.update_policy(rollout, ppo_epochs=4, minibatch_size=128)
        val_metrics = trainer.evaluate_val()
        trainer.scheduler.step()

        elapsed = time.time() - t0
        logger.info(
            "Epoch %2d/%2d (%.1fs) | Rew: %+.3f | PolLoss: %+.4f | ValLoss: %.4f | Ent: %.3f | Val Acc: %.2f%% | Val F1: %.4f | CatAcc: %.2f%%",
            ep, epochs, elapsed,
            rollout["mean_reward"], up_metrics["policy_loss"], up_metrics["value_loss"], up_metrics["entropy"],
            val_metrics["val_accuracy"] * 100, val_metrics["val_f1"], val_metrics["val_category_acc"] * 100,
        )

        if val_metrics["val_balanced_acc"] > best_val_score:
            best_val_score = val_metrics["val_balanced_acc"]
            best_metrics = {
                "epoch": ep,
                "reward": rollout["mean_reward"],
                **val_metrics,
                **up_metrics,
            }
            # Save PPO v3 checkpoint
            checkpoint_payload = {
                "version": out.stem,
                "model_state_dict": trainer.network.state_dict(),
                "actor_state_dict": trainer.network.actor_net.state_dict(),
                "actor_log_std": trainer.network.actor_log_std.cpu().tolist(),
                "critic_state_dict": trainer.network.critic_net.state_dict(),
                "normalizer": normalizer.to_dict(),
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "categories": CATEGORIES,
                "metrics": best_metrics,
                "training_metadata": {
                    "dataset": "FakeAVCeleb v1.2",
                    "provenance": "fusion_train strictly disjoint (100 IDs), selected on fusion_val (50 IDs)",
                    "detector_checkpoints": {
                        "spatial": "spatial_agent_lc.pth",
                        "temporal": "temporal_agent_lc.pth",
                        "audio": "audio_agent_lc.pth",
                    },
                    "seed": seed,
                    "epochs": epochs,
                    "selection_metric": "val_balanced_acc",
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            }
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint_payload, out)
            logger.info(">>> New best PPO checkpoint saved to %s (Val BalAcc: %.4f)", out, best_val_score)

    total_time = time.time() - start_time
    logger.info("=" * 80)
    logger.info("PPO v3 Training complete in %.1f seconds.", total_time)
    logger.info("Best Validation Metrics: %s", best_metrics)
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", type=Path, default=CHECKPOINT_OUT)
    args = parser.parse_args()
    run_ppo_training(args.seed, args.epochs, args.out)
