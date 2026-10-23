"""
evaluate_ppo_v3_frozen.py
=========================
Frozen empirical benchmark on the 217 pure held-out test clips:
Compares:
  1. Legacy Fixed Fusion (naive arithmetic average)
  2. Semantically Corrected Fixed Fusion (independent modality probability formula)
  3. PPO v3 Adaptive Fusion Policy (trained with out-of-sample features)

IMPORTANT — Dual decision surfaces in PPO apply_action_to_state:
  - Binary metrics below use verdict: pred_fake = (verdict == "DEEPFAKE").
    UNCERTAIN maps to binary not-fake.
  - The 4-way confusion matrix uses predicted_category from the SAME forward pass.
  - In the UNCERTAIN deadband, predicted_category may be a manipulation class while
    verdict is non-DEEPFAKE, so category-implied binary metrics can differ from
    verdict-based binary metrics. Do not treat them as interchangeable.
  - For a full dual-surface reconciliation (record-level CSV + audit), run:
      reconcile_ppo_v3_evaluation.py
    which writes ppo_v3_*_recomputed / consistency_audit artifacts and does NOT
    overwrite this script's historical JSON/report.

Reports:
  - Exact counts: unique test videos, eligible, evaluated, failed, skipped.
  - Binary metrics (verdict-based) + category-based binary (explicit).
  - Per-GT binary accuracy AND 4-way categorical recall/precision/F1.
  - 4-way attribution confusion matrix.
  - Demographic fairness across 10 strata (5 races x 2 genders).
  - Paired McNemar's test (PPO v3 vs Corrected Fixed).
  - Statistical confidence intervals (Wilson score 95% CI).
  - Inference latency (mean ± std ms).
  - Separate diagnostic evaluation of 00014_fake.mp4 (fusion_train identity).
  - Documentation of the 49 historical 3-way test clips and their exclusion rationale.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_ppo_v3_frozen")

_HERE = Path(__file__).resolve().parent
_BACKEND_DIR = _HERE.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fusion_policy import (
    CATEGORIES,
    CATEGORY_TO_IDX,
    IDX_TO_CATEGORY,
    STATE_DIM,
    ACTION_DIM,
    ActorCriticNetwork,
    FeatureNormalizer,
    apply_action_to_state,
    compute_verdict_corrected_fixed,
    compute_verdict_legacy,
)

MANIFEST_PATH = _HERE / "leakage_controlled_split_manifest.json"  # pinned to v1: historical final_test results (see README contamination warning)
HISTORICAL_MANIFEST_PATH = _HERE / "pure_identity_disjoint_manifest.json"
FEATURES_CACHE_PATH = _HERE / "cache" / "fusion_features_lc_v3.json"
PPO_V3_CKPT_PATH = _BACKEND_DIR / "models" / "ppo_fusion_policy_v3.pth"
EVAL_RESULTS_PATH = _HERE / "ppo_v3_test_results.json"


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.95996
    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2 * total)) / denom
    half = (z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * total)) / total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


class PPOv3PolicyWrapper:
    def __init__(self, checkpoint_path: Path):
        assert checkpoint_path.exists(), f"Missing PPO v3 checkpoint: {checkpoint_path}"
        self.device = torch.device("cpu")
        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)

        self.network = ActorCriticNetwork(STATE_DIM, ACTION_DIM).to(self.device)
        # v3 checkpoint stores model_state_dict (full combined) as well as
        # actor_state_dict/critic_state_dict separately — use combined form.
        self.network.load_state_dict(ckpt["model_state_dict"])
        self.network.eval()

        norm_data = ckpt.get("normalizer") or ckpt.get("normalizer_state")
        self.normalizer = FeatureNormalizer.from_dict(norm_data)
        logger.info("Loaded PPO v3 Policy from %s", checkpoint_path)

    @torch.no_grad()
    def fuse(self, state_vec: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        s_norm = self.normalizer.transform(state_vec)
        s_t = torch.tensor(s_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.network.get_action_dist(s_t)
        action = dist.mean.squeeze(0).cpu().numpy()
        return apply_action_to_state(action, state_vec, metadata)


def run_frozen_benchmark():
    logger.info("=" * 80)
    logger.info("VERITAS PPO v3 FROZEN BENCHMARK EVALUATION")
    logger.info("=" * 80)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(FEATURES_CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)
    records = cache["records"]

    ppo_v3 = PPOv3PolicyWrapper(PPO_V3_CKPT_PATH)

    test_clips_raw = []
    for cat_name, clist in manifest["clips"]["final_test"].items():
        for item in clist:
            test_clips_raw.append(item)

    logger.info("Loaded %d manifest-listed final_test clips", len(test_clips_raw))

    results = []
    latencies = []

    for item in test_clips_raw:
        p = item["full_path"]
        if p not in records:
            logger.error("Missing feature record for %s!", p)
            continue
        rec = records[p]
        agents = rec["agents"]
        state_vec = np.array(rec["state_vec"], dtype=np.float32)
        metadata = rec["metadata"]
        gt_cat = rec["ground_truth_category"]
        gt_is_fake = 0 if gt_cat == 0 else 1

        # 1. Legacy Fixed Fusion
        conf_leg, verd_leg, _, _, det_leg = compute_verdict_legacy(agents)
        pred_leg_fake = 1 if verd_leg == "DEEPFAKE" else 0

        # 2. Corrected Fixed Fusion
        conf_corr, verd_corr, _, _, det_corr = compute_verdict_corrected_fixed(agents, media_type="video")
        pred_corr_fake = 1 if verd_corr == "DEEPFAKE" else 0

        # 3. PPO v3 Policy
        t_start = time.perf_counter()
        det_ppo = ppo_v3.fuse(state_vec, metadata)
        t_lat = (time.perf_counter() - t_start) * 1000.0  # ms
        latencies.append(t_lat)

        verd_ppo = det_ppo["verdict"]
        conf_ppo = det_ppo["overall_confidence"]
        # Surface A: binary from verdict (UNCERTAIN -> not-fake)
        pred_ppo_fake = 1 if verd_ppo == "DEEPFAKE" else 0
        ppo_cat = det_ppo["predicted_category"]
        ppo_cat_idx = CATEGORY_TO_IDX.get(ppo_cat, 0)
        # Surface B: binary implied by category (AUTHENTIC -> not-fake)
        pred_ppo_fake_from_cat = 0 if ppo_cat == "AUTHENTIC" else 1

        results.append({
            "path": p,
            "type": item["type"],
            "race": item.get("race", ""),
            "gender": item.get("gender", ""),
            "source": item.get("source", ""),
            "gt_cat": gt_cat,
            "gt_is_fake": gt_is_fake,
            # Legacy
            "leg_conf": conf_leg,
            "leg_verd": verd_leg,
            "leg_fake": pred_leg_fake,
            # Corrected Fixed
            "corr_conf": conf_corr,
            "corr_verd": verd_corr,
            "corr_fake": pred_corr_fake,
            # PPO v3
            "ppo_conf": conf_ppo,
            "ppo_verd": verd_ppo,
            "ppo_fake": pred_ppo_fake,
            "ppo_fake_from_category": pred_ppo_fake_from_cat,
            "ppo_binary_discrepant": bool(pred_ppo_fake != pred_ppo_fake_from_cat),
            "ppo_cat": ppo_cat,
            "ppo_cat_idx": ppo_cat_idx,
            "ppo_cat_probs": det_ppo["category_probabilities"],
            "weights": det_ppo["dynamic_weights"],
            "mismatch_score": det_ppo["mismatch_score"],
            "decision_threshold": det_ppo["decision_threshold"],
        })

    total_eval = len(results)
    y_true = np.array([r["gt_is_fake"] for r in results])
    y_leg = np.array([r["leg_fake"] for r in results])
    y_corr = np.array([r["corr_fake"] for r in results])
    y_ppo = np.array([r["ppo_fake"] for r in results])
    y_ppo_cat = np.array([r["ppo_fake_from_category"] for r in results])
    n_discrepant = int(np.sum([r["ppo_binary_discrepant"] for r in results]))
    if n_discrepant:
        logger.warning(
            "PPO dual-surface discrepancy on %d/%d samples "
            "(verdict-based binary != category-based binary). "
            "See reconcile_ppo_v3_evaluation.py for the audit.",
            n_discrepant,
            total_eval,
        )

    def calc_metrics(y_t, y_p):
        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        tn = int(np.sum((y_t == 0) & (y_p == 0)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        acc = (tp + tn) / max(len(y_t), 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        spec = tn / max(tn + fp, 1)
        bal_acc = 0.5 * (rec + spec)
        ci_low, ci_high = wilson_score_interval(tp + tn, len(y_t))
        return {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "specificity": spec, "balanced_accuracy": bal_acc,
            "ci_95": (ci_low, ci_high),
        }

    m_leg = calc_metrics(y_true, y_leg)
    m_corr = calc_metrics(y_true, y_corr)
    m_ppo = calc_metrics(y_true, y_ppo)
    m_ppo_from_cat = calc_metrics(y_true, y_ppo_cat)

    # McNemar's paired test between PPO v3 (verdict) and Corrected Fixed
    # b: Corrected correct, PPO incorrect
    # c: PPO correct, Corrected incorrect
    # NOTE: non-significance is NOT proof of equivalence.
    corr_correct = (y_corr == y_true)
    ppo_correct = (y_ppo == y_true)
    b = int(np.sum(corr_correct & (~ppo_correct)))
    c = int(np.sum(ppo_correct & (~corr_correct)))
    if (b + c) > 0:
        chi2 = float((abs(b - c) - 1.0)**2 / (b + c))
        # 1-df chi-square survival function
        p_val = float(stats.chi2.sf(chi2, df=1))
    else:
        chi2 = 0.0
        p_val = 1.0

    # 4x4 Confusion Matrix for PPO v3 (from predicted_category)
    cm4 = np.zeros((4, 4), dtype=int)
    for r in results:
        cm4[r["gt_cat"], r["ppo_cat_idx"]] += 1

    cat_attribution_acc = float(np.trace(cm4) / max(total_eval, 1))

    # Per-GT binary accuracy (NOT the same as 4-way categorical recall)
    # plus explicit categorical recall/precision from cm4 diagonal.
    cat_metrics = {}
    for cidx, cname in enumerate(CATEGORIES):
        sub = [r for r in results if r["gt_cat"] == cidx]
        if not sub:
            continue
        y_sub_t = np.array([r["gt_is_fake"] for r in sub])
        y_sub_leg = np.array([r["leg_fake"] for r in sub])
        y_sub_corr = np.array([r["corr_fake"] for r in sub])
        y_sub_ppo = np.array([r["ppo_fake"] for r in sub])
        y_sub_ppo_c = np.array([r["ppo_fake_from_category"] for r in sub])
        diag = int(cm4[cidx, cidx])
        support = int(cm4[cidx, :].sum())
        pred_col = int(cm4[:, cidx].sum())
        cat_prec = diag / max(pred_col, 1)
        cat_rec = diag / max(support, 1)
        cat_f1 = 2 * cat_prec * cat_rec / max(cat_prec + cat_rec, 1e-8)

        cat_metrics[cname] = {
            "count": len(sub),
            "legacy_binary_acc": float(np.mean(y_sub_t == y_sub_leg)),
            "corrected_binary_acc": float(np.mean(y_sub_t == y_sub_corr)),
            "ppo_verdict_binary_acc": float(np.mean(y_sub_t == y_sub_ppo)),
            "ppo_category_binary_acc": float(np.mean(y_sub_t == y_sub_ppo_c)),
            # Back-compat aliases (binary, NOT categorical recall)
            "legacy_acc": float(np.mean(y_sub_t == y_sub_leg)),
            "corrected_acc": float(np.mean(y_sub_t == y_sub_corr)),
            "ppo_acc": float(np.mean(y_sub_t == y_sub_ppo)),
            "ppo_categorical_precision": float(cat_prec),
            "ppo_categorical_recall": float(cat_rec),
            "ppo_categorical_f1": float(cat_f1),
            "ppo_categorical_diagonal": diag,
        }

    # Demographic fairness breakdown
    strata_metrics = {}
    for r in results:
        key = f"{r['race']} ({r['gender']})"
        if key not in strata_metrics:
            strata_metrics[key] = {"total": 0, "ppo_correct": 0, "corr_correct": 0}
        strata_metrics[key]["total"] += 1
        if r["ppo_fake"] == r["gt_is_fake"]:
            strata_metrics[key]["ppo_correct"] += 1
        if r["corr_fake"] == r["gt_is_fake"]:
            strata_metrics[key]["corr_correct"] += 1

    for k, v in strata_metrics.items():
        v["ppo_acc"] = v["ppo_correct"] / max(v["total"], 1)
        v["corr_acc"] = v["corr_correct"] / max(v["total"], 1)

    # Diagnostic evaluation: 00014_fake.mp4
    diagnostic_result = None
    diag_path = r"D:\VERITAS_TRAINING\data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2\RealVideo-FakeAudio\African\men\id00014\00014_fake.mp4"
    if os.path.exists(diag_path) and diag_path in records:
        diag_rec = records[diag_path]
        diag_ppo = ppo_v3.fuse(np.array(diag_rec["state_vec"], dtype=np.float32), diag_rec["metadata"])
        diagnostic_result = {
            "path": diag_path,
            "note": "DIAGNOSTIC ONLY — identity id00014 belongs to fusion_train identity pool; excluded from frozen test evidence.",
            "fusion_output": diag_ppo,
            "verdict": diag_ppo["verdict"],
            "confidence": diag_ppo["overall_confidence"],
            "predicted_category": diag_ppo["predicted_category"],
        }

    # Auditing the 49 historical test clips
    historical_49_summary = {
        "historical_3way_pure_test_total": 266,
        "pure_5way_retained_test_total": 217,
        "excluded_clips_count": 49,
        "exclusion_reason": "Cross-partition donor edge isolation: clips paired a test source identity with a donor identity allocated to detector_train, detector_val, or fusion_train. Retaining them would violate 0% donor leakage across the 5 partitions.",
    }

    eval_payload = {
        "evaluation_name": "PPO v3 Frozen Leakage-Controlled Evaluation",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_samples": total_eval,
        "decision_surface_note": (
            "ppo_v3 binary_summary is verdict-based (UNCERTAIN->not-fake). "
            "ppo_v3_category_based uses predicted_category != AUTHENTIC. "
            "confusion_matrix_4way uses predicted_category. "
            "per_category_metrics.ppo_acc is binary accuracy within GT class, "
            "not 4-way categorical recall (see ppo_categorical_recall)."
        ),
        "dual_surface_discrepant_count": n_discrepant,
        "binary_summary": {
            "legacy_fixed": m_leg,
            "corrected_fixed": m_corr,
            "ppo_v3": m_ppo,
            "ppo_v3_category_based": m_ppo_from_cat,
            "fixed_fusion_four_category": "unavailable",
        },
        "mcnemar_test": {
            "b_corrected_correct_ppo_wrong": b,
            "c_ppo_correct_corrected_wrong": c,
            "chi2": chi2,
            "p_value": p_val,
            "statistically_significant": bool(p_val < 0.05),
            "interpretation": (
                "Non-significance is not proof of equivalence between systems."
            ),
        },
        "per_category_metrics": cat_metrics,
        "categorical_attribution_accuracy": cat_attribution_acc,
        "confusion_matrix_4way": cm4.tolist(),
        "demographic_fairness": strata_metrics,
        "latency_ms": {
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "p95": float(np.percentile(latencies, 95)),
        },
        "diagnostic_example": diagnostic_result,
        "historical_49_audit": historical_49_summary,
    }

    parser = argparse.ArgumentParser(description="PPO v3 frozen evaluation")
    parser.add_argument(
        "--output",
        type=Path,
        default=EVAL_RESULTS_PATH,
        help=(
            "JSON output path. Default overwrites ppo_v3_test_results.json. "
            "For audit without clobbering the historical file, use "
            "reconcile_ppo_v3_evaluation.py instead."
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Compute metrics but do not write JSON (preserves existing artifacts).",
    )
    args, _unknown = parser.parse_known_args()

    if not args.no_write:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(eval_payload, f, indent=2)
        logger.info("Frozen evaluation results exported to %s", out_path)
    else:
        logger.info("Skipping JSON write (--no-write)")

    logger.info("Legacy Fixed Acc:    %.2f%%", m_leg["accuracy"] * 100)
    logger.info("Corrected Fixed Acc: %.2f%%", m_corr["accuracy"] * 100)
    logger.info("PPO v3 (verdict) Acc: %.2f%%", m_ppo["accuracy"] * 100)
    logger.info("PPO v3 (category) Acc: %.2f%%", m_ppo_from_cat["accuracy"] * 100)
    logger.info(
        "Dual-surface discrepant samples: %d", n_discrepant,
    )
    logger.info("McNemar chi2=%.4f (p=%.4f, sig=%s)", chi2, p_val, p_val < 0.05)


if __name__ == "__main__":
    run_frozen_benchmark()

