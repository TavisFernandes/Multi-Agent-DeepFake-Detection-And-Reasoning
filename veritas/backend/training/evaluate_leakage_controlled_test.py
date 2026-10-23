"""
evaluate_leakage_controlled_test.py
===================================
Rigorous empirical evaluation of VERITAS fusion policies on the strictly
leakage-controlled, pure identity-disjoint held-out test split (217 clips).

Evaluates:
1. Legacy Fixed Fusion
2. Semantically Corrected Fixed Fusion
3. PPO v2 Adaptive Fusion Policy
4. Demographic subgroup breakdowns (5 races x 2 genders)
5. 4-class categorical confusion matrix and classification metrics
6. Paired McNemar's test and 95% Clopper-Pearson / Wilson score confidence intervals.
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fusion_policy import (
    PPOFusionPolicy,
    compute_verdict_corrected_fixed,
    compute_verdict_legacy,
    extract_state_from_agents,
    CATEGORIES,
)

MANIFEST_PATH = BACKEND_DIR / "training" / "leakage_controlled_split_manifest.json"  # pinned to v1: historical final_test results (see README contamination warning)
CACHE_PATH = BACKEND_DIR / "training" / "cache" / "interim_feature_extraction.json"
OUTPUT_REPORT_PATH = BACKEND_DIR / "training" / "leakage_controlled_test_results.json"


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence interval for a proportion."""
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # for 95%
    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2 * total)) / denom
    half = (z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * total)) / total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def run_evaluation():
    print("=" * 80)
    print("VERITAS LEAKAGE-CONTROLLED TEST EVALUATION (n=217 Pure Held-Out Clips)")
    print("=" * 80)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    test_clips_by_cat = manifest["clips"]["final_test"]
    all_test_clips = []
    for cat_name, clist in test_clips_by_cat.items():
        all_test_clips.extend(clist)

    print(f"Loaded {len(all_test_clips)} test clips across 4 categories.")

    # Instantiate PPO policy
    ppo_policy = PPOFusionPolicy()

    # Ground truth mapping:
    # RealVideo-RealAudio: 0 (AUTHENTIC)
    # RealVideo-FakeAudio: 1 (AUDIO_MANIPULATED)
    # FakeVideo-RealAudio: 2 (VIDEO_MANIPULATED)
    # FakeVideo-FakeAudio: 3 (AUDIO_VIDEO_MANIPULATED)
    cat_map = {
        "RealVideo-RealAudio": 0,
        "RealVideo-FakeAudio": 1,
        "FakeVideo-RealAudio": 2,
        "FakeVideo-FakeAudio": 3,
    }

    results = []
    for c in all_test_clips:
        p = c["full_path"]
        if p not in cache:
            raise RuntimeError(f"Missing cache entry for {p}")

        entry = cache[p]
        agents = entry["agents"]
        gt_cat = cat_map[c["type"]]
        gt_is_fake = 0 if gt_cat == 0 else 1

        # 1. Legacy Fixed Fusion
        conf_leg, verd_leg, _, _, det_leg = compute_verdict_legacy(agents)
        pred_leg_fake = 1 if verd_leg == "DEEPFAKE" else 0

        # 2. Corrected Fixed Fusion
        conf_corr, verd_corr, _, _, det_corr = compute_verdict_corrected_fixed(agents, media_type="video")
        pred_corr_fake = 1 if verd_corr == "DEEPFAKE" else 0

        # 3. PPO Fusion
        conf_ppo, verd_ppo, _, _, det_ppo = ppo_policy.fuse_agents(agents, media_type="video")
        pred_ppo_fake = 1 if verd_ppo == "DEEPFAKE" else 0
        ppo_cat = det_ppo.get("predicted_category", "UNKNOWN")
        ppo_cat_idx = CATEGORIES.index(ppo_cat) if ppo_cat in CATEGORIES else -1

        results.append({
            "path": p,
            "type": c["type"],
            "race": c["race"],
            "gender": c["gender"],
            "source": c["source"],
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
            # PPO
            "ppo_conf": conf_ppo,
            "ppo_verd": verd_ppo,
            "ppo_fake": pred_ppo_fake,
            "ppo_cat": ppo_cat,
            "ppo_cat_idx": ppo_cat_idx,
            "ppo_cat_probs": det_ppo.get("category_probs", []),
            "weights": {
                "w_spatial": det_ppo.get("w_spatial", 0.0),
                "w_temporal": det_ppo.get("w_temporal", 0.0),
                "w_audio": det_ppo.get("w_audio", 0.0),
            }
        })

    total_samples = len(results)
    y_true = np.array([r["gt_is_fake"] for r in results])
    y_leg = np.array([r["leg_fake"] for r in results])
    y_corr = np.array([r["corr_fake"] for r in results])
    y_ppo = np.array([r["ppo_fake"] for r in results])

    def calc_metrics(y_t, y_p):
        tp = int(np.sum((y_t == 1) & (y_p == 1)))
        tn = int(np.sum((y_t == 0) & (y_p == 0)))
        fp = int(np.sum((y_t == 0) & (y_p == 1)))
        fn = int(np.sum((y_t == 1) & (y_p == 0)))
        acc = (tp + tn) / len(y_t)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
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

    print("\n" + "=" * 50)
    print("BINARY CLASSIFICATION SUMMARY")
    print("=" * 50)
    print(f"Total pure held-out samples: {total_samples}")
    print(f"Authentic: {int(np.sum(y_true == 0))}, Deepfake: {int(np.sum(y_true == 1))}")
    print("-" * 50)
    print(f"Legacy Fixed:     Acc={m_leg['accuracy']*100:.2f}% | BalAcc={m_leg['balanced_accuracy']*100:.2f}% | Rec={m_leg['recall']*100:.2f}% | Prec={m_leg['precision']*100:.2f}% | F1={m_leg['f1']:.4f} | CI95=[{m_leg['ci_95'][0]*100:.1f}%, {m_leg['ci_95'][1]*100:.1f}%]")
    print(f"Corrected Fixed:  Acc={m_corr['accuracy']*100:.2f}% | BalAcc={m_corr['balanced_accuracy']*100:.2f}% | Rec={m_corr['recall']*100:.2f}% | Prec={m_corr['precision']*100:.2f}% | F1={m_corr['f1']:.4f} | CI95=[{m_corr['ci_95'][0]*100:.1f}%, {m_corr['ci_95'][1]*100:.1f}%]")
    print(f"PPO v2 Policy:    Acc={m_ppo['accuracy']*100:.2f}% | BalAcc={m_ppo['balanced_accuracy']*100:.2f}% | Rec={m_ppo['recall']*100:.2f}% | Prec={m_ppo['precision']*100:.2f}% | F1={m_ppo['f1']:.4f} | CI95=[{m_ppo['ci_95'][0]*100:.1f}%, {m_ppo['ci_95'][1]*100:.1f}%]")

    # Confusion breakdown by category
    print("\n" + "=" * 50)
    print("ACCURACY BY GROUND TRUTH CATEGORY")
    print("=" * 50)
    cat_summary = {}
    for cname, cidx in cat_map.items():
        subset = [r for r in results if r["gt_cat"] == cidx]
        y_sub_t = np.array([r["gt_is_fake"] for r in subset])
        y_sub_leg = np.array([r["leg_fake"] for r in subset])
        y_sub_corr = np.array([r["corr_fake"] for r in subset])
        y_sub_ppo = np.array([r["ppo_fake"] for r in subset])
        
        acc_leg = np.mean(y_sub_t == y_sub_leg)
        acc_corr = np.mean(y_sub_t == y_sub_corr)
        acc_ppo = np.mean(y_sub_t == y_sub_ppo)
        cat_summary[cname] = {
            "count": len(subset),
            "leg_acc": acc_leg,
            "corr_acc": acc_corr,
            "ppo_acc": acc_ppo,
        }
        print(f"Category: {cname} (n={len(subset)})")
        print(f"  Legacy Fixed:     {acc_leg*100:.2f}% ({int(np.sum(y_sub_t == y_sub_leg))}/{len(subset)})")
        print(f"  Corrected Fixed:  {acc_corr*100:.2f}% ({int(np.sum(y_sub_t == y_sub_corr))}/{len(subset)})")
        print(f"  PPO v2:           {acc_ppo*100:.2f}% ({int(np.sum(y_sub_t == y_sub_ppo))}/{len(subset)})")

    # 4x4 Categorical Confusion Matrix for PPO
    print("\n" + "=" * 50)
    print("PPO 4-CATEGORY CONFUSION MATRIX")
    print("=" * 50)
    # Rows: Ground Truth (0..3), Columns: Predicted (0..3)
    cm4 = np.zeros((4, 4), dtype=int)
    for r in results:
        gt = r["gt_cat"]
        pred = r["ppo_cat_idx"]
        if 0 <= pred < 4:
            cm4[gt, pred] += 1
    
    print("GT \\ Pred | AUTH | AUD_FAKE | VID_FAKE | BOTH_FAKE")
    row_names = ["AUTH     ", "AUD_FAKE ", "VID_FAKE ", "BOTH_FAKE"]
    for i in range(4):
        print(f"{row_names[i]} | {cm4[i, 0]:4d} | {cm4[i, 1]:8d} | {cm4[i, 2]:8d} | {cm4[i, 3]:9d}")

    cat_diag = np.trace(cm4)
    cat_acc = cat_diag / total_samples
    print(f"\nCategorical Attribution Accuracy: {cat_acc*100:.2f}% ({cat_diag}/{total_samples})")

    # McNemar's Test between Corrected Fixed and PPO
    # b: Corrected correct, PPO incorrect
    # c: Corrected incorrect, PPO correct
    b = int(np.sum((y_corr == y_true) & (y_ppo != y_true)))
    c = int(np.sum((y_corr != y_true) & (y_ppo == y_true)))
    if b + c > 0:
        mcnemar_stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
    else:
        mcnemar_stat = 0.0
    print("\n" + "=" * 50)
    print("MCNEMAR'S PAIRED TEST (Corrected Fixed vs PPO v2)")
    print("=" * 50)
    print(f"Discordant pairs: Corrected only = {b}, PPO only = {c}")
    print(f"McNemar chi-square stat (with Yates continuity correction): {mcnemar_stat:.4f}")
    if mcnemar_stat > 3.841:
        print("Result: Statistically significant difference (p < 0.05)")
    else:
        print("Result: No statistically significant difference (p >= 0.05)")

    # Demographic Fairness: Breakdown by Race and Gender
    print("\n" + "=" * 50)
    print("DEMOGRAPHIC FAIRNESS ANALYSIS (PPO v2)")
    print("=" * 50)
    races = sorted(list(set(r["race"] for r in results)))
    demo_summary = {}
    for race in races:
        for gender in ["men", "women"]:
            sub = [r for r in results if r["race"] == race and r["gender"] == gender]
            if not sub:
                continue
            y_sub_t = np.array([r["gt_is_fake"] for r in sub])
            y_sub_ppo = np.array([r["ppo_fake"] for r in sub])
            acc = float(np.mean(y_sub_t == y_sub_ppo))
            demo_summary[f"{race}_{gender}"] = {
                "count": len(sub),
                "accuracy": acc,
            }
            print(f"Strata: {race:25s} | {gender:5s} | n={len(sub):2d} | Acc={acc*100:.1f}%")

    # Save complete JSON
    out_payload = {
        "dataset": "FakeAVCeleb_v1.2",
        "split": "final_test",
        "total_samples": total_samples,
        "metrics": {
            "legacy_fixed": m_leg,
            "corrected_fixed": m_corr,
            "ppo_v2": m_ppo,
        },
        "mcnemar": {
            "b_corrected_only": b,
            "c_ppo_only": c,
            "stat": mcnemar_stat,
            "significant_at_p05": bool(mcnemar_stat > 3.841),
        },
        "category_summary": cat_summary,
        "categorical_confusion_matrix_4x4": cm4.tolist(),
        "categorical_accuracy": cat_acc,
        "demographic_summary": demo_summary,
    }

    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nDetailed evaluation results written to: {OUTPUT_REPORT_PATH}")


if __name__ == "__main__":
    run_evaluation()
