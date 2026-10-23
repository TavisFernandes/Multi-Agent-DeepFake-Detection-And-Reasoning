"""
reconcile_ppo_v3_evaluation.py
==============================
Deterministic recompute of PPO v3 frozen-test metrics with explicit handling of
the dual decision surfaces in apply_action_to_state:

  Surface A — Binary verdict:
      DEEPFAKE  iff overall_confidence >= decision_threshold + 0.05
      AUTHENTIC iff overall_confidence <= decision_threshold - 0.05
      else UNCERTAIN (evaluator maps non-DEEPFAKE -> binary authentic / not-fake)

  Surface B — Four-category attribution:
      predicted_category from the same forward pass; in the UNCERTAIN band,
      category may be a manipulation class even when verdict != DEEPFAKE.

This script does NOT overwrite:
  - ppo_v3_lc_evaluation_report.md
  - ppo_v3_test_results.json
  - ppo_fusion_policy_v3.pth

Outputs:
  - ppo_v3_record_level_predictions.csv
  - ppo_v3_test_results_recomputed.json
  - ppo_v3_consistency_audit.md
"""

from __future__ import annotations

import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy import stats

_HERE = Path(__file__).resolve().parent
_BACKEND_DIR = _HERE.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fusion_policy import (  # noqa: E402
    CATEGORIES,
    CATEGORY_TO_IDX,
    STATE_DIM,
    ACTION_DIM,
    ActorCriticNetwork,
    FeatureNormalizer,
    apply_action_to_state,
    compute_verdict_corrected_fixed,
    compute_verdict_legacy,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reconcile_ppo_v3")

MANIFEST_PATH = _HERE / "leakage_controlled_split_manifest.json"  # pinned to v1: historical final_test results (see README contamination warning)
FEATURES_CACHE_PATH = _HERE / "cache" / "fusion_features_lc_v3.json"
PPO_V3_CKPT_PATH = _BACKEND_DIR / "models" / "ppo_fusion_policy_v3.pth"
OLD_RESULTS_PATH = _HERE / "ppo_v3_test_results.json"

OUT_CSV = _HERE / "ppo_v3_record_level_predictions.csv"
OUT_JSON = _HERE / "ppo_v3_test_results_recomputed.json"
OUT_AUDIT = _HERE / "ppo_v3_consistency_audit.md"

# Short display aliases used in the original report headers (cosmetic only)
CAT_DISPLAY = {
    "AUTHENTIC": "AUTHENTIC",
    "AUDIO_MANIPULATED": "AUDIO_FAKE",
    "VIDEO_MANIPULATED": "VIDEO_FAKE",
    "AUDIO_VIDEO_MANIPULATED": "BOTH_FAKE",
}


def wilson_score_interval(successes: int, total: int) -> Tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.95996
    p_hat = successes / total
    denom = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2 * total)) / denom
    half = (z * math.sqrt((p_hat * (1 - p_hat) + (z**2) / (4 * total)) / total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def calc_binary_metrics(y_t: np.ndarray, y_p: np.ndarray) -> Dict[str, Any]:
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    n = len(y_t)
    acc = (tp + tn) / max(n, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    spec = tn / max(tn + fp, 1)
    bal_acc = 0.5 * (rec + spec)
    ci_low, ci_high = wilson_score_interval(tp + tn, n)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": n,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": spec,
        "ci_95": [ci_low, ci_high],
    }


def mcnemar(y_true: np.ndarray, y_a: np.ndarray, y_b: np.ndarray) -> Dict[str, Any]:
    """Compare system A vs B; b = A correct & B wrong, c = B correct & A wrong."""
    a_ok = y_a == y_true
    b_ok = y_b == y_true
    b = int(np.sum(a_ok & (~b_ok)))
    c = int(np.sum(b_ok & (~a_ok)))
    if (b + c) > 0:
        chi2 = float((abs(b - c) - 1.0) ** 2 / (b + c))
        p_val = float(stats.chi2.sf(chi2, df=1))
    else:
        chi2 = 0.0
        p_val = 1.0
    return {
        "b": b,
        "c": c,
        "chi2": chi2,
        "p_value": p_val,
        "significant_at_0_05": bool(p_val < 0.05),
        "note": (
            "Non-significance is NOT proof of equivalence; "
            "it only fails to reject equal discordant error rates."
        ),
    }


def per_class_prf(cm: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(CATEGORIES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        support = int(cm[i, :].sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        out[name] = {
            "support": support,
            "tp_diagonal": tp,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
    return out


def verdict_zone(p_fake: float, threshold: float) -> str:
    if p_fake >= threshold + 0.05:
        return "HIGH_FAKE"
    if p_fake <= threshold - 0.05:
        return "HIGH_AUTH"
    return "UNCERTAIN"


class PPOv3PolicyWrapper:
    def __init__(self, checkpoint_path: Path):
        assert checkpoint_path.exists(), f"Missing PPO v3 checkpoint: {checkpoint_path}"
        self.device = torch.device("cpu")
        ckpt = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        self.network = ActorCriticNetwork(STATE_DIM, ACTION_DIM).to(self.device)
        self.network.load_state_dict(ckpt["model_state_dict"])
        self.network.eval()
        norm_data = ckpt.get("normalizer") or ckpt.get("normalizer_state")
        self.normalizer = FeatureNormalizer.from_dict(norm_data)
        self.checkpoint_keys = sorted(list(ckpt.keys()))
        self.best_epoch = ckpt.get("best_epoch") or ckpt.get("epoch")
        self.val_metrics = ckpt.get("val_metrics") or ckpt.get("metrics")
        logger.info("Loaded PPO v3 from %s (keys=%s)", checkpoint_path, self.checkpoint_keys)

    @torch.no_grad()
    def fuse(self, state_vec: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        s_norm = self.normalizer.transform(state_vec)
        s_t = torch.tensor(s_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.network.get_action_dist(s_t)
        action = dist.mean.squeeze(0).cpu().numpy()
        return apply_action_to_state(action, state_vec, metadata)


def run_reconcile() -> Dict[str, Any]:
    logger.info("=" * 80)
    logger.info("PPO v3 METRICS RECONCILIATION (dual-surface audit)")
    logger.info("=" * 80)

    assert PPO_V3_CKPT_PATH.exists(), f"Missing checkpoint {PPO_V3_CKPT_PATH}"
    assert MANIFEST_PATH.exists(), f"Missing manifest {MANIFEST_PATH}"
    assert FEATURES_CACHE_PATH.exists(), f"Missing cache {FEATURES_CACHE_PATH}"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(FEATURES_CACHE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)
    records = cache["records"]

    ppo = PPOv3PolicyWrapper(PPO_V3_CKPT_PATH)

    test_clips_raw: List[Dict[str, Any]] = []
    for _cat_name, clist in manifest["clips"]["final_test"].items():
        for item in clist:
            test_clips_raw.append(item)

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []

    for sample_id, item in enumerate(test_clips_raw):
        p = item["full_path"]
        if p not in records:
            missing.append(p)
            logger.error("Missing feature record for %s", p)
            continue

        rec = records[p]
        agents = rec["agents"]
        state_vec = np.array(rec["state_vec"], dtype=np.float32)
        metadata = rec["metadata"]
        gt_cat = int(rec["ground_truth_category"])
        gt_is_fake = 0 if gt_cat == 0 else 1
        gt_cat_name = CATEGORIES[gt_cat]

        conf_leg, verd_leg, _, _, _ = compute_verdict_legacy(agents)
        pred_leg_fake = 1 if verd_leg == "DEEPFAKE" else 0

        conf_corr, verd_corr, _, _, _ = compute_verdict_corrected_fixed(
            agents, media_type="video"
        )
        pred_corr_fake = 1 if verd_corr == "DEEPFAKE" else 0

        det = ppo.fuse(state_vec, metadata)
        verd = det["verdict"]
        conf = float(det["overall_confidence"])
        p_fake = float(det.get("p_fake", conf))
        thr = float(det["decision_threshold"])
        pred_cat = det["predicted_category"]
        pred_cat_idx = CATEGORY_TO_IDX.get(pred_cat, -1)
        zone = verdict_zone(p_fake, thr)

        fake_from_verdict = 1 if verd == "DEEPFAKE" else 0
        fake_from_category = 0 if pred_cat == "AUTHENTIC" else 1
        discrepant = fake_from_verdict != fake_from_category

        cat_probs = det["category_probabilities"]
        weights = det["dynamic_weights"]

        # Stable relative path for CSV (dataset-relative)
        rel = item.get("rel_path") or item.get("path")
        if not rel:
            # Derive from FakeAVCeleb layout markers
            markers = (
                "RealVideo-RealAudio",
                "RealVideo-FakeAudio",
                "FakeVideo-RealAudio",
                "FakeVideo-FakeAudio",
            )
            rel = p
            for m in markers:
                if m in p:
                    rel = p[p.index(m) :]
                    break

        rows.append(
            {
                "sample_id": sample_id,
                "full_path": p,
                "rel_path": rel,
                "type": item.get("type", ""),
                "race": item.get("race", ""),
                "gender": item.get("gender", ""),
                "source": item.get("source", ""),
                "gt_cat_idx": gt_cat,
                "gt_cat_name": gt_cat_name,
                "gt_is_fake": gt_is_fake,
                "ppo_verdict": verd,
                "ppo_predicted_category": pred_cat,
                "ppo_pred_cat_idx": pred_cat_idx,
                "ppo_p_fake": p_fake,
                "ppo_overall_confidence": conf,
                "ppo_threshold": thr,
                "ppo_verdict_zone": zone,
                "ppo_fake_from_verdict": fake_from_verdict,
                "ppo_fake_from_category": fake_from_category,
                "ppo_binary_discrepant": discrepant,
                "ppo_p_authentic": float(cat_probs.get("AUTHENTIC", 0.0)),
                "ppo_p_audio_manip": float(cat_probs.get("AUDIO_MANIPULATED", 0.0)),
                "ppo_p_video_manip": float(cat_probs.get("VIDEO_MANIPULATED", 0.0)),
                "ppo_p_both": float(cat_probs.get("AUDIO_VIDEO_MANIPULATED", 0.0)),
                "ppo_w_spatial": float(weights.get("spatial", 0.0)),
                "ppo_w_temporal": float(weights.get("temporal", 0.0)),
                "ppo_w_audio": float(weights.get("audio", 0.0)),
                "ppo_mismatch_score": float(det.get("mismatch_score", 0.0)),
                "corr_verdict": verd_corr,
                "corr_fake": pred_corr_fake,
                "corr_confidence": float(conf_corr),
                "leg_verdict": verd_leg,
                "leg_fake": pred_leg_fake,
                "leg_confidence": float(conf_leg),
            }
        )

    n = len(rows)
    if missing:
        logger.warning("Missing %d feature records", len(missing))
    else:
        assert n == 217, f"Expected 217 evaluated samples, got {n}"

    y_true = np.array([r["gt_is_fake"] for r in rows], dtype=int)
    y_leg = np.array([r["leg_fake"] for r in rows], dtype=int)
    y_corr = np.array([r["corr_fake"] for r in rows], dtype=int)
    y_ppo_v = np.array([r["ppo_fake_from_verdict"] for r in rows], dtype=int)
    y_ppo_c = np.array([r["ppo_fake_from_category"] for r in rows], dtype=int)

    m_leg = calc_binary_metrics(y_true, y_leg)
    m_corr = calc_binary_metrics(y_true, y_corr)
    m_ppo_v = calc_binary_metrics(y_true, y_ppo_v)
    m_ppo_c = calc_binary_metrics(y_true, y_ppo_c)

    # 4-way CM from predicted_category (same records)
    cm4 = np.zeros((4, 4), dtype=int)
    for r in rows:
        assert r["ppo_pred_cat_idx"] >= 0, f"Unknown category {r['ppo_predicted_category']}"
        cm4[r["gt_cat_idx"], r["ppo_pred_cat_idx"]] += 1

    cat_prf = per_class_prf(cm4)
    cat_acc = float(np.trace(cm4) / max(n, 1))
    macro_f1 = float(np.mean([cat_prf[c]["f1"] for c in CATEGORIES]))

    # Category-implied binary CM must match y_ppo_c
    implied_tn = int(cm4[0, 0])
    implied_fp = int(cm4[0, 1:].sum())
    implied_fn = int(cm4[1:, 0].sum())
    implied_tp = int(cm4[1:, 1:].sum())
    assert (
        implied_tp == m_ppo_c["tp"]
        and implied_tn == m_ppo_c["tn"]
        and implied_fp == m_ppo_c["fp"]
        and implied_fn == m_ppo_c["fn"]
    ), "Category-implied binary CM disagrees with category-based binary metrics"
    assert int(cm4.sum()) == n, "4-way CM does not sum to n"
    assert m_ppo_v["n"] == n and m_ppo_c["n"] == n
    assert m_ppo_v["tp"] + m_ppo_v["tn"] + m_ppo_v["fp"] + m_ppo_v["fn"] == n
    assert m_ppo_c["tp"] + m_ppo_c["tn"] + m_ppo_c["fp"] + m_ppo_c["fn"] == n

    for i, name in enumerate(CATEGORIES):
        assert cat_prf[name]["tp_diagonal"] == int(cm4[i, i])
        assert cat_prf[name]["recall"] == (
            cm4[i, i] / max(cm4[i, :].sum(), 1)
        ) or cat_prf[name]["support"] == 0

    discrepant_rows = [r for r in rows if r["ppo_binary_discrepant"]]
    for r in discrepant_rows:
        assert r["ppo_verdict"] == "UNCERTAIN / INSUFFICIENT EVIDENCE"
        assert r["ppo_predicted_category"] != "AUTHENTIC"
        assert r["ppo_verdict_zone"] == "UNCERTAIN"

    # Per-GT-category binary accuracy (what the old report called "per-category recall")
    per_gt: Dict[str, Any] = {}
    for cidx, cname in enumerate(CATEGORIES):
        sub = [r for r in rows if r["gt_cat_idx"] == cidx]
        if not sub:
            continue
        yt = np.array([r["gt_is_fake"] for r in sub])
        per_gt[cname] = {
            "count": len(sub),
            "legacy_binary_acc": float(np.mean(yt == np.array([r["leg_fake"] for r in sub]))),
            "corrected_binary_acc": float(
                np.mean(yt == np.array([r["corr_fake"] for r in sub]))
            ),
            "ppo_verdict_binary_acc": float(
                np.mean(yt == np.array([r["ppo_fake_from_verdict"] for r in sub]))
            ),
            "ppo_category_binary_acc": float(
                np.mean(yt == np.array([r["ppo_fake_from_category"] for r in sub]))
            ),
            "ppo_category_recall": cat_prf[cname]["recall"],
            "ppo_category_precision": cat_prf[cname]["precision"],
            "ppo_category_f1": cat_prf[cname]["f1"],
            "ppo_cat_correct": cat_prf[cname]["tp_diagonal"],
        }

    # Demographics (verdict-based and category-based)
    strata: Dict[str, Any] = {}
    for r in rows:
        key = f"{r['race']} ({r['gender']})"
        if key not in strata:
            strata[key] = {
                "total": 0,
                "ppo_verdict_correct": 0,
                "ppo_category_correct": 0,
                "corr_correct": 0,
            }
        strata[key]["total"] += 1
        if r["ppo_fake_from_verdict"] == r["gt_is_fake"]:
            strata[key]["ppo_verdict_correct"] += 1
        if r["ppo_fake_from_category"] == r["gt_is_fake"]:
            strata[key]["ppo_category_correct"] += 1
        if r["corr_fake"] == r["gt_is_fake"]:
            strata[key]["corr_correct"] += 1
    for v in strata.values():
        t = max(v["total"], 1)
        v["ppo_verdict_acc"] = v["ppo_verdict_correct"] / t
        v["ppo_category_acc"] = v["ppo_category_correct"] / t
        v["corr_acc"] = v["corr_correct"] / t

    mcn_v = mcnemar(y_true, y_corr, y_ppo_v)
    mcn_c = mcnemar(y_true, y_corr, y_ppo_c)

    old_payload = None
    if OLD_RESULTS_PATH.exists():
        with open(OLD_RESULTS_PATH, "r", encoding="utf-8") as f:
            old_payload = json.load(f)

    old_ppo = (old_payload or {}).get("binary_summary", {}).get("ppo_v3", {})
    verdict_matches_old = (
        old_ppo.get("tp") == m_ppo_v["tp"]
        and old_ppo.get("tn") == m_ppo_v["tn"]
        and old_ppo.get("fp") == m_ppo_v["fp"]
        and old_ppo.get("fn") == m_ppo_v["fn"]
    )

    # Promotion assessment (no threshold tuning; no PPO as default)
    promotion = {
        "activate_ppo_v3_as_default": False,
        "keep_corrected_fixed_as_default": True,
        "rationale": [
            (
                f"Corrected fixed binary accuracy {m_corr['accuracy']*100:.2f}% "
                f"> PPO verdict-based {m_ppo_v['accuracy']*100:.2f}% "
                f"> PPO category-based {m_ppo_c['accuracy']*100:.2f}%."
            ),
            (
                f"False positives: corrected fixed FP={m_corr['fp']}, "
                f"PPO verdict FP={m_ppo_v['fp']}, PPO category FP={m_ppo_c['fp']} "
                f"(on n={n} authentic-capable binary labels; authentic n="
                f"{per_gt['AUTHENTIC']['count']})."
            ),
            (
                f"Recall is tied at {m_ppo_v['recall']*100:.2f}% (verdict) / "
                f"{m_ppo_c['recall']*100:.2f}% (category) vs corrected "
                f"{m_corr['recall']*100:.2f}% — high recall alone does not justify promotion."
            ),
            (
                f"McNemar vs corrected fixed: verdict-based p={mcn_v['p_value']:.4f} "
                f"(not significant); category-based p={mcn_c['p_value']:.4f} "
                f"({'significant' if mcn_c['significant_at_0_05'] else 'not significant'}). "
                "Non-significance is not equivalence."
            ),
            (
                f"4-way categorical accuracy {cat_acc*100:.2f}% with AUTHENTIC "
                f"categorical recall {cat_prf['AUTHENTIC']['recall']*100:.2f}% "
                f"({cat_prf['AUTHENTIC']['tp_diagonal']}/{per_gt['AUTHENTIC']['count']}) "
                "shows material authentic over-attribution to AUDIO_MANIPULATED."
            ),
            (
                f"{len(discrepant_rows)} sample(s) violate "
                "binary_fake==(predicted_category!='AUTHENTIC') because the UNCERTAIN "
                "band allows manip categories under a non-DEEPFAKE verdict."
            ),
        ],
        "status": "METRICS RECONCILED — PPO NOT READY FOR PROMOTION",
    }

    payload: Dict[str, Any] = {
        "evaluation_name": "PPO v3 Recomputed Consistency Audit",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(PPO_V3_CKPT_PATH.resolve()),
        "checkpoint_keys": ppo.checkpoint_keys,
        "cache": str(FEATURES_CACHE_PATH.resolve()),
        "manifest": str(MANIFEST_PATH.resolve()),
        "total_test_samples": n,
        "missing_from_cache": missing,
        "design_note": {
            "dual_decision_surfaces": True,
            "invariant_binary_eq_category_ne_authentic": (
                "Holds for HIGH_FAKE and HIGH_AUTH zones; intentionally may fail "
                "in the UNCERTAIN deadband where predicted_category uses "
                "p_cat[0]>=0.5 while verdict remains non-DEEPFAKE."
            ),
            "binary_from_verdict_rule": "pred_fake = 1 iff verdict == 'DEEPFAKE'",
            "binary_from_category_rule": "pred_fake = 1 iff predicted_category != 'AUTHENTIC'",
            "justification": (
                "apply_action_to_state documents a three-zone threshold with an "
                "UNCERTAIN band for operational abstention; category attribution "
                "still emits a hard 4-way label for diagnostics/reward. Metrics "
                "must never silently mix the two surfaces."
            ),
        },
        "discrepant_count": len(discrepant_rows),
        "discrepant_samples": [
            {
                "sample_id": r["sample_id"],
                "rel_path": r["rel_path"],
                "gt_cat": r["gt_cat_name"],
                "verdict": r["ppo_verdict"],
                "pred_cat": r["ppo_predicted_category"],
                "p_fake": r["ppo_p_fake"],
                "threshold": r["ppo_threshold"],
                "verdict_zone": r["ppo_verdict_zone"],
            }
            for r in discrepant_rows
        ],
        "invariant_satisfied_verdict_vs_category_all_samples": len(discrepant_rows) == 0,
        "binary_summary": {
            "legacy_fixed": m_leg,
            "corrected_fixed": m_corr,
            "ppo_v3_verdict_based": m_ppo_v,
            "ppo_v3_category_based": m_ppo_c,
            "fixed_fusion_four_category": "unavailable",
        },
        "confusion_matrix_4way": {
            "labels": CATEGORIES,
            "display_aliases": [CAT_DISPLAY[c] for c in CATEGORIES],
            "matrix": cm4.tolist(),
            "row_is_ground_truth": True,
            "col_is_prediction": True,
            "sum": int(cm4.sum()),
        },
        "categorical_attribution_accuracy": cat_acc,
        "categorical_macro_f1": macro_f1,
        "per_category_classification_metrics": cat_prf,
        "per_gt_category_binary_and_attribution": per_gt,
        "mcnemar_tests": {
            "corrected_fixed_vs_ppo_verdict": mcn_v,
            "corrected_fixed_vs_ppo_category": mcn_c,
        },
        "demographic_fairness": strata,
        "assertions": {
            "category_binary_matches_4way_margins": True,
            "per_category_diagonal_matches_4way": True,
            "all_confusion_matrices_sum_to_n": True,
            "verdict_based_matches_original_json": verdict_matches_old,
        },
        "old_report_comparison": {
            "old_binary_ppo": old_ppo,
            "old_cm4": (old_payload or {}).get("confusion_matrix_4way"),
            "old_per_category_called_acc_was_binary": True,
            "old_authentic_84pct_equals_verdict_binary_acc": (
                abs(float((old_payload or {}).get("per_category_metrics", {})
                    .get("AUTHENTIC", {}).get("ppo_acc", -1)) - 0.84) < 1e-9
            ),
            "old_authentic_categorical_recall_should_be": (
                f"{cat_prf['AUTHENTIC']['tp_diagonal']}/{per_gt['AUTHENTIC']['count']} "
                f"= {cat_prf['AUTHENTIC']['recall']*100:.2f}%"
            ),
            "root_cause": (
                "Original evaluator derived binary metrics from verdict "
                "(UNCERTAIN counted as not-fake) while the 4-way matrix used "
                "predicted_category. Two AUTHENTIC clips were UNCERTAIN + "
                "AUDIO_MANIPULATED, so category-implied FP=10/TN=40 while "
                "verdict-based FP=8/TN=42. Per-category '84%' was binary "
                "accuracy on AUTHENTIC (42/50), not categorical recall (40/50)."
            ),
        },
        "promotion_assessment": promotion,
        "status": promotion["status"],
    }

    # --- Write CSV ---
    csv_fields = [
        "sample_id",
        "rel_path",
        "type",
        "race",
        "gender",
        "gt_cat_idx",
        "gt_cat_name",
        "gt_is_fake",
        "ppo_verdict",
        "ppo_predicted_category",
        "ppo_pred_cat_idx",
        "ppo_p_fake",
        "ppo_overall_confidence",
        "ppo_threshold",
        "ppo_verdict_zone",
        "ppo_fake_from_verdict",
        "ppo_fake_from_category",
        "ppo_binary_discrepant",
        "ppo_p_authentic",
        "ppo_p_audio_manip",
        "ppo_p_video_manip",
        "ppo_p_both",
        "ppo_w_spatial",
        "ppo_w_temporal",
        "ppo_w_audio",
        "ppo_mismatch_score",
        "corr_verdict",
        "corr_fake",
        "corr_confidence",
        "leg_verdict",
        "leg_fake",
        "leg_confidence",
    ]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    logger.info("Wrote %s (%d rows)", OUT_CSV, n)

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", OUT_JSON)

    _write_audit_md(payload, rows)
    logger.info("Wrote %s", OUT_AUDIT)
    logger.info("STATUS: %s", payload["status"])
    return payload


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _write_audit_md(payload: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    m_corr = payload["binary_summary"]["corrected_fixed"]
    m_leg = payload["binary_summary"]["legacy_fixed"]
    m_v = payload["binary_summary"]["ppo_v3_verdict_based"]
    m_c = payload["binary_summary"]["ppo_v3_category_based"]
    cm = payload["confusion_matrix_4way"]["matrix"]
    labels = payload["confusion_matrix_4way"]["labels"]
    aliases = payload["confusion_matrix_4way"]["display_aliases"]
    disc = payload["discrepant_samples"]
    mcn_v = payload["mcnemar_tests"]["corrected_fixed_vs_ppo_verdict"]
    mcn_c = payload["mcnemar_tests"]["corrected_fixed_vs_ppo_category"]
    per_gt = payload["per_gt_category_binary_and_attribution"]
    cat_prf = payload["per_category_classification_metrics"]

    lines: List[str] = []
    a = lines.append

    a("# VERITAS PPO v3 — Binary / Four-Category Consistency Audit")
    a("")
    a(f"**Date:** {payload['date']}  ")
    a(f"**Status:** `{payload['status']}`  ")
    a(f"**Checkpoint:** `{payload['checkpoint']}`  ")
    a(f"**Features:** `{payload['cache']}`  ")
    a(f"**Manifest:** `{payload['manifest']}`  ")
    a(f"**Evaluated samples:** {payload['total_test_samples']}  ")
    a("")
    a("Original report/JSON were **not** overwritten. PPO v3 was **not** promoted.")
    a("")
    a("---")
    a("")
    a("## 1. Root Cause")
    a("")
    a(payload["old_report_comparison"]["root_cause"])
    a("")
    a("### Dual decision surfaces (intentional policy design)")
    a("")
    a("`apply_action_to_state` emits two related but not identical decisions:")
    a("")
    a("1. **Binary verdict** with a deadband: `DEEPFAKE` if `p_fake >= τ+0.05`,")
    a("   `AUTHENTIC` if `p_fake <= τ-0.05`, else `UNCERTAIN / INSUFFICIENT EVIDENCE`.")
    a("2. **Four-category `predicted_category`**: hard AUTHENTIC / AUDIO / VIDEO / BOTH.")
    a("   In the UNCERTAIN band, category uses `p_cat[0] >= 0.5` independently, so a")
    a("   sample can be **UNCERTAIN** (binary not-fake) while `predicted_category` is a")
    a("   manipulation class.")
    a("")
    a("This is intentional (abstention on binary verdict while still attributing a")
    a("diagnostic category for reward / analytics). It is **not** a silent bug in the")
    a("policy. The evaluation bug was **mixing surfaces** in one report without labeling them.")
    a("")
    a("Invariant `binary_fake == (predicted_category != AUTHENTIC)`:")
    a(f"- Satisfied on all samples: **{payload['invariant_satisfied_verdict_vs_category_all_samples']}**")
    a(f"- Discrepant samples: **{payload['discrepant_count']}** (all in UNCERTAIN zone)")
    a("")
    if disc:
        a("| sample_id | rel_path | GT | verdict | pred_cat | p_fake | τ |")
        a("| ---: | :--- | :--- | :--- | :--- | ---: | ---: |")
        for d in disc:
            a(
                f"| {d['sample_id']} | `{d['rel_path']}` | {d['gt_cat']} | "
                f"{d['verdict']} | {d['pred_cat']} | {d['p_fake']:.4f} | {d['threshold']:.4f} |"
            )
        a("")
    a("---")
    a("")
    a("## 2. Evaluator Trace (same forward pass per sample)")
    a("")
    a("| Quantity | Source |")
    a("| :--- | :--- |")
    a("| `predicted_category` | `apply_action_to_state` → argmax / UNCERTAIN rule |")
    a("| `verdict` | same call → three-zone threshold on `p_fake` |")
    a("| `overall_confidence` | `round(clip(p_fake))` from same call |")
    a("| Binary CM (original report) | `pred_fake = (verdict == DEEPFAKE)` |")
    a("| 4-way CM | `CATEGORY_TO_IDX[predicted_category]` |")
    a("| Per-category “acc” in original | **binary** accuracy within each GT class |")
    a("| Checkpoint | `models/ppo_fusion_policy_v3.pth` (`model_state_dict` + normalizer) |")
    a("| Split / features | `leakage_controlled_split_manifest.json` → `final_test`; `cache/fusion_features_lc_v3.json` |")
    a("")
    a("Deterministic inference uses the actor mean action (no sampling).")
    a("")
    a("---")
    a("")
    a("## 3. Recomputed Metrics (n=217, same videos & labels)")
    a("")
    a("### 3.1 Binary classification")
    a("")
    a("| System | Acc | Bal Acc | Prec | Rec | Spec | F1 | TP | TN | FP | FN | CI 95% |")
    a("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    def row(name: str, m: Dict[str, Any]) -> None:
        ci = m["ci_95"]
        a(
            f"| {name} | {_pct(m['accuracy'])} | {_pct(m['balanced_accuracy'])} | "
            f"{_pct(m['precision'])} | {_pct(m['recall'])} | {_pct(m['specificity'])} | "
            f"{m['f1']:.4f} | {m['tp']} | {m['tn']} | {m['fp']} | {m['fn']} | "
            f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%] |"
        )

    row("Legacy Fixed", m_leg)
    row("Corrected Fixed", m_corr)
    row("PPO v3 (verdict-based)", m_v)
    row("PPO v3 (category-based)", m_c)
    a("")
    a("Fixed fusion **four-category metrics: unavailable** (binary-only emitters).")
    a("")
    a("Category-based binary (TP/TN/FP/FN) is exactly implied by the 4-way matrix margins:")
    a(
        f"`TN=cm[0,0]={cm[0][0]}`, `FP=sum(cm[0,1:])={sum(cm[0][1:])}`, "
        f"`FN=sum(cm[1:,0])={sum(cm[i][0] for i in range(1,4))}`, "
        f"`TP=sum(cm[1:,1:])={sum(cm[i][j] for i in range(1,4) for j in range(1,4))}` "
        f"→ accuracy {_pct(m_c['accuracy'])}."
    )
    a("")
    a("### 3.2 Four-category confusion matrix (PPO v3)")
    a("")
    a("Rows = ground truth; columns = prediction. Code labels / report aliases:")
    a("")
    header = "| GT \\ Pred | " + " | ".join(aliases) + " |"
    a(header)
    a("| :--- | " + " | ".join([":---:"] * 4) + " |")
    for i, lab in enumerate(labels):
        a(f"| **{lab}** ({aliases[i]}) | " + " | ".join(str(cm[i][j]) for j in range(4)) + " |")
    a("")
    a(
        f"**4-way accuracy:** {_pct(payload['categorical_attribution_accuracy'])} "
        f"({int(np.trace(np.array(cm)))}/{payload['total_test_samples']})  "
    )
    a(f"**4-way macro-F1:** {payload['categorical_macro_f1']:.4f}")
    a("")
    a("### 3.3 Per-category precision / recall / F1 (from same 4-way matrix)")
    a("")
    a("| Category | n | Precision | Recall | F1 | Diagonal |")
    a("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for name in CATEGORIES:
        p = cat_prf[name]
        a(
            f"| {name} | {p['support']} | {_pct(p['precision'])} | "
            f"{_pct(p['recall'])} | {p['f1']:.4f} | {p['tp_diagonal']} |"
        )
    a("")
    a("### 3.4 Per-GT binary accuracy vs categorical recall (AUTHENTIC example)")
    a("")
    a("| Metric | Value | Notes |")
    a("| :--- | :--- | :--- |")
    auth = per_gt["AUTHENTIC"]
    a(
        f"| Original report “AUTHENTIC 84%” | 84.0% | "
        f"Actually PPO **verdict** binary acc = {auth['ppo_verdict_binary_acc']*100:.1f}% "
        f"({int(round(auth['ppo_verdict_binary_acc']*auth['count']))}/{auth['count']}) |"
    )
    a(
        f"| Categorical AUTHENTIC recall | {_pct(auth['ppo_category_recall'])} | "
        f"{auth['ppo_cat_correct']}/{auth['count']} = CM diagonal |"
    )
    a(
        f"| Category-based binary AUTHENTIC acc | {_pct(auth['ppo_category_binary_acc'])} | "
        f"Equals categorical recall when mapping non-AUTHENTIC→fake |"
    )
    a("")
    a("### 3.5 McNemar (Corrected Fixed vs PPO)")
    a("")
    a("| Comparison | b (corr✓ ppo✗) | c (ppo✓ corr✗) | χ² | p | Significant? |")
    a("| :--- | :---: | :---: | :---: | :---: | :---: |")
    a(
        f"| Verdict-based PPO | {mcn_v['b']} | {mcn_v['c']} | {mcn_v['chi2']:.4f} | "
        f"{mcn_v['p_value']:.4f} | {mcn_v['significant_at_0_05']} |"
    )
    a(
        f"| Category-based PPO | {mcn_c['b']} | {mcn_c['c']} | {mcn_c['chi2']:.4f} | "
        f"{mcn_c['p_value']:.4f} | {mcn_c['significant_at_0_05']} |"
    )
    a("")
    a(
        "> Non-significant McNemar does **not** prove equivalence. Corrected fixed "
        "still has fewer false positives and higher accuracy on this frozen set."
    )
    a("")
    a("---")
    a("")
    a("## 4. Production-readiness conclusion")
    a("")
    a("**Do not promote PPO v3 to production default.** Keep corrected fixed fusion.")
    a("")
    for r in payload["promotion_assessment"]["rationale"]:
        a(f"- {r}")
    a("")
    a("PPO v3 remains valuable for continued evaluation of 4-way attribution, but")
    a("authentic false positives / UNCERTAIN–category mismatches must be addressed")
    a("before any default switch. No threshold retuning on the test set was performed.")
    a("")
    a("---")
    a("")
    a("## 5. Deliverables & assertions")
    a("")
    a("| File | Role |")
    a("| :--- | :--- |")
    a("| `ppo_v3_record_level_predictions.csv` | Record-level dual-surface table |")
    a("| `ppo_v3_test_results_recomputed.json` | Machine-readable recomputed metrics |")
    a("| `ppo_v3_consistency_audit.md` | This document |")
    a("| `ppo_v3_test_results.json` | Preserved original (verdict-based) JSON |")
    a("| `ppo_v3_lc_evaluation_report.md` | Preserved original narrative report |")
    a("")
    a("Assertions passed:")
    a("")
    for k, v in payload["assertions"].items():
        a(f"- `{k}`: **{v}**")
    a("")
    a(f"**Final status:** `{payload['status']}`")
    a("")

    OUT_AUDIT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_reconcile()
