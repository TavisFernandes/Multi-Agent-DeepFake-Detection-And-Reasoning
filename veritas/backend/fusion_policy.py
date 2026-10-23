"""
fusion_policy.py
================
Genuine PPO-Based Adaptive Fusion Policy for VERITAS Multi-Agent Deepfake Detection.

Mathematical Architecture & Operational Specification:
-------------------------------------------------------
1. State Representation (14-Dimensional Continuous Vector s in R^14):
   - s[0]: Spatial fake probability p_sp in [0, 1] (or 0.5 if unavailable)
   - s[1]: Spatial uncertainty u_sp = 1 - 2*|p_sp - 0.5| in [0, 1]
   - s[2]: Spatial valid flag v_sp in {0, 1}
   - s[3]: Temporal fake probability p_te in [0, 1] (or 0.5 if unavailable)
   - s[4]: Temporal uncertainty u_te = 1 - 2*|p_te - 0.5| in [0, 1]
   - s[5]: Temporal valid flag v_te in {0, 1}
   - s[6]: Audio fake probability p_au in [0, 1] (or 0.5 if unavailable)
   - s[7]: Audio uncertainty u_au = 1 - 2*|p_au - 0.5| in [0, 1]
   - s[8]: Audio valid flag v_au in {0, 1}
   - s[9]: Context cross-modal mismatch m_ctx = |p_vis_est - p_au| in [0, 1]
   - s[10]: Context valid flag v_ctx in {0, 1}
   - s[11]: Audio duration normalized d_au = min(duration / 30.0, 1.0)
   - s[12]: Container has_audio flag in {0, 1}
   - s[13]: Container has_video flag in {0, 1}

2. Action Space (5-Dimensional Continuous Policy a in R^5):
   - a[0]: Spatial visual gating logit
   - a[1]: Temporal visual gating logit
   - a[2]: Audio trust logit (controls w_au = sigmoid(a[2]) in (0, 1))
   - a[3]: Cross-modal mismatch sensitivity gain g_ctx = 2.0 * sigmoid(a[3])
   - a[4]: Adaptive decision threshold offset Delta tau = 0.15 * tanh(a[4])

3. Exact Action Application & Fusion Mathematics:
   - Visual Fusion:
       If v_sp = 1 and v_te = 1:
           w_sp = exp(a[0]) / (exp(a[0]) + exp(a[1]))
           w_te = exp(a[1]) / (exp(a[0]) + exp(a[1]))
           P_v = w_sp * p_sp + w_te * p_te
       Elif v_sp = 1: w_sp = 1.0, w_te = 0.0, P_v = p_sp
       Elif v_te = 1: w_sp = 0.0, w_te = 1.0, P_v = p_te
       Else: w_sp = 0.0, w_te = 0.0, P_v = None

   - Audio Trust Modulation:
       If v_au = 1:
           w_au = sigmoid(a[2])
           P_a = w_au * p_au + (1.0 - w_au) * 0.5
       Else:
           w_au = 0.0, P_a = None

   - Joint Four-Category Probability Distribution:
       If P_v is not None and P_a is not None:
           P_base(AUTHENTIC)              = (1 - P_v) * (1 - P_a)
           P_base(AUDIO_MANIPULATED)       = (1 - P_v) * P_a
           P_base(VIDEO_MANIPULATED)       = P_v * (1 - P_a)
           P_base(AUDIO_VIDEO_MANIPULATED) = P_v * P_a

           m_ctx = |P_v - P_a|
           g_ctx = 2.0 * sigmoid(a[3])

           z0 = ln(P_base(AUTHENTIC) + 1e-6) - g_ctx * m_ctx
           z1 = ln(P_base(AUDIO_MANIPULATED) + 1e-6) + g_ctx * m_ctx * 1(P_a > P_v)
           z2 = ln(P_base(VIDEO_MANIPULATED) + 1e-6) + g_ctx * m_ctx * 1(P_v > P_a)
           z3 = ln(P_base(AUDIO_VIDEO_MANIPULATED) + 1e-6) - g_ctx * m_ctx * 1(m_ctx > 0.4)

           p_cat = Softmax([z0, z1, z2, z3])
           P_fake = 1.0 - p_cat[0] = p_cat[1] + p_cat[2] + p_cat[3]

       Elif P_v is not None (Audio Missing / Silent / Image):
           p_cat = [1.0 - P_v, 0.0, P_v, 0.0]
           P_fake = P_v

       Elif P_a is not None (Video Missing / Audio-Only):
           p_cat = [1.0 - P_a, P_a, 0.0, 0.0]
           P_fake = P_a

       Else:
           p_cat = [0.0, 0.0, 0.0, 0.0]
           P_fake = 0.0

   - Verdict Determination & Parity:
       tau = clip(0.50 + 0.15 * tanh(a[4]), 0.35, 0.65)
       deadband = 0.05 (frozen architectural operating constant; not tuned on test)
       If P_fake >= tau + deadband -> "DEEPFAKE", category = argmax_{k in {1,2,3}} p_cat[k]
       If P_fake <= tau - deadband -> "AUTHENTIC", category = "AUTHENTIC" (k=0)
       Else -> "UNCERTAIN / INSUFFICIENT EVIDENCE", category = leading hypothesis
               (AUTHENTIC if p_cat[0] >= 0.5 else argmax_{k in {1,2,3}} p_cat[k])

       Dual-surface note: outside the UNCERTAIN deadband,
       (verdict == DEEPFAKE) <=> (predicted_category != AUTHENTIC).
       Inside UNCERTAIN, binary abstention coexists with a *tentative* category
       hypothesis. Evaluation must never count UNCERTAIN as AUTHENTIC silently.
       See decision_contract.py for field semantics and selective metrics.

   - Probability status:
       category_probabilities are model-implied (softmax / modality marginals).
       They are NOT claimed to be externally calibrated unless measured on val.

4. Single-Step Contextual Bandit PPO:
   - Discount factor gamma = 0. Advantage A(s, a) = R(s, a) - V(s).
   - Clipped surrogate objective, value loss, entropy regularization.
   - Deterministic production inference uses distribution mean mu.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from decision_contract import (
    DEFAULT_DEADBAND,
    enrich_decision_fields,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & Taxonomy
# ---------------------------------------------------------------------------
STATE_DIM = 14
ACTION_DIM = 5  # a0: w_sp_logit, a1: w_te_logit, a2: w_au_logit, a3: g_ctx_logit, a4: delta_tau_logit

CATEGORIES = [
    "AUTHENTIC",
    "AUDIO_MANIPULATED",
    "VIDEO_MANIPULATED",
    "AUDIO_VIDEO_MANIPULATED",
]

CATEGORY_TO_IDX = {cat: idx for idx, cat in enumerate(CATEGORIES)}
IDX_TO_CATEGORY = {idx: cat for idx, cat in enumerate(CATEGORIES)}

_HERE = Path(__file__).resolve().parent
# v4: train_ppo_v3.py --seed 2 --epochs 300, on scores from the *_lc detectors production now loads.
DEFAULT_V4_CHECKPOINT_PATH = _HERE / "models" / "ppo_fusion_policy_v4.pth"
DEFAULT_V2_CHECKPOINT_PATH = _HERE / "models" / "ppo_fusion_policy_v2.pth"
DEFAULT_CHECKPOINT_PATH = _HERE / "models" / "ppo_fusion_policy.pth"


# ---------------------------------------------------------------------------
# Feature Normalizer
# ---------------------------------------------------------------------------
class FeatureNormalizer:
    """
    Feature normalizer fitted strictly on the training split.
    Frozen during validation, test evaluation, and production inference.
    """

    def __init__(self, state_dim: int = STATE_DIM):
        self.state_dim = state_dim
        self.mean = np.zeros(state_dim, dtype=np.float32)
        self.std = np.ones(state_dim, dtype=np.float32)
        self.count = 0

    def fit(self, X: np.ndarray) -> None:
        """Fit normalization parameters on training array X of shape (N, D)."""
        if len(X) == 0:
            return
        X = np.asarray(X, dtype=np.float32)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero on constant features (e.g. constant flags)
        self.std = np.where(self.std < 1e-4, 1.0, self.std)
        self.count = len(X)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Normalize feature vector or batch."""
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / (self.std + 1e-7)

    def to_dict(self) -> dict:
        return {
            "state_dim": self.state_dim,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "count": self.count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> FeatureNormalizer:
        norm = cls(state_dim=d.get("state_dim", STATE_DIM))
        norm.mean = np.array(d["mean"], dtype=np.float32)
        norm.std = np.array(d["std"], dtype=np.float32)
        norm.count = d.get("count", 0)
        return norm


# ---------------------------------------------------------------------------
# State Extraction Helper
# ---------------------------------------------------------------------------
def extract_state_from_agents(
    agents: List[Dict[str, Any]],
    media_type: str = "",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Converts list of raw agent dictionaries into a 14-dimensional state vector.
    
    Returns:
        state_vec: shape (14,), raw unnormalized features.
        metadata: auxiliary information extracted from agents.
    """
    agent_by_name: Dict[str, Dict[str, Any]] = {}
    for a in agents:
        name = (a.get("agent") or a.get("agent_name") or "").lower()
        if "spatial" in name:
            agent_by_name["spatial"] = a
        elif "temporal" in name:
            agent_by_name["temporal"] = a
        elif "audio" in name:
            agent_by_name["audio"] = a
        elif "context" in name:
            agent_by_name["context"] = a

    unavailable_statuses = {
        "not_applicable",
        "unavailable",
        "no_audio",
        "untrained",
        "error",
        "failed",
        "mocked",
    }

    # 1. Spatial Agent
    sp = agent_by_name.get("spatial")
    sp_valid = 0.0
    sp_prob = 0.5
    sp_uncertainty = 1.0
    if sp and sp.get("is_fake") is not None and sp.get("status") not in unavailable_statuses:
        sp_valid = 1.0
        sp_prob = float(np.clip(sp.get("confidence", 0.5), 0.0, 1.0))
        sp_uncertainty = 1.0 - 2.0 * abs(sp_prob - 0.5)

    # 2. Temporal Agent
    te = agent_by_name.get("temporal")
    te_valid = 0.0
    te_prob = 0.5
    te_uncertainty = 1.0
    if te and te.get("is_fake") is not None and te.get("status") not in unavailable_statuses:
        te_valid = 1.0
        te_prob = float(np.clip(te.get("confidence", 0.5), 0.0, 1.0))
        te_uncertainty = 1.0 - 2.0 * abs(te_prob - 0.5)

    # 3. Audio Agent
    au = agent_by_name.get("audio")
    au_valid = 0.0
    au_prob = 0.5
    au_uncertainty = 1.0
    au_duration = 0.0
    has_audio_flag = 0.0
    if au:
        has_audio_flag = 1.0 if au.get("has_audio", False) else 0.0
        dur = float(au.get("duration", 0.0) or 0.0)
        au_duration = float(np.clip(dur / 30.0, 0.0, 1.0))
        if au.get("is_fake") is not None and au.get("status") not in unavailable_statuses:
            au_valid = 1.0
            au_prob = float(np.clip(au.get("confidence", 0.5), 0.0, 1.0))
            au_uncertainty = 1.0 - 2.0 * abs(au_prob - 0.5)

    # Visual aggregate estimate for context divergence
    if sp_valid and te_valid:
        v_prob_est = (sp_prob + te_prob) / 2.0
    elif sp_valid:
        v_prob_est = sp_prob
    elif te_valid:
        v_prob_est = te_prob
    else:
        v_prob_est = None

    # 4. Context Mismatch
    ctx_mismatch = 0.0
    ctx_valid = 0.0
    if v_prob_est is not None and au_valid:
        ctx_valid = 1.0
        ctx_mismatch = float(abs(v_prob_est - au_prob))

    # 5. Container media flags
    mtype = (media_type or "").lower().strip()
    if mtype == "video":
        has_video_flag = 1.0
        if not au:
            has_audio_flag = 1.0
    elif mtype == "audio":
        has_video_flag = 0.0
        has_audio_flag = 1.0
    elif mtype == "image":
        has_video_flag = 1.0
        has_audio_flag = 0.0
    else:
        has_video_flag = 1.0 if (sp_valid or te_valid or sp or te) else 0.0
        if not au:
            has_audio_flag = 0.0

    state_vec = np.array([
        sp_prob,            # 0
        sp_uncertainty,     # 1
        sp_valid,           # 2
        te_prob,            # 3
        te_uncertainty,     # 4
        te_valid,           # 5
        au_prob,            # 6
        au_uncertainty,     # 7
        au_valid,           # 8
        ctx_mismatch,       # 9
        ctx_valid,          # 10
        au_duration,        # 11
        has_audio_flag,     # 12
        has_video_flag,     # 13
    ], dtype=np.float32)

    metadata = {
        "spatial_valid": bool(sp_valid),
        "temporal_valid": bool(te_valid),
        "audio_valid": bool(au_valid),
        "context_valid": bool(ctx_valid),
        "has_audio": bool(has_audio_flag),
        "has_video": bool(has_video_flag),
        "raw_spatial_prob": sp_prob if sp_valid else None,
        "raw_temporal_prob": te_prob if te_valid else None,
        "raw_audio_prob": au_prob if au_valid else None,
    }

    return state_vec, metadata


# ---------------------------------------------------------------------------
# Actor-Critic Network Architecture
# ---------------------------------------------------------------------------
class ActorCriticNetwork(nn.Module):
    """
    Continuous-action Actor-Critic Network for PPO Adaptive Fusion.
    Actor outputs mean mu for 5 continuous actions with learnable log_std.
    Critic outputs scalar state-value estimate V(s).
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Policy Network (Actor)
        self.actor_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
        )
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5, dtype=torch.float32))

        # Value Network (Critic)
        self.critic_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor_net[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_net[-1].weight, gain=1.0)

    def get_action_dist(self, state: torch.Tensor) -> Normal:
        """Return Gaussian action distribution."""
        mu = self.actor_net(state)
        std = torch.exp(torch.clamp(self.actor_log_std, min=-2.0, max=1.0))
        return Normal(mu, std)

    def forward(self, state: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        dist = self.get_action_dist(state)
        val = self.critic_net(state).squeeze(-1)
        return dist, val

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic_net(state).squeeze(-1)


# ---------------------------------------------------------------------------
# Action Mapping & Fusion Decision Logic
# ---------------------------------------------------------------------------
def apply_action_to_state(
    action: np.ndarray,
    state_vec: np.ndarray,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Maps continuous PPO action vector to fusion weights, modality probabilities,
    calibrated 4-category probabilities, and the final verdict.
    
    Action mapping:
      a[0]: Spatial visual gating logit
      a[1]: Temporal visual gating logit
      a[2]: Audio trust logit (directly modulates p_audio via sigmoid(a[2]))
      a[3]: Cross-modal mismatch sensitivity gain g_ctx in [0, 2.0]
      a[4]: Decision threshold offset Delta tau in [-0.15, +0.15]
    """
    sp_valid = metadata.get("spatial_valid", False)
    te_valid = metadata.get("temporal_valid", False)
    au_valid = metadata.get("audio_valid", False)

    sp_prob = state_vec[0] if sp_valid else None
    te_prob = state_vec[3] if te_valid else None
    au_prob = state_vec[6] if au_valid else None

    # 1. Visual Modality Gating (Spatial vs Temporal)
    w_sp = 0.0
    w_te = 0.0
    p_visual: Optional[float] = None

    if sp_valid and te_valid:
        logits = np.array([action[0], action[1]], dtype=np.float32)
        exp_logits = np.exp(logits - np.max(logits))
        w = exp_logits / (np.sum(exp_logits) + 1e-7)
        w_sp = float(w[0])
        w_te = float(w[1])
        p_visual = float(w_sp * sp_prob + w_te * te_prob)
    elif sp_valid:
        w_sp = 1.0
        w_te = 0.0
        p_visual = float(sp_prob)
    elif te_valid:
        w_sp = 0.0
        w_te = 1.0
        p_visual = float(te_prob)
    else:
        p_visual = None

    # 2. Audio Modality Gating & Trust Modulation
    w_au = 0.0
    p_audio: Optional[float] = None
    if au_valid and au_prob is not None:
        # Sigmoid trust weight in (0, 1)
        w_au = float(1.0 / (1.0 + np.exp(-action[2])))
        # Meaningfully scale audio fake probability towards uninformative prior (0.5) when trust is low
        p_audio = float(w_au * au_prob + (1.0 - w_au) * 0.5)

    # 3. Category Probabilities (4-way decomposition)
    category_probs = np.zeros(4, dtype=np.float32)
    ctx_mismatch = 0.0

    if (p_visual is not None) and (p_audio is not None):
        pv = float(np.clip(p_visual, 1e-4, 1.0 - 1e-4))
        pa = float(np.clip(p_audio, 1e-4, 1.0 - 1e-4))
        ctx_mismatch = float(abs(pv - pa))

        # Base independent marginal joint distribution
        p_auth_base = (1.0 - pv) * (1.0 - pa)
        p_aud_base = (1.0 - pv) * pa
        p_vid_base = pv * (1.0 - pa)
        p_both_base = pv * pa

        # Mismatch sensitivity modulation (g_ctx in [0, 2.0])
        g_ctx = float(2.0 / (1.0 + np.exp(-action[3])))
        z0 = math.log(max(p_auth_base, 1e-6)) - g_ctx * ctx_mismatch
        z1 = math.log(max(p_aud_base, 1e-6)) + g_ctx * ctx_mismatch * (1.0 if pa > pv else 0.0)
        z2 = math.log(max(p_vid_base, 1e-6)) + g_ctx * ctx_mismatch * (1.0 if pv > pa else 0.0)
        z3 = math.log(max(p_both_base, 1e-6)) - g_ctx * ctx_mismatch * (1.0 if ctx_mismatch > 0.4 else 0.0)

        logits_cat = np.array([z0, z1, z2, z3], dtype=np.float32)
        exp_cat = np.exp(logits_cat - np.max(logits_cat))
        category_probs = exp_cat / np.sum(exp_cat)

        p_fake = float(1.0 - category_probs[0])

    elif p_visual is not None:
        # Audio unavailable / missing modality
        pv = float(np.clip(p_visual, 0.0, 1.0))
        category_probs[0] = float(1.0 - pv)
        category_probs[1] = 0.0
        category_probs[2] = float(pv)  # Video Manipulated
        category_probs[3] = 0.0
        p_fake = float(pv)

    elif p_audio is not None:
        # Video unavailable / missing modality
        pa = float(np.clip(p_audio, 0.0, 1.0))
        category_probs[0] = float(1.0 - pa)
        category_probs[1] = float(pa)  # Audio Manipulated
        category_probs[2] = 0.0
        category_probs[3] = 0.0
        p_fake = float(pa)

    else:
        # All modalities unavailable
        p_fake = 0.0

    # 4. Adaptive Threshold & Verdict Determination
    delta_tau = float(np.tanh(action[4]) * 0.15) if len(action) > 4 else 0.0
    decision_threshold = float(np.clip(0.50 + delta_tau, 0.35, 0.65))

    contributing = []
    unavailable = []
    if sp_valid:
        contributing.append("Spatial Analysis")
    else:
        unavailable.append("Spatial Analysis")

    if te_valid:
        contributing.append("Temporal Analysis")
    else:
        unavailable.append("Temporal Analysis")

    if au_valid:
        contributing.append("Audio Analysis")
    else:
        unavailable.append("Audio Analysis")
    # Context is explanation-only (mock_agents.py), so it is never listed as contributing.

    if not contributing:
        verdict = "UNVERIFIED MODALITY"
        overall_confidence = 0.0
        predicted_category = "UNVERIFIED"
    else:
        # overall_confidence retained as the fused fake score for API backward compat
        overall_confidence = round(float(np.clip(p_fake, 0.0, 1.0)), 4)
        if overall_confidence >= decision_threshold + DEFAULT_DEADBAND:
            verdict = "DEEPFAKE"
            # Argmax over manipulated categories to preserve semantic parity
            manip_indices = [1, 2, 3]
            best_manip_idx = manip_indices[int(np.argmax(category_probs[manip_indices]))]
            predicted_category = IDX_TO_CATEGORY[best_manip_idx]
        elif overall_confidence <= decision_threshold - DEFAULT_DEADBAND:
            verdict = "AUTHENTIC"
            predicted_category = "AUTHENTIC"
        else:
            verdict = "UNCERTAIN / INSUFFICIENT EVIDENCE"
            # Leading category hypothesis only — not a definitive authenticity claim
            if category_probs[0] >= 0.50:
                predicted_category = "AUTHENTIC"
            else:
                manip_indices = [1, 2, 3]
                best_manip_idx = manip_indices[int(np.argmax(category_probs[manip_indices]))]
                predicted_category = IDX_TO_CATEGORY[best_manip_idx]

    result = {
        "verdict": verdict,
        "overall_confidence": overall_confidence,
        "p_fake": float(p_fake),
        "p_visual": p_visual,
        "p_audio": p_audio,
        "category_probabilities": {
            cat: round(float(category_probs[idx]), 4)
            for idx, cat in enumerate(CATEGORIES)
        },
        "predicted_category": predicted_category,
        "decision_threshold": round(decision_threshold, 4),
        "contributing_agents": contributing,
        "unavailable_agents": unavailable,
        "dynamic_weights": {
            "spatial": round(w_sp, 4),
            "temporal": round(w_te, 4),
            "audio": round(w_au, 4),
        },
        "mismatch_score": round(float(ctx_mismatch), 4),
        "method": "ppo",
    }
    return enrich_decision_fields(
        result,
        tau=decision_threshold,
        deadband=DEFAULT_DEADBAND,
        probabilities_are_calibrated=False,
    )


# ---------------------------------------------------------------------------
# Defensible Ground-Truth Reward Function
# ---------------------------------------------------------------------------
def compute_reward(
    fusion_output: Dict[str, Any],
    ground_truth_category: int,
) -> float:
    """
    Computes reward signal based strictly on verified ground truth.
    
    ground_truth_category:
        0: AUTHENTIC (RealVideo-RealAudio) -> y_fake = 0
        1: AUDIO_MANIPULATED (RealVideo-FakeAudio) -> y_fake = 1
        2: VIDEO_MANIPULATED (FakeVideo-RealAudio) -> y_fake = 1
        3: AUDIO_VIDEO_MANIPULATED (FakeVideo-FakeAudio) -> y_fake = 1
    """
    gt_is_fake = 1 if ground_truth_category > 0 else 0
    verdict = fusion_output["verdict"]
    pred_cat_idx = CATEGORY_TO_IDX.get(fusion_output["predicted_category"], 0)

    # 1. Binary Decision Payoff Matrix
    if verdict == "DEEPFAKE":
        if gt_is_fake == 1:
            r_binary = +1.5   # True Positive
        else:
            r_binary = -2.0   # False Positive on authentic media
    elif verdict == "AUTHENTIC":
        if gt_is_fake == 0:
            r_binary = +1.5   # True Negative
        else:
            r_binary = -3.0   # False Negative: missed deepfake (highest penalty)
    else:
        # Uncertain / Insufficient evidence
        r_binary = -0.3

    # 2. Fine-grained 4-Category Attribution Reward
    r_cat = +1.0 if (pred_cat_idx == ground_truth_category) else -0.5

    return float(r_binary + r_cat)


# ---------------------------------------------------------------------------
# PPO Fusion Policy (Production Model Wrapper)
# ---------------------------------------------------------------------------
class PPOFusionPolicy:
    """
    High-level inference and lifecycle management for the PPO Adaptive Fusion policy.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = ActorCriticNetwork(STATE_DIM, ACTION_DIM).to(self.device)
        self.normalizer = FeatureNormalizer(STATE_DIM)
        self.is_loaded = False
        self.checkpoint_metadata: Dict[str, Any] = {}
        self.checkpoint_path: Optional[Path] = None

        # Prefer v4, then v2, then the original v1
        if checkpoint_path:
            ckpt = Path(checkpoint_path)
        elif DEFAULT_V4_CHECKPOINT_PATH.exists():
            ckpt = DEFAULT_V4_CHECKPOINT_PATH
        elif DEFAULT_V2_CHECKPOINT_PATH.exists():
            ckpt = DEFAULT_V2_CHECKPOINT_PATH
        else:
            ckpt = DEFAULT_CHECKPOINT_PATH

        if ckpt.exists():
            try:
                self.load_checkpoint(ckpt)
            except Exception as e:
                logger.error("Failed to load PPO checkpoint from %s: %s", ckpt, e)

    def load_checkpoint(self, path: Union[str, Path]) -> None:
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PPO checkpoint not found at: {path}")

        logger.info("Loading PPO Fusion Policy from %s", path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Detect action dimension in checkpoint
        loaded_action_dim = checkpoint.get("action_dim", ACTION_DIM)
        if loaded_action_dim != self.network.action_dim:
            self.network = ActorCriticNetwork(STATE_DIM, loaded_action_dim).to(self.device)

        if "model_state_dict" in checkpoint:
            self.network.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            self.network.load_state_dict(checkpoint["state_dict"])
        else:
            raise KeyError("Checkpoint missing model_state_dict")

        if "normalizer" in checkpoint:
            self.normalizer = FeatureNormalizer.from_dict(checkpoint["normalizer"])
        else:
            logger.warning("Normalizer missing from checkpoint; using identity normalizer.")

        self.checkpoint_metadata = checkpoint.get("metadata", {})
        self.checkpoint_path = path
        self.network.eval()
        self.is_loaded = True
        logger.info("PPO Fusion Policy loaded successfully (action_dim=%d).", loaded_action_dim)

    def save_checkpoint(
        self,
        path: Union[str, Path],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_state_dict": self.network.state_dict(),
            "normalizer": self.normalizer.to_dict(),
            "state_dim": self.network.state_dim,
            "action_dim": self.network.action_dim,
            "metadata": extra_metadata or {},
        }
        torch.save(payload, path)
        logger.info("Saved PPO Fusion Policy to %s", path)

    def predict(
        self,
        state_vec: np.ndarray,
        metadata: Dict[str, Any],
        deterministic: bool = True,
    ) -> Dict[str, Any]:
        """Runs deterministic or sampled inference on a single state vector."""
        self.network.eval()
        norm_state = self.normalizer.transform(state_vec)
        state_tensor = torch.from_numpy(norm_state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            dist, val = self.network(state_tensor)
            if deterministic:
                action_tensor = dist.mean
            else:
                action_tensor = dist.sample()
            action = action_tensor.squeeze(0).cpu().numpy()

        return apply_action_to_state(action, state_vec, metadata)

    def fuse_agents(
        self,
        agents: List[Dict[str, Any]],
        media_type: str = "",
    ) -> Tuple[float, str, List[str], List[str], Dict[str, Any]]:
        """
        Production fusion interface conforming to VERITAS multi-agent pipeline.
        
        Returns:
            overall_confidence: float in [0, 1]
            verdict: "AUTHENTIC" | "DEEPFAKE" | "UNCERTAIN / INSUFFICIENT EVIDENCE" | "UNVERIFIED MODALITY"
            contributing_agents: list of agent names that participated
            unavailable_agents: list of agent names excluded
            details: rich diagnostic details dictionary
        """
        state_vec, metadata = extract_state_from_agents(agents, media_type)
        res = self.predict(state_vec, metadata, deterministic=True)

        return (
            res["overall_confidence"],
            res["verdict"],
            res["contributing_agents"],
            res["unavailable_agents"],
            res,
        )


# ---------------------------------------------------------------------------
# Semantically Corrected Fixed-Weight Fusion Baseline
# ---------------------------------------------------------------------------
def compute_verdict_corrected_fixed(
    agents: List[Dict[str, Any]],
    media_type: str = "",
) -> Tuple[float, str, List[str], List[str], Dict[str, Any]]:
    """
    Semantically corrected reference baseline.
    Does NOT blindly add context mismatch into the linear average.
    Combines visual signals first, then joins visual with audio via logical manipulation rule:
        P(Manipulated) = 1 - (1 - P_visual) * (1 - P_audio)
    """
    state_vec, metadata = extract_state_from_agents(agents, media_type)

    sp_valid = metadata["spatial_valid"]
    te_valid = metadata["temporal_valid"]
    au_valid = metadata["audio_valid"]

    p_sp = state_vec[0] if sp_valid else None
    p_te = state_vec[3] if te_valid else None
    p_au = state_vec[6] if au_valid else None

    # Visual combination
    if sp_valid and te_valid:
        p_vis = (0.40 * p_sp + 0.30 * p_te) / 0.70
    elif sp_valid:
        p_vis = p_sp
    elif te_valid:
        p_vis = p_te
    else:
        p_vis = None

    # Overall fusion
    if p_vis is not None and au_valid:
        p_fake = float(1.0 - (1.0 - p_vis) * (1.0 - p_au))
    elif p_vis is not None:
        p_fake = float(p_vis)
    elif au_valid:
        p_fake = float(p_au)
    else:
        p_fake = 0.0

    contributing = []
    unavailable = []
    if sp_valid:
        contributing.append("Spatial Analysis")
    else:
        unavailable.append("Spatial Analysis")
    if te_valid:
        contributing.append("Temporal Analysis")
    else:
        unavailable.append("Temporal Analysis")
    if au_valid:
        contributing.append("Audio Analysis")
    else:
        unavailable.append("Audio Analysis")
    # Context is explanation-only (mock_agents.py), so it is never listed as contributing.

    if not contributing:
        details = enrich_decision_fields(
            {
                "method": "corrected_fixed",
                "verdict": "UNVERIFIED MODALITY",
                "p_fake": 0.0,
                "overall_confidence": 0.0,
                "predicted_category": "UNVERIFIED",
                "category_probabilities": {c: 0.0 for c in CATEGORIES},
                "contributing_agents": contributing,
                "unavailable_agents": unavailable,
            },
            probabilities_are_calibrated=False,
        )
        return 0.0, "UNVERIFIED MODALITY", contributing, unavailable, details

    overall = round(p_fake, 4)
    if overall > 0.55:
        verdict = "DEEPFAKE"
    elif overall < 0.45:
        verdict = "AUTHENTIC"
    else:
        verdict = "UNCERTAIN / INSUFFICIENT EVIDENCE"

    # Independent joint category decomposition (model-implied, not calibrated)
    category_probs = np.zeros(4, dtype=np.float32)
    if p_vis is not None and au_valid and p_au is not None:
        pv = float(np.clip(p_vis, 0.0, 1.0))
        pa = float(np.clip(p_au, 0.0, 1.0))
        category_probs[0] = (1.0 - pv) * (1.0 - pa)
        category_probs[1] = (1.0 - pv) * pa
        category_probs[2] = pv * (1.0 - pa)
        category_probs[3] = pv * pa
    elif p_vis is not None:
        pv = float(np.clip(p_vis, 0.0, 1.0))
        category_probs[0] = 1.0 - pv
        category_probs[2] = pv
    elif au_valid and p_au is not None:
        pa = float(np.clip(p_au, 0.0, 1.0))
        category_probs[0] = 1.0 - pa
        category_probs[1] = pa

    if verdict == "DEEPFAKE":
        manip_indices = [1, 2, 3]
        best_manip_idx = manip_indices[int(np.argmax(category_probs[manip_indices]))]
        predicted_category = IDX_TO_CATEGORY[best_manip_idx]
    elif verdict == "AUTHENTIC":
        predicted_category = "AUTHENTIC"
    else:
        # Tentative leading hypothesis under abstention
        if float(category_probs[0]) >= 0.50:
            predicted_category = "AUTHENTIC"
        else:
            manip_indices = [1, 2, 3]
            best_manip_idx = manip_indices[int(np.argmax(category_probs[manip_indices]))]
            predicted_category = IDX_TO_CATEGORY[best_manip_idx]

    details = {
        "method": "corrected_fixed",
        "verdict": verdict,
        "p_visual": p_vis,
        "p_audio": p_au,
        "p_fake": p_fake,
        "overall_confidence": overall,
        "contributing_agents": contributing,
        "unavailable_agents": unavailable,
        "category_probabilities": {
            cat: round(float(category_probs[idx]), 4)
            for idx, cat in enumerate(CATEGORIES)
        },
        "predicted_category": predicted_category,
        "decision_threshold": 0.50,
    }
    details = enrich_decision_fields(
        details,
        tau=0.50,
        deadband=DEFAULT_DEADBAND,
        probabilities_are_calibrated=False,
    )
    return overall, verdict, contributing, unavailable, details


# ---------------------------------------------------------------------------
# Legacy Fixed-Weight Fusion (Reference Baseline)
# ---------------------------------------------------------------------------
_LEGACY_WEIGHTS = {
    "Spatial Analysis": 0.40,
    "Temporal Analysis": 0.30,
    "Audio Analysis": 0.20,
    "Context Verification": 0.10,
    "spatial": 0.40,
    "temporal": 0.30,
    "audio": 0.20,
    "context": 0.10,
}


def compute_verdict_legacy(
    agents: List[Dict[str, Any]],
) -> Tuple[float, str, List[str], List[str], Dict[str, Any]]:
    """Original fixed-weight fusion with dynamic denominator."""
    def _agent_key(a: dict) -> str:
        return a.get("agent") or a.get("agent_name", "")

    contributing_agents = []
    unavailable_agents = []
    active_weights_sum = 0.0
    weighted_score_sum = 0.0

    for a in agents:
        agent_name = _agent_key(a)
        is_unavailable = (
            a.get("is_fake") is None
            or a.get("status") in (
                "not_applicable",
                "unavailable",
                "no_audio",
                "untrained",
                "error",
                "failed",
            )
        )
        if not is_unavailable:
            contributing_agents.append(agent_name)
            weight = _LEGACY_WEIGHTS.get(agent_name, 0.0)
            active_weights_sum += weight
            weighted_score_sum += float(a.get("confidence", 0.0)) * weight
        else:
            unavailable_agents.append(agent_name)

    if active_weights_sum == 0.0:
        return 0.0, "UNVERIFIED MODALITY", contributing_agents, unavailable_agents, {"method": "legacy"}

    overall = round(weighted_score_sum / active_weights_sum, 4)
    if overall > 0.55:
        verdict = "DEEPFAKE"
    elif overall < 0.45:
        verdict = "AUTHENTIC"
    else:
        verdict = "UNCERTAIN / INSUFFICIENT EVIDENCE"

    details = {
        "method": "legacy",
        "weighted_score": overall,
        "active_weights_sum": active_weights_sum,
    }
    return overall, verdict, contributing_agents, unavailable_agents, details


# Module singleton for production inference
_GLOBAL_POLICY: Optional[PPOFusionPolicy] = None


def get_fusion_policy() -> PPOFusionPolicy:
    global _GLOBAL_POLICY
    if _GLOBAL_POLICY is None:
        _GLOBAL_POLICY = PPOFusionPolicy()
    return _GLOBAL_POLICY
