"""
`ppo` mode must load ppo_fusion_policy_v4.pth (not v2/v3), and production must still default to corrected_fixed.
Run: python -m pytest test_ppo_v4_loading.py
"""
import torch

import fusion_policy
from main import _compute_verdict

V4 = fusion_policy.DEFAULT_V4_CHECKPOINT_PATH
AGENTS = [
    {"agent": "Spatial Analysis", "confidence": 0.9, "is_fake": True, "status": "trained"},
    {"agent": "Temporal Analysis", "confidence": 0.8, "is_fake": True, "status": "trained"},
    {"agent": "Audio Analysis", "confidence": 0.1, "is_fake": False, "status": "trained", "has_audio": True, "duration": 6.0},
]


def test_default_policy_loads_the_v4_file():
    policy = fusion_policy.PPOFusionPolicy(device=torch.device("cpu"))
    assert policy.is_loaded
    assert policy.checkpoint_path == V4.resolve()
    expected = torch.load(V4, map_location="cpu", weights_only=False)["model_state_dict"]
    for name, tensor in policy.network.state_dict().items():
        assert torch.equal(tensor.cpu(), expected[name]), name


def test_ppo_mode_uses_the_v4_policy(monkeypatch):
    monkeypatch.setattr(fusion_policy, "_GLOBAL_POLICY", None)
    *_, details = _compute_verdict(AGENTS, "video", return_details=True, mode_override="ppo")
    assert details["method"] == "ppo"
    assert fusion_policy.get_fusion_policy().checkpoint_path == V4.resolve()


def test_production_default_is_still_corrected_fixed(monkeypatch):
    monkeypatch.delenv("VERITAS_FUSION_MODE", raising=False)
    *_, details = _compute_verdict(AGENTS, "video", return_details=True)
    assert details["method"] == "corrected_fixed"
