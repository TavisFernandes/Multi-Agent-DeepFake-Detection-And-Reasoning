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


def test_ppo_mode_uses_the_v4_policy_with_its_training_detectors(monkeypatch):
    import main
    monkeypatch.setattr(fusion_policy, "_GLOBAL_POLICY", None)
    policy = fusion_policy.get_fusion_policy()
    assert policy.trained_on == {"spatial": "spatial_agent_lc.pth", "temporal": "temporal_agent_lc.pth",
                                 "audio": "audio_agent_lc.pth"}
    monkeypatch.setattr(main, "_loaded_models", lambda: dict(policy.trained_on))
    *_, details = _compute_verdict(AGENTS, "video", return_details=True, mode_override="ppo")
    assert details["method"] == "ppo"
    assert policy.checkpoint_path == V4.resolve()


def test_ppo_mode_falls_back_when_other_detectors_are_loaded(monkeypatch):
    import main
    from main import _decision_payload
    monkeypatch.setattr(fusion_policy, "_GLOBAL_POLICY", None)
    monkeypatch.setattr(main, "_loaded_models", lambda: {"spatial": "spatial_agent_lc_v2_aug_s0.pth",
                                                         "temporal": "temporal_agent_lc.pth",
                                                         "audio": "audio_agent_lc_v2_aug_s1.pth"})
    *_, details = _compute_verdict(AGENTS, "video", return_details=True, mode_override="ppo")
    assert details["method"] == "corrected_fixed"
    assert "PPO policy was trained on detectors" in _decision_payload(details)["fusion_fallback_reason"]


def test_missing_v4_is_an_error_not_a_silent_fallback_to_v2(monkeypatch, tmp_path):
    assert fusion_policy.DEFAULT_V2_CHECKPOINT_PATH.exists()  # an older policy is available to fall back to...
    monkeypatch.setattr(fusion_policy, "DEFAULT_V4_CHECKPOINT_PATH", tmp_path / "missing_v4.pth")
    policy = fusion_policy.PPOFusionPolicy(device=torch.device("cpu"))
    assert not policy.is_loaded and policy.checkpoint_path is None  # ...but it must not be used
    assert policy.load_error.startswith("FileNotFoundError")


def test_corrupt_checkpoint_reports_the_error(tmp_path):
    bad = tmp_path / "corrupt.pth"
    bad.write_bytes(b"not a checkpoint")
    policy = fusion_policy.PPOFusionPolicy(checkpoint_path=bad, device=torch.device("cpu"))
    assert not policy.is_loaded and policy.load_error


def test_ppo_mode_reports_the_fallback_when_the_policy_cannot_load(monkeypatch, tmp_path):
    from main import _decision_payload
    monkeypatch.setattr(fusion_policy, "DEFAULT_V4_CHECKPOINT_PATH", tmp_path / "missing_v4.pth")
    monkeypatch.setattr(fusion_policy, "_GLOBAL_POLICY", None)
    *_, details = _compute_verdict(AGENTS, "video", return_details=True, mode_override="ppo")
    assert details["method"] == "corrected_fixed"
    assert "PPO policy not loaded (FileNotFoundError" in _decision_payload(details)["fusion_fallback_reason"]


def test_production_default_is_still_corrected_fixed(monkeypatch):
    monkeypatch.delenv("VERITAS_FUSION_MODE", raising=False)
    *_, details = _compute_verdict(AGENTS, "video", return_details=True)
    assert details["method"] == "corrected_fixed"
