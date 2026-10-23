# These test_*.py files are standalone scripts (module-level checks, sys.exit) — run them with
# `python <file>`; collecting them under pytest executes and exits mid-collection.
collect_ignore = [
    "smoke_test.py",  # needs the backend running on :8000
    "test_audio_agent_production_integration.py",
    "test_audio_agent_readiness.py",
    "test_audio_routing.py",
    "test_context_agent.py",
    "test_fusion.py",
    "test_integration_routing_fusion.py",
    "test_ppo_fusion.py",
    "test_score_pipeline.py",
    "test_spatial.py",
    "test_temporal.py",
    "test_video_routing.py",
]
