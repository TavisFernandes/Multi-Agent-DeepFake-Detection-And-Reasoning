"""
cache/fusion_features_lc_v4.json: exactly the v3 fusion_train/fusion_val clips (no final_test), and every audio
score traces to a WAV decoded from that record's own source file.
Run: python -m pytest test_feature_cache_v4.py   (skips when the local cache or dataset is absent)
"""
import json
import sys
from pathlib import Path

import pytest

_TRAINING = Path(__file__).resolve().parent / "training"
sys.path.insert(0, str(_TRAINING))

import build_manifest_v3  # noqa: E402
import wav_cache  # noqa: E402

CACHE = _TRAINING / "cache" / "fusion_features_lc_v4.json"
pytestmark = pytest.mark.skipif(not CACHE.exists() or not wav_cache.CACHE_DIR.exists(),
                                reason="local feature cache / WAV cache not on this machine")


@pytest.fixture(scope="module")
def cache():
    return json.loads(CACHE.read_bytes())


def test_records_are_v3_train_val_only(cache):
    v3 = json.loads(build_manifest_v3.V3_PATH.read_bytes())
    expected = {c["full_path"]: s for s in ("fusion_train", "fusion_val") for l in v3["clips"][s].values() for c in l}
    assert {fp: r["split"] for fp, r in cache["records"].items()} == expected
    prov = cache["metadata"]["provenance"]
    assert prov["manifest"]["sha256"] == wav_cache.file_sha256(build_manifest_v3.V3_PATH)
    assert prov["wav_preprocessing"] == wav_cache.PREPROCESSING
    for name, sha in prov["checkpoints_sha256"].items():
        assert wav_cache.file_sha256(_TRAINING.parent / "models" / name) == sha


def test_each_audio_score_comes_from_its_own_file(cache):
    wavs = set()
    for fp, r in cache["records"].items():
        assert r["source_sha256"] == wav_cache.file_sha256(fp)
        audio = next(a for a in r["agents"] if a["agent"] == "Audio Analysis")
        if audio["status"] == "no_audio":
            continue
        wp = audio["wav_provenance"]
        assert wp["rel_path"] == r["rel_path"] and wp["source_sha256"] == r["source_sha256"]
        wav = wav_cache.CACHE_DIR / f"{wav_cache.cache_key(r['rel_path'])}.wav"
        assert wav_cache.file_sha256(wav) == wp["wav_sha256"]
        wavs.add(wav)
    assert len(wavs) == len(cache["records"])  # no two records share a WAV
