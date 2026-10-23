"""
train_temporal_lc.get_cached_or_sample_temporal: namesakes in two category folders get separate frame-cache
entries, and an entry is rebuilt when its source file changes. Run: python -m pytest test_temporal_cache.py
"""
import json

from test_wav_cache import REL_FAKE, REL_REAL, clips, make_clip  # noqa: F401  (clips is a fixture; puts training/ on sys.path)
import wav_cache  # noqa: E402


def test_temporal_frame_cache_namesakes(clips, monkeypatch):
    import train_temporal_lc
    root, cache = clips
    monkeypatch.setattr(train_temporal_lc, "TEMPORAL_CACHE_DIR", cache)
    for rel in (REL_FAKE, REL_REAL):
        assert train_temporal_lc.get_cached_or_sample_temporal(str(root / rel), rel).shape == (16, 112, 112, 3)
    sidecars = {json.loads(p.read_text())["rel_path"] for p in cache.glob("*.json")}
    assert sidecars == {REL_FAKE, REL_REAL} and len(list(cache.glob("*.npy"))) == 2
    make_clip(root / REL_FAKE, 2000)  # source changed at the same path -> sidecar no longer matches
    train_temporal_lc.get_cached_or_sample_temporal(str(root / REL_FAKE), REL_FAKE)
    side = json.loads(next(p for p in cache.glob("FakeVideo-FakeAudio__*.json")).read_text())
    assert side["source_sha256"] == wav_cache.file_sha256(root / REL_FAKE)
