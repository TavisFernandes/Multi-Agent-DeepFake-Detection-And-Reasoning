"""
wav_cache.py: two category folders with the same identity and stem get separate, correct WAVs, and a cached WAV
is never reused after its source, its bytes or the preprocessing change.
Run: python -m pytest test_wav_cache.py
"""
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "training"))

import wav_cache  # noqa: E402

REL_FAKE = r"FakeVideo-FakeAudio\African\men\id00987\00160_id00476_wavtolip.mp4"
REL_REAL = r"FakeVideo-RealAudio\African\men\id00987\00160_id00476_wavtolip.mp4"


def make_clip(path, freq):
    """2 s black video + sine tone at `freq` Hz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([wav_cache._FFMPEG, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=black:s=64x64:d=2",
                    "-f", "lavfi", "-i", f"sine=frequency={freq}:duration=2", "-shortest", "-c:v", "libx264",
                    "-c:a", "aac", str(path)], check=True, timeout=60)


def dominant_hz(wav_path):
    with wave.open(wav_path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
        rate = w.getframerate()
    return np.argmax(np.abs(np.fft.rfft(x))) * rate / len(x)


@pytest.fixture
def clips(tmp_path):
    root = tmp_path / "data"
    make_clip(root / REL_FAKE, 440)
    make_clip(root / REL_REAL, 1000)
    return root, tmp_path / "cache"


def test_same_identity_and_stem_in_two_categories(clips):
    root, cache = clips
    old_key = lambda rel: f"{Path(rel).parent.name}_{Path(rel).stem}"  # previous naming
    assert old_key(REL_FAKE) == old_key(REL_REAL)
    assert wav_cache.cache_key(REL_FAKE) != wav_cache.cache_key(REL_REAL)
    assert wav_cache.cache_key(REL_FAKE).startswith("FakeVideo-FakeAudio__id00987__")

    fake_wav, fake_prov = wav_cache.get_wav(str(root / REL_FAKE), REL_FAKE, cache)
    real_wav, real_prov = wav_cache.get_wav(str(root / REL_REAL), REL_REAL, cache)
    assert fake_wav != real_wav and fake_prov["wav_sha256"] != real_prov["wav_sha256"]
    assert abs(dominant_hz(fake_wav) - 440) < 5 and abs(dominant_hz(real_wav) - 1000) < 5
    assert fake_prov["source_sha256"] == wav_cache.file_sha256(root / REL_FAKE)
    assert fake_prov["rel_path"] == REL_FAKE and fake_prov["preprocessing"] == wav_cache.PREPROCESSING
    # Reuse returns the same validated entry without re-extracting.
    mtime = Path(fake_wav).stat().st_mtime_ns
    assert wav_cache.get_wav(str(root / REL_FAKE), REL_FAKE, cache) == (fake_wav, fake_prov)
    assert Path(fake_wav).stat().st_mtime_ns == mtime


def test_stale_entries_are_re_extracted(clips, monkeypatch):
    root, cache = clips
    src = root / REL_FAKE
    wav, prov = wav_cache.get_wav(str(src), REL_FAKE, cache)

    make_clip(src, 2000)  # source replaced at the same path
    wav, prov2 = wav_cache.get_wav(str(src), REL_FAKE, cache)
    assert prov2["source_sha256"] != prov["source_sha256"] and abs(dominant_hz(wav) - 2000) < 5

    Path(wav).write_bytes(Path(wav).read_bytes()[:-100])  # cached WAV damaged
    wav, prov3 = wav_cache.get_wav(str(src), REL_FAKE, cache)
    assert prov3["wav_sha256"] == wav_cache.file_sha256(wav) == prov2["wav_sha256"]

    monkeypatch.setattr(wav_cache, "PREPROCESSING", {**wav_cache.PREPROCESSING, "version": 99})
    _, prov4 = wav_cache.get_wav(str(src), REL_FAKE, cache)
    assert prov4["preprocessing"]["version"] == 99
