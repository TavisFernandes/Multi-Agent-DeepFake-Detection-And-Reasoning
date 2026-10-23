"""
wav_cache.py
============
Collision-free 16 kHz mono WAV cache shared by train_audio_lc.py and extract_fusion_features_lc_v3.py.

The previous cache (D:\\VERITAS_TRAINING\\wav_cache, named `{identity}_{stem}.wav`) dropped the category folder,
so e.g. FakeVideo-FakeAudio\\...\\id00987\\00160_id00476_wavtolip.mp4 and its FakeVideo-RealAudio namesake shared
one WAV. That directory is left untouched for audit; this cache lives in its own versioned directory.

Each WAV is keyed by its dataset-relative path (category kept in the name) and has a JSON sidecar recording the
source path, source SHA-256, preprocessing and WAV SHA-256. A WAV is reused only if all of them still match;
otherwise it is re-extracted.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path, PureWindowsPath
from typing import Dict, Optional, Tuple

import imageio_ffmpeg

CACHE_DIR = Path(r"D:\VERITAS_TRAINING\wav_cache_v2")
PREPROCESSING = {"version": 2, "codec": "pcm_s16le", "sample_rate": 16000, "channels": 1}
_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cache_key(rel_path: str) -> str:
    """`{category}__{identity}__{stem}__{sha256(rel_path)[:16]}`: readable, and unique per relative path."""
    p = PureWindowsPath(rel_path)
    return f"{p.parts[0]}__{p.parent.name}__{p.stem}__{hashlib.sha256(p.as_posix().encode()).hexdigest()[:16]}"


def get_wav(full_path: str, rel_path: str, cache_dir: Path = CACHE_DIR) -> Optional[Tuple[str, Dict]]:
    """(wav_path, provenance) for this source file, or None if it has no decodable audio."""
    source_sha = file_sha256(full_path)
    wav = cache_dir / f"{cache_key(rel_path)}.wav"
    sidecar = wav.with_suffix(".json")
    try:
        prov = json.loads(sidecar.read_text(encoding="utf-8"))
        if (prov["rel_path"] == rel_path and prov["source_sha256"] == source_sha
                and prov["preprocessing"] == PREPROCESSING and prov["wav_sha256"] == file_sha256(wav)):
            return str(wav), prov
    except (OSError, ValueError, KeyError):
        pass  # missing, unreadable or stale: re-extract below

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = wav.with_name(wav.stem + ".partial.wav")  # never leave a half-written WAV under the real name
    cmd = [_FFMPEG, "-y", "-nostdin", "-loglevel", "error", "-i", full_path, "-vn",
           "-acodec", PREPROCESSING["codec"], "-ar", str(PREPROCESSING["sample_rate"]),
           "-ac", str(PREPROCESSING["channels"]), str(tmp)]
    try:
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
    except subprocess.TimeoutExpired:
        res = None
    if res is None or res.returncode != 0 or not tmp.exists() or tmp.stat().st_size <= 1000:
        tmp.unlink(missing_ok=True)
        return None  # same "no audio" rule as the previous cache (<= 1000 bytes)
    os.replace(tmp, wav)
    prov = {"rel_path": rel_path, "source_sha256": source_sha, "preprocessing": PREPROCESSING,
            "wav_sha256": file_sha256(wav)}
    sidecar.write_text(json.dumps(prov, indent=1), encoding="utf-8")
    return str(wav), prov
