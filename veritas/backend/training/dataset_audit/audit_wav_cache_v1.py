"""
Read-only audit of the old WAV cache (D:\\VERITAS_TRAINING\\wav_cache, `{identity}_{stem}.wav`, shared by
train_audio_lc.py and extract_fusion_features_lc_v3.py). For every cache name claimed by more than one v1 manifest
file, decode each claimant fresh and record which file the cached PCM actually came from.
Writes wav_cache_v1_audit.json next to this script. Run: python audit_wav_cache_v1.py
"""
import json
import subprocess
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
import wav_cache  # noqa: E402

OLD_CACHE = Path(r"D:\VERITAS_TRAINING\wav_cache")
V1 = json.loads((_HERE.parent / "leakage_controlled_split_manifest.json").read_bytes())


def pcm(mp4):
    return subprocess.run([wav_cache._FFMPEG, "-nostdin", "-loglevel", "error", "-i", mp4, "-vn", "-f", "s16le",
                           "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-"], capture_output=True,
                          check=True, timeout=60).stdout


claimants = defaultdict(dict)  # old cache name -> {rel_path: (split, full_path)}
for split, cats in V1["clips"].items():
    for clist in cats.values():
        for c in clist:
            p = PureWindowsPath(c["rel_path"])
            claimants[f"{p.parent.name}_{p.stem}.wav"][c["rel_path"]] = (split, c["full_path"])

rows = []
for name, files in sorted(claimants.items()):
    if len(files) < 2:
        continue
    cached = OLD_CACHE / name
    with wave.open(str(cached)) as w:
        data = w.readframes(w.getnframes())
    owners = [rel for rel, (_, fp) in files.items() if pcm(fp) == data]
    for rel, (split, _) in sorted(files.items()):
        rows.append({"cache_name": name, "rel_path": rel, "split": split, "cached_wav_is_own_audio": rel in owners,
                     "cached_wav_owner": owners})

wrong = [r for r in rows if not r["cached_wav_is_own_audio"]]
summary = {"colliding_cache_names": len({r["cache_name"] for r in rows}), "files_sharing_a_name": len(rows),
           "files_given_another_files_audio": len(wrong),
           "wrong_audio_by_split": dict(sorted(Counter(r["split"] for r in wrong).items())),
           "wrong_audio_by_category": dict(sorted(Counter(r["rel_path"].split("\\")[0] for r in wrong).items())),
           "names_matching_no_claimant": sorted({r["cache_name"] for r in rows if not r["cached_wav_owner"]})}
(_HERE / "wav_cache_v1_audit.json").write_bytes((json.dumps({"summary": summary, "files": rows}, indent=1) + "\n").encode())
print(json.dumps(summary, indent=1))
