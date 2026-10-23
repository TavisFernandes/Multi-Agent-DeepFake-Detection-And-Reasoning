"""
leakage_controlled_split_manifest_v2.json: the 26 mislabelled whole-file copies are not eligible training or
validation examples, v1 is untouched, and every train/val loader reads v2.
Run: python -m pytest test_manifest_v2.py   (hash checks skip when the dataset is not on disk)
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

_TRAINING = Path(__file__).resolve().parent / "training"
sys.path.insert(0, str(_TRAINING))

import build_manifest_v2 as b  # noqa: E402

V1_SHA256 = "86f84ce7948f9e0d1a11b4b90d8cccb00e2a08456abf88227c82c7f9aba932d4"  # independent_evaluation_protocol.md
V1 = json.loads(b.V1_PATH.read_text(encoding="utf-8"))
V2 = json.loads(b.V2_PATH.read_text(encoding="utf-8"))
EXCL = json.loads(b.EXCLUSIONS_PATH.read_text(encoding="utf-8"))["exclusions"]
EXCLUDED = {e["rel_path"] for e in EXCL}


def rows(m):
    return [(s, cat, c["rel_path"]) for s, cat, c in b.clip_rows(m)]


def test_v1_unchanged():
    assert hashlib.sha256(b.V1_PATH.read_bytes()).hexdigest() == V1_SHA256
    assert V2["metadata"]["derived_from"]["sha256"] == V1_SHA256


def test_exclusion_records():
    assert len(EXCL) == len(EXCLUDED) == 26
    assert V2["metadata"]["excluded_by_split"] == {"detector_train": 14, "detector_val": 2, "final_test": 1,
                                                   "fusion_train": 5, "fusion_val": 4}
    v1 = set(rows(V1))
    for e in EXCL:
        assert e["original_label"] != b.AUTHENTIC and e["reason"] and len(e["sha256"]) == 64
        assert (e["split"], e["original_label"], e["rel_path"]) in v1
        # Not relabelled: the authentic original stays, once, in the same split.
        assert (e["split"], b.AUTHENTIC, e["authentic_source_rel_path"]) in set(rows(V2))


def test_v2_is_v1_minus_exclusions():
    assert rows(V2) == [r for r in rows(V1) if r[2] not in EXCLUDED]
    for split, s in V2["summary"].items():
        assert s["total_clips"] == sum(map(len, V2["clips"][split].values()))
    assert V2["identities"] == V1["identities"]


def test_train_val_loaders_use_v2():
    import train_audio_lc, train_spatial_lc, train_temporal_lc, extract_fusion_features_lc_v3, train_ppo_v3
    for mod in (train_audio_lc, train_spatial_lc, train_temporal_lc, extract_fusion_features_lc_v3, train_ppo_v3):
        assert Path(mod.MANIFEST_PATH).resolve() == b.V2_PATH, mod.__name__


def test_ppo_drops_excluded_cache_records():
    from train_ppo_v3 import eligible_records
    by_rel = {c["rel_path"]: (s, c["full_path"]) for s, _, c in b.clip_rows(V1)}
    records = {fp: {"split": s, "full_path": fp, "rel_path": rel} for rel, (s, fp) in by_rel.items()}
    for split in ("fusion_train", "fusion_val"):
        kept = {r["rel_path"] for r in eligible_records(records, split)}
        assert kept and not kept & EXCLUDED
        assert {r for s, _, r in rows(V2) if s == split} == kept


@pytest.mark.skipif(not Path(V2["metadata"]["dataset_root"]).exists(), reason="FakeAVCeleb dataset not on disk")
def test_excluded_files_are_byte_identical_to_authentic_source():
    root = Path(V2["metadata"]["dataset_root"])
    for e in EXCL:
        md5, sha = b.digests(root / e["rel_path"])
        assert (md5, sha) == (e["md5"], e["sha256"]) == b.digests(root / e["authentic_source_rel_path"])
