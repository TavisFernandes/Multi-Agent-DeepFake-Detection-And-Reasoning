"""
leakage_controlled_split_manifest_v3.json: no file counted twice, every removal recorded, labels and split
separation preserved, v1/v2 untouched, loaders read v3, and the builder is byte-for-byte repeatable.
Run: python -m pytest test_manifest_v3.py   (generation check skips when the dataset is not on disk)
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path, PureWindowsPath

import pytest

_TRAINING = Path(__file__).resolve().parent / "training"
sys.path.insert(0, str(_TRAINING))

import build_manifest_v2 as b2  # noqa: E402
import build_manifest_v3 as b3  # noqa: E402

V2_SHA256 = "86db55017d1c8512cdc32c69cea268d82fd85fe5378075106ccfc25ed9e7316c"  # as committed in f6ee889
V2 = json.loads(b2.V2_PATH.read_bytes())
V3 = json.loads(b3.V3_PATH.read_bytes())
EXCL = json.loads(b3.EXCLUSIONS_PATH.read_bytes())["exclusions"]


def rows(m):
    return [(s, cat, c) for s, cat, c in b2.clip_rows(m)]


def test_v2_unchanged_and_recorded():
    assert hashlib.sha256(b2.V2_PATH.read_bytes()).hexdigest() == V2_SHA256
    assert V3["metadata"]["derived_from"]["sha256"] == V2_SHA256


def test_each_file_listed_once():
    paths = [c["rel_path"] for _, _, c in rows(V3)]
    assert len(paths) == len(set(paths))  # within and across splits
    for split, s in V3["summary"].items():
        assert s["total_clips"] == s["unique_files"] == sum(map(len, V3["clips"][split].values()))


def test_exclusion_records():
    assert Counter(e["duplicate_type"] for e in EXCL) == {"repeated_row": 11}
    assert Counter(e["split"] for e in EXCL) == {"detector_train": 8, "fusion_train": 2, "fusion_val": 1}
    v2_rows = {(s, cat, json.dumps(c, sort_keys=True)) for s, cat, c in rows(V2)}
    v3_where = {c["rel_path"]: (s, cat) for s, cat, c in rows(V3)}
    for e in EXCL:
        assert len(e["sha256"]) == 64 and len(e["md5"]) == 32 and e["reason"]
        assert (e["split"], e["label"], json.dumps(e["removed_row"], sort_keys=True)) in v2_rows
        assert v3_where[e["canonical_rel_path"]] == (e["split"], e["label"])


def test_v3_is_v2_with_first_occurrence_kept():
    seen, expected = set(), []
    for s, cat, c in rows(V2):
        if (s, c["rel_path"]) not in seen:
            seen.add((s, c["rel_path"]))
            expected.append((s, cat, c))
    assert rows(V3) == expected


def test_labels_and_split_separation_preserved():
    v2_label = {(s, c["rel_path"]): cat for s, cat, c in rows(V2)}
    assert {(s, c["rel_path"]): cat for s, cat, c in rows(V3)} == v2_label  # same files, same labels
    assert V3["identities"] == V2["identities"]
    owner = {i: s for s, ids in V3["identities"].items() for i in ids}
    for s, _, c in rows(V3):
        assert owner[PureWindowsPath(c["rel_path"]).parts[3]] == s
        for donor in (c["target1"], c["target2"]):
            assert owner.get(donor, s) == s, f"donor {donor} of {c['rel_path']} is in another split"


def test_train_val_loaders_use_v3():
    import train_audio_lc, train_spatial_lc, train_temporal_lc, extract_fusion_features_lc_v3, train_ppo_v3
    for mod in (train_audio_lc, train_spatial_lc, train_temporal_lc, extract_fusion_features_lc_v3, train_ppo_v3):
        assert Path(mod.MANIFEST_PATH).resolve() == b3.V3_PATH, mod.__name__


def test_ppo_keeps_only_v3_records():
    from train_ppo_v3 import eligible_records
    v1 = json.loads(b2.V1_PATH.read_bytes())
    records = {c["full_path"]: {"split": s, "full_path": c["full_path"], "rel_path": c["rel_path"]}
               for s, _, c in rows(v1)}
    v2_excluded = {e["rel_path"] for e in json.loads(b2.EXCLUSIONS_PATH.read_bytes())["exclusions"]}
    for split in ("fusion_train", "fusion_val"):
        kept = [r["rel_path"] for r in eligible_records(records, split)]
        assert sorted(kept) == sorted(c["rel_path"] for s, _, c in rows(V3) if s == split)
        assert not set(kept) & v2_excluded


def test_v3_written_lf():
    assert b"\r" not in b3.V3_PATH.read_bytes() and b"\r" not in b3.EXCLUSIONS_PATH.read_bytes()


@pytest.mark.skipif(not Path(V3["metadata"]["dataset_root"]).exists(), reason="FakeAVCeleb dataset not on disk")
def test_builder_is_repeatable(tmp_path):
    b3.main(tmp_path / "v3.json", tmp_path / "excl.json")
    assert (tmp_path / "v3.json").read_bytes() == b3.V3_PATH.read_bytes()
    assert (tmp_path / "excl.json").read_bytes() == b3.EXCLUSIONS_PATH.read_bytes()
