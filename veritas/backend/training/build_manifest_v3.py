"""
build_manifest_v3.py
====================
Derives leakage_controlled_split_manifest_v3.json from v2 so no media file gets double weight within a split:
  - repeated_row:       the same path listed more than once in a split (meta_data.csv lists 11 FakeVideo-FakeAudio
                        files twice, once per `method`); the first occurrence is kept, later rows are dropped.
  - byte_identical_path: distinct paths with byte-identical media and the same label in one split; the
                        lexicographically smallest path is kept. A label conflict stops the build for review.
Only byte identity counts as a duplicate (no similarity matching). v1, v2 and the source videos are only read.

Deterministic: same ordering as v2, json indent=1, ASCII, LF line endings (pinned in .gitattributes).

Outputs:
  leakage_controlled_split_manifest_v3.json      eligible model-development clips
  dataset_audit/manifest_v3_exclusions.json      one auditable record per dropped row / excluded path

Run from veritas/backend/training/:  python build_manifest_v3.py
"""

import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from build_manifest_v2 import V2_PATH, clip_rows, digests, duplicate_groups

_HERE = Path(__file__).resolve().parent
V3_PATH = _HERE / "leakage_controlled_split_manifest_v3.json"
EXCLUSIONS_PATH = _HERE / "dataset_audit" / "manifest_v3_exclusions.json"
EXPECTED_REPEATED_ROWS = 11  # 8 detector_train, 2 fusion_train, 1 fusion_val (audit of v2)


def write_lf(path, obj):
    path.write_bytes((json.dumps(obj, indent=1) + "\n").encode("ascii"))


def main(v3_path=V3_PATH, exclusions_path=EXCLUSIONS_PATH):
    v2_bytes = V2_PATH.read_bytes()
    v2 = json.loads(v2_bytes)
    root = Path(v2["metadata"]["dataset_root"])
    v3 = copy.deepcopy(v2)
    exclusions = []

    def record(split, cat, row, dup_type, canonical, reason):
        md5, sha = digests(root / row["rel_path"])
        exclusions.append({"rel_path": row["rel_path"], "sha256": sha, "md5": md5, "split": split, "label": cat,
                           "duplicate_type": dup_type, "canonical_rel_path": canonical, "removed_row": row,
                           "reason": reason})

    # 1. Repeated rows: keep the first occurrence of each path per split.
    for split, cats in v3["clips"].items():
        first = {}
        for cat in cats:
            kept = []
            for row in cats[cat]:
                rel = row["rel_path"]
                if rel not in first:
                    first[rel] = cat
                    kept.append(row)
                    continue
                assert first[rel] == cat, f"{rel} listed under two labels in {split}: {first[rel]}, {cat}"
                record(split, cat, row, "repeated_row", rel,
                       "same file listed again in the same split (meta_data.csv has one row per method); "
                       "one occurrence kept, label unchanged")
            cats[cat] = kept

    # 2. Distinct paths with byte-identical media in one split: keep the smallest path.
    where = {c["rel_path"]: (s, cat) for s, cat, c in clip_rows(v3)}
    for sha, members in sorted(duplicate_groups(root).items()):
        listed = sorted(r for r, _ in members if r in where)
        if len(listed) < 2:
            continue
        assert len({where[r] for r in listed}) == 1, f"byte-identical files with different split/label: {listed}"
        split, cat = where[listed[0]]
        for rel in listed[1:]:
            row = next(c for c in v3["clips"][split][cat] if c["rel_path"] == rel)
            v3["clips"][split][cat].remove(row)
            record(split, cat, row, "byte_identical_path", listed[0],
                   "byte-identical to the canonical path, same label and split; canonical kept, label unchanged")

    repeated = sum(e["duplicate_type"] == "repeated_row" for e in exclusions)
    assert repeated == EXPECTED_REPEATED_ROWS, f"found {repeated} repeated rows, expected {EXPECTED_REPEATED_ROWS}"

    for split, cats in v3["clips"].items():
        paths = [c["rel_path"] for cl in cats.values() for c in cl]
        assert len(paths) == len(set(paths)), f"{split} still has repeated paths"
        v3["summary"][split] = {"total_clips": len(paths), "unique_files": len(set(paths)),
                                "by_category": {cat: len(l) for cat, l in cats.items()}}

    v3["metadata"].update({
        "manifest_name": "leakage_controlled_split_manifest_v3",
        "manifest_version": 3,
        "derived_from": {"file": V2_PATH.name, "sha256": hashlib.sha256(v2_bytes).hexdigest(),
                         "v1": v2["metadata"]["derived_from"]},
        "exclusions_file": "dataset_audit/manifest_v3_exclusions.json",
        "v2_exclusions_file": v2["metadata"]["exclusions_file"],
        "removed_in_v3_by_split": dict(sorted(Counter(e["split"] for e in exclusions).items())),
    })
    for k in ("excluded_clips_count", "excluded_by_split"):
        v3["metadata"]["v2_" + k] = v3["metadata"].pop(k)

    write_lf(exclusions_path, {
        "manifest": v3["metadata"]["manifest_name"],
        "derived_from": v3["metadata"]["derived_from"],
        "dataset_root": str(root),
        "exclusions": exclusions,
    })
    write_lf(v3_path, v3)

    print(f"removed rows: {len(exclusions)}  {Counter(e['duplicate_type'] for e in exclusions)}  "
          f"{v3['metadata']['removed_in_v3_by_split']}")
    for split, s in v3["summary"].items():
        print(f"  {split:15s} {s['total_clips']:5d} clips (v2 {v2['summary'][split]['total_clips']})  {s['by_category']}")


if __name__ == "__main__":
    main()
