"""
build_manifest_v2.py
====================
Derives leakage_controlled_split_manifest_v2.json from v1 by excluding manipulated-labelled clips that are
byte-identical copies of an authentic RealVideo-RealAudio video (see dataset_audit/duplicate_label_audit.md).

Reproducible: hashes every .mp4 under the dataset root (MD5/SHA-256 only where file sizes collide), so rerunning
on the same data rewrites both outputs byte-for-byte. v1 and the source videos are only read, never modified.

Outputs:
  leakage_controlled_split_manifest_v2.json      eligible model-development clips
  dataset_audit/manifest_v2_exclusions.json      one auditable record per excluded file

Run from veritas/backend/training/:  python build_manifest_v2.py
"""

import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath

_HERE = Path(__file__).resolve().parent
V1_PATH = _HERE / "leakage_controlled_split_manifest.json"
V2_PATH = _HERE / "leakage_controlled_split_manifest_v2.json"
EXCLUSIONS_PATH = _HERE / "dataset_audit" / "manifest_v2_exclusions.json"
AUTHENTIC = "RealVideo-RealAudio"
EXPECTED_EXCLUSIONS = 26  # confirmed in dataset_audit/duplicate_label_audit.md
REASON = ("byte-identical copy of an authentic RealVideo-RealAudio video filed under a manipulated-media "
          "category; not relabelled (the authentic original is already in the manifest with its correct label)")


def digests(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def duplicate_groups(root):
    """Groups (by SHA-256) of >1 byte-identical .mp4 files, as root-relative Windows-style paths."""
    by_size = defaultdict(list)
    for p in root.rglob("*.mp4"):
        by_size[p.stat().st_size].append(p)
    groups = defaultdict(list)
    for paths in by_size.values():
        if len(paths) > 1:
            for p in paths:
                md5, sha = digests(p)
                groups[sha].append((str(PureWindowsPath(p.relative_to(root))), md5))
    return {sha: sorted(m) for sha, m in groups.items() if len(m) > 1}


def clip_rows(manifest):
    for split, cats in manifest["clips"].items():
        for cat, clist in cats.items():
            for c in clist:
                yield split, cat, c


def main():
    v1_bytes = V1_PATH.read_bytes()
    v1 = json.loads(v1_bytes)
    root = Path(v1["metadata"]["dataset_root"])
    groups = duplicate_groups(root)

    where = defaultdict(set)  # rel_path -> {(split, category)} in v1
    for split, cat, c in clip_rows(v1):
        where[c["rel_path"]].add((split, cat))

    exclusions = []
    for sha, members in groups.items():
        rels = [r for r, _ in members]
        originals = [r for r in rels if r.split("\\")[0] == AUTHENTIC]
        # Every byte-identical group must sit in one split, otherwise it is cross-split leakage.
        splits = {s for r in rels for s, _ in where[r]}
        assert len(splits) <= 1, f"byte-identical files span splits {splits}: {rels}"
        if not originals:
            continue
        assert len(originals) == 1, f"multiple authentic originals: {originals}"
        for rel, md5 in members:
            if rel in originals or rel not in where:
                continue
            (split, cat), = where[rel]
            assert (split, AUTHENTIC) in where[originals[0]], f"authentic source of {rel} not in {split}"
            exclusions.append({
                "rel_path": rel,
                "sha256": sha,
                "md5": md5,
                "authentic_source_rel_path": originals[0],
                "original_label": cat,
                "split": split,
                "reason": REASON,
            })
    exclusions.sort(key=lambda e: e["rel_path"])
    assert len(exclusions) == EXPECTED_EXCLUSIONS, f"found {len(exclusions)} exclusions, expected {EXPECTED_EXCLUSIONS}"

    excluded = {e["rel_path"] for e in exclusions}
    v2 = copy.deepcopy(v1)
    for split, cats in v2["clips"].items():
        for cat in cats:
            cats[cat] = [c for c in cats[cat] if c["rel_path"] not in excluded]
        v2["summary"][split] = {"total_clips": sum(map(len, cats.values())),
                                "by_category": {cat: len(l) for cat, l in cats.items()}}
    for split, ids in v2["identities"].items():
        left = {PureWindowsPath(c["rel_path"]).parts[3] for cl in v2["clips"][split].values() for c in cl}
        assert set(ids) <= left, f"{split}: identities left without clips: {set(ids) - left}"

    v2["metadata"].update({
        "manifest_name": "leakage_controlled_split_manifest_v2",
        "manifest_version": 2,
        "created_at": "2026-10-23",
        "derived_from": {"file": V1_PATH.name, "sha256": hashlib.sha256(v1_bytes).hexdigest()},
        "exclusions_file": "dataset_audit/manifest_v2_exclusions.json",
        "excluded_clips_count": len(exclusions),
        "excluded_by_split": dict(sorted(Counter(e["split"] for e in exclusions).items())),
        "note": ("Excluded files remain on disk and in v1 for audit, but are not eligible model-development "
                 "samples. final_test was scored repeatedly under v1: do not rescore it as a fresh result or "
                 "use it for model or threshold selection."),
        "byte_identical_groups_scanned": len(groups),
    })

    EXCLUSIONS_PATH.write_text(json.dumps({
        "manifest": v2["metadata"]["manifest_name"],
        "derived_from": v2["metadata"]["derived_from"],
        "dataset_root": str(root),
        "method": "SHA-256 of every .mp4 under dataset_root whose file size collides with another; "
                  "groups containing a RealVideo-RealAudio file exclude every other member listed in v1",
        "exclusions": exclusions,
    }, indent=1) + "\n", encoding="utf-8")
    V2_PATH.write_text(json.dumps(v2, indent=1) + "\n", encoding="utf-8")

    print(f"byte-identical groups: {len(groups)}  excluded: {len(exclusions)}  {v2['metadata']['excluded_by_split']}")
    for split, s in v2["summary"].items():
        unique = {c["full_path"] for cl in v2["clips"][split].values() for c in cl}
        print(f"  {split:15s} rows {s['total_clips']:5d} (v1 {v1['summary'][split]['total_clips']}) "
              f"unique {len(unique)}  {s['by_category']}")


if __name__ == "__main__":
    main()
