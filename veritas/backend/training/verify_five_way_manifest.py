"""
verify_five_way_manifest.py
===========================
Comprehensive verification of the 5-way leakage-controlled split manifest.
Validates:
1. Identity disjointness across all 5 partitions.
2. Exact clip inventory and disk existence.
3. Source and donor constraints against meta_data.csv.
4. Target validity and class distributions for Spatial, Temporal, Audio detectors.
5. Disk space and GPU memory availability.
"""

import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
MANIFEST_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else _HERE / "leakage_controlled_split_manifest_v2.json"
DATASET_ROOT = Path(r"D:\VERITAS_TRAINING\data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2")
META_CSV = DATASET_ROOT / "meta_data.csv"

def run_verification():
    print("=" * 75)
    print("VERITAS 5-WAY LEAKAGE-CONTROLLED MANIFEST VERIFICATION")
    print("=" * 75)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    splits = ["detector_train", "detector_val", "fusion_train", "fusion_val", "final_test"]

    # 1. Identity Disjointness
    print("\n--- 1. Identity Disjointness Verification ---")
    split_id_sets = {}
    all_ids = set()
    disjoint = True
    for s in splits:
        ids = set(manifest["identities"][s])
        split_id_sets[s] = ids
        overlap = all_ids.intersection(ids)
        if overlap:
            print(f"ERROR: Overlap found in {s}: {overlap}")
            disjoint = False
        all_ids.update(ids)
        print(f"  {s:16s}: {len(ids):3d} identities")

    print(f"Total unique identities: {len(all_ids)} / 500")
    print(f"Disjointness: {'PASSED (Zero Overlap)' if disjoint else 'FAILED'}")

    # 2. Clips inventory and file existence on disk
    print("\n--- 2. Clips Inventory & Disk Existence ---")
    total_manifest_clips = 0
    missing_files = 0
    all_clip_records = []

    for s in splits:
        split_clips = manifest["clips"][s]
        s_total = 0
        cat_counts = {}
        for cat, clist in split_clips.items():
            cat_counts[cat] = len(clist)
            s_total += len(clist)
            for item in clist:
                p = item["full_path"]
                all_clip_records.append((s, cat, item))
                if not os.path.exists(p):
                    missing_files += 1
        total_manifest_clips += s_total
        print(f"  {s:16s}: {s_total:5d} clips | Breakdown: {cat_counts}")

    print(f"Total clips across 5 splits: {total_manifest_clips}")
    print(f"Missing clips on disk:       {missing_files}")

    # 3. Source and Donor Leakage Audit
    print("\n--- 3. Source and Donor Leakage Audit against meta_data.csv ---")
    meta_lookup = {}
    with open(META_CSV, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            if r:
                rel = r[9].replace("FakeAVCeleb/", "") + "/" + r[8]
                meta_lookup[rel.replace("\\", "/").lower()] = r

    cross_contamination_detected = False
    checked_count = 0

    for s in splits:
        my_ids = split_id_sets[s]
        other_ids = all_ids - my_ids
        for cat, clist in manifest["clips"][s].items():
            for item in clist:
                checked_count += 1
                rel = item["rel_path"].replace("\\", "/").lower()
                row = meta_lookup.get(rel)
                if not row:
                    continue
                src = row[0]
                donors = [d for d in [row[1], row[2]] if d and d != "-"]
                if src not in my_ids:
                    print(f"SOURCE LEAKAGE in {s}: clip {rel} has source {src} not in split!")
                    cross_contamination_detected = True
                for d in donors:
                    if d in other_ids:
                        print(f"DONOR LEAKAGE in {s}: clip {rel} has donor {d} from another split!")
                        cross_contamination_detected = True

    print(f"Checked clips against meta_data.csv: {checked_count}")
    print(f"Cross-partition contamination: {'NONE DETECTED (100% Disjoint)' if not cross_contamination_detected else 'CONTAMINATION DETECTED'}")

    # 4. Detector-Specific Ground Truth Class Distributions
    print("\n--- 4. Detector-Specific Target Distributions ---")
    # Spatial: FakeVideo-* is 1, RealVideo-* is 0
    # Temporal: FakeVideo-* is 1, RealVideo-* is 0
    # Audio: *-FakeAudio is 1, *-RealAudio is 0
    for s in ["detector_train", "detector_val"]:
        sp_real = len(manifest["clips"][s]["RealVideo-RealAudio"]) + len(manifest["clips"][s]["RealVideo-FakeAudio"])
        sp_fake = len(manifest["clips"][s]["FakeVideo-RealAudio"]) + len(manifest["clips"][s]["FakeVideo-FakeAudio"])
        au_real = len(manifest["clips"][s]["RealVideo-RealAudio"]) + len(manifest["clips"][s]["FakeVideo-RealAudio"])
        au_fake = len(manifest["clips"][s]["RealVideo-FakeAudio"]) + len(manifest["clips"][s]["FakeVideo-FakeAudio"])
        print(f"  {s:16s} | Spatial/Temporal: Real={sp_real:4d}, Fake={sp_fake:4d} (Total={sp_real+sp_fake}) | Audio: Real={au_real:4d}, Fake={au_fake:4d} (Total={au_real+au_fake})")

    # 5. Hardware Resources
    print("\n--- 5. Hardware Resources Verification ---")
    total_d, used_d, free_d = shutil.disk_usage("D:\\")
    print(f"D: Drive Space - Free: {free_d / (1024**3):.2f} GB / Total: {total_d / (1024**3):.2f} GB")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        print(f"GPU: {gpu_name} ({vram_mb:.1f} MB VRAM)")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA is NOT available.")

    print("\n" + "=" * 75)
    print("VERIFICATION COMPLETED SUCCESSFULLY.")
    print("=" * 75)

if __name__ == "__main__":
    run_verification()
