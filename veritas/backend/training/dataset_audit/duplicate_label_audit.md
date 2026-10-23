# FakeAVCeleb v1.2 — byte-identical cross-category duplicates

**Date:** 2026-10-23 · **Status:** evidence recorded, no data relabelled or removed.

## Finding
Every `.mp4` under the dataset root (21,560 files) was MD5-hashed where file sizes collided.
25 groups of **byte-identical files** sit in different category folders: 24 pairs
`RealVideo-RealAudio` = `FakeVideo-RealAudio`, and 1 triple that also includes a `RealVideo-FakeAudio` copy.
That is **26 copies labelled manipulated whose content is an unmodified authentic video** — all 26 are in
`leakage_controlled_split_manifest.json`.

Why this is a packaging error, not partial manipulation: every frame is identical (whole file is identical),
the copies keep the authentic clip's filename (`00065.mp4`), and their manifest rows have `target1 = "-"`,
whereas genuine face-swaps carry a donor id in the filename and `target1` (e.g. `00027_id02586_wavtolip.mp4`).

| Split | Mislabelled copies |
|---|---|
| detector_train | 14 |
| detector_val | 2 |
| fusion_train | 5 |
| fusion_val | 4 |
| final_test | 1 |

## Measured impact (fusion_val only)
`corrected_fixed` with the `*_lc` detectors reports 5 "fake called AUTHENTIC" errors on fusion_val;
**4 of those 5 are these copies** (fused scores 0.03–0.24). The detectors were right; the labels were wrong.
Historical false-authentic rates in earlier reports are therefore overstated; the single `final_test` copy
was not re-scored (final_test is frozen).

## Proposed treatment (needs approval — not applied)
1. Exclude the 26 manipulated-labelled copies from all splits (keep the `RealVideo-RealAudio` originals,
   which already appear once with the correct label). Do not relabel them to authentic: that would just
   duplicate the original clip.
2. Regenerate the manifest under a new version name (`..._v2`), never overwriting v1.
3. Re-score fusion_val; leave historical final_test results as-is with a pointer to this note.
4. Detector retraining is not required for this alone (14 noisy labels among 5,056 detector_train clips),
   but should use the v2 manifest whenever retraining happens.

Reproduce: hash all `.mp4` under the dataset root grouped by size; group by MD5; keep groups spanning >1
top-level category folder.

## Full list
| # | Authentic original (RealVideo-RealAudio) | Byte-identical copy filed as manipulated | Split |
|---|---|---|---|
| 1 | `RealVideo-RealAudio\African\women\id00220\00027.mp4` | `FakeVideo-RealAudio\African\women\id00220\00027.mp4` | detector_train |
| 2 | `RealVideo-RealAudio\African\women\id00460\00005.mp4` | `FakeVideo-RealAudio\African\women\id00460\00005.mp4` | detector_train |
| 3 | `RealVideo-RealAudio\African\women\id00577\00010.mp4` | `FakeVideo-RealAudio\African\women\id00577\00010.mp4` | final_test |
| 4 | `RealVideo-RealAudio\African\women\id00592\00017.mp4` | `FakeVideo-RealAudio\African\women\id00592\00017.mp4` | fusion_train |
| 5 | `RealVideo-RealAudio\African\women\id00829\00271.mp4` | `FakeVideo-RealAudio\African\women\id00829\00271.mp4` | fusion_train |
| 6 | `RealVideo-RealAudio\African\women\id01532\00065.mp4` | `FakeVideo-RealAudio\African\women\id01532\00065.mp4` | detector_train |
| 7 | `RealVideo-RealAudio\African\women\id01661\00059.mp4` | `FakeVideo-RealAudio\African\women\id01661\00059.mp4` | fusion_train |
| 8 | `RealVideo-RealAudio\African\women\id01783\00015.mp4` | `FakeVideo-RealAudio\African\women\id01783\00015.mp4` | detector_train |
| 9 | `RealVideo-RealAudio\African\women\id02617\00028.mp4` | `FakeVideo-RealAudio\African\women\id02617\00028.mp4` | detector_val |
| 10 | `RealVideo-RealAudio\Asian (East)\men\id03965\00051.mp4` | `FakeVideo-RealAudio\Asian (East)\men\id03965\00051.mp4` | detector_train |
| 11 | `RealVideo-RealAudio\Asian (East)\men\id04789\002121.mp4` | `FakeVideo-RealAudio\Asian (East)\men\id04789\002121.mp4` | detector_train |
| 12 | `RealVideo-RealAudio\Asian (East)\men\id05268\00010.mp4` | `FakeVideo-RealAudio\Asian (East)\men\id05268\00010.mp4` | detector_train |
| 13 | `RealVideo-RealAudio\Asian (East)\men\id05332\00065.mp4` | `FakeVideo-RealAudio\Asian (East)\men\id05332\00065.mp4` | fusion_val |
| 14 | `RealVideo-RealAudio\Asian (East)\men\id05383\00015.mp4` | `FakeVideo-RealAudio\Asian (East)\men\id05383\00015.mp4` | detector_train |
| 15 | `RealVideo-RealAudio\Asian (East)\men\id06152\06152.mp4` | `FakeVideo-RealAudio\Asian (East)\men\id06152\06152.mp4` | detector_val |
| 16 | `RealVideo-RealAudio\Asian (East)\men\id06591\00021.mp4` | `FakeVideo-RealAudio\Asian (East)\men\id06591\00021.mp4` | fusion_train |
| 17 | `RealVideo-RealAudio\Asian (East)\men\id06594\00002.mp4` | `FakeVideo-RealAudio\Asian (East)\men\id06594\00002.mp4` | detector_train |
| 18 | `RealVideo-RealAudio\Asian (East)\men\id06776\00021.mp4` | `FakeVideo-RealAudio\Asian (East)\men\id06776\00021.mp4` | detector_train |
| 18 | `RealVideo-RealAudio\Asian (East)\men\id06776\00021.mp4` | `RealVideo-FakeAudio\Asian (East)\men\id06776\00021.mp4` | detector_train |
| 19 | `RealVideo-RealAudio\Asian (East)\men\id06807\00015.mp4` | `FakeVideo-RealAudio\Asian (East)\men\id06807\00015.mp4` | fusion_val |
| 20 | `RealVideo-RealAudio\Asian (East)\men\id06878\00001.mp4` | `FakeVideo-RealAudio\Asian (East)\men\id06878\00001.mp4` | fusion_train |
| 21 | `RealVideo-RealAudio\Asian (East)\men\id07338\00003.mp4` | `FakeVideo-RealAudio\Asian (East)\men\id07338\00003.mp4` | fusion_val |
| 22 | `RealVideo-RealAudio\Asian (East)\men\id08613\00074.mp4` | `FakeVideo-RealAudio\Asian (East)\men\id08613\00074.mp4` | detector_train |
| 23 | `RealVideo-RealAudio\Asian (East)\men\id08652\00006.mp4` | `FakeVideo-RealAudio\Asian (East)\men\id08652\00006.mp4` | detector_train |
| 24 | `RealVideo-RealAudio\Asian (East)\women\id07901\00040.mp4` | `FakeVideo-RealAudio\Asian (East)\women\id07901\00040.mp4` | detector_train |
| 25 | `RealVideo-RealAudio\Asian (East)\women\id08139\00067.mp4` | `FakeVideo-RealAudio\Asian (East)\women\id08139\00067.mp4` | fusion_val |
