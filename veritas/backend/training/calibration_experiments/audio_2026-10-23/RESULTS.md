# Audio calibration experiment (2026-10-23) — archived, NOT deployed

**Decision:** production keeps raw audio scores (`sigmoid(logit)` vs the checkpoint `decision_threshold` = 0.50).
Neither Platt nor isotonic is applied anywhere in the serving path. Kept here for comparison after Phase 3.

**Pre-registered rule** (in `audio_calib.py`, fixed before results): pick lowest fusion_val Brier, *provided
fake-audio-called-real does not increase* vs raw. Applied to the cohort below, Platt (3→4) and isotonic (3→4)
both **fail**, so raw is kept. The rule is not changed after seeing results.

## Setup
- Model: `models/audio_agent_lc.pth` (trained on detector_train, selected on detector_val).
- Fit: fusion_train, cached scores `state_vec[6]` from `cache/fusion_features_lc_v3.json`, n = 795 (331 fake audio).
- Eval: fusion_val minus the 26 mislabelled copies (= v2 fusion_val), **215 unique clips, all with audio:
  129 real audio (RealVideo-RealAudio 50 + FakeVideo-RealAudio 79), 86 fake audio (RealVideo-FakeAudio 50 + FakeVideo-FakeAudio 36)**.
- Threshold 0.5 on the (calibrated) probability. ECE: 10 equal-width bins. Platt: `p = sigmoid(1.6171·logit(raw) − 0.7362)`.
- Reproduce (from `veritas/backend`, sklearn is only in py -3.10):
  `py -3.10 training/calibration_experiments/audio_2026-10-23/audio_calib.py . training/calibration_experiments/audio_2026-10-23`
  → `audio_calib_output.txt`; rewrites `audio_platt.json` identically.

## A. Cached feature scores (the cohort the rule was applied to)
| Scores | Brier | ECE | AUC | real called fake (FP) | fake called real (FN) |
|---|---|---|---|---|---|
| raw | 0.0724 | 0.1597 | 0.9827 | 16/129 | 3/86 |
| Platt | 0.0455 | 0.0728 | 0.9827 | 8/129 | 4/86 |
| isotonic | 0.0283 | 0.0372 | 0.9819 | 2/129 | 4/86 |

## B. Live re-decode per format (`val_diag.json`, same 215 clips, same calibrators), FP/129 · FN/86
| Decoding | raw | Platt | isotonic |
|---|---|---|---|
| original, centre 3 s | 16 · 0 | 8 · 0 | 2 · 0 |
| MP3 64k | 20 · 0 | 4 · 0 | 0 · **6** |
| AAC 64k | 14 · 0 | 6 · 0 | 3 · **3** |
| WAV 44.1k stereo | 29 · 0 | 12 · 0 | 1 · 0 |
| gain −6 dB | 17 · 0 | 7 · 0 | 1 · **3** |
| gain +6 dB | 11 · 0 | 7 · 0 | 3 · **1** |

## Reconciliation: "4/86 under Platt vs 3/86 raw" vs "fake-as-real stays zero across decoding variants"
Same 215 clips, same threshold, same calibrators. What differs is the **score source**:
- Table A uses the cached `fusion_features_lc_v3` scores. Table B re-decodes each file's own audio.
- They agree to within 1e-4 on 211/215 clips. The 4 that differ are all FakeVideo-FakeAudio clips whose cached
  score came from **another file's audio**: `extract_fusion_features_lc_v3.py` caches WAVs as
  `{identity}_{stem}.wav`, dropping the category folder, so `FakeVideo-FakeAudio\…\id00987\00160_id00476_wavtolip.mp4`
  reused the WAV of the real-audio `FakeVideo-RealAudio\…\id00987\00160_id00476_wavtolip.mp4`.

| Clip (FakeVideo-FakeAudio) | cached raw | Platt(cached) | live raw |
|---|---|---|---|
| `African\men\id00987\00160_id00476_wavtolip.mp4` | 0.380 | 0.178 | 0.893 |
| `Asian (East)\women\id06427\00138_id08139_wavtolip.mp4` | 0.161 | 0.032 | 0.836 |
| `Asian (South)\women\id00747\00053_id06343_wavtolip.mp4` | 0.142 | 0.025 | 0.908 |
| `Caucasian (European)\men\id00192\00078_id00981_wavtolip.mp4` | 0.608 | 0.493 | 0.874 |

So every fake-as-real error in Table A (raw's 3, Platt's 4) is a cache artefact, not a model error. The earlier
summary was also wrong for **isotonic**, which is *not* zero across variants (up to 6/86 on MP3).

## Why these numbers cannot be used to decide calibration
- The fit data has the same bug: 78 fusion_train cache records (16 fusion_val, 4 final_test) share a WAV key
  with another file, so the fitted calibrators are not valid out-of-sample.
- Phase 3 changes checkpoints/preprocessing. Redo with: fixed WAV cache key, fresh out-of-sample scores, fit on
  fusion_train, compare on fusion_val incl. per-codec FP and FN. Never on final_test.
