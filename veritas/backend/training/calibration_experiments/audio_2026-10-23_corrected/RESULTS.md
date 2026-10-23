# Audio calibration on collision-free features (2026-10-23) — evaluation only, NOT deployed

No calibrator is selected or deployed. Production keeps raw `audio_agent_lc.pth` scores, threshold 0.50,
and fusion is unchanged. `final_test` is not used.

## Setup
- Scores: `cache/fusion_features_lc_v4.json`, SHA-256 `8d3db421…0621` (local, gitignored). It was extracted
  through `wav_cache.py`, and each record's WAV traces to its own source file (`test_feature_cache_v4.py`).
- Fit: fusion_train (manifest v3), **795 clips, all with audio: 331 fake audio, 464 real**.
- Eval: fusion_val (manifest v3), **215 clips, all with audio: 86 fake audio, 129 real**.
- Threshold 0.5. ECE uses 10 equal-width bins. Fitted Platt: `p = sigmoid(6.5799·logit(raw) − 9.0543)`.
- Formats: the same calibrators applied to live re-decodes of the same fusion_val files
  (`../audio_2026-10-23/val_diag.json`, decoded directly, no cache). The live centre decode matches the v4
  cache to within 5e-5.
- Reproduce: `py -3.10 training/calibration_experiments/audio_2026-10-23_corrected/audio_calib.py` from
  `veritas/backend`. Its output is in `audio_calib_output.txt`.

## Results — FP = real audio called fake (/129), FN = fake audio called real (/86)
| Format | raw Brier · ECE · FP · FN | Platt Brier · ECE · FP · FN | isotonic Brier · ECE · FP · FN |
|---|---|---|---|
| original (centre 3 s) | 0.0634 · 0.1745 · 16 · 0 | 0.0061 · 0.0111 · 1 · 0 | 0.0082 · 0.0083 · 2 · 0 |
| MP3 64k | 0.0743 · 0.2112 · 20 · 0 | 0.0217 · 0.0359 · 0 · **7** | 0.0247 · 0.0347 · 0 · **6** |
| AAC 64k | 0.0620 · 0.1590 · 14 · 0 | 0.0196 · 0.0278 · 2 · **3** | 0.0261 · 0.0180 · 3 · **3** |
| WAV 44.1k stereo | 0.0941 · 0.2390 · 29 · 0 | 0.0065 · 0.0128 · 1 · 0 | 0.0046 · 0.0038 · 1 · 0 |
| gain −6 dB | 0.0666 · 0.1857 · 17 · 0 | 0.0170 · 0.0365 · 1 · **4** | 0.0194 · 0.0225 · 1 · **3** |
| gain +6 dB | 0.0535 · 0.1417 · 11 · 0 | 0.0117 · 0.0083 · 2 · **1** | 0.0140 · 0.0112 · 3 · **1** |

AUC on the original audio: raw 0.9997, Platt 0.9997, isotonic 0.9992.

## Reading
- With correct audio, raw scores miss **no** fakes in any format. Earlier "fake-as-real" errors were
  cache artefacts (see `../audio_2026-10-23/RESULTS.md`).
- Raw scores are badly over-confident towards "fake" on real speech: 11–29 of 129 false-DEEPFAKE depending
  on format, ECE 0.14–0.24.
- Calibration fitted on clean fusion_train audio fixes the clean case. Under MP3/AAC and gain change it
  converts false-DEEPFAKE into missed fakes (up to 7/86). The score shift is codec-dependent, which a
  single monotone map fitted on clean audio cannot correct.
- The model under test was itself trained with 269 wrong-audio detector_train examples
  (`../../dataset_audit/wav_cache_v1_audit.json`). The next step is retraining the audio detector on
  correct audio with codec/gain augmentation, then redoing this evaluation. Calibrating this checkpoint is
  not worth deploying.
