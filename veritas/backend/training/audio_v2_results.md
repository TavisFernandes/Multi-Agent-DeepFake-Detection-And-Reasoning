# Audio detector v2 — corrected audio, with/without augmentation (2026-10-23)

**Status: development comparison only. Production still loads `models/audio_agent_lc.pth` (unchanged).**
fusion_val was used to compare these runs, so its numbers are optimistic for whichever run is picked. Each
result needs confirming on an independent evaluation collection before any real-world claim. `final_test` was
not used.

## Runs
Every run uses the same architecture, optimiser and 5-epoch schedule. All train on detector_train (manifest v3,
5,034 clips) with collision-free WAVs (`wav_cache.py`). Selection: detector_val F1, with ties broken by lower
val loss. detector_val (106 clips) saturates at F1 = 1.0 from epoch 1. `--augment` = codec/resample variant
(MP3 64k, AAC 64k, 44.1 kHz stereo) with p 0.5, random 3 s window, gain ±6 dB with p 0.5, white noise at
SNR 15–40 dB with p 0.3.

| Run | Checkpoint (committed in `models/`) | SHA-256 | Epoch kept | Train time | Peak RAM | Min free commit | Peak GPU |
|---|---|---|---|---|---|---|---|
| production | `audio_agent_lc.pth` | `cc47d83b…` | — | — | — | — | — |
| noaug s0 | `audio_agent_lc_v2_noaug_s0.pth` | `31e70e4d…` | 3 | 171 s | 8.0 GB | 7.0 GB | 2.7 GB |
| noaug s1 | `audio_agent_lc_v2_noaug_s1.pth` | `154b3cc2…` | 3 | 169 s | 8.0 GB | 8.0 GB | 2.7 GB |
| aug s0 | `audio_agent_lc_v2_aug_s0.pth` | `8b7916f5…` | 3 | 419 s | 5.0 GB | 10.0 GB | 2.8 GB |
| aug s1 | `audio_agent_lc_v2_aug_s1.pth` | `d40911c9…` | 3 | 325 s | 5.0 GB | 11.0 GB | 2.5 GB |

RTX 4050 Laptop 6 GB, `D:\VERITAS_TRAINING\.venv` (torch 2.6.0+cu124). Runs were sequential, under a commit watchdog.

The four v2 candidates are committed so later system-level evaluation can use these exact files. Each checkpoint
also stores its own `training_provenance` (manifest file and SHA-256, WAV preprocessing, augmentation config,
seed, epochs, clip counts). Full SHA-256:

```
cc47d83bcff5f119b2703d0162322dd46a3dd6d8ac054eca7d66998a0a0e9f72  audio_agent_lc.pth (production, unchanged)
31e70e4dfcb5e6c0ab1180983aa3e09792357486881dbfccbb8211f894e9bd90  audio_agent_lc_v2_noaug_s0.pth
154b3cc24ea543907e8ea4b4644f7a8556127e6c7054068d215c0b83a8caf38a  audio_agent_lc_v2_noaug_s1.pth
8b7916f51a3e588b03e3c742f759db513fee63d785988c0e593771e6d4485443  audio_agent_lc_v2_aug_s0.pth
d40911c9979eddfe6137a912a49d3cc51967f265a3ee8e3f1e4c4e15f95ad673  audio_agent_lc_v2_aug_s1.pth
```
Training manifest for all four: `leakage_controlled_split_manifest_v3.json`, SHA-256 `9313dcc7…357b17`.

## fusion_val (215 clips: 129 real audio, 86 fake audio), threshold 0.5
`python eval_audio_robustness.py OUT.json <checkpoints…>`: one run scores every checkpoint on the same
inputs. Formats: original; MP3/AAC 64k and 44.1 kHz stereo through `wav_cache.VARIANTS`; gain ±6 dB; white
noise at 20 dB SNR with a fixed per-clip seed.

**Real audio called fake (/129) · fake audio called real (/86)**

| Format | production | noaug s0 | noaug s1 | aug s0 | aug s1 |
|---|---|---|---|---|---|
| original | 16 · 0 | 0 · 0 | 0 · 0 | 1 · 0 | 0 · 0 |
| MP3 64k | 20 · 0 | 0 · 0 | 1 · 0 | 1 · 0 | 0 · 0 |
| AAC 64k | 14 · 0 | 0 · **4** | 0 · 0 | 2 · 0 | 0 · 0 |
| 44.1 kHz stereo | 29 · 0 | 0 · 0 | 1 · 0 | 1 · 0 | 0 · 0 |
| gain −6 dB | 17 · 0 | 0 · 0 | 1 · 0 | 2 · 0 | 0 · 0 |
| gain +6 dB | 11 · 0 | 0 · 0 | 1 · 0 | 0 · 0 | 0 · 0 |
| noise 20 dB | 6 · **11** | 0 · **12** | 0 · **43** | 3 · 0 | 1 · 0 |

**ECE (10 equal-width bins)**

| Format | production | noaug s0 | noaug s1 | aug s0 | aug s1 |
|---|---|---|---|---|---|
| original | 0.1746 | 0.0020 | 0.0055 | 0.0089 | 0.0043 |
| MP3 64k | 0.2114 | 0.0065 | 0.0096 | 0.0084 | 0.0051 |
| AAC 64k | 0.1591 | 0.0217 | 0.0130 | 0.0111 | 0.0042 |
| 44.1 kHz stereo | 0.2391 | 0.0034 | 0.0067 | 0.0107 | 0.0048 |
| gain −6 dB | 0.1859 | 0.0041 | 0.0085 | 0.0142 | 0.0046 |
| gain +6 dB | 0.1417 | 0.0031 | 0.0074 | 0.0052 | 0.0034 |
| noise 20 dB | 0.0936 | 0.0736 | 0.1866 | 0.0181 | 0.0093 |

Brier and AUC per format are in the evaluation JSON. AUC is ≥ 0.998 for every v2 run in every format, except
noise for noaug (0.9985 / 0.9902).

## Reading
1. **The corrected audio accounts for most of the production model's false-DEEPFAKE errors.** With the label
   fix alone (noaug), real audio called fake falls from 11–29/129 to 0–1/129 on original, codec, resample and
   gain. This fits the 269 wrong-audio detector_train examples acting as label noise.
2. **Without augmentation the model is not robust to noise, and it varies by seed.** At 20 dB noise it misses
   12/86 (s0) and 43/86 (s1) fakes. noaug s0 also misses 4/86 on AAC.
3. **Augmentation fixes noise, and the result holds across both seeds.** Both aug runs miss 0/86 fakes in every
   format, with at most 2/129 false-DEEPFAKE and ECE ≤ 0.018.
4. The augmentation covers the codec, gain and noise families tested here. These results are therefore
   in-distribution for the recipe, on held-out identities. They are not evidence about other codecs,
   real microphones or other generators.

**Candidate:** the augmented recipe. aug s1 has the lowest errors and ECE here, but the two aug seeds are within
a few clips of each other, and this sample (215 clips) cannot rank them reliably. Next step before any
promotion: an independent evaluation collection.
