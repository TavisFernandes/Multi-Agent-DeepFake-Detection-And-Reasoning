# Spatial detector v2, temporal robustness, and the 2026-10-24 model selection

**Status (2026-10-24): production loads `spatial_agent_lc_v2_aug_s0.pth` and `audio_agent_lc_v2_aug_s1.pth`.
Temporal is unchanged (`temporal_agent_lc.pth`). Fusion stays `corrected_fixed`; PPO stays experimental.**
Rollback: `VERITAS_SPATIAL_CHECKPOINT=spatial_agent_lc.pth`, `VERITAS_AUDIO_CHECKPOINT=audio_agent_lc.pth`.

**Evidence level.** Every number below comes from fusion_val (manifest v3, 215 clips, held-out identities), and
fusion_val was used to compare and select these candidates. That makes it development data, and the counts are
optimistic for the selected models. They are not a real-world accuracy estimate. `final_test` was not used. A
genuinely independent collection is still needed to assess the finished system.

Raw outputs (per-clip scores, training logs): `robustness_eval_2026-10-24/`.

## Spatial runs
All runs train on detector_train (5,034 clips) with the same EfficientNet-B0 head, AdamW 3e-4 and 5 epochs.
- `--augment`: 3 frames per clip, plus one H.264 re-encode per clip (CRF 24–38, p 0.4). Then JPEG q25–95 (p 0.4),
  rescale ×0.3–0.9 (p 0.3), Gaussian blur σ 0.3–1.8 (p 0.25), crop-margin jitter keeping 0.8–1.0 of the frame
  side at a random offset (p 0.4), colour (p 0.3) and flip. Checkpoint selection uses the mean detector_val AUC over
  clean / JPEG q40 / ×0.5 / H.264 CRF 32 copies.
- Frames are cached on disk (`D:\VERITAS_TRAINING\spatial_frames_v1`, one `.npy` per clip with a sidecar recording
  the source SHA-256 and settings) instead of about 4.5 GB in RAM. Building the cache took about 9 min; each run
  took about 4.5–5 min on an RTX 4050 6 GB.
- Each checkpoint stores its `training_provenance` (manifest hash, augmentation config, selection rule, seed).

| Run | File | SHA-256 | Epoch kept |
|---|---|---|---|
| previous production | `spatial_agent_lc.pth` | `a1babdc2581a4dcc10359b71d73880e586b5ac564b557db1b497fcc0a5ed6be6` | — |
| noaug s0 (retrain control) | `spatial_agent_lc_v2_noaug_s0.pth` | `b94fc31b0106c3be5ab0798cfa714ed645509ec50505227d119e78971e14fb50` | 3 |
| **aug s0 (selected)** | `spatial_agent_lc_v2_aug_s0.pth` | `2c99c7381d20284c9d9953a0bcec9b00cf171e482f1f2ba42c9c6bb61d4eff53` | 5 |
| aug s1 | `spatial_agent_lc_v2_aug_s1.pth` | `a1de7408213bed5561d4814a46d18f429dffaea37faa767a7f3e1abdf99e5efe` | 4 |

Manifest: `leakage_controlled_split_manifest_v3.json`, SHA-256 `9313dcc7…357b17`.

## Spatial on fusion_val (production face-crop path), threshold 0.5
`python eval_spatial_robustness.py OUT.json <checkpoints…>`. The conditions are the existing degradation tests.

**Real video called DEEPFAKE (/100) · fake video called AUTHENTIC (/115) · AUC**

| Condition | previous | noaug s0 | **aug s0** | aug s1 |
|---|---|---|---|---|
| original | 0 · 0 · 1.000 | 0 · 0 · 1.000 | 1 · 5 · 0.996 | 1 · 6 · 0.994 |
| JPEG q30 | 97 · 2 · 0.658 | 96 · 1 · 0.671 | 5 · 26 · 0.962 | 3 · 23 · 0.940 |
| JPEG q60 | 75 · 0 · 0.834 | 71 · 1 · 0.892 | 1 · 14 · 0.981 | 2 · 20 · 0.974 |
| downscale ×0.5 | 98 · 0 · 0.457 | 98 · 0 · 0.691 | 5 · 11 · 0.979 | 0 · 24 · 0.960 |
| blur σ 1.5 | 100 · 0 · 0.459 | 100 · 0 · 0.503 | 9 · 31 · 0.907 | 2 · 43 · 0.878 |
| H.264 CRF 32 | 16 · 78 · 0.701 | 5 · 90 · 0.707 | 8 · 32 · 0.911 | 10 · 32 · 0.913 |
| 720p, face 25% | 98 · 0 · 0.561 | 94 · 0 · 0.911 | 8 · 5 · 0.983 | 2 · 6 · 0.982 |
| 720p, face 25%, H.264 CRF 28 | 79 · 6 · 0.647 | 45 · 28 · 0.692 | 8 · 42 · 0.866 | 5 · 52 · 0.855 |

Reading: the previous model, and a plain retrain of it, are near-perfect on native clips and collapse under any
re-encoding. They call almost every degraded real face fake. Augmentation keeps AUC at 0.87–0.98 in every
condition, at the cost of 5/115 native fakes missed. Retraining without augmentation does not fix robustness, so the
gain comes from the augmentation. aug s0 has the higher AUC in 7 of 8 conditions and fewer missed fakes in 6 of 8 (one tie,
one worse). aug s1 has slightly fewer false DEEPFAKEs. The seeds are close, and 215 clips cannot rank them
reliably.

## Temporal (R3D-18, unchanged) on the same tests
`python eval_temporal_robustness.py OUT.json` runs `temporal_agent.run_temporal_agent` on rebuilt clips, so it uses
the production face crop, frame sampling and the checkpoint threshold (0.35).

| Condition | real → DEEPFAKE /100 | fake → AUTHENTIC /115 | AUC |
|---|---|---|---|
| original | 2 | 1 | 0.998 |
| JPEG q30 | 93 | 0 | 0.784 |
| JPEG q60 | 22 | 4 | 0.969 |
| downscale ×0.5 | 69 | 0 | 0.971 |
| blur σ 1.5 | 99 | 0 | 0.778 |
| H.264 CRF 32 | 7 | 85 | 0.717 |
| 720p, face 25% | 19 | 13 | 0.944 |
| 720p, face 25%, H.264 CRF 28 | 13 | 60 | 0.768 |

**Temporal shows the same failure pattern the previous spatial model had.** JPEG and blur make real clips look
fake, and H.264 makes fake clips look real. This calls for augmented retraining, with the same recipe as spatial
(frame-level degradations plus whole-clip H.264), before the temporal score can be trusted on uploaded media. No
training was done in this sprint.

## System level: corrected_fixed on fusion_val
`python eval_system_fusion_val.py OUT.json spatial.json temporal.json audio.json` feeds the per-clip scores through
`fusion_policy.compute_verdict_corrected_fixed`. A clip is fake unless it is RealVideo-RealAudio (50 real,
165 fake). Each scenario degrades video and audio together.

**Real called DEEPFAKE (/50) · fake called AUTHENTIC (/165) · UNCERTAIN (real / fake)**

| Scenario | previous spatial + previous audio | previous spatial + aug s1 audio | **aug s0 spatial + aug s1 audio** |
|---|---|---|---|
| native | 4 · 0 · (2/0) | 0 · 0 · (0/0) | 0 · 3 · (0/1) |
| JPEG q60 + MP3 64k | 41 · 0 · (3/0) | 35 · 0 · (4/0) | 1 · 4 · (1/3) |
| ×0.5 + AAC 64k | 49 · 0 · (0/0) | 49 · 0 · (0/0) | 4 · 2 · (5/5) |
| H.264 CRF 32 + AAC 64k | 9 · 32 · (2/8) | 2 · 49 · (2/4) | 1 · 33 · (1/13) |
| 720p face 25% H.264 CRF 28 + AAC 64k | 35 · 3 · (2/2) | 33 · 4 · (2/3) | 0 · 22 · (3/12) |
| **total over 5 scenarios** | **138 · 35** | **119 · 53** | **6 · 64** |

All fake→AUTHENTIC errors of the selected combination are FakeVideo-RealAudio clips (3 native, 33 and 22 under
H.264). The audio detector catches every fake-audio clip in every scenario, so the remaining misses are
visual-only fakes after heavy compression, where temporal is the weaker detector.

## Selection
- **Audio → `audio_agent_lc_v2_aug_s1.pth`.** It makes 0/129 false DEEPFAKE and 0/86 missed fakes in every format
  except noise (1/129), with ECE ≤ 0.01, against 6–29/129 false DEEPFAKE for the previous model
  (`audio_v2_results.md`). The error profile is better on both sides, so the switch is justified.
- **Spatial → `spatial_agent_lc_v2_aug_s0.pth`.** This is a trade-off, accepted on purpose. Across the five
  system scenarios, false DEEPFAKE falls from 119 to 6 of 250, and missed fakes rise from 53 to 64 of 825.
  On native clips, 3 visual-only fakes are now missed (0 before). Uploaded media is almost always re-encoded, and the
  previous model called most re-encoded real faces fake, so the robust model is the safer demo default.
- **Temporal → unchanged**; it is the next model to retrain.
- **Fusion → corrected_fixed unchanged.** No fusion or uncertainty change was made. Any rule tuned on these counts
  would be one more selection on fusion_val. PPO v4 was trained on the previous detectors' scores and has not
  been re-validated with v2.

## Known limitations
- Development evidence only (see top). The degradations are synthetic and cover FakeAVCeleb generators only.
- The audio detector has no speech check. A pure 1 kHz tone (`test_audio.wav`) scores 0.99 fake (0.61 with the
  previous model, also DEEPFAKE).
- Crop-margin augmentation only zooms in. Zooming out would need background the training clips do not have.
