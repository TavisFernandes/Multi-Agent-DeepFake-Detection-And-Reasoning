# VERITAS — Training and Evaluation

## Current pipeline: leakage-controlled (`*_lc`) — what production loads

Production loads `spatial_agent_lc.pth`, `temporal_agent_lc.pth` and `audio_agent_lc.pth`; `ppo` mode loads
`ppo_fusion_policy_v4.pth`. All of them come from the 5-way identity-disjoint split in
`leakage_controlled_split_manifest.json` (detector_train / detector_val / fusion_train / fusion_val / final_test).

**Environment:** `pip install -r ../requirements-lock.txt -r ../requirements-training.txt`.
Detector training needs a CUDA build of torch (the locked one is CPU-only); PPO and evaluation run on CPU.

**Paths (not yet configurable):** the manifest stores absolute paths under
`D:\VERITAS_TRAINING\data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2`, and `train_temporal_lc.py`, `train_audio_lc.py`
and `extract_fusion_features_lc_v3.py` cache to `D:\VERITAS_TRAINING\{temporal_cache,wav_cache}`.
Another machine must use the same layout until these become settings.

**Missing:** the script that generated `leakage_controlled_split_manifest.json` is not in the repo, so the
split can be reused but not regenerated. See `dataset_audit/duplicate_label_audit.md` for 26 known
mislabelled clips in it.

Run from `veritas/backend/training/`, in order:

```bash
python train_spatial_lc.py           # -> models/spatial_agent_lc.pth   (selected on detector_val)
python train_temporal_lc.py          # -> models/temporal_agent_lc.pth  (selected on detector_val)
python train_audio_lc.py             # -> models/audio_agent_lc.pth     (selected on detector_val)
python extract_fusion_features_lc_v3.py   # -> cache/fusion_features_lc_v3.json (out-of-sample scores)
python train_ppo_v3.py --seed 2 --epochs 300 --out ../models/ppo_fusion_policy_v4.pth
```

The detector scripts and `train_ppo_v3.py` without `--out` **overwrite** their default checkpoint; pass a new
name (PPO) or copy the existing file aside first (detectors). Use `fusion_val` for every choice; `final_test`
has been scored repeatedly and must not be used for tuning.

---

# Legacy: Retraining the Agents on FakeAVCeleb (Kaggle GPU)

> These produced the pre-October checkpoints (`spatial_agent.pth`, `temporal_agent_checkpoint.pth`,
> `audio_agent_checkpoint.pth`). Their "identity-aware" split groups by the race folder, not the identity,
> so they are kept for reference only.

These scripts retrain the three neural agents on the FakeAVCeleb dataset. They are
written for a **Kaggle Notebook with a GPU** (T4 or P100) because:

- The dataset already lives on Kaggle (https://www.kaggle.com/datasets/shreyaty08/fakeavceleb) — no download.
- Each model needs a GPU; CPU training is not viable (weeks, not hours).

Each script produces a checkpoint in the **exact format the inference code loads**, so
after training you just rename the file and drop it into `backend/models/`.

| Script | Agent | Output file (Kaggle) | Rename to | Load key |
|--------|-------|----------------------|-----------|----------|
| `train_spatial_kaggle.py`  | Spatial (EfficientNet-B0)  | `spatial_agent_best.pth`  | `spatial_agent.pth`             | `state_dict` |
| `train_temporal_kaggle.py` | Temporal (R3D-18)          | `temporal_agent_best.pth` | `temporal_agent_checkpoint.pth` | `model_state_dict` |
| `train_audio_kaggle.py`    | Audio (EfficientNet-B0)    | `audio_agent_best.pth`    | `audio_agent_checkpoint.pth`    | `model_state_dict` |

> The **Context** agent is *not* trained. It is a cross-modal consistency check
> (see `backend/mock_agents.py`) that compares the visual verdict against the audio
> verdict at inference time. There is nothing to train and no checkpoint to produce.

---

## Step-by-step (per agent)

1. Go to https://www.kaggle.com/code → **New Notebook**.
2. **Add data** (right panel) → search `shreyaty08/fakeavceleb` → add it.
3. **Settings → Accelerator → GPU T4 x1** (or P100). Also enable **Internet** (needed
   once, to download the ImageNet / Kinetics-400 pretrained backbone).
4. Confirm the dataset path. In a cell run:
   ```python
   !find /kaggle/input -maxdepth 3 -type d | head -50
   ```
   Find the folder that directly contains the four category dirs
   (`RealVideo-RealAudio`, `RealVideo-FakeAudio`, `FakeVideo-RealAudio`,
   `FakeVideo-FakeAudio`) and set `DATASET_ROOT` at the top of the script to match.
   The default assumes:
   ```
   /kaggle/input/fakeavceleb/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2
   ```
5. Paste the whole script into a cell (or `%load` it) and **Run**.
   - Spatial: ~1–2 h. Temporal: ~2–4 h. Audio: ~2–4 h (first epoch is slow while it
     extracts + caches WAVs).
6. When it finishes, the last log lines print a **VERIFICATION** block confirming the
   checkpoint reloads `strict=True` and produces a valid probability.
7. Download the `*_best.pth` from the **Output** tab (`/kaggle/working/checkpoints/`).
8. Rename per the table above and place in `veritas/backend/models/`.

You can run all three in one notebook sequentially, or three separate notebooks in
parallel (Kaggle allows multiple GPU sessions on some accounts — check your quota).

---

## Label conventions

**Video label** (spatial + temporal) — is the *face/video* fake?
```
RealVideo-*  → 0 (REAL)
FakeVideo-*  → 1 (FAKE)
```

**Audio label** (audio agent) — is the *audio* fake? Independent of the video:
```
*-RealAudio  → 0 (REAL)
*-FakeAudio  → 1 (FAKE)
```
So `RealVideo-FakeAudio` is video-REAL but audio-FAKE, and `FakeVideo-RealAudio` is
video-FAKE but audio-REAL. This is exactly the mismatch the context agent looks for.

---

## Important notes

- **Preprocessing must match inference.** The audio script copies
  `audio_agent._wav_to_tensor` verbatim (mel params, 3 s clip, 224×224, `(x-0.5)/0.5`,
  3-channel). The temporal script uses the checkpoint-authoritative normalisation
  constants. Do not change these without changing the matching inference module.
- **Identity-aware split.** All three scripts split train/val by identity (the folder
  under each category), so the same person never appears in both — this prevents the
  inflated metrics you get from a naive random split.
- **Decision threshold.** The temporal script selects its threshold on the validation
  set (best-F1 sweep) and saves it as `decision_threshold`; `temporal_agent.py` reads
  it at load. FakeAVCeleb's heavy FAKE skew tends to push this well below 0.5 (the
  shipped checkpoint used ~0.05).
- **The old `backend/train_temporal_agent.py` is superseded** by
  `train_temporal_kaggle.py`. The old one never saved `normalization` /
  `decision_threshold`, which `temporal_agent.py` needs — using it would silently
  degrade inference. Prefer the new script.

## After you drop the checkpoints in

From `backend/`:
```bash
python verify_temporal_checkpoint.py     # sanity-checks the temporal .pth
python main.py                           # start the backend; watch the startup logs
```
All three agents should now load without "checkpoint not found" warnings, and the
audio agent will report `status: "trained"` instead of `"untrained"`.
