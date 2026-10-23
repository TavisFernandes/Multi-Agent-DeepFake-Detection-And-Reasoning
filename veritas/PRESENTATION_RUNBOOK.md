# VERITAS — Presentation Runbook & Demonstration Guide

This runbook provides the verified, step-by-step procedures for launching, operating, demonstrating, and troubleshooting the VERITAS Deepfake Detection System for the college presentation.

---

## 1. System Architecture & Topology

```
   ┌──────────────────────────────────────────────────────────────┐
   │                  Browser Client (:5173)                      │
   │  React 19 + Vite 7 + Tailwind CSS (Interactive Console)      │
   └──────────────────────────────┬───────────────────────────────┘
                                  │  /api/analyse/stream → /analyse/stream
                                  │  (Vite dev proxy; FastAPI streams NDJSON back)
                                  ▼
   ┌──────────────────────────────────────────────────────────────┐
   │               Inference Backend (:8000)                      │
   │  FastAPI + PyTorch 2.6 (Python 3.10)                         │
   ├──────────────────────────────┬───────────────────────────────┤
   │  Spatial Agent               │ EfficientNet-B0 + GradCAM     │
   │  Temporal Agent              │ R3D-18 (16-frame 3D CNN)      │
   │  Audio Agent                 │ EfficientNet-B0 Mel-Spec      │
   │  Context Agent               │ Cross-modal consistency logic │
   │  Fusion Policy               │ corrected_fixed (Default)     │
   └──────────────────────────────┴───────────────────────────────┘
```

---

## 2. Prerequisites & Verification

Before launching the services, verify the required tools are present:

### 2.1 Python Environment
* Python **3.10** is required for PyTorch checkpoint compatibility.
* On Windows systems with multiple Python versions installed, use `py -3.10` to target Python 3.10.
* Verify PyTorch is available:
  ```powershell
  py -3.10 -c "import torch; print('PyTorch:', torch.__version__)"
  ```

### 2.2 Node.js Environment
* Node.js **18+** (Node 20+ recommended) with npm.
* Verify Node.js and npm:
  ```powershell
  node -v
  npm -v
  ```

### 2.3 FFmpeg
* FFmpeg must be installed on the system PATH for container probing and audio extraction:
  ```powershell
  ffmpeg -version
  ```

### 2.4 Verify Trained Checkpoints
Check that the files the backend loads are in `veritas/backend/models/`:
* `spatial_agent_lc_v2_aug_s0.pth` (~16.8 MB) — spatial default since 2026-10-24 (`backend/training/spatial_v2_results.md`)
* `temporal_agent_lc.pth` (~126.6 MB)
* `audio_agent_lc_v2_aug_s1.pth` (~15.6 MB) — audio default since 2026-10-24 (`backend/training/audio_v2_results.md`)
* `face_detection_yunet_2023mar.onnx` (~0.2 MB)
* `ppo_fusion_policy_v4.pth` (only used when `VERITAS_FUSION_MODE=ppo`)

The previous detectors (`spatial_agent_lc.pth`, `audio_agent_lc.pth`) are kept for rollback. To run with them, set
these before starting the backend:
```powershell
$env:VERITAS_SPATIAL_CHECKPOINT = "spatial_agent_lc.pth"
$env:VERITAS_AUDIO_CHECKPOINT   = "audio_agent_lc.pth"
```

---

## 3. Exact Startup Commands (Two Terminals)

Open two separate PowerShell terminals and run each service in its dedicated directory:

### Terminal 1 — Backend (FastAPI + PyTorch)
```powershell
cd veritas/backend
py -3.10 main.py
```
* **Expected Port**: `8000`
* **Success Output**:
  ```
  INFO:     Application startup complete.
  INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
  ```

### Terminal 2 — Frontend (React + Vite)
```powershell
cd veritas/frontend
npm run dev
```
* **Expected Port**: `5173`
* **Success Output**:
  ```
    VITE v7.x ready in ... ms
    ➜  Local:   http://localhost:5173/
  ```

---

## 4. Verification & Health Checks

Verify both tiers are communicating before beginning your presentation:

### 4.1 Backend Health Check
```powershell
py -3.10 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/healthz').read().decode())"
```
* **Expected Response**: `{"status":"ok","models":{"spatial":"spatial_agent_lc_v2_aug_s0.pth","temporal":"temporal_agent_lc.pth","audio":"audio_agent_lc_v2_aug_s1.pth"}}`
* A `null` model means that detector failed to load; check the backend log (`Detector checkpoint: ...` lines).

### 4.2 Browser Access
Open Google Chrome or Microsoft Edge and navigate to:
```
http://localhost:5173/console
```

---

## 5. Live Demonstration Procedure

Follow these steps for a clean, glitch-free presentation:

1. **Open the Console**:
   * Navigate to `http://localhost:5173/console`.
   * Observe the automated boot sequence:
     - Spatial Analysis Module loaded (`EfficientNet-B0`)
     - Temporal Analysis Module loaded (`R3D-18`)
     - Audio Analysis Module loaded (`MelSpectrogram CNN`)
     - Context Analysis Module loaded
     - GradCAM engine initialized
     - `[READY] VERITAS v1.0 — awaiting media input`
   * Once complete, the left sidebar becomes interactive.

2. **Select Modality & Upload Media**:
   * For video demonstrations: Click the **Video** tab in the sidebar.
   * Click the drop zone or drag-and-drop a sample file (e.g. `veritas/backend/00025_fake.mp4`).
   * Verify the file name and size appear with the green checkmark ("Content loaded").

3. **Trigger Multi-Agent Analysis**:
   * Click **Analyse with VERITAS**.
   * Observe the button transition to `Analysing with Multi-Agent Pipeline...` with an active spinner (duplicate clicks are safely prevented).
   * Notice the four agent cards executing sequentially in real time:
     1. **Spatial Analyst**: Evaluates visual frame artifacts via EfficientNet-B0.
     2. **Temporal Analyst**: Assesses frame-to-frame coherence via R3D-18.
     3. **Audio Analyst**: Detects speech synthesis artifacts via Mel-Spectrogram CNN.
     4. **Context Analyst**: Cross-verifies visual-auditory consistency.

4. **Review the Verdict & Category Attribution**:
   * When inference finishes, the final verdict card appears:
     - **Headline**: `DEEPFAKE DETECTED` (Red), `AUTHENTIC MEDIA` (Green), or `INCONCLUSIVE — UNCERTAIN` (Amber).
     - **Attribution Hypothesis**: Displays manipulation subtype (e.g., `VIDEO MANIPULATED`, `AUDIO MANIPULATED`).
     - **Fused Fake Score**: The composite confidence percentage.
     - **Category Attribution**: Full calibrated breakdown across categories.
     - **Dynamic Weights / Decision Thresholds**: Exact mathematical audit trail.

5. **Test Another Sample**:
   * Click **Verify Another Sample** in the sidebar.
   * The console resets cleanly to accept new content without reloading the page.

---

## 6. Honest Detection Semantics & Answering Panel Questions

Be prepared to explain the detection semantics to examiners:

1. **Three-State Decision Space**:
   - `AUTHENTIC`: Fused fake score is decisively below threshold ($\tau_{low}$).
   - `DEEPFAKE`: Fused fake score is decisively above threshold ($\tau_{high}$).
   - `UNCERTAIN`: Abstention zone ($[0.45, 0.55]$ deadband). The system abstains rather than falsely claiming authenticity when evidence is borderline.

2. **Why `corrected_fixed` is Production Default**:
   - Visual score = $(0.40 \cdot$ Spatial $+ 0.30 \cdot$ Temporal$) / 0.70$; fused score = $1 - (1 - \text{visual})(1 - \text{audio})$. Unavailable streams (e.g. videos without audio) are left out rather than counted as authentic. Context is explanation-only and has no weight.
   - Missing modalities never lower the score, and a video with no usable face can never be called AUTHENTIC.
   - PPO v4 is experimental (`VERITAS_FUSION_MODE=ppo`) and disabled in production.

3. **Missing Modality Handling**:
   - Image uploads mark Temporal as `not_applicable` and Audio as `no_audio`.
   - Videos without audio streams honestly report Audio as `no_audio` and exclude it from the denominator.

---

## 7. Clean Shutdown & Restart

### 7.1 Graceful Shutdown
In each of the two terminals, press `Ctrl + C` once.

### 7.2 Resolving Port Conflicts (If a service won't start)
If a port is already bound, identify and terminate the orphaned process:

```powershell
# Check for active listeners on VERITAS ports
netstat -ano | findstr "8000 5173"

# Terminate process by PID (replace <PID> with the number from the last column)
taskkill /F /PID <PID>
```

---

## 8. Troubleshooting Guide

| Issue | Cause | Fix |
|---|---|---|
| **`python: command not found` or pytest closed-file crash** | System defaulted to experimental Python 3.14 | Launch using `py -3.10 main.py` or run `py -3.10 -m pytest -s` |
| **Console says "Can't reach the analysis service"** | Backend FastAPI server is not running on port 8000 | Start Terminal 1: `cd veritas/backend && py -3.10 main.py` |
| **Audio Agent reports `no_audio` on video** | Video container has no audio track | Normal behavior; audio correctly excluded from fusion score |
| **Large or long video is rejected or slow** | Files over 50MB are rejected; long videos take longer on CPU | Use files under 50MB; short clips analyse fastest |
| **Port 8000 or 5173 already in use** | Stale previous run still bound to port | Run `netstat -ano \| findstr "8000 5173"` and `taskkill /F /PID <PID>` |
