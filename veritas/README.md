# VERITAS — Deepfake Detection System

A multi-agent AI system that detects deepfakes in images, videos, and audio using four specialised neural network agents.

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **pip** (comes with Python)
- **npm** (comes with Node.js)
- **FFmpeg** (required for audio analysis of video files)
  - Windows: `winget install ffmpeg`
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`

## Setup & Running

Open **two separate terminals** and run each service:

### Terminal 1 — Backend (FastAPI + PyTorch)

```bash
cd backend
pip install -r requirements.txt
python main.py
```

→ Runs on **http://localhost:8000**

> First run will take longer as PyTorch downloads are heavy. Subsequent runs are instant.

### Terminal 2 — Frontend (React + Vite)

```bash
cd frontend
npm install
npm run dev
```

→ Open **http://localhost:5173** in your browser

---

## Architecture

```
Browser → Vite dev server (5173)
    │  /api/* forwarded with the /api prefix stripped (vite.config.ts)
    ▼
Backend — FastAPI (8000)
    │  Streams NDJSON: one JSON object per line
    │  Multi-agent inference pipeline
    ├── Spatial Agent   (EfficientNet-B0 + GradCAM)
    ├── Temporal Agent  (R3D-18 video model)
    ├── Audio Agent     (EfficientNet-B0 + Mel Spectrogram)
    └── Context Agent   (metadata analysis)
```

## Agents

| Agent | Model | Input | Detects |
|-------|-------|-------|---------|
| **Spatial** | EfficientNet-B0 + GradCAM | Image/Video frame | Pixel-level artifacts, blending boundaries |
| **Temporal** | R3D-18 (16-frame sampling) | Video | Inter-frame inconsistencies |
| **Audio** | EfficientNet-B0 + Mel Spectrogram | Audio/Video | Speech synthesis artifacts |
| **Context** | Derived from the other scores | Video with audio | Describes whether visual and audio scores agree (explanation only, never votes) |

## Model Checkpoints

Files the backend loads from `backend/models/` (leakage-controlled training, see `backend/training/README.md`):

| Agent | File loaded | Notes |
|-------|-------------|-------|
| Spatial | `spatial_agent_lc.pth` | EfficientNet-B0, threshold 0.50 |
| Temporal | `temporal_agent_lc.pth` | R3D-18, threshold 0.35 (from the checkpoint) |
| Audio | `audio_agent_lc.pth` | EfficientNet-B0 on mel spectrogram, threshold 0.50 |
| Face detector | `face_detection_yunet_2023mar.onnx` | OpenCV YuNet, crops faces for Spatial/Temporal |
| PPO fusion (`VERITAS_FUSION_MODE=ppo` only) | `ppo_fusion_policy_v4.pth` | Experimental; production uses `corrected_fixed` |
| Context | none | Explanation only: restates the visual–audio score gap, never votes |

The older `spatial_agent.pth`, `temporal_agent_checkpoint.pth`, `audio_agent_checkpoint.pth` and
`ppo_fusion_policy{,_v2,_v3}.pth` are kept for reference and are not loaded.

If a checkpoint is missing, that agent reports itself unavailable and is left out of fusion; no score is invented.

## Tech Stack

- **Frontend**: React 19 + Vite 7 + Tailwind CSS 4 + Framer Motion
- **Backend**: FastAPI + PyTorch 2.6

## Notes

- The 50MB file size limit is enforced in the browser and in the backend (requests with a larger `Content-Length` are rejected before the upload is read).
- Vite's `/api` proxy is for local development only. A production deployment needs a web server (e.g. nginx or Caddy) to serve `frontend/dist` and forward `/api` to FastAPI, with its own upload limit and timeouts.
- GradCAM heatmaps are rendered in the UI when a spatial checkpoint is available.
- FFmpeg must be installed on the system for audio extraction from video files.
