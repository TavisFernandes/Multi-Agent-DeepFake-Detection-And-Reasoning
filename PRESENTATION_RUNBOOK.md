# VERITAS — Presentation Runbook & Demonstration Guide

This runbook provides the verified, step-by-step procedures for launching, operating, demonstrating, and troubleshooting the VERITAS Deepfake Detection System for the college presentation.

See the complete documentation in [veritas/PRESENTATION_RUNBOOK.md](file:///d:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/PRESENTATION_RUNBOOK.md).

---

## Quick Launch Procedure

Open two separate PowerShell terminals:

### Terminal 1 — Backend (Port 8000)
```powershell
cd veritas/backend
py -3.10 main.py
```

### Terminal 2 — Frontend (Port 5173)
```powershell
cd veritas/frontend
npm run dev
```

Open in browser: **http://localhost:5173/console**

---

## Health Checks
```powershell
# Backend Health
py -3.10 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/healthz').read().decode())"
```
