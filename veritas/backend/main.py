"""
main.py

FastAPI service for VERITAS deepfake detection.
Port: 8000

Endpoints
---------
GET  /healthz              — liveness probe
POST /analyse              — multipart upload, returns full multi-agent verdict JSON (legacy)
POST /analyse/stream       — multipart upload, returns newline-delimited JSON stream,
                             one line per agent result then one final verdict line

On startup the Spatial Agent model is loaded once as a global singleton.
The Temporal, Audio, and Context agents are currently mocked; swap them out
for real models by replacing the functions in mock_agents.py.
"""

import asyncio
import cv2
import json
import logging
import os
import time
import tempfile
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

import spatial_agent
from temporal_agent import run_temporal_agent
from audio_agent import run_audio_agent
from mock_agents import context_agent_result
import fusion_policy
from decision_contract import VERDICT_UNCERTAIN_FULL, enrich_decision_fields
from fusion_policy import (
    get_fusion_policy,
    compute_verdict_corrected_fixed,
    compute_verdict_legacy,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="VERITAS Deepfake Detection API",
    description="Multi-agent deepfake detection service with PPO Adaptive Fusion.",
    version="1.2.0",
)

# Multipart framing (boundary lines, part headers) adds a little on top of the 50 MB file itself.
_MAX_REQUEST_BYTES = 50 * 1024 * 1024 + 1024 * 1024


class RejectOversizedRequests:
    """Answer 413 from Content-Length before Starlette reads the multipart body.
    The routes' own 50 MB check stays authoritative: Content-Length can be absent (chunked uploads)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            length = dict(scope["headers"]).get(b"content-length", b"")
            if length.isdigit() and int(length) > _MAX_REQUEST_BYTES:
                response = JSONResponse({"detail": "File exceeds 50MB limit."}, status_code=413)
                return await response(scope, receive, send)
        await self.app(scope, receive, send)


# Added before CORS so CORS wraps it and the 413 still carries CORS headers.
app.add_middleware(RejectOversizedRequests)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup — load the models once
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Starting VERITAS backend service …")
    try:
        spatial_agent.load_model()
    except Exception as _se:
        logger.error("Spatial Agent startup error: %s", _se)
    try:
        import temporal_agent as _ta
        if not _ta._model_loaded:
            _ta.load_model()
    except Exception as _te:
        logger.error("Temporal Agent startup error: %s", _te)
    try:
        import audio_agent as _aa
        if not _aa._model_loaded:
            _aa._load_model()
    except Exception as _ae:
        logger.error("Audio Agent startup error: %s", _ae)
    try:
        policy = get_fusion_policy()
        if policy.is_loaded:
            logger.info("PPO Fusion Policy loaded successfully.")
    except Exception as _fe:
        logger.warning("PPO Fusion Policy startup load notice: %s", _fe)
    for name, loaded in _loaded_models().items():
        logger.info("Detector checkpoint: %s", f"{name} = {loaded}")
    logger.info("Startup complete.")


def _loaded_models() -> dict:
    """Checkpoint file each detector loaded (None if not loaded); audio/temporal load at import, before logging is set up."""
    import audio_agent as _aa
    import temporal_agent as _ta
    return {name: m._CHECKPOINT_PATH.name if m._model_loaded else None
            for name, m in (("spatial", spatial_agent), ("temporal", _ta), ("audio", _aa))}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"}
_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}

# Weighted contribution of each agent to the overall verdict
_WEIGHTS = {
    "Spatial Analysis":    0.40,
    "Temporal Analysis":   0.30,
    "Audio Analysis":      0.20,
    "Context Verification": 0.10,
    "spatial":             0.40,
    "temporal":            0.30,
    "audio":               0.20,
    "context":             0.10,
}


def _detect_media_type(filename: str) -> str | None:
    ext = Path(filename).suffix.lower()
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _VIDEO_EXTS:
        return "video"
    if ext in _AUDIO_EXTS:
        return "audio"
    return None


def _resolve_media_type(filename: str, hint: str) -> str:
    """Use a valid hint, else the extension; unknown types get 415 instead of being guessed as images."""
    hint = hint.strip().lower()
    resolved = hint if hint in ("image", "video", "audio") else _detect_media_type(filename)
    if resolved is None:
        supported = ", ".join(sorted(_IMAGE_EXTS | _VIDEO_EXTS | _AUDIO_EXTS))
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{Path(filename).suffix or filename}'. Supported: {supported}.")
    return resolved


def _build_summary(
    agents: list[dict],
    overall_conf: float,
    verdict: str,
    contributing: list[str],
    details: dict | None = None,
) -> str:
    if not contributing:
        return "No trained agents were available to assess this media. Authenticity cannot be verified."

    details = details or {}
    pred_cat = details.get("predicted_category")
    verdict_u = (verdict or "").upper()
    is_uncertain = "UNCERTAIN" in verdict_u or "INSUFFICIENT" in verdict_u
    is_unverified = "UNVERIFIED" in verdict_u
    fake_p = float(details.get("fake_probability", overall_conf) or overall_conf)

    if is_unverified:
        return (
            "No usable modality evidence was available. "
            "Authenticity cannot be verified."
        )

    if is_uncertain:
        if pred_cat and pred_cat not in ("AUTHENTIC", "UNVERIFIED"):
            hyp = pred_cat.replace("_", " ").lower()
            return (
                f"Evidence is inconclusive for a definitive authenticity verdict "
                f"(fused fake score {fake_p:.0%}). "
                f"Leading category hypothesis: {hyp}. "
                f"This is not a definitive authentic or deepfake decision."
            )
        return (
            f"Evidence is inconclusive for a definitive authenticity verdict "
            f"(fused fake score {fake_p:.0%}). "
            f"The framework abstains rather than claiming authenticity."
        )

    agents_str = ", ".join(contributing)
    full_consensus = {"Spatial Analysis", "Temporal Analysis", "Audio Analysis"} <= set(contributing) and not details.get("unavailable_agents")
    modality_note = (
        "Multi-agent consensus reached."
        if full_consensus
        else f"Assessment based on available modalities: {agents_str}."
    )

    if pred_cat == "AUDIO_MANIPULATED" and not is_uncertain:
        return (
            f"VERITAS multi-agent fusion detected synthetic audio manipulation on authentic video. "
            f"{modality_note} "
            f"Fused fake score {fake_p:.0%}; verdict {verdict}."
        )
    if pred_cat == "VIDEO_MANIPULATED" and not is_uncertain:
        return (
            f"VERITAS multi-agent fusion detected visual/facial deepfake manipulation with authentic audio. "
            f"{modality_note} "
            f"Fused fake score {fake_p:.0%}; verdict {verdict}."
        )
    if pred_cat == "AUDIO_VIDEO_MANIPULATED" and not is_uncertain:
        return (
            f"VERITAS multi-agent fusion detected complete multi-modal deepfake manipulation across both visual and audio streams. "
            f"{modality_note} "
            f"Fused fake score {fake_p:.0%}; verdict {verdict}."
        )
    if (pred_cat == "AUTHENTIC" or verdict_u == "AUTHENTIC") and not is_uncertain:
        if full_consensus:
            return (
                f"VERITAS multi-agent fusion verified authentic content across evaluated modalities. "
                f"Multi-agent consensus reached. "
                f"Fused fake score {fake_p:.0%}; verdict {verdict}."
            )
        return (
            f"VERITAS has analysed the submitted content. "
            f"{modality_note} "
            f"Fused fake score {fake_p:.0%}; verdict {verdict}."
        )

    return (
        f"VERITAS has analysed the submitted content. "
        f"{modality_note} "
        f"Fused fake score {fake_p:.0%}; verdict {verdict}."
    )


def _decision_payload(details: dict) -> dict:
    """Contract fields exposed on analyse / stream responses."""
    return {
        "fake_probability": details.get("fake_probability", details.get("p_fake")),
        "decision_confidence": details.get("decision_confidence"),
        "decision_thresholds": details.get("decision_thresholds"),
        "verdict_class": details.get("verdict_class"),
        "is_abstention": details.get("is_abstention"),
        "leading_category_is_hypothesis": details.get("leading_category_is_hypothesis"),
        "category_probabilities_calibrated": details.get(
            "category_probabilities_calibrated", False
        ),
        "fusion_fallback_reason": details.get("fusion_fallback_reason"),
    }


def _extract_video_frame(video_path: str) -> str | None:
    """
    Extract the middle frame from a video and save it to a temporary PNG file
    (lossless, so the spatial agent sees the decoded frame as in training).
    Returns the path to the temporary file, or None if extraction fails.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Could not open video %s for frame extraction", video_path)
        return None
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            target_frame = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
        else:
            ret, frame = cap.read()
            
        if not ret or frame is None:
            logger.warning("Could not read any frames from video %s", video_path)
            return None
            
        fd, tmp_img_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        
        success = cv2.imwrite(tmp_img_path, frame)
        if not success:
            logger.warning("Failed to save extracted frame to %s", tmp_img_path)
            if os.path.exists(tmp_img_path):
                os.remove(tmp_img_path)
            return None
            
        return tmp_img_path
        
    except Exception as e:
        logger.error("Error extracting video frame: %s", e, exc_info=True)
        return None
    finally:
        cap.release()


def _build_agents(tmp_path: str, resolved_media_type: str) -> list[dict]:
    """Run all agents and return their results in order."""
    if resolved_media_type == "audio":
        s_result = {
            "agent":       "Spatial Analysis",
            "status":      "not_applicable",
            "color":       "cyan",
            "confidence":  0.0,
            "is_fake":     None,
            "finding":     "Spatial analysis is not applicable to audio content.",
            "gradcam_b64": None,
        }
    elif resolved_media_type == "video":
        frame_path = _extract_video_frame(tmp_path)
        if frame_path:
            try:
                s_result = spatial_agent.predict(frame_path)
                s_result.setdefault("status", "trained")  # keep "no_face" from predict()
            except Exception as exc:
                logger.error("Spatial agent prediction error: %s", exc)
                s_result = {
                    "agent":       "Spatial Analysis",
                    "status":      "error",
                    "color":       "cyan",
                    "confidence":  0.0,
                    "is_fake":     None,
                    "finding":     f"Spatial analysis error: {exc}",
                    "gradcam_b64": None,
                }
            finally:
                if os.path.exists(frame_path):
                    try: os.remove(frame_path)
                    except OSError: pass
        else:
            s_result = {
                "agent":       "Spatial Analysis",
                "status":      "error",
                "color":       "cyan",
                "confidence":  0.0,
                "is_fake":     None,
                "finding":     "Failed to extract frame from video for spatial analysis.",
                "gradcam_b64": None,
            }
    else:
        try:
            s_result = spatial_agent.predict(tmp_path)
            s_result.setdefault("status", "trained")  # keep "no_face" from predict()
        except Exception as exc:
            logger.error("Spatial agent prediction error: %s", exc)
            s_result = {
                "agent":       "Spatial Analysis",
                "status":      "error",
                "color":       "cyan",
                "confidence":  0.0,
                "is_fake":     None,
                "finding":     f"Spatial analysis error: {exc}",
                "gradcam_b64": None,
            }

    if resolved_media_type != "video":
        t_result = {
            "agent":       "Temporal Analysis",
            "status":      "not_applicable",
            "confidence":  0.0,
            "is_fake":     None,
            "finding":     "Temporal analysis not applicable to non-video media.",
            "gradcam_b64": None,
        }
    else:
        t_result = run_temporal_agent(tmp_path)

    if resolved_media_type == "image":
        a_result = {
            "agent":       "Audio Analysis",
            "status":      "no_audio",
            "has_audio":   False,
            "codec":       "",
            "sample_rate": 0,
            "channels":    0,
            "duration":    0.0,
            "is_fake":     None,
            "confidence":  0,
            "finding":     "Audio analysis is not applicable to image media."
        }
    else:
        a_result = run_audio_agent(tmp_path)

    return [
        s_result,
        t_result,
        a_result,
        context_agent_result(s_result, t_result, a_result),
    ]


def _compute_verdict(
    agents: list[dict],
    media_type: str = "",
    return_details: bool = False,
    mode_override: str = "",
) -> tuple:
    """
    Compute confidence and verdict from valid agent results.
    Configurable via mode_override or VERITAS_FUSION_MODE environment variable:
      - 'corrected_fixed': Semantically corrected fixed-weight fusion (safe default).
      - 'ppo': Learned PPO adaptive fusion policy (explicitly selectable for controlled experiments).
      - 'legacy': Original fixed-weight linear average (0.40/0.30/0.20/0.10).
    """
    mode = (mode_override or os.environ.get("VERITAS_FUSION_MODE", "corrected_fixed")).lower().strip()

    if mode == "ppo":
        policy = get_fusion_policy()
        if policy.is_loaded:
            overall, verdict, contributing, unavailable, details = policy.fuse_agents(agents, media_type)
            details["method"] = "ppo"
        else:
            logger.warning("PPO policy not loaded (%s); using corrected fixed-weight fusion.", policy.load_error)
            overall, verdict, contributing, unavailable, details = compute_verdict_corrected_fixed(agents, media_type)
            details["fusion_fallback_reason"] = f"PPO policy not loaded ({policy.load_error}); corrected_fixed was used."
    elif mode == "legacy":
        overall, verdict, contributing, unavailable, details = compute_verdict_legacy(agents)
    elif mode == "corrected_fixed":
        overall, verdict, contributing, unavailable, details = compute_verdict_corrected_fixed(agents, media_type)
    else:
        logger.warning("Unknown fusion mode '%s', defaulting to corrected fixed-weight fusion.", mode)
        overall, verdict, contributing, unavailable, details = compute_verdict_corrected_fixed(agents, media_type)

    # A video whose picture was never checked (no usable face, undecodable frames) must not read as AUTHENTIC.
    visual_checked = any(a.get("is_fake") is not None for a in agents if a.get("agent") in ("Spatial Analysis", "Temporal Analysis"))
    if verdict == "AUTHENTIC" and (
        any(a.get("status") == "no_face" for a in agents) or (media_type == "video" and not visual_checked)
    ):
        verdict = VERDICT_UNCERTAIN_FULL
        details["verdict"] = verdict
        enrich_decision_fields(details)

    if return_details:
        return overall, verdict, contributing, unavailable, details
    return overall, verdict, contributing, unavailable


async def _stream_agents(
    tmp_path: str,
    resolved_media_type: str,
    mode_override: str = "",
) -> AsyncIterator[str]:
    """
    Async generator that yields newline-delimited JSON lines:
      - One line per agent result (type: "agent_result")
      - One final verdict line  (type: "final_verdict")
      - One completion line     (type: "done")

    Each line is valid JSON terminated by a single newline so Express can
    split on \\n and forward events immediately.
    """
    agents_done: list[dict] = []

    # ── 1. Spatial agent — runs the actual model (may be slow) ────────────────
    if resolved_media_type == "audio":
        s_result = {
            "agent":       "Spatial Analysis",
            "status":      "not_applicable",
            "color":       "cyan",
            "confidence":  0.0,
            "is_fake":     None,
            "finding":     "Spatial analysis is not applicable to audio content.",
            "gradcam_b64": None,
        }
    elif resolved_media_type == "video":
        frame_path = _extract_video_frame(tmp_path)
        if frame_path:
            try:
                s_result = await asyncio.get_event_loop().run_in_executor(
                    None, spatial_agent.predict, frame_path
                )
                s_result.setdefault("status", "trained")  # keep "no_face" from predict()
            except Exception as exc:
                logger.error("Spatial stream prediction error: %s", exc)
                s_result = {
                    "agent":       "Spatial Analysis",
                    "status":      "error",
                    "color":       "cyan",
                    "confidence":  0.0,
                    "is_fake":     None,
                    "finding":     f"Spatial analysis error: {exc}",
                    "gradcam_b64": None,
                }
            finally:
                if os.path.exists(frame_path):
                    try: os.remove(frame_path)
                    except OSError: pass
        else:
            s_result = {
                "agent":       "Spatial Analysis",
                "status":      "error",
                "color":       "cyan",
                "confidence":  0.0,
                "is_fake":     None,
                "finding":     "Failed to extract frame from video for spatial analysis.",
                "gradcam_b64": None,
            }
    else:
        # Offload CPU-bound work to a thread so we don't block the event loop
        try:
            s_result = await asyncio.get_event_loop().run_in_executor(
                None, spatial_agent.predict, tmp_path
            )
            s_result.setdefault("status", "trained")  # keep "no_face" from predict()
        except Exception as exc:
            logger.error("Spatial stream prediction error: %s", exc)
            s_result = {
                "agent":       "Spatial Analysis",
                "status":      "error",
                "color":       "cyan",
                "confidence":  0.0,
                "is_fake":     None,
                "finding":     f"Spatial analysis error: {exc}",
                "gradcam_b64": None,
            }

    agents_done.append(s_result)
    yield json.dumps({"type": "agent_result", **s_result}) + "\n"

    # Small artificial delay so the UI can animate between agents when mocks
    # respond instantly (remove if real models add enough latency).
    await asyncio.sleep(0.4)

    # ── 2. Temporal agent ────────────────────────────────────────────────────
    loop = asyncio.get_event_loop()
    if resolved_media_type != "video":
        t_result = {
            "agent":       "Temporal Analysis",
            "status":      "not_applicable",
            "confidence":  0.0,
            "is_fake":     None,
            "finding":     "Temporal analysis not applicable to non-video media.",
            "gradcam_b64": None,
        }
    else:
        t_result = await loop.run_in_executor(None, run_temporal_agent, tmp_path)
    agents_done.append(t_result)
    yield json.dumps({"type": "agent_result", **t_result}) + "\n"
    await asyncio.sleep(0.4)

    # ── 3. Audio agent ───────────────────────────────────────────────────────
    if resolved_media_type == "image":
        a_result = {
            "agent":       "Audio Analysis",
            "status":      "no_audio",
            "has_audio":   False,
            "codec":       "",
            "sample_rate": 0,
            "channels":    0,
            "duration":    0.0,
            "is_fake":     None,
            "confidence":  0,
            "finding":     "Audio analysis is not applicable to image media."
        }
    else:
        a_result = await loop.run_in_executor(None, run_audio_agent, tmp_path)
    agents_done.append(a_result)
    yield json.dumps({"type": "agent_result", **a_result}) + "\n"
    await asyncio.sleep(0.4)

    # ── 4. Context agent — cross-modal consistency over prior results ─────────
    c_result = context_agent_result(s_result, t_result, a_result)
    agents_done.append(c_result)
    yield json.dumps({"type": "agent_result", **c_result}) + "\n"

    # ── 5. Final verdict ─────────────────────────────────────────────────────
    overall_confidence, verdict, contributing, unavailable, details = _compute_verdict(
        agents_done, resolved_media_type, return_details=True, mode_override=mode_override
    )
    
    yield json.dumps({
        "type": "final_verdict",
        "verdict": verdict,
        "overall_confidence": overall_confidence,
        "contributing_agents": contributing,
        "unavailable_agents": unavailable,
        "summary": _build_summary(agents_done, overall_confidence, verdict, contributing, details),
        "agents": agents_done,
        "fusion_method": details.get("method", "corrected_fixed"),
        "predicted_category": details.get("predicted_category"),
        "category_probabilities": details.get("category_probabilities"),
        "dynamic_weights": details.get("dynamic_weights"),
        "decision_threshold": details.get("decision_threshold"),
        **_decision_payload(details),
    }) + "\n"

    # ── 6. Signal completion ─────────────────────────────────────────────────
    yield json.dumps({"type": "done"}) + "\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "models": _loaded_models()}


# ── Legacy blocking endpoint (unchanged behaviour) ───────────────────────────
@app.post("/analyse")
async def analyse(
    file: UploadFile = File(...),
    media_type: str = Form(default=""),
    fusion_mode: str = Form(default=""),
) -> dict:
    """
    Analyse an uploaded file for deepfake indicators.
    Returns a single JSON object when all agents have finished.

    Form fields
    -----------
    file        : The media file to analyse (image / video / audio).
    media_type  : Optional hint ("image", "video", "audio").
                  Inferred from the file extension when not provided.
    fusion_mode : Optional override ("corrected_fixed", "ppo", "legacy").
    """
    start_ms = time.time()
    tmp_path: str | None = None

    try:
        filename = file.filename or "upload"
        resolved_media_type = _resolve_media_type(filename, media_type)

        contents = await file.read()
        if len(contents) > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File exceeds 50MB limit.")
        suffix = Path(filename).suffix or ".tmp"
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(tmp_fd, "wb") as fh:  # the finally block below removes tmp_path
            fh.write(contents)

        agents = _build_agents(tmp_path, resolved_media_type)
        overall_confidence, verdict, contributing, unavailable, details = _compute_verdict(
            agents, resolved_media_type, return_details=True, mode_override=fusion_mode
        )
        processing_time_ms = round((time.time() - start_ms) * 1000)

        return {
            "verdict": verdict,
            "overall_confidence": overall_confidence,
            "contributing_agents": contributing,
            "unavailable_agents": unavailable,
            "summary": _build_summary(agents, overall_confidence, verdict, contributing, details),
            "agents": agents,
            "media_type": resolved_media_type,
            "processing_time_ms": processing_time_ms,
            "fusion_method": details.get("method", "corrected_fixed"),
            "predicted_category": details.get("predicted_category"),
            "category_probabilities": details.get("category_probabilities"),
            "dynamic_weights": details.get("dynamic_weights"),
            "decision_threshold": details.get("decision_threshold"),
            **_decision_payload(details),
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.error("Error in /analyse: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as cleanup_err:
                logger.warning("Could not delete temp file %s: %s", tmp_path, cleanup_err)


# ── NEW streaming endpoint ────────────────────────────────────────────────────
@app.post("/analyse/stream")
async def analyse_stream(
    file: UploadFile = File(...),
    media_type: str = Form(default=""),
    fusion_mode: str = Form(default=""),
):
    """
    Streaming version of /analyse.

    Returns a StreamingResponse of newline-delimited JSON (NDJSON).
    Each line is a JSON object with a "type" field:
      { "type": "agent_result",   "agent": "...", "confidence": 0.87, ... }
      { "type": "final_verdict",  "verdict": "FAKE", "overall_confidence": 0.81, ... }
      { "type": "done" }

    The Express BFF converts each line into a named SSE event for the frontend.
    """
    filename = file.filename or "upload"
    resolved_media_type = _resolve_media_type(filename, media_type)

    # Save the upload to a temp file up front so the stream can reference it
    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded file.")
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 50MB limit.")

    suffix = Path(filename).suffix or ".tmp"
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            fh.write(contents)
    except Exception:
        os.remove(tmp_path)  # no stream will run, so nothing else will clean it up
        raise HTTPException(status_code=500, detail="Could not store uploaded file.")

    async def generate():
        try:
            async for line in _stream_agents(tmp_path, resolved_media_type, mode_override=fusion_mode):
                yield line
        finally:
            # Always clean up the temp file, even if the client disconnects
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as e:
                    logger.warning("Could not delete temp file %s: %s", tmp_path, e)

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            # Tell Express/proxies not to buffer the response
            "X-Accel-Buffering": "no",
            "Cache-Control": "no-cache",
        },
    )


# ---------------------------------------------------------------------------
# Entry-point (run directly with `python main.py`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
