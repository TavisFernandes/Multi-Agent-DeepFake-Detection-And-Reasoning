"""
audio_agent.py

Deepfake audio detection via mel-spectrogram + EfficientNet-B0.

The model treats a 3-second audio clip as a 224×224 image (mel spectrogram)
and classifies it as real (0) or synthesised/deepfake (1).

Architecture
------------
  torchvision.models.efficientnet_b0(pretrained=False)
  model.classifier[1] = nn.Linear(1280, 1)

Audio pipeline
--------------
  any media (incl. WAV/MP3) → ffmpeg → mono 16 kHz 16-bit WAV
  centre 3 s crop (wrap-pad if shorter), matching train_audio_lc.py
  WAV → torchaudio → MelSpectrogram (128 bins) → AmplitudeToDB
      → resize (224, 224) → normalise → 3-channel tensor

Public interface
----------------
  run_audio_agent(media_path: str) -> dict
      {"agent": "Audio Analysis", "confidence": float, "finding": str}
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import wave
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torchvision.models import efficientnet_b0
import imageio_ffmpeg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & device
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
# Default: augmented v2 (training/audio_v2_results.md); rollback: VERITAS_AUDIO_CHECKPOINT=audio_agent_lc.pth
_CHECKPOINT_NAME = os.environ.get("VERITAS_AUDIO_CHECKPOINT", "audio_agent_lc_v2_aug_s1.pth")
_CANDIDATE_PATHS = [
    _HERE / "models" / _CHECKPOINT_NAME,
    _HERE / ".." / "backend" / "models" / _CHECKPOINT_NAME,
    _HERE / ".." / ".." / "backend" / "models" / _CHECKPOINT_NAME,
]
_CHECKPOINT_PATH = next(
    (p for p in _CANDIDATE_PATHS if p.resolve().exists()), _CANDIDATE_PATHS[0]
)

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DECISION_THRESHOLD: float = 0.50

# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------
_SAMPLE_RATE   = 16_000
_CLIP_SAMPLES  = 48_000          # 3 seconds × 16 kHz

# MelSpectrogram parameters
_MEL_TRANSFORM = T.MelSpectrogram(
    sample_rate=_SAMPLE_RATE,
    n_fft=1024,
    hop_length=512,
    n_mels=128,
)
_DB_TRANSFORM  = T.AmplitudeToDB()

# ---------------------------------------------------------------------------
# Singleton model
# ---------------------------------------------------------------------------
_model: Optional[nn.Module] = None
_model_loaded: bool = False


def _load_model(custom_path: Optional[str] = None) -> None:
    """Build EfficientNet-B0, replace the classifier head, load checkpoint."""
    global _model, _model_loaded, _DECISION_THRESHOLD

    checkpoint_path = Path(custom_path).resolve() if custom_path else _CHECKPOINT_PATH.resolve()
    if not checkpoint_path.exists():
        logger.warning(
            "Audio agent checkpoint not found at %s. "
            "Falling back to mock responses.",
            checkpoint_path,
        )
        return

    try:
        logger.info("Loading Audio Agent checkpoint from %s …", checkpoint_path)

        model = efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(1280, 1)

        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        model.to(_DEVICE)
        model.eval()

        threshold = ckpt.get("decision_threshold")
        if threshold is not None:
            _DECISION_THRESHOLD = float(threshold)
            logger.info("Audio Agent decision threshold loaded from checkpoint: %.4f", _DECISION_THRESHOLD)
        else:
            _DECISION_THRESHOLD = 0.50
            logger.info("Audio Agent: decision_threshold not in checkpoint — using default 0.50")

        _model = model
        _model_loaded = True
        logger.info(
            "Audio Agent loaded successfully (epoch %s, threshold=%.4f).",
            ckpt.get("epoch", "?"), _DECISION_THRESHOLD
        )

    except Exception as exc:
        logger.error("Failed to load Audio Agent checkpoint: %s", exc, exc_info=True)


def load_model(checkpoint_path: Optional[str] = None) -> None:
    """Public function to load or reload the model checkpoint."""
    _load_model(checkpoint_path)


# Load eagerly at import time
_load_model()


# ---------------------------------------------------------------------------
# Audio extraction helpers
# ---------------------------------------------------------------------------

def _extract_wav(media_path: str) -> Optional[str]:
    """
    Return a path to a 16 kHz mono WAV file for the given media.

    Always runs ffmpeg, even for .wav/.mp3: the reader below only handles
    16-bit PCM WAV, so MP3, 24-bit and float WAVs must be converted.

    Returns None on ffmpeg failure (non-zero exit or exception).
    The caller is responsible for deleting the temp file created here
    (the path is a NamedTemporaryFile that has been closed but not deleted).
    """
    # Create a temp WAV and extract audio via ffmpeg
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    tmp.close()

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-i", media_path,
        "-vn",                 # strip video
        "-acodec", "pcm_s16le",
        "-ar", str(_SAMPLE_RATE),
        "-ac", "1",            # mono
        tmp_path,
        "-y",
        "-loglevel", "quiet",
    ]

    try:
        result = subprocess.run(cmd, timeout=60, capture_output=True)
        if result.returncode != 0:
            logger.warning(
                "ffmpeg returned %d for %s", result.returncode, media_path
            )
            os.unlink(tmp_path)
            return None
        return tmp_path

    except Exception as exc:
        logger.warning("ffmpeg failed: %s", exc)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return None


def _probe_audio_stream(media_path: str) -> dict:
    """
    Probe the media file using ffmpeg to detect if an audio stream exists.
    Returns:
        {
            "has_audio": bool,
            "codec": str,
            "sample_rate": int,
            "channels": int,
            "duration": float
        }
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg_exe, "-i", media_path]
    try:
        # ffmpeg prints info to stderr when run without an output file
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        output = result.stderr
        
        has_audio = False
        codec = ""
        sample_rate = 0
        channels = 0
        duration = 0.0
        
        for line in output.splitlines():
            line = line.strip()
            # Duration parse: "Duration: 00:00:04.81, start: 0.000000, bitrate: 1045 kb/s"
            if line.startswith("Duration:"):
                parts = line.split(",")
                if len(parts) > 0:
                    dur_str = parts[0].replace("Duration:", "").strip()
                    try:
                        h, m, s = dur_str.split(":")
                        duration = float(h) * 3600 + float(m) * 60 + float(s)
                    except ValueError:
                        pass

            # Audio stream parse: "Stream #0:1(und): Audio: mp3 (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 128 kb/s"
            if "Stream " in line and "Audio:" in line:
                has_audio = True
                parts = line.split("Audio:")[1].split(",")
                if len(parts) > 0:
                    codec = parts[0].strip().split()[0]
                for p in parts:
                    if "Hz" in p:
                        try:
                            sample_rate = int(p.replace("Hz", "").strip())
                        except ValueError:
                            pass
                    if "stereo" in p:
                        channels = 2
                    elif "mono" in p:
                        channels = 1
                break
                
        return {
            "has_audio": has_audio,
            "codec": codec,
            "sample_rate": sample_rate,
            "channels": channels,
            "duration": duration,
        }
    except Exception as e:
        logger.error(f"Error probing audio stream for {media_path}: {e}")
        return {"has_audio": False, "codec": "", "sample_rate": 0, "channels": 0, "duration": 0.0}

# ---------------------------------------------------------------------------
# Mel-spectrogram preprocessing
# ---------------------------------------------------------------------------

def _wav_to_tensor(wav_path: str) -> torch.Tensor:
    """
    Load a 16 kHz mono WAV file (produced by _extract_wav) and convert
    it to a (1, 3, 224, 224) tensor ready for EfficientNet-B0 inference.

    Uses the stdlib 'wave' module + numpy for I/O, avoiding the
    torchaudio.load dependency that requires torchcodec in torchaudio 2.11+.

    Steps
    -----
    1. wave.open → raw PCM bytes → float32 numpy array in [-1, 1]
    2. Mono-mix (always mono after ffmpeg -ac 1, but defensive)
    3. Clip to _CLIP_SAMPLES, zero-pad if shorter
    4. MelSpectrogram (torchaudio.transforms) → AmplitudeToDB → (1, 128, T)
    5. Resize to (1, 1, 224, 224) via F.interpolate
    6. Normalise: (x − 0.5) / 0.5
    7. Repeat channel dim ×3 → (1, 3, 224, 224)
    """
    # ── 1. Load via stdlib wave + numpy ──────────────────────────────────────
    with wave.open(wav_path, "rb") as wf:
        n_channels  = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()   # bytes per sample (should be 2 for pcm_s16le)
        n_frames    = wf.getnframes()
        raw_bytes   = wf.readframes(n_frames)

    # Decode 16-bit signed LE PCM → float32 in [-1, 1]
    audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    # Reshape to (n_channels, n_samples)
    audio_np = audio_np.reshape(-1, n_channels).T   # (channels, samples)

    # Convert to torch tensor (C, N)
    waveform = torch.from_numpy(audio_np.copy())

    # ── 2. Resample if needed (ffmpeg already produces 16 kHz; guard only) ───
    if sample_rate != _SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=_SAMPLE_RATE)
        waveform = resampler(waveform)

    # ── 3. Mono-mix ──────────────────────────────────────────────────────────
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)   # (1, N)

    # ── 4. Centre-crop / wrap-pad to _CLIP_SAMPLES (as train_audio_lc.py) ───
    n = waveform.shape[1]
    if n == 0:
        raise ValueError("Audio track is empty.")
    if n >= _CLIP_SAMPLES:
        start = (n - _CLIP_SAMPLES) // 2
        waveform = waveform[:, start:start + _CLIP_SAMPLES]
    else:
        waveform = waveform.repeat(1, -(-_CLIP_SAMPLES // n))[:, :_CLIP_SAMPLES]

    # ── 5. Mel spectrogram → dB ──────────────────────────────────────────────
    mel    = _MEL_TRANSFORM(waveform)   # (1, 128, T)
    mel_db = _DB_TRANSFORM(mel)         # (1, 128, T)

    # ── 6. Resize to (224, 224) ──────────────────────────────────────────────
    mel_4d     = mel_db.unsqueeze(0)           # (1, 1, 128, T)
    mel_resized = F.interpolate(
        mel_4d, size=(224, 224), mode="bilinear", align_corners=False
    )                                          # (1, 1, 224, 224)
    mel_2d = mel_resized.squeeze(0)            # (1, 224, 224)

    # ── 7. Normalise ─────────────────────────────────────────────────────────
    mel_norm = (mel_2d - 0.5) / 0.5           # (1, 224, 224)

    # ── 8. Repeat to 3 channels ──────────────────────────────────────────────
    tensor = mel_norm.repeat(3, 1, 1)         # (3, 224, 224)
    tensor = tensor.unsqueeze(0)              # (1, 3, 224, 224)

    return tensor.to(_DEVICE)


# On the 3 s crop the model sees, over 6,150 dataset WAVs: level -49.2 to -7.4 dBFS; energy share of the 6 strongest
# FFT bins 0.009-0.954 (the most tonal clips are all fake audio). A pure tone scores ~1.0.
_SILENCE_DBFS = -60.0
_TONE_FRACTION = 0.995


def _no_speech_reason(wav_path: str) -> Optional[str]:
    """Why the track cannot be speech (silent or a pure tone), or None. The model was trained on speech only."""
    with wave.open(wav_path, "rb") as wf:
        x = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if x.size == 0:
        return "Audio track is empty"
    start = max(0, (x.size - _CLIP_SAMPLES) // 2)
    x = np.resize(x[start:start + _CLIP_SAMPLES], _CLIP_SAMPLES)  # same centre crop / wrap-pad as the model input
    if 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12) < _SILENCE_DBFS:
        return "Audio track is silent"
    power = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    if np.sort(power)[-6:].sum() / power.sum() > _TONE_FRACTION:
        return "Audio track is a pure tone, not speech"
    return None


# ---------------------------------------------------------------------------
# Finding strings
# ---------------------------------------------------------------------------

def _make_finding_trained(confidence: float) -> str:
    """Human-readable finding for a TRAINED model's output."""
    if confidence > 0.70:
        return (
            "Synthesized speech artifacts detected. Unnatural "
            "frequency patterns and formant transitions inconsistent with organic "
            "vocal production."
        )
    if confidence >= 0.40:
        return (
            "Moderate audio anomalies present. Some frequency "
            "components show characteristics of voice synthesis."
        )
    return (
        "Audio frequency patterns consistent with authentic human "
        "speech. No synthesis artifacts detected."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_audio_agent(media_path: str) -> dict:
    """
    Run the Audio Agent on any media file (video or audio).

    Always probes the container first to determine audio stream metadata.

    If the model checkpoint is NOT loaded (untrained), returns:
      status='untrained', is_fake=None, confidence=0
    — no authenticity prediction, zero fusion contribution.

    If the model IS loaded, runs inference and returns:
      status='trained', is_fake=bool, confidence=float in [0,1]

    Parameters
    ----------
    media_path : str
        Path to a video or audio file.

    Returns
    -------
    dict
    """
    tmp_wav_path: Optional[str] = None

    try:
        # ── 1. Probe container for audio stream metadata ────────────────────────────
        probe = _probe_audio_stream(media_path)
        logger.info(
            "Audio probe [%s]: has_audio=%s codec=%s sr=%s ch=%s dur=%.2fs",
            Path(media_path).name,
            probe["has_audio"], probe["codec"],
            probe["sample_rate"], probe["channels"], probe["duration"],
        )

        # ── 2. No audio stream in container ──────────────────────────────────────
        if not probe["has_audio"]:
            return {
                "agent":       "Audio Analysis",
                "status":      "no_audio",
                "has_audio":   False,
                "codec":       "",
                "sample_rate": 0,
                "channels":    0,
                "duration":    0.0,
                "is_fake":     None,
                "confidence":  0,
                "finding":     "No audio stream detected in media container.",
            }

        # ── 3. Untrained model ──────────────────────────────────────────────────
        # Return probe metadata only — DO NOT fabricate a prediction.
        if _model is None:
            return {
                "agent":       "Audio Analysis",
                "status":      "untrained",
                "has_audio":   probe["has_audio"],
                "codec":       probe["codec"],
                "sample_rate": probe["sample_rate"],
                "channels":    probe["channels"],
                "duration":    round(probe["duration"], 3),
                "is_fake":     None,
                "confidence":  0,
                "finding":     (
                    f"Audio stream detected: {probe['codec'].upper()} "
                    f"{probe['sample_rate']} Hz "
                    f"{probe['channels']}-ch "
                    f"{probe['duration']:.2f}s. "
                    "No trained model available — authenticity not assessed."
                ),
            }

        # ── 4. Trained model — extract WAV and run inference ────────────────────
        wav_path = _extract_wav(media_path)
        if wav_path is None:
            return {
                "agent":       "Audio Analysis",
                "status":      "error",
                "has_audio":   probe["has_audio"],
                "codec":       probe["codec"],
                "sample_rate": probe["sample_rate"],
                "channels":    probe["channels"],
                "duration":    round(probe["duration"], 3),
                "is_fake":     None,
                "confidence":  0,
                "finding":     "Failed to extract audio track for analysis.",
            }

        tmp_wav_path = wav_path

        no_speech = _no_speech_reason(wav_path)
        if no_speech:
            return {
                "agent":       "Audio Analysis",
                "status":      "unavailable",
                "has_audio":   probe["has_audio"],
                "codec":       probe["codec"],
                "sample_rate": probe["sample_rate"],
                "channels":    probe["channels"],
                "duration":    round(probe["duration"], 3),
                "is_fake":     None,
                "confidence":  0,
                "finding":     f"{no_speech} — the audio model only assesses speech, so no score was produced.",
            }

        tensor = _wav_to_tensor(wav_path)

        _model.eval()
        with torch.no_grad():
            logit      = _model(tensor)                     # (1, 1)
            confidence = torch.sigmoid(logit.squeeze()).item()

        is_fake = confidence >= _DECISION_THRESHOLD

        return {
            "agent":              "Audio Analysis",
            "status":             "trained",
            "has_audio":          probe["has_audio"],
            "codec":              probe["codec"],
            "sample_rate":        probe["sample_rate"],
            "channels":           probe["channels"],
            "duration":           round(probe["duration"], 3),
            "is_fake":            is_fake,
            "confidence":         round(confidence, 4),
            "fake_probability":   round(confidence, 4),
            "decision_threshold": _DECISION_THRESHOLD,
            "finding":            _make_finding_trained(confidence),
        }

    except Exception as exc:
        logger.error("Audio Agent error: %s", exc, exc_info=True)
        return {
            "agent":       "Audio Analysis",
            "status":      "error",
            "has_audio":   False,
            "codec":       "",
            "sample_rate": 0,
            "channels":    0,
            "duration":    0.0,
            "is_fake":     None,
            "confidence":  0,
            "finding":     f"Audio analysis failed: {exc}",
        }

    finally:
        if tmp_wav_path and os.path.exists(tmp_wav_path):
            try:
                os.unlink(tmp_wav_path)
            except OSError as e:
                logger.warning("Could not delete temp WAV %s: %s", tmp_wav_path, e)
