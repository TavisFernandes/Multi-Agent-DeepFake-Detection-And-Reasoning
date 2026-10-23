"""
mock_agents.py

Context Verification agent.

Unlike the spatial/temporal/audio agents, "context" is NOT a trained neural
network — FakeAVCeleb has no context-classification target, and a model trained
on its folder structure would merely leak the label. Instead, the context agent
performs *cross-modal consistency analysis*: it compares what the visual agents
(spatial + temporal) concluded against what the audio agent concluded.

This is the exact signal FakeAVCeleb's category taxonomy encodes:
    RealVideo-RealAudio  → both modalities agree REAL
    RealVideo-FakeAudio  → visual REAL, audio FAKE   (mismatch → manipulated)
    FakeVideo-RealAudio  → visual FAKE, audio REAL   (mismatch → manipulated)
    FakeVideo-FakeAudio  → both modalities agree FAKE

Explanation only
----------------
The agent only re-describes the other agents' scores, so it adds no independent
evidence (on fusion_val, adding the visual–audio gap to a logistic fusion moved
ROC-AUC 0.9844 → 0.9852). It therefore never votes: is_fake is always None,
status is "explanation" when it can describe the scores, and confidence carries
the visual–audio score gap |P_visual − P_audio| (not a fake probability).
- Fewer than two modalities → "not_applicable", nothing to compare.
- Both available → a neutral description of whether the scores agree.
"""

from typing import Optional

_AGENT_NAME = "Context Verification"
_COLOR = "purple"


def _visual_fake_prob(spatial: Optional[dict], temporal: Optional[dict]) -> Optional[float]:
    """
    Combine the visual agents into a single P(FAKE), weighting spatial:temporal
    2:1 (mirrors the 0.40:0.30 fusion weights). Returns None if neither made a
    definitive prediction.
    """
    parts: list[tuple[float, float]] = []  # (weight, fake_prob)
    if spatial and spatial.get("is_fake") is not None:
        parts.append((0.40, float(spatial.get("confidence", 0.0))))
    if temporal and temporal.get("is_fake") is not None:
        parts.append((0.30, float(temporal.get("confidence", 0.0))))
    if not parts:
        return None
    wsum = sum(w for w, _ in parts)
    return sum(w * p for w, p in parts) / wsum


def _unavailable(reason: str) -> dict:
    """Old-stub-compatible response: no authenticity contribution."""
    return {
        "agent":      _AGENT_NAME,
        "status":     "not_applicable",
        "color":      _COLOR,
        "confidence": 0,
        "is_fake":    None,
        "finding":    reason,
    }


def context_agent_result(
    spatial_result: Optional[dict] = None,
    temporal_result: Optional[dict] = None,
    audio_result: Optional[dict] = None,
) -> dict:
    """
    Explanation-only description of visual vs audio agreement (never votes).

    Parameters
    ----------
    spatial_result, temporal_result, audio_result : dict | None
        The upstream agent results produced earlier in the pipeline. Any may be
        None or carry is_fake=None when a modality does not apply.

    Returns
    -------
    dict with the standard agent shape; is_fake is always None.
    """
    # Called with no context (e.g. legacy callers) → behave like the old stub.
    if spatial_result is None and temporal_result is None and audio_result is None:
        return _unavailable(
            "Context agent requires upstream agent results. Authenticity cannot be verified."
        )

    visual_p = _visual_fake_prob(spatial_result, temporal_result)
    audio_has = bool(audio_result) and audio_result.get("is_fake") is not None
    audio_p = float(audio_result.get("confidence", 0.0)) if audio_has else None

    # Need both a visual and an audio verdict to reason about consistency.
    if visual_p is None or audio_p is None:
        if visual_p is None and audio_p is None:
            reason = "No modality verdicts available for cross-modal consistency analysis."
        elif audio_p is None:
            reason = (
                "Only a visual verdict is available (no audio track). "
                "Cross-modal consistency cannot be assessed."
            )
        else:
            reason = (
                "Only an audio verdict is available (no visual analysis). "
                "Cross-modal consistency cannot be assessed."
            )
        return _unavailable(reason)

    divergence = abs(visual_p - audio_p)   # 0 (identical) … 1 (opposite)
    direction = "point the same way" if (visual_p >= 0.5) == (audio_p >= 0.5) else "point in different directions"
    return {
        "agent":      _AGENT_NAME,
        "status":     "explanation",
        "color":      _COLOR,
        "confidence": round(divergence, 4),
        "is_fake":    None,
        "finding":    (
            f"The visual score ({visual_p:.0%}) and audio score ({audio_p:.0%}) {direction}, "
            f"a gap of {divergence:.0%}. This restates the other models' outputs; "
            f"it is not separate evidence and does not change the verdict."
        ),
    }
