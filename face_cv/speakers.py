"""Speaker match result. The vectors themselves live in Qdrant ``visitor_voices``."""

from __future__ import annotations

from dataclasses import dataclass

SPEAKER_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
SPEAKER_DIM = 192
DEFAULT_SPEAKER_THRESHOLD = 0.25
VOICE_COLLECTION = "visitor_voices"


@dataclass(frozen=True)
class SpeakerMatch:
    matched: bool
    name: str
    confidence: float
    relationship: str = ""


def match_from_hit(
    *,
    name: str | None,
    score: float,
    threshold: float,
    embedding_model: str | None = None,
    relationship: str | None = None,
) -> SpeakerMatch:
    """Turn one Qdrant hit into a name or a miss.

    A point enrolled with a different model is a miss. Those vectors are not
    in the same space as the care-PC ECAPA embedding.
    """
    relation = (relationship or "").strip()
    if embedding_model and embedding_model != SPEAKER_MODEL:
        return SpeakerMatch(False, "", float(max(score, 0.0)), relation)
    label = (name or "").strip()
    if label and score >= threshold:
        return SpeakerMatch(True, label, float(score), relation)
    return SpeakerMatch(False, "", float(max(score, 0.0)), relation)
