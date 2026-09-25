"""Streaming visitor decisions: same streak + cooldown policy as the fall worker.

A face sitting in front of the door camera for ten seconds is one visitor, not
fifty. Require NEED consecutive frames agreeing on the same identity, then hold
a per-identity cooldown so the caregiver gets one event per visit.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pipeline import UNKNOWN_LABEL

DEFAULT_NEED = 5
DEFAULT_COOLDOWN_S = 120.0


@dataclass
class VisitorEvent:
    name: str
    confidence: float
    known: bool
    frame_idx: int


class LiveFaceRecognizer:
    def __init__(
        self,
        *,
        need: int = DEFAULT_NEED,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
    ) -> None:
        self.need = need
        self.cooldown_s = cooldown_s

        self._streak_name: str | None = None
        self._run = 0
        self._last_name: str | None = None
        self._last_score = 0.0
        self._last_event_t: dict[str, float] = {}
        self.frame_idx = 0

    @property
    def last_name(self) -> str | None:
        return self._last_name

    @property
    def last_score(self) -> float:
        return self._last_score

    @property
    def consecutive_hits(self) -> int:
        return self._run

    def reset(self) -> None:
        self._streak_name = None
        self._run = 0
        self._last_name = None
        self._last_score = 0.0
        self.frame_idx = 0

    def push(
        self,
        name: str | None,
        score: float,
        now_s: float,
    ) -> VisitorEvent | None:
        """Push one frame's identity (``None`` = no face in frame).

        Returns a VisitorEvent when the streak is met and cooldown is clear.
        """
        self.frame_idx += 1

        if name is None:
            self._streak_name = None
            self._run = 0
            self._last_name = None
            self._last_score = 0.0
            return None

        if name == self._streak_name:
            self._run += 1
        else:
            self._streak_name = name
            self._run = 1

        self._last_name = name
        self._last_score = score

        if self._run < self.need:
            return None

        last_t = self._last_event_t.get(name)
        if last_t is not None and (now_s - last_t) < self.cooldown_s:
            return None

        self._last_event_t[name] = now_s
        self._run = 0
        return VisitorEvent(
            name=name,
            confidence=score,
            known=name != UNKNOWN_LABEL,
            frame_idx=self.frame_idx,
        )
