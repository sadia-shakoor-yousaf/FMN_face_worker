"""One visit, one event. Face decides. Voice is a fallback, not a second voter.

A known face still announces on a short streak. An unknown face has to stay
unknown for ``unknown_hold_s`` (brief dropouts do not reset that clock) before
the microphone is opened. The mic then waits up to ``voice_wait_s`` for speech
to *start*. A few seconds of that speech are matched elsewhere; this class
only decides which single visitor event to emit.

If the face becomes known while the mic is waiting, the face wins and the
wait ends. After an event, nothing new is emitted until that face has left
and the per-identity cooldown has cleared.
"""

from __future__ import annotations

from dataclasses import dataclass

UNKNOWN_LABEL = "unknown"

DEFAULT_NEED = 5
DEFAULT_COOLDOWN_S = 120.0
DEFAULT_UNKNOWN_HOLD_S = 10.0
DEFAULT_VOICE_WAIT_S = 30.0
DEFAULT_DROPOUT_S = 1.0
# Cold ECAPA load can take a while after speech has already started.
DEFAULT_SPEECH_GRACE_S = 90.0


@dataclass(frozen=True)
class EmittedVisit:
    name: str
    confidence: float
    known: bool
    via: str  # "face" | "voice" | "silence"


@dataclass(frozen=True)
class VisitStep:
    start_listen: bool = False
    stop_listen: bool = False
    event: EmittedVisit | None = None


class VisitDecider:
    def __init__(
        self,
        *,
        need: int = DEFAULT_NEED,
        cooldown_s: float = DEFAULT_COOLDOWN_S,
        unknown_hold_s: float = DEFAULT_UNKNOWN_HOLD_S,
        voice_wait_s: float = DEFAULT_VOICE_WAIT_S,
        dropout_s: float = DEFAULT_DROPOUT_S,
        speech_grace_s: float = DEFAULT_SPEECH_GRACE_S,
        voice: bool = True,
    ) -> None:
        self.need = need
        self.cooldown_s = cooldown_s
        self.unknown_hold_s = unknown_hold_s
        self.voice_wait_s = voice_wait_s
        self.dropout_s = dropout_s
        self.speech_grace_s = speech_grace_s
        self.voice = voice

        self.phase = "watch"
        self.frame_idx = 0
        self._run = 0
        self._streak_name: str | None = None
        self._unknown_since: float | None = None
        self._last_unknown_t: float | None = None
        self._last_unknown_score = 0.0
        self._listen_started: float | None = None
        self._speech = False
        self._speech_deadline: float | None = None
        self._last_event_t: dict[str, float] = {}
        self._busy_visit = False
        self._absent_since: float | None = None

    @property
    def consecutive_hits(self) -> int:
        return self._run

    def unknown_elapsed(self, now_s: float) -> float:
        if self.phase != "watch" or self._unknown_since is None:
            return 0.0
        return max(0.0, now_s - self._unknown_since)

    def listen_elapsed(self, now_s: float) -> float:
        if self.phase != "listen" or self._listen_started is None:
            return 0.0
        return max(0.0, now_s - self._listen_started)

    def push(self, name: str | None, score: float, now_s: float) -> VisitStep:
        """One processed frame. ``None`` means the detector found no face."""
        self.frame_idx += 1
        if name is None:
            return self._on_no_face(now_s)
        self._absent_since = None
        if name == UNKNOWN_LABEL:
            return self._on_unknown(score, now_s)
        return self._on_known(name, score, now_s)

    def tick(self, now_s: float) -> VisitStep:
        """Fire the silence timeout even when the face label has not changed."""
        if self.phase != "listen" or self._listen_started is None:
            return VisitStep()
        if self._speech:
            if self._speech_deadline is not None and now_s >= self._speech_deadline:
                return self._emit_unknown(self._last_unknown_score, now_s, via="silence")
            return VisitStep()
        if now_s - self._listen_started >= self.voice_wait_s:
            return self._emit_unknown(self._last_unknown_score, now_s, via="silence")
        return VisitStep()

    def note_speech(self, now_s: float) -> None:
        """Speech has started. The 30s clock stops; the clip still has to finish."""
        if self.phase != "listen" or self._speech:
            return
        self._speech = True
        self._speech_deadline = now_s + self.speech_grace_s

    def resolve_voice(
        self,
        *,
        matched: bool,
        name: str | None,
        confidence: float,
        now_s: float,
    ) -> VisitStep:
        """Result of the short clip. Ignored if the face already ended the wait."""
        if self.phase != "listen":
            return VisitStep()
        if matched and name and name != UNKNOWN_LABEL:
            if self._cooling(name, now_s):
                self._leave_listen()
                self._last_event_t[UNKNOWN_LABEL] = now_s
                self._busy_visit = True
                return VisitStep(stop_listen=True)
            self._leave_listen()
            self._last_event_t[name] = now_s
            self._last_event_t[UNKNOWN_LABEL] = now_s
            self._busy_visit = True
            return VisitStep(
                stop_listen=True,
                event=EmittedVisit(
                    name=name,
                    confidence=confidence,
                    known=True,
                    via="voice",
                ),
            )
        score = confidence if confidence > 0 else self._last_unknown_score
        return self._emit_unknown(score, now_s, via="voice")

    def _on_no_face(self, now_s: float) -> VisitStep:
        self._reset_streak()
        if self._absent_since is None:
            self._absent_since = now_s
        gone_for = now_s - self._absent_since
        if gone_for >= self.dropout_s:
            if self._busy_visit:
                self._busy_visit = False
            if self.phase != "listen":
                self._unknown_since = None
                self._last_unknown_t = None
        return VisitStep()

    def _on_unknown(self, score: float, now_s: float) -> VisitStep:
        self._reset_streak()
        self._last_unknown_score = score
        if self.phase == "listen":
            return VisitStep()
        if self._busy_visit:
            return VisitStep()
        if self._unknown_since is None:
            self._unknown_since = now_s
        self._last_unknown_t = now_s
        if (now_s - self._unknown_since) < self.unknown_hold_s:
            return VisitStep()
        if self._cooling(UNKNOWN_LABEL, now_s):
            return VisitStep()
        if not self.voice:
            return self._emit_unknown(score, now_s, via="face")
        self.phase = "listen"
        self._listen_started = now_s
        self._speech = False
        self._speech_deadline = None
        return VisitStep(start_listen=True)

    def _on_known(self, name: str, score: float, now_s: float) -> VisitStep:
        if name == self._streak_name:
            self._run += 1
        else:
            self._streak_name = name
            self._run = 1
        if self._run < self.need:
            if self.phase != "listen":
                self._unknown_since = None
                self._last_unknown_t = None
            return VisitStep()
        if self._cooling(name, now_s):
            # Already announced this person. Do not let the silence timeout
            # turn a recovered face into an unknown alert.
            if self.phase == "listen":
                self._leave_listen()
                self._busy_visit = True
                self._reset_streak()
                return VisitStep(stop_listen=True)
            return VisitStep()
        stopping = self.phase == "listen"
        self._leave_listen()
        self._last_event_t[name] = now_s
        self._busy_visit = True
        self._reset_streak()
        return VisitStep(
            stop_listen=stopping,
            event=EmittedVisit(
                name=name,
                confidence=score,
                known=True,
                via="face",
            ),
        )

    def _emit_unknown(self, score: float, now_s: float, *, via: str) -> VisitStep:
        self._leave_listen()
        self._last_event_t[UNKNOWN_LABEL] = now_s
        self._busy_visit = True
        self._reset_streak()
        return VisitStep(
            stop_listen=True,
            event=EmittedVisit(
                name=UNKNOWN_LABEL,
                confidence=score,
                known=False,
                via=via,
            ),
        )

    def _leave_listen(self) -> None:
        self.phase = "watch"
        self._listen_started = None
        self._speech = False
        self._speech_deadline = None
        self._unknown_since = None
        self._last_unknown_t = None

    def _reset_streak(self) -> None:
        self._streak_name = None
        self._run = 0

    def _cooling(self, name: str, now_s: float) -> bool:
        last = self._last_event_t.get(name)
        return last is not None and (now_s - last) < self.cooldown_s
