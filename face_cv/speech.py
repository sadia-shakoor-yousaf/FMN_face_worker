"""Energy gate for the voice fallback.

The 30 second clock is how long we wait for someone to start talking.
Once speech is under way, a few seconds of audio is enough for ECAPA.
This module does not score speakers.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_RMS_THRESHOLD = 0.02
DEFAULT_ONSET_S = 0.25
DEFAULT_CLIP_S = 3.0
DEFAULT_PREROLL_S = 0.3


def rms(samples: np.ndarray) -> float:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio * audio)))


class SpeechGate:
    """Turns a mic stream into one short clip once speech has started."""

    def __init__(
        self,
        *,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        rms_threshold: float = DEFAULT_RMS_THRESHOLD,
        onset_s: float = DEFAULT_ONSET_S,
        clip_s: float = DEFAULT_CLIP_S,
        preroll_s: float = DEFAULT_PREROLL_S,
    ) -> None:
        self.sample_rate = sample_rate
        self.rms_threshold = rms_threshold
        self.onset_n = max(1, int(onset_s * sample_rate))
        self.clip_n = max(1, int(clip_s * sample_rate))
        self.preroll_n = max(0, int(preroll_s * sample_rate))
        self.reset()

    @property
    def capturing(self) -> bool:
        return self._capturing

    def reset(self) -> None:
        self._preroll: list[np.ndarray] = []
        self._preroll_n = 0
        self._hot = 0
        self._capturing = False
        self._buf: list[np.ndarray] = []
        self._buf_n = 0

    def push(self, samples: np.ndarray) -> np.ndarray | None:
        """Feed one mic block. Returns the clip once it is long enough."""
        audio = np.asarray(samples, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return None
        if self._capturing:
            return self._capture(audio)
        self._remember(audio)
        if rms(audio) >= self.rms_threshold:
            self._hot += int(audio.size)
        else:
            self._hot = 0
        if self._hot < self.onset_n:
            return None
        self._capturing = True
        rolled = self._rolled()
        self._buf = [rolled]
        self._buf_n = int(rolled.size)
        if self._buf_n >= self.clip_n:
            return self._finish()
        return None

    def _remember(self, audio: np.ndarray) -> None:
        self._preroll.append(audio)
        self._preroll_n += int(audio.size)
        while self._preroll and (self._preroll_n - int(self._preroll[0].size)) >= self.preroll_n:
            self._preroll_n -= int(self._preroll.pop(0).size)

    def _rolled(self) -> np.ndarray:
        if not self._preroll:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._preroll)

    def _capture(self, audio: np.ndarray) -> np.ndarray | None:
        self._buf.append(audio)
        self._buf_n += int(audio.size)
        if self._buf_n >= self.clip_n:
            return self._finish()
        return None

    def _finish(self) -> np.ndarray:
        clip = np.concatenate(self._buf)[: self.clip_n]
        self.reset()
        return clip


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    ints = (pcm * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(ints.tobytes())
