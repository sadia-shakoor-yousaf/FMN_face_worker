"""Microphone session for the voice fallback.

Opened only after the face has been unknown long enough. Closed as soon as
the visit decision is made. A speech onset is reported immediately; the
speaker match follows once a few seconds of that speech are captured. The
worker writes that clip under data/live_logs. It is not uploaded.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

from face_cv.speakers import VOICE_COLLECTION, SpeakerMatch
from face_cv.speech import DEFAULT_SAMPLE_RATE, SpeechGate


@dataclass(frozen=True)
class SpeechStarted:
    pass


@dataclass(frozen=True)
class SpeechMatched:
    match: SpeakerMatch
    audio: np.ndarray | None = None
    sample_rate: int = DEFAULT_SAMPLE_RATE


class Ear:
    def __init__(
        self,
        *,
        patient_id: int | None,
        threshold: float,
        log: logging.Logger,
        collection: str = VOICE_COLLECTION,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        rms_threshold: float = 0.02,
        clip_s: float = 3.0,
    ) -> None:
        self._patient_id = patient_id
        self._threshold = threshold
        self._collection = collection
        self._log = log
        self._sample_rate = sample_rate
        self._rms_threshold = rms_threshold
        self._clip_s = clip_s
        self._out: queue.Queue[SpeechStarted | SpeechMatched] = queue.Queue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._session = 0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.stop()
        with self._lock:
            self._session += 1
            session = self._session
            self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(session,),
            name="fmn-ear",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def poll(self) -> list[SpeechStarted | SpeechMatched]:
        notes: list[SpeechStarted | SpeechMatched] = []
        while True:
            try:
                notes.append(self._out.get_nowait())
            except queue.Empty:
                return notes

    def _emit(self, session: int, note: SpeechStarted | SpeechMatched) -> None:
        with self._lock:
            if session != self._session or self._stop.is_set():
                return
            self._out.put(note)

    def _run(self, session: int) -> None:
        try:
            import sounddevice as sd
        except Exception as e:  # noqa: BLE001 — missing PortAudio is a runtime fact
            self._log.error("microphone unavailable: %s", e)
            return

        gate = SpeechGate(
            sample_rate=self._sample_rate,
            rms_threshold=self._rms_threshold,
            clip_s=self._clip_s,
        )
        clip: dict[str, object] = {}

        def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            del frames, time_info
            if status:
                self._log.debug("mic status: %s", status)
            if self._stop.is_set() or session != self._session:
                raise sd.CallbackStop()
            mono = indata[:, 0].copy()
            was_capturing = gate.capturing
            audio = gate.push(mono)
            if gate.capturing and not was_capturing:
                self._emit(session, SpeechStarted())
            if audio is not None:
                clip["audio"] = audio
                raise sd.CallbackStop()

        try:
            with sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._sample_rate // 10,
                callback=callback,
            ):
                while (
                    not self._stop.is_set()
                    and session == self._session
                    and "audio" not in clip
                ):
                    time.sleep(0.05)
        except Exception as e:  # noqa: BLE001 — device open / stream errors
            self._log.error("microphone stream failed: %s", e)
            return

        audio = clip.get("audio")
        if audio is None or self._stop.is_set() or session != self._session:
            return
        self._match_clip(session, np.asarray(audio, dtype=np.float32))

    def _match_clip(self, session: int, audio: np.ndarray) -> None:
        clip = np.asarray(audio, dtype=np.float32)
        note = SpeechMatched(SpeakerMatch(False, "", 0.0), audio=clip, sample_rate=self._sample_rate)
        if not self._patient_id:
            self._log.warning("speech captured but --patient-id is missing; treating as no match")
            self._emit(session, note)
            return
        try:
            from face_cv.speaker_embed import embed_numpy
            from face_cv.voice_qdrant import search_voice

            embedding = embed_numpy(clip, self._sample_rate)
            found = search_voice(
                patient_id=self._patient_id,
                embedding=embedding,
                threshold=self._threshold,
                collection=self._collection,
            )
            note = SpeechMatched(found, audio=clip, sample_rate=self._sample_rate)
        except Exception as e:  # noqa: BLE001 — missing SpeechBrain or Qdrant
            self._log.warning("voice match failed: %s", e)
        self._emit(session, note)
