"""Speech onset: wait until energy shows up, then keep a short clip."""

from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from face_cv.speech import SpeechGate, write_wav


class SpeechGateTests(unittest.TestCase):
    def test_silence_never_opens_a_clip(self) -> None:
        gate = SpeechGate(sample_rate=1600, onset_s=0.2, clip_s=0.5, preroll_s=0.1)
        silence = np.zeros(160, dtype=np.float32)
        for _ in range(20):
            self.assertIsNone(gate.push(silence))
        self.assertFalse(gate.capturing)

    def test_onset_then_a_few_seconds_returns_one_clip(self) -> None:
        rate = 1600
        gate = SpeechGate(
            sample_rate=rate,
            rms_threshold=0.02,
            onset_s=0.2,
            clip_s=0.5,
            preroll_s=0.1,
        )
        hop = 160  # 0.1s
        loud = np.full(hop, 0.2, dtype=np.float32)
        self.assertIsNone(gate.push(loud))
        self.assertFalse(gate.capturing)
        self.assertIsNone(gate.push(loud))
        self.assertTrue(gate.capturing)
        clip = None
        for _ in range(10):
            clip = gate.push(loud)
            if clip is not None:
                break
        self.assertIsNotNone(clip)
        assert clip is not None
        self.assertEqual(clip.shape[0], int(0.5 * rate))
        self.assertFalse(gate.capturing)

    def test_write_wav_roundtrip(self) -> None:
        samples = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "clip.wav"
            write_wav(path, samples, 16_000)
            with wave.open(str(path), "rb") as handle:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getframerate(), 16_000)
                self.assertEqual(handle.getnframes(), 1600)


if __name__ == "__main__":
    unittest.main()
