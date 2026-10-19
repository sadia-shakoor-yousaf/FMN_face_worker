"""Decision rules for one visitor visit. No camera, mic, or model."""

from __future__ import annotations

import unittest

from face_cv.visit import UNKNOWN_LABEL, VisitDecider


def _decider(**kwargs) -> VisitDecider:
    defaults = dict(need=5, cooldown_s=120, unknown_hold_s=10, voice_wait_s=30, dropout_s=1)
    defaults.update(kwargs)
    return VisitDecider(**defaults)


class VisitDeciderTests(unittest.TestCase):
    def test_known_face_emits_on_streak_without_listening(self) -> None:
        decider = _decider()
        for i in range(4):
            step = decider.push("Ammi", 0.8, i * 0.2)
            self.assertIsNone(step.event)
            self.assertFalse(step.start_listen)
        step = decider.push("Ammi", 0.81, 1.0)
        self.assertIsNotNone(step.event)
        assert step.event is not None
        self.assertTrue(step.event.known)
        self.assertEqual(step.event.name, "Ammi")
        self.assertEqual(step.event.via, "face")
        self.assertFalse(step.start_listen)

    def test_unknown_hold_opens_the_mic_and_does_not_alert(self) -> None:
        decider = _decider()
        decider.push(UNKNOWN_LABEL, 0.4, 0.0)
        early = decider.push(UNKNOWN_LABEL, 0.4, 9.9)
        self.assertFalse(early.start_listen)
        self.assertIsNone(early.event)
        started = decider.push(UNKNOWN_LABEL, 0.4, 10.0)
        self.assertTrue(started.start_listen)
        self.assertIsNone(started.event)
        self.assertEqual(decider.phase, "listen")

    def test_brief_dropout_does_not_reset_the_unknown_clock(self) -> None:
        decider = _decider()
        decider.push(UNKNOWN_LABEL, 0.4, 0.0)
        decider.push(None, 0.0, 0.4)
        started = decider.push(UNKNOWN_LABEL, 0.4, 10.0)
        self.assertTrue(started.start_listen)

    def test_long_gap_restarts_the_unknown_clock(self) -> None:
        decider = _decider()
        decider.push(UNKNOWN_LABEL, 0.4, 0.0)
        decider.push(None, 0.0, 0.2)
        decider.push(None, 0.0, 1.5)
        self.assertFalse(decider.push(UNKNOWN_LABEL, 0.4, 1.6).start_listen)
        self.assertFalse(decider.push(UNKNOWN_LABEL, 0.4, 11.5).start_listen)
        self.assertTrue(decider.push(UNKNOWN_LABEL, 0.4, 11.6).start_listen)

    def test_silence_for_the_full_wait_alerts_unknown(self) -> None:
        decider = _decider()
        decider.push(UNKNOWN_LABEL, 0.4, 0.0)
        decider.push(UNKNOWN_LABEL, 0.4, 10.0)
        self.assertIsNone(decider.tick(39.9).event)
        step = decider.tick(40.0)
        self.assertIsNotNone(step.event)
        assert step.event is not None
        self.assertFalse(step.event.known)
        self.assertEqual(step.event.via, "silence")
        self.assertTrue(step.stop_listen)
        self.assertEqual(decider.phase, "watch")

    def test_speech_blocks_the_silence_timeout_until_the_clip_resolves(self) -> None:
        decider = _decider(speech_grace_s=15)
        decider.push(UNKNOWN_LABEL, 0.4, 0.0)
        decider.push(UNKNOWN_LABEL, 0.4, 10.0)
        decider.note_speech(39.0)
        self.assertIsNone(decider.tick(40.0).event)
        step = decider.resolve_voice(matched=True, name="Ammi", confidence=0.55, now_s=42.0)
        self.assertIsNotNone(step.event)
        assert step.event is not None
        self.assertEqual(step.event.name, "Ammi")
        self.assertEqual(step.event.via, "voice")
        self.assertTrue(step.event.known)
        self.assertTrue(step.stop_listen)

    def test_speech_that_does_not_match_alerts_unknown(self) -> None:
        decider = _decider()
        decider.push(UNKNOWN_LABEL, 0.4, 0.0)
        decider.push(UNKNOWN_LABEL, 0.4, 10.0)
        decider.note_speech(12.0)
        step = decider.resolve_voice(matched=False, name=None, confidence=0.1, now_s=15.0)
        self.assertIsNotNone(step.event)
        assert step.event is not None
        self.assertFalse(step.event.known)
        self.assertEqual(step.event.via, "voice")

    def test_known_face_during_the_wait_cancels_voice(self) -> None:
        decider = _decider()
        decider.push(UNKNOWN_LABEL, 0.4, 0.0)
        decider.push(UNKNOWN_LABEL, 0.4, 10.0)
        step = None
        for i in range(5):
            step = decider.push("Ammi", 0.9, 11.0 + i)
        assert step is not None and step.event is not None
        self.assertEqual(step.event.name, "Ammi")
        self.assertEqual(step.event.via, "face")
        self.assertTrue(step.stop_listen)
        self.assertEqual(decider.phase, "watch")
        self.assertIsNone(decider.tick(50.0).event)

    def test_voice_result_after_face_wins_is_ignored(self) -> None:
        decider = _decider()
        decider.push(UNKNOWN_LABEL, 0.4, 0.0)
        decider.push(UNKNOWN_LABEL, 0.4, 10.0)
        for i in range(5):
            decider.push("Ammi", 0.9, 11.0 + i)
        late = decider.resolve_voice(matched=True, name="Doctor", confidence=0.7, now_s=16.0)
        self.assertIsNone(late.event)
        self.assertFalse(late.stop_listen)

    def test_same_visit_does_not_listen_again_until_the_face_leaves(self) -> None:
        decider = _decider()
        for i in range(5):
            decider.push("Ammi", 0.9, float(i))
        held = decider.push(UNKNOWN_LABEL, 0.2, 20.0)
        self.assertFalse(held.start_listen)
        decider.push(None, 0.0, 21.0)
        decider.push(None, 0.0, 22.5)
        self.assertFalse(decider.push(UNKNOWN_LABEL, 0.2, 23.0).start_listen)
        self.assertTrue(decider.push(UNKNOWN_LABEL, 0.2, 33.0).start_listen)

    def test_no_voice_posts_unknown_when_the_hold_ends(self) -> None:
        decider = _decider(voice=False)
        decider.push(UNKNOWN_LABEL, 0.3, 0.0)
        step = decider.push(UNKNOWN_LABEL, 0.3, 10.0)
        self.assertFalse(step.start_listen)
        self.assertIsNotNone(step.event)
        assert step.event is not None
        self.assertFalse(step.event.known)
        self.assertEqual(step.event.via, "face")


if __name__ == "__main__":
    unittest.main()
