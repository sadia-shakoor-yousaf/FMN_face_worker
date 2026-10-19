"""Qdrant hit → name. No network."""

from __future__ import annotations

import unittest

from face_cv.speakers import SPEAKER_MODEL, match_from_hit


class MatchFromHitTests(unittest.TestCase):
    def test_score_over_threshold_names_them(self) -> None:
        found = match_from_hit(
            name="Ammi",
            score=0.42,
            threshold=0.25,
            embedding_model=SPEAKER_MODEL,
        )
        self.assertTrue(found.matched)
        self.assertEqual(found.name, "Ammi")

    def test_below_threshold_is_unknown(self) -> None:
        found = match_from_hit(name="Ammi", score=0.1, threshold=0.25)
        self.assertFalse(found.matched)
        self.assertEqual(found.name, "")

    def test_wrong_model_is_unknown(self) -> None:
        found = match_from_hit(
            name="Ammi",
            score=0.9,
            threshold=0.25,
            embedding_model="some-other-model",
        )
        self.assertFalse(found.matched)


if __name__ == "__main__":
    unittest.main()
