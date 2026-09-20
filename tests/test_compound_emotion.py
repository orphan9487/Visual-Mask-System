"""Unit tests for presentation-level compound emotion resolution."""

from __future__ import annotations

import unittest

from src.reasoning.compound import resolve_emotions


class CompoundEmotionTests(unittest.TestCase):
    def test_incongruent_text_and_emoji_produce_primary_and_secondary(self):
        result = resolve_emotions({
            "emotion": "sadness",
            "text_emotion": "sadness",
            "incongruent": True,
            "emoji_signal": {"emotion": "joy", "strength": 0.8},
        })

        self.assertEqual(result["compound_name"], "口是心非")
        self.assertEqual(
            [item["emotion"] for item in result["emotions"]],
            ["sadness", "joy"],
        )
        self.assertEqual(result["emotions"][0]["role"], "primary")
        self.assertEqual(result["emotions"][1]["role"], "secondary")

    def test_congruent_signal_remains_single_emotion(self):
        result = resolve_emotions({
            "emotion": "joy",
            "source": "fused",
            "incongruent": False,
            "emoji_signal": {"emotion": "joy", "strength": 0.9},
        })

        self.assertIsNone(result["compound_name"])
        self.assertEqual(result["emotions"], [
            {"emotion": "joy", "role": "primary", "source": "fused"}
        ])

    def test_weak_secondary_signal_is_not_presented_as_compound(self):
        result = resolve_emotions({
            "emotion": "fear",
            "incongruent": True,
            "emoji_signal": {"emotion": "surprise", "strength": 0.2},
        })

        self.assertIsNone(result["compound_name"])
        self.assertEqual(len(result["emotions"]), 1)


if __name__ == "__main__":
    unittest.main()
