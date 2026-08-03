import unittest

from src.perception.emoji_emotion import encode_text_emojis
from src.perception.multimodal_fusion import perceive


class EmojiEmotionTests(unittest.TestCase):
    def test_new_sadness_emoji_is_recognized(self):
        signal = encode_text_emojis("今天放學下雨🥲")

        self.assertEqual(signal.emotion, "sadness")
        self.assertEqual(signal.emojis, ["🥲"])

    def test_sadness_emoji_overrides_neutral_text_prediction(self):
        result = perceive("今天放學下雨🥲", "neutral", text_confident=True)

        self.assertEqual(result.emotion, "sadness")
        self.assertEqual(result.source, "emoji")

    def test_ambiguous_emoji_is_not_forced_into_an_emotion(self):
        signal = encode_text_emojis("今天累爆了😅")

        self.assertIsNone(signal.emotion)
        self.assertEqual(signal.emojis, ["😅"])


if __name__ == "__main__":
    unittest.main()
