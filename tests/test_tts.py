from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.integrations.tts import (
    EdgeTextToSpeechAdapter,
    TextToSpeechError,
    sanitize_tts_text,
)


class _FakeCommunicator:
    def __init__(self, payload=b"mp3-data"):
        self.payload = payload

    async def save(self, path):
        Path(path).write_bytes(self.payload)


class TextToSpeechTests(unittest.IsolatedAsyncioTestCase):
    async def test_synthesizes_mp3_atomically(self):
        calls = []

        def factory(text, voice, **kwargs):
            calls.append((text, voice, kwargs))
            return _FakeCommunicator()

        adapter = EdgeTextToSpeechAdapter(
            voice="zh-TW-YunJheNeural",
            communicate_factory=factory,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "speech.mp3"
            result = await adapter.synthesize("我很開心", output, emotion="joy")

            self.assertEqual(result.output_path, output.resolve())
            self.assertEqual(output.read_bytes(), b"mp3-data")
            self.assertFalse((Path(directory) / "speech.mp3.part").exists())

        self.assertEqual(calls[0][0], "我很開心")
        self.assertEqual(calls[0][1], "zh-TW-YunJheNeural")
        self.assertEqual(calls[0][2]["rate"], "+12%")
        self.assertEqual(calls[0][2]["volume"], "+5%")
        self.assertEqual(calls[0][2]["pitch"], "+12Hz")
        self.assertEqual(result.emotion, "joy")

    async def test_sadness_uses_slower_lower_profile(self):
        calls = []

        def factory(text, voice, **kwargs):
            calls.append(kwargs)
            return _FakeCommunicator()

        adapter = EdgeTextToSpeechAdapter(communicate_factory=factory)
        with tempfile.TemporaryDirectory() as directory:
            await adapter.synthesize(
                "我很難過", Path(directory) / "speech.mp3", emotion="sadness"
            )

        self.assertEqual(
            calls[0], {"rate": "-18%", "volume": "-8%", "pitch": "-12Hz"}
        )

    async def test_rejects_text_over_limit(self):
        adapter = EdgeTextToSpeechAdapter(
            max_text_chars=3,
            communicate_factory=lambda *args, **kwargs: _FakeCommunicator(),
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TextToSpeechError, "exceeds"):
                await adapter.synthesize("1234", Path(directory) / "speech.mp3")

    def test_removes_unicode_emoji_and_sticker_names(self):
        self.assertEqual(
            sanitize_tts_text("早安😀 [貼圖：熊大揮手] 今天加油🥳"),
            "早安 今天加油",
        )
        self.assertEqual(sanitize_tts_text("Hello (love) world"), "Hello world")

    async def test_emoji_only_text_is_not_synthesized(self):
        adapter = EdgeTextToSpeechAdapter(
            communicate_factory=lambda *args, **kwargs: _FakeCommunicator(),
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TextToSpeechError, "no speakable"):
                await adapter.synthesize(
                    "😀🥳 [貼圖]", Path(directory) / "speech.mp3", emotion="joy"
                )


if __name__ == "__main__":
    unittest.main()
