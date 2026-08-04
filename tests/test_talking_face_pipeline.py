from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.integrations.talking_face import TalkingFaceResult
from src.integrations.tts import TextToSpeechResult
from src.services.pipeline_service import PipelineResult
from src.services.talking_face_pipeline import (
    TalkingFacePipelineError,
    TalkingFacePipelineService,
)


class _FakeTTS:
    def __init__(self):
        self.calls = []

    async def synthesize(self, text, output_path, *, emotion="neutral"):
        self.calls.append((text, emotion))
        output = Path(output_path)
        output.write_bytes(b"mp3")
        return TextToSpeechResult(output.resolve(), 100, "test-voice")


class _FakeTalkingFace:
    def __init__(self, *, healthy=True):
        self.healthy = healthy
        self.calls = []

    async def health(self):
        return self.healthy

    async def generate(self, face_path, audio_path, output_path):
        self.calls.append((Path(face_path), Path(audio_path)))
        output = Path(output_path)
        output.write_bytes(b"\x00\x00\x00\x18ftypmp42video")
        return TalkingFaceResult(output.resolve(), 200, "video/mp4")


class TalkingFacePipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_composes_tts_and_video_with_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "instruction_runs"
            runs.mkdir()
            image = runs / "sticker_joy.png"
            image.write_bytes(b"png")
            image_result = PipelineResult(
                "joy", "henry", image, "instruction_runs/sticker_joy.png",
                0.8, 0.8, 1000,
            )
            tts = _FakeTTS()
            talking_face = _FakeTalkingFace()
            service = TalkingFacePipelineService(
                tts=tts, talking_face=talking_face, output_root=root
            )

            result = await service.generate("我很開心", image_result)

            self.assertEqual(result.output_path.suffix, ".mp4")
            self.assertTrue(result.output_path.is_file())
            self.assertTrue(
                result.output_path.with_name(
                    f"{result.output_path.stem}_preview.png"
                ).is_file()
            )
            self.assertEqual(result.processing_ms, 1300)
            self.assertEqual(tts.calls, [("我很開心", "joy")])
            self.assertEqual(list(service.jobs_root.iterdir()), [])

    async def test_unready_sidecar_fails_before_tts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            image.write_bytes(b"png")
            result = PipelineResult(
                "joy", "henry", image, "image.png", 0.8, 0.8, 100,
            )
            tts = _FakeTTS()
            service = TalkingFacePipelineService(
                tts=tts,
                talking_face=_FakeTalkingFace(healthy=False),
                output_root=root,
            )

            with self.assertRaisesRegex(TalkingFacePipelineError, "not ready"):
                await service.generate("hello", result)
            self.assertEqual(tts.calls, [])


if __name__ == "__main__":
    unittest.main()
