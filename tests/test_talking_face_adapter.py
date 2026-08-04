from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import httpx

from src.integrations.talking_face import HttpTalkingFaceAdapter, TalkingFaceError


MP4_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"test-video"


class TalkingFaceAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_uploads_inputs_and_saves_mp4(self):
        received = {}

        def handler(request: httpx.Request) -> httpx.Response:
            received["method"] = request.method
            received["path"] = request.url.path
            received["authorization"] = request.headers.get("authorization")
            received["content_type"] = request.headers.get("content-type", "")
            received["body"] = request.content
            return httpx.Response(
                200,
                headers={"content-type": "video/mp4"},
                content=MP4_BYTES,
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            base_url="http://talking-face.test", transport=transport
        ) as client:
            adapter = HttpTalkingFaceAdapter(
                "http://talking-face.test", token="secret", client=client
            )
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                face = root / "henry.png"
                audio = root / "speech.wav"
                output = root / "result.mp4"
                face.write_bytes(b"png-data")
                audio.write_bytes(b"wav-data")

                result = await adapter.generate(face, audio, output)

                self.assertEqual(result.output_path, output.resolve())
                self.assertEqual(output.read_bytes(), MP4_BYTES)
                self.assertFalse(output.with_suffix(".mp4.part").exists())

        self.assertEqual(received["method"], "POST")
        self.assertEqual(received["path"], "/generate")
        self.assertEqual(received["authorization"], "Bearer secret")
        self.assertIn("multipart/form-data", received["content_type"])
        self.assertIn(b'name="face"', received["body"])
        self.assertIn(b'name="audio"', received["body"])

    async def test_rejects_non_video_response_without_creating_output(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"detail": "not a video"})

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            base_url="http://talking-face.test", transport=transport
        ) as client:
            adapter = HttpTalkingFaceAdapter(
                "http://talking-face.test", client=client
            )
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                face = root / "face.jpg"
                audio = root / "audio.m4a"
                output = root / "result.mp4"
                face.write_bytes(b"jpg")
                audio.write_bytes(b"m4a")

                with self.assertRaisesRegex(
                    TalkingFaceError, "unexpected content type"
                ):
                    await adapter.generate(face, audio, output)

                self.assertFalse(output.exists())

    async def test_health_returns_false_when_service_is_unavailable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(
            base_url="http://talking-face.test", transport=transport
        ) as client:
            adapter = HttpTalkingFaceAdapter(
                "http://talking-face.test", client=client
            )
            self.assertFalse(await adapter.health())


if __name__ == "__main__":
    unittest.main()
