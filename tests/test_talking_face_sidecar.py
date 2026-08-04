from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from src.integrations.talking_face_partner import (
    PartnerSystemError,
    PartnerWav2LipBackend,
)
from src.interfaces.talking_face_sidecar import create_app


MP4_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"sidecar-video"


class _FakeBackend:
    def __init__(self):
        self.received = None
        self.closed = False

    async def generate(self, face_path, audio_path, output_path):
        self.received = (
            face_path.read_bytes(),
            audio_path.read_bytes(),
        )
        output_path.write_bytes(MP4_BYTES)

    async def close(self):
        self.closed = True


class TalkingFaceSidecarTests(unittest.TestCase):
    def test_generate_returns_mp4_and_cleans_job_files(self):
        backend = _FakeBackend()
        with tempfile.TemporaryDirectory() as directory:
            jobs = Path(directory) / "jobs"
            app = create_app(
                backend,
                token="secret",
                work_dir=jobs,
                max_input_bytes=1024,
            )
            with TestClient(app) as client:
                response = client.post(
                    "/generate",
                    headers={"Authorization": "Bearer secret"},
                    files={
                        "face": ("henry.png", b"face-data", "image/png"),
                        "audio": ("speech.wav", b"audio-data", "audio/wav"),
                    },
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "video/mp4")
            self.assertEqual(response.content, MP4_BYTES)
            self.assertEqual(backend.received, (b"face-data", b"audio-data"))
            self.assertEqual(list(jobs.iterdir()), [])
            self.assertTrue(backend.closed)

    def test_token_is_required_when_configured(self):
        backend = _FakeBackend()
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(backend, token="secret", work_dir=directory)
            with TestClient(app) as client:
                self.assertEqual(client.get("/health").status_code, 401)
                response = client.get(
                    "/health", headers={"Authorization": "Bearer secret"}
                )
                self.assertEqual(response.status_code, 200)

    def test_rejects_unsupported_input_type(self):
        backend = _FakeBackend()
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(backend, work_dir=directory)
            with TestClient(app) as client:
                response = client.post(
                    "/generate",
                    files={
                        "face": ("face.gif", b"gif", "image/gif"),
                        "audio": ("speech.wav", b"wav", "audio/wav"),
                    },
                )
            self.assertEqual(response.status_code, 422)

    def test_partner_loader_reports_missing_weights_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "api_server.py").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(
                PartnerSystemError,
                "wav2lip_gan.pth.*s3fd.pth",
            ):
                import asyncio

                asyncio.run(PartnerWav2LipBackend.load(root))


if __name__ == "__main__":
    unittest.main()
