"""Tests for LINE webhook signature validation and text replies."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import sys
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from linebot.v3.messaging import ImageMessage, TextMessage, VideoMessage

from src.interfaces.line_webhook import create_line_router, _result_messages
from src.services.pipeline_service import PipelineResult


CHANNEL_SECRET = "test-channel-secret"


def _signature(body: bytes) -> str:
    digest = hmac.new(CHANNEL_SECRET.encode(), body, hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


def _event_body(*, text: str = "你好", source: dict | None = None) -> bytes:
    payload = {
        "destination": "U0123456789",
        "events": [
            {
                "type": "message",
                "message": {
                    "type": "text",
                    "id": "1",
                    "quoteToken": "quote-token",
                    "text": text,
                },
                "webhookEventId": "01H00000000000000000000000",
                "deliveryContext": {"isRedelivery": False},
                "timestamp": 1700000000000,
                "source": source or {"type": "user", "userId": "U123"},
                "replyToken": "reply-token",
                "mode": "active",
            }
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


class LineWebhookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.replies: list[tuple[str, str]] = []
        self.pushes: list[tuple[str, list]] = []
        self.analysis_calls: list[dict] = []
        self.generation_calls: list[object] = []
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        output_path = Path(self.temp_dir.name) / "generated.png"
        output_path.write_bytes(b"png")

        class FakePipeline:
            async def analyze(_self, text, history, **kwargs):
                self.analysis_calls.append({
                    "text": text,
                    "history": history,
                    **kwargs,
                })

                class Analysis:
                    emotion = "joy"

                return Analysis()

            async def generate(_self, analysis, **kwargs):
                self.generation_calls.append(analysis)
                return PipelineResult(
                    emotion=analysis.emotion,
                    mask_id="human",
                    output_path=output_path,
                    public_path="instruction_runs/generated.png",
                    base_intensity=0.7,
                    effective_intensity=0.7,
                    processing_ms=1234,
                )

        app = FastAPI()
        app.include_router(
            create_line_router(
                channel_secret=CHANNEL_SECRET,
                channel_access_token="test-access-token",
                reply_text=lambda token, text: self.replies.append((token, text)),
                push_messages=lambda target, messages: self.pushes.append(
                    (target, messages)
                ),
                public_base_url="https://example.test",
                default_mask_id="ethan",
                pipeline_service=FakePipeline(),
            )
        )
        self.client = TestClient(app)

    def test_valid_signature_pushes_generated_image(self):
        body = _event_body()
        response = self.client.post(
            "/line/callback",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Line-Signature": _signature(body),
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})
        self.assertEqual(self.replies, [])
        self.assertEqual(len(self.pushes), 1)
        target, messages = self.pushes[0]
        self.assertEqual(target, "U123")
        self.assertIsInstance(messages[0], TextMessage)
        self.assertIsInstance(messages[1], ImageMessage)
        self.assertEqual(
            messages[1].original_content_url,
            "https://example.test/output/instruction_runs/generated.png",
        )
        self.assertEqual(
            self.analysis_calls,
            [{
                "text": "你好",
                "history": [],
                "user_id": "U123",
                "mask_id": "ethan",
            }],
        )

    def test_invalid_signature_is_rejected(self):
        response = self.client.post(
            "/line/callback",
            content=_event_body(),
            headers={
                "Content-Type": "application/json",
                "X-Line-Signature": "not-a-valid-signature",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.replies, [])
        self.assertEqual(self.pushes, [])

    def test_group_id_is_used_as_push_target_and_conversation_key(self):
        source = {"type": "group", "groupId": "G123", "userId": "U123"}
        for text in ("第一句", "第二句"):
            body = _event_body(text=text, source=source)
            response = self.client.post(
                "/line/callback",
                content=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Line-Signature": _signature(body),
                },
            )
            self.assertEqual(response.status_code, 200)

        self.assertEqual([target for target, _ in self.pushes], ["G123", "G123"])
        self.assertEqual(self.analysis_calls[0]["history"], [])
        self.assertEqual(
            self.analysis_calls[1]["history"],
            [
                {"role": "user", "content": "第一句"},
                {"role": "assistant", "content": "joy"},
            ],
        )

    def test_generation_failure_pushes_emotion_fallback(self):
        class FailingPipeline:
            async def analyze(_self, text, history, **kwargs):
                class Analysis:
                    emotion = "sadness"

                return Analysis()

            async def generate(_self, analysis, **kwargs):
                raise RuntimeError("generation failed")

        pushes = []
        app = FastAPI()
        app.include_router(create_line_router(
            channel_secret=CHANNEL_SECRET,
            channel_access_token="test-access-token",
            push_messages=lambda target, messages: pushes.append((target, messages)),
            public_base_url="https://example.test",
            pipeline_service=FailingPipeline(),
        ))
        client = TestClient(app)
        body = _event_body()
        response = client.post(
            "/line/callback",
            content=body,
            headers={"X-Line-Signature": _signature(body)},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(pushes[0][0], "U123")
        self.assertIsInstance(pushes[0][1][0], TextMessage)
        self.assertIn("sadness", pushes[0][1][0].text)

    def test_mp4_result_uses_video_message_and_preview(self):
        root = Path(self.temp_dir.name)
        video = root / "video_joy.mp4"
        preview = root / "video_joy_preview.png"
        video.write_bytes(b"mp4")
        preview.write_bytes(b"png")
        result = PipelineResult(
            emotion="joy",
            mask_id="human",
            output_path=video,
            public_path="instruction_runs/video_joy.mp4",
            base_intensity=0.7,
            effective_intensity=0.7,
            processing_ms=2000,
        )

        messages = _result_messages(result, "https://example.test")

        self.assertIsInstance(messages[1], VideoMessage)
        self.assertEqual(
            messages[1].preview_image_url,
            "https://example.test/output/instruction_runs/video_joy_preview.png",
        )


if __name__ == "__main__":
    unittest.main()
