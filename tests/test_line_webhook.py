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

from src.interfaces.line_webhook import (
    _remove_line_emoji_spans,
    _result_messages,
    create_line_router,
)
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
        self.replies: list[tuple[str, list]] = []
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

        self.pipeline = FakePipeline()
        app = FastAPI()
        app.include_router(
            create_line_router(
                channel_secret=CHANNEL_SECRET,
                channel_access_token="test-access-token",
                reply_messages=lambda token, messages: self.replies.append(
                    (token, messages)
                ),
                push_messages=lambda target, messages: self.pushes.append(
                    (target, messages)
                ),
                public_base_url="https://example.test",
                default_mask_id="ethan",
                delivery_mode="reply",
                pipeline_service=self.pipeline,
            )
        )
        self.client = TestClient(app)

    def test_valid_signature_replies_with_generated_image(self):
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
        self.assertEqual(self.pushes, [])
        self.assertEqual(len(self.replies), 1)
        reply_token, messages = self.replies[0]
        self.assertEqual(reply_token, "reply-token")
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

    def test_line_emoji_metadata_removes_utf16_alternative_text(self):
        from types import SimpleNamespace

        text = "嗨😀(love)今天"
        emoji_metadata = [SimpleNamespace(index=3, length=6)]

        self.assertEqual(
            _remove_line_emoji_spans(text, emoji_metadata),
            "嗨😀今天",
        )

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

        self.assertEqual(
            [token for token, _ in self.replies],
            ["reply-token", "reply-token"],
        )
        self.assertEqual(self.analysis_calls[0]["history"], [])
        self.assertEqual(
            self.analysis_calls[1]["history"],
            [
                {"role": "user", "content": "第一句"},
                {"role": "assistant", "content": "joy"},
            ],
        )

    def test_generation_failure_replies_with_emotion_fallback(self):
        class FailingPipeline:
            async def analyze(_self, text, history, **kwargs):
                class Analysis:
                    emotion = "sadness"

                return Analysis()

            async def generate(_self, analysis, **kwargs):
                raise RuntimeError("generation failed")

        replies = []
        app = FastAPI()
        app.include_router(create_line_router(
            channel_secret=CHANNEL_SECRET,
            channel_access_token="test-access-token",
            reply_messages=lambda token, messages: replies.append(
                (token, messages)
            ),
            public_base_url="https://example.test",
            delivery_mode="reply",
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
        self.assertEqual(replies[0][0], "reply-token")
        self.assertIsInstance(replies[0][1][0], TextMessage)
        self.assertIn("sadness", replies[0][1][0].text)

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

    def test_talking_face_pipeline_replies_with_video(self):
        root = Path(self.temp_dir.name)
        image = root / "generated.png"
        image.write_bytes(b"png")

        class FakeImagePipeline:
            async def analyze(_self, text, history, **kwargs):
                class Analysis:
                    emotion = "joy"
                return Analysis()

            async def generate(_self, analysis, **kwargs):
                return PipelineResult(
                    "joy", "henry", image,
                    "instruction_runs/generated.png", 0.8, 0.8, 100,
                )

        class FakeTalkingFacePipeline:
            async def generate(_self, text, image_result):
                video = root / "talking_face_generated.mp4"
                preview = root / "talking_face_generated_preview.png"
                video.write_bytes(b"mp4")
                preview.write_bytes(b"png")
                return PipelineResult(
                    "joy", "henry", video,
                    "instruction_runs/talking_face_generated.mp4",
                    0.8, 0.8, 300,
                )

        replies = []
        app = FastAPI()
        app.include_router(create_line_router(
            channel_secret=CHANNEL_SECRET,
            channel_access_token="test-access-token",
            reply_messages=lambda token, messages: replies.append(
                (token, messages)
            ),
            public_base_url="https://example.test",
            delivery_mode="reply",
            pipeline_service=FakeImagePipeline(),
            talking_face_enabled=True,
            talking_face_service=FakeTalkingFacePipeline(),
        ))
        client = TestClient(app)
        body = _event_body(text="我很開心")
        response = client.post(
            "/line/callback",
            content=body,
            headers={"X-Line-Signature": _signature(body)},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(replies[0][1][1], VideoMessage)

    def test_talking_face_failure_falls_back_to_generated_image(self):
        root = Path(self.temp_dir.name)
        image = root / "fallback.png"
        image.write_bytes(b"png")

        class FakeImagePipeline:
            async def analyze(_self, text, history, **kwargs):
                class Analysis:
                    emotion = "disgust"
                return Analysis()

            async def generate(_self, analysis, **kwargs):
                return PipelineResult(
                    "disgust", "henry", image,
                    "instruction_runs/fallback.png", 0.6, 0.6, 100,
                )

        class FailingTalkingFacePipeline:
            async def generate(_self, text, image_result):
                raise RuntimeError("sidecar offline")

        replies = []
        app = FastAPI()
        app.include_router(create_line_router(
            channel_secret=CHANNEL_SECRET,
            channel_access_token="test-access-token",
            reply_messages=lambda token, messages: replies.append(
                (token, messages)
            ),
            public_base_url="https://example.test",
            delivery_mode="reply",
            pipeline_service=FakeImagePipeline(),
            talking_face_enabled=True,
            talking_face_service=FailingTalkingFacePipeline(),
        ))
        client = TestClient(app)
        body = _event_body(text="我很不開心")
        response = client.post(
            "/line/callback",
            content=body,
            headers={"X-Line-Signature": _signature(body)},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(replies[0][1][1], ImageMessage)

    def test_push_mode_remains_available(self):
        pushes = []
        replies = []
        app = FastAPI()
        app.include_router(create_line_router(
            channel_secret=CHANNEL_SECRET,
            channel_access_token="test-access-token",
            reply_messages=lambda token, messages: replies.append(
                (token, messages)
            ),
            push_messages=lambda target, messages: pushes.append(
                (target, messages)
            ),
            public_base_url="https://example.test",
            delivery_mode="push",
            pipeline_service=self.pipeline,
        ))
        client = TestClient(app)
        body = _event_body()

        response = client.post(
            "/line/callback",
            content=body,
            headers={"X-Line-Signature": _signature(body)},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(replies, [])
        self.assertEqual(pushes[0][0], "U123")
        self.assertIsInstance(pushes[0][1][1], ImageMessage)


if __name__ == "__main__":
    unittest.main()
