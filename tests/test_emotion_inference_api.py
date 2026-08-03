from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.interfaces.emotion_inference_app import create_app
from src.services import emotion_service
from src.services.emotion_inference_client import RemoteEmotionInferenceError


class EmotionInferenceApiTests(unittest.TestCase):
    def test_analyze_returns_raw_model_result_and_history(self):
        received = {}

        async def predictor(text, history):
            received["text"] = text
            received["history"] = history
            return {
                "emotion": "disgust",
                "rationale": "negation detected",
                "parse_ok": True,
                "source": "llm",
            }

        with patch.dict(os.environ, {"VMS_EMOTION_API_TOKEN": ""}):
            client = TestClient(create_app(predictor))
        response = client.post(
            "/analyze",
            json={
                "text": "我很不開心",
                "history": [{"role": "A", "content": "我很開心"}],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["emotion"], "disgust")
        self.assertEqual(response.json()["source"], "llm")
        self.assertEqual(received["text"], "我很不開心")
        self.assertEqual(received["history"][0]["role"], "A")

    def test_token_is_required_when_configured(self):
        async def predictor(text, history):
            return {"emotion": "neutral", "parse_ok": True}

        with patch.dict(os.environ, {"VMS_EMOTION_API_TOKEN": "secret"}):
            client = TestClient(create_app(predictor))

        self.assertEqual(client.post("/analyze", json={"text": "hi"}).status_code, 401)
        response = client.post(
            "/analyze",
            json={"text": "hi"},
            headers={"Authorization": "Bearer secret"},
        )
        self.assertEqual(response.status_code, 200)


class EmotionServiceRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_remote_service_is_used_when_configured(self):
        remote_result = {
            "emotion": "disgust",
            "rationale": "remote",
            "parse_ok": True,
            "source": "llm",
        }
        with (
            patch.object(emotion_service.config, "EMOTION_API_URL", "http://127.0.0.1:8010"),
            patch.object(emotion_service.config, "EMOTION_API_TOKEN", "token"),
            patch.object(emotion_service.config, "EMOTION_API_TIMEOUT", 3.0),
            patch(
                "src.services.emotion_inference_client.predict_remote",
                new=AsyncMock(return_value=remote_result),
            ) as remote,
        ):
            result = await emotion_service.predict_emotion("我很不開心", [])

        self.assertEqual(result, remote_result)
        remote.assert_awaited_once()

    async def test_remote_failure_can_fall_back_locally(self):
        with (
            patch.object(emotion_service.config, "EMOTION_API_URL", "http://127.0.0.1:8010"),
            patch.object(emotion_service.config, "EMOTION_API_FALLBACK_LOCAL", True),
            patch(
                "src.services.emotion_inference_client.predict_remote",
                new=AsyncMock(side_effect=RemoteEmotionInferenceError("offline")),
            ),
            patch.object(
                emotion_service,
                "predict_emotion_local",
                new=AsyncMock(return_value={"emotion": "neutral", "source": "mock"}),
            ) as local,
        ):
            result = await emotion_service.predict_emotion("hello", [])

        self.assertEqual(result["source"], "mock")
        local.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
