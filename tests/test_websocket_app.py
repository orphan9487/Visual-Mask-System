"""WebSocket integration tests using two in-process browser clients."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fastapi.testclient import TestClient

    import src.interfaces.websocket_app as websocket_app
    from src.services.pipeline_service import PipelineAnalysis, PipelineResult
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "requirements-dev.txt is not installed")
class WebSocketAppTests(unittest.TestCase):
    def test_health_and_two_clients_receive_generated_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "result.png"
            output.write_bytes(b"image")

            class FakePipeline:
                async def analyze(self, text, history, **kwargs):
                    return PipelineAnalysis(
                        payload={
                            "emotion": "joy",
                            "emotions": [
                                {"emotion": "joy", "role": "primary", "source": "context"},
                                {"emotion": "surprise", "role": "secondary", "source": "surface"},
                            ],
                            "compound_name": "驚喜",
                            "instruction": {
                                "identity": {"mask_id": kwargs["mask_id"]},
                                "emotion": "joy",
                                "intensity": 0.8,
                                "source": {"rationale": "integration test"},
                            },
                        },
                        text=text,
                        history=history,
                        user_id=kwargs["user_id"],
                    )

                async def generate(self, analysis, **_kwargs):
                    return PipelineResult(
                        emotion=analysis.emotion,
                        mask_id=analysis.mask_id,
                        output_path=output,
                        public_path="integration/result.png",
                        base_intensity=0.8,
                        effective_intensity=0.8,
                        processing_ms=1,
                    )

            app = websocket_app.create_app(FakePipeline())
            with (
                patch.object(websocket_app.inference_config, "EMOTION_API_URL", ""),
                patch.object(websocket_app, "get_user_active_lora", return_value="human"),
                patch.object(websocket_app, "get_intensity_multiplier", return_value=1.0),
                patch.object(websocket_app, "save_emotion_feedback") as save_feedback,
                TestClient(app) as client,
            ):
                health = client.get("/health")
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json()["transport"], "websocket")
                self.assertEqual(health.json()["emotion_inference"], "local")

                with (
                    client.websocket_connect("/ws/alice") as alice,
                    client.websocket_connect("/ws/bob") as bob,
                ):
                    alice.send_text("happy today")
                    alice_generating = alice.receive_json()
                    bob_generating = bob.receive_json()
                    alice_done = alice.receive_json()
                    bob_done = bob.receive_json()
                    response = client.post(
                        "/api/emotion_feedback",
                        json={
                            "username": "alice",
                            "correct_label": "anger",
                            "msg_id": alice_generating["msg_id"],
                        },
                    )

                self.assertEqual(response.status_code, 200)
                saved_context = save_feedback.call_args.args
                self.assertEqual(saved_context[0], "alice")
                self.assertIn("happy today", saved_context[1])
                self.assertEqual(saved_context[2:], ("joy", "anger"))
                same_label = client.post(
                    "/api/emotion_feedback",
                    json={
                        "username": "alice",
                        "correct_label": "joy",
                        "msg_id": alice_generating["msg_id"],
                    },
                )
                self.assertEqual(same_label.status_code, 400)

            self.assertEqual(alice_generating["status"], "generating")
            self.assertEqual(alice_generating["compound_name"], "驚喜")
            self.assertEqual(len(alice_generating["emotions"]), 2)
            self.assertEqual(bob_generating["status"], "generating")
            self.assertEqual(alice_done["status"], "done")
            self.assertEqual(bob_done["status"], "done")
            self.assertEqual(alice_done["msg_id"], alice_generating["msg_id"])
            self.assertEqual(bob_done["mask_file"], "integration/result.png")

    def test_feedback_rejects_unknown_label_and_missing_message(self):
        app = websocket_app.create_app()
        with TestClient(app) as client:
            invalid = client.post(
                "/api/emotion_feedback",
                json={"username": "alice", "correct_label": "confused"},
            )
            missing = client.post(
                "/api/emotion_feedback",
                json={"username": "alice", "correct_label": "anger"},
            )
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(missing.status_code, 409)


if __name__ == "__main__":
    unittest.main()
