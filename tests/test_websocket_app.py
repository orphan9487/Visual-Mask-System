"""WebSocket integration tests using two in-process browser clients."""

from __future__ import annotations

import json
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

    def test_context_is_shared_room_history_without_labels(self):
        """語境＝聊天室所有人的發言（含對方），不含模型預測標籤，且受視窗大小限制。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "result.png"
            output.write_bytes(b"image")
            seen_contexts = []

            class RecordingPipeline:
                async def analyze(self, text, history, **kwargs):
                    seen_contexts.append(list(history))
                    return PipelineAnalysis(
                        payload={
                            "emotion": "anger",
                            "emotions": [{"emotion": "anger", "role": "primary",
                                          "source": "context"}],
                            "compound_name": "",
                            "instruction": {
                                "identity": {"mask_id": kwargs["mask_id"]},
                                "emotion": "anger",
                                "intensity": 0.5,
                                "source": {"rationale": ""},
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
                        base_intensity=0.5,
                        effective_intensity=0.5,
                        processing_ms=1,
                    )

            # 視窗設 2，以便用少量訊息驗證「超出視窗的舊發言會被捨棄」
            with patch.object(websocket_app.inference_config, "ERC_HISTORY_TURNS", 2):
                app = websocket_app.create_app(RecordingPipeline())
            with (
                patch.object(websocket_app.inference_config, "EMOTION_API_URL", ""),
                patch.object(websocket_app, "get_user_active_lora", return_value="human"),
                patch.object(websocket_app, "get_intensity_multiplier", return_value=1.0),
                TestClient(app) as client,
            ):
                with (
                    client.websocket_connect("/ws/alice") as alice,
                    client.websocket_connect("/ws/bob") as bob,
                ):
                    def say(sender, text):
                        sender.send_text(text)
                        for ws in (alice, bob):   # 每則訊息廣播 generating + done
                            ws.receive_json()
                            ws.receive_json()

                    say(alice, "你把我的資料庫刪光了")
                    say(bob, "真好啊")
                    say(alice, "對不起啦")
                    say(bob, "哼")

        a1 = {"role": "alice", "content": "你把我的資料庫刪光了"}
        b1 = {"role": "bob", "content": "真好啊"}
        a2 = {"role": "alice", "content": "對不起啦"}
        self.assertEqual(seen_contexts[0], [])
        self.assertEqual(seen_contexts[1], [a1])        # bob 看得到 alice 的話
        self.assertEqual(seen_contexts[2], [a1, b1])
        self.assertEqual(seen_contexts[3], [b1, a2])    # 視窗 2：最舊的 a1 被捨棄
        for ctx in seen_contexts:                        # 語境裡不得出現預測標籤
            self.assertTrue(all(turn["role"] in ("alice", "bob") for turn in ctx))

    def test_explain_on_demand_uses_message_context_and_caches(self):
        """按需解釋：用該則訊息當下的語境與標籤呼叫解釋器；成功才快取，失敗可重試。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "result.png"
            output.write_bytes(b"image")
            calls = []
            replies = [
                {"explanation": "（說明產生失敗，請稍後再試）", "source": "fallback"},
                {"explanation": "前文說資料庫被刪，這句是反話。", "source": "llm"},
            ]

            async def explainer(text, history, emotion, speaker=None):
                calls.append({"text": text, "history": history,
                              "emotion": emotion, "speaker": speaker})
                return replies[len(calls) - 1]

            class Pipeline:
                async def analyze(self, text, history, **kwargs):
                    return PipelineAnalysis(
                        payload={
                            "emotion": "disgust",
                            "emotions": [{"emotion": "disgust", "role": "primary",
                                          "source": "context"}],
                            "compound_name": "",
                            "instruction": {
                                "identity": {"mask_id": kwargs["mask_id"]},
                                "emotion": "disgust", "intensity": 0.5,
                                "source": {"rationale": ""},
                            },
                        },
                        text=text, history=history, user_id=kwargs["user_id"],
                    )

                async def generate(self, analysis, **_kwargs):
                    return PipelineResult(
                        emotion=analysis.emotion, mask_id=analysis.mask_id,
                        output_path=output, public_path="integration/result.png",
                        base_intensity=0.5, effective_intensity=0.5, processing_ms=1,
                    )

            app = websocket_app.create_app(Pipeline(), explainer=explainer)
            with (
                patch.object(websocket_app.inference_config, "EMOTION_API_URL", ""),
                patch.object(websocket_app, "get_user_active_lora", return_value="human"),
                patch.object(websocket_app, "get_intensity_multiplier", return_value=1.0),
                TestClient(app) as client,
            ):
                missing = client.post("/api/explain", json={"msg_id": "nope"})
                self.assertEqual(missing.status_code, 404)

                with (
                    client.websocket_connect("/ws/alice") as alice,
                    client.websocket_connect("/ws/bob") as bob,
                ):
                    def say(sender, text):
                        sender.send_text(text)
                        first = alice.receive_json()
                        alice.receive_json()
                        bob.receive_json()
                        bob.receive_json()
                        return first["msg_id"]

                    say(alice, "你把我的資料庫刪光了")
                    msg_id = say(bob, "真好啊")

                failed = client.post("/api/explain", json={"msg_id": msg_id}).json()
                ok = client.post("/api/explain", json={"msg_id": msg_id}).json()
                cached = client.post("/api/explain", json={"msg_id": msg_id}).json()

        self.assertFalse(failed["cached"])                 # 失敗的說明不快取 → 可重試
        self.assertEqual(ok["explanation"], "前文說資料庫被刪，這句是反話。")
        self.assertTrue(cached["cached"])
        self.assertEqual(cached["explanation"], ok["explanation"])
        self.assertEqual(len(calls), 2)                    # 第三次走快取，沒再呼叫模型
        self.assertEqual(calls[0], {
            "text": "真好啊",
            "history": [{"role": "alice", "content": "你把我的資料庫刪光了"}],
            "emotion": "disgust",
            "speaker": "bob",
        })

    def test_chat_record_appends_messages_when_enabled(self):
        """VMS_CHAT_RECORD 有設時，每則發言附加寫入 JSONL（建真實聊天測試集用）。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            record = Path(temp_dir) / "rec" / "recordings.jsonl"

            class FailingPipeline:          # 分析失敗也要先錄到（收到就記）
                async def analyze(self, text, history, **kwargs):
                    raise RuntimeError("no model")

            with patch.object(websocket_app.inference_config, "CHAT_RECORD_PATH", record):
                app = websocket_app.create_app(FailingPipeline())
            with (
                patch.object(websocket_app, "get_user_active_lora", return_value="human"),
                TestClient(app) as client,
                client.websocket_connect("/ws/alice") as alice,
            ):
                alice.send_text("早安")
                alice.receive_json()
            rows = [json.loads(l) for l in record.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["speaker"], rows[0]["text"]), ("alice", "早安"))
        self.assertIn("session", rows[0])

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
