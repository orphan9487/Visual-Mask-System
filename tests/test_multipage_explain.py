"""新版聊天室（multipage_app）的按需解釋：判讀當下的語境被保存，成員才能查詢，成功結果會快取。"""

import os
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import src.interfaces.multipage_app as mp
from src.security.tokens import issue_token

SECRET = {"VMS_SESSION_SECRET": "t" * 40}


class _Analysis:
    emotion = "disgust"
    history = [{"role": "Bob", "content": "你把我的資料庫刪光了"}]


class _Result:
    emotion, public_path = "disgust", "instruction_runs/x.png"


class MultipageExplainTests(unittest.TestCase):
    def test_explain_after_message_and_membership_check(self):
        users = {1: {"id": 1, "username": "Alice"}, 2: {"id": 2, "username": "Eve"}}
        explainer = AsyncMock(return_value={"explanation": "前文說資料庫被刪，這句是反話。", "source": "llm"})
        with (
            patch.dict(os.environ, SECRET),
            patch.object(mp, "get_user", lambda uid: users.get(uid)),
            patch.object(mp, "is_room_member", lambda room, uid: uid == 1),
            patch.object(mp, "room_members", lambda room: [1]),
            patch.object(mp, "save_message", lambda *a: None),
            patch.object(mp, "update_message_generation", lambda *a: None),
            patch.object(mp.portrait_masks, "active_key", lambda uid: "portrait_1_" + "a" * 32),
            patch.object(mp.pipeline, "analyze", AsyncMock(return_value=_Analysis())),
            patch.object(mp.pipeline, "generate", AsyncMock(return_value=_Result())),
            patch("src.services.emotion_service.explain_emotion", explainer),
        ):
            app = mp.create_app()
            client = TestClient(app)
            token = issue_token(user_id=1, username="Alice")
            with client.websocket_connect(f"/ws/room1?token={token}") as ws:
                ws.send_text('{"text": "真好啊"}')
                first = ws.receive_json()
                ws.receive_json()
            mid = first["msg_id"]
            url = f"/api/chats/room1/messages/{mid}/explain"
            auth = {"Authorization": f"Bearer {token}"}
            ok = client.post(url, headers=auth).json()
            cached = client.post(url, headers=auth).json()
            outsider = client.post(url, headers={"Authorization": "Bearer " + issue_token(user_id=2, username="Eve")})
            missing = client.post("/api/chats/room1/messages/nope/explain", headers=auth)

        self.assertEqual(ok["explanation"], "前文說資料庫被刪，這句是反話。")
        self.assertTrue(cached["cached"])
        self.assertEqual(explainer.await_count, 1)
        self.assertEqual(explainer.await_args.args[2], "disgust")
        self.assertEqual(outsider.status_code, 403)      # 非聊天室成員不能查
        self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
