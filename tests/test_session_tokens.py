"""登入金鑰：不合格就拒絕簽發/驗證；合格時可往返且防竄改。"""

import os
import unittest
from unittest.mock import patch

from src.security import tokens

STRONG = "x" * 20 + "a-strong-random-value-123"


class SessionTokenTests(unittest.TestCase):
    def test_weak_or_missing_secret_is_rejected(self):
        for value in ("", "change-this-to-a-long-random-secret",
                      "visual-mask-local-development-secret", "short"):
            with patch.dict(os.environ, {"VMS_SESSION_SECRET": value}):
                self.assertIsNotNone(tokens.session_secret_problem())
                with self.assertRaises(RuntimeError):
                    tokens.issue_token(user_id=1, username="a")
                with self.assertRaises(RuntimeError):
                    tokens.verify_token("abc.def")

    def test_round_trip_and_tamper_detection(self):
        with patch.dict(os.environ, {"VMS_SESSION_SECRET": STRONG}):
            token = tokens.issue_token(user_id=7, username="alice")
            self.assertEqual(tokens.verify_token(token), {"user_id": 7, "username": "alice"})
            payload, signature = token.split(".")
            forged = tokens._b64(b'{"uid":1,"username":"x","exp":9999999999}')
            self.assertIsNone(tokens.verify_token(f"{forged}.{signature}"))
        # 換了金鑰，舊 token 失效（也代表攻擊者無法用預設值簽出有效 token）
        with patch.dict(os.environ, {"VMS_SESSION_SECRET": STRONG + "-rotated"}):
            self.assertIsNone(tokens.verify_token(token))


if __name__ == "__main__":
    unittest.main()
