"""Small signed-token implementation with only standard-library dependencies."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


# 範例檔與舊版程式碼裡出現過的預設值：任何人都看得到，用它簽章等於沒有簽章。
_KNOWN_WEAK = {
    "change-this-to-a-long-random-secret",
    "visual-mask-local-development-secret",
}
_MIN_LENGTH = 32


def session_secret_problem() -> str | None:
    """回傳登入金鑰的問題描述；沒問題回 None。"""
    secret = os.getenv("VMS_SESSION_SECRET", "").strip()
    if not secret:
        return "VMS_SESSION_SECRET 未設定"
    if secret in _KNOWN_WEAK:
        return "VMS_SESSION_SECRET 仍是公開的預設值"
    if len(secret) < _MIN_LENGTH:
        return f"VMS_SESSION_SECRET 太短（至少 {_MIN_LENGTH} 個字元）"
    return None


def require_session_secret() -> None:
    problem = session_secret_problem()
    if problem:
        raise RuntimeError(
            f"{problem}。請在 .env 設定隨機金鑰，例如執行："
            'python -c "import secrets; print(secrets.token_urlsafe(48))"'
        )


def _secret() -> bytes:
    # 金鑰不合格時直接拒絕簽發與驗證（fail closed），不退回任何預設值，
    # 否則忘記設定的部署會讓任何人都能偽造登入。
    require_session_secret()
    return os.environ["VMS_SESSION_SECRET"].strip().encode()


def issue_token(*, user_id: int, username: str, ttl_seconds: int = 60 * 60 * 12) -> str:
    payload = {"uid": user_id, "username": username, "exp": int(time.time()) + ttl_seconds}
    encoded = _b64(json.dumps(payload, separators=(",", ":")).encode())
    signature = _b64(hmac.new(_secret(), encoded.encode(), hashlib.sha256).digest())
    return f"{encoded}.{signature}"


def verify_token(token: str) -> dict | None:
    try:
        encoded, signature = token.split(".", 1)
        expected = _b64(hmac.new(_secret(), encoded.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(expected, signature):
            return None
        payload = json.loads(_unb64(encoded))
        if int(payload["exp"]) < time.time():
            return None
        return {"user_id": int(payload["uid"]), "username": str(payload["username"])}
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
