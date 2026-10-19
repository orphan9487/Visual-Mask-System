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


def _secret() -> bytes:
    return os.getenv("VMS_SESSION_SECRET", "visual-mask-local-development-secret").encode()


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
