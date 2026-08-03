"""HTTP client for the dependency-isolated emotion inference service."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib import error, request


class RemoteEmotionInferenceError(RuntimeError):
    """Raised when the isolated inference service cannot return a valid result."""


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    token: str = "",
    timeout: float = 120.0,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except (error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RemoteEmotionInferenceError(str(exc)) from exc

    if not isinstance(decoded, dict) or not decoded.get("emotion"):
        raise RemoteEmotionInferenceError("inference service returned an invalid payload")
    return decoded


async def predict_remote(
    text: str,
    history: list[dict],
    *,
    base_url: str,
    token: str = "",
    timeout: float = 120.0,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        _post_json,
        f"{base_url.rstrip('/')}/analyze",
        {"text": text, "history": history},
        token=token,
        timeout=timeout,
    )
