"""LINE Messaging API webhook adapter.

This module deliberately contains only transport concerns.  The shared emotion
and image pipeline can be connected after the LINE round-trip is verified.
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import quote

from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, status
from linebot.v3 import WebhookParser
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    ImageMessage,
    MessagingApi,
    PushMessageRequest,
    ReplyMessageRequest,
    TextMessage,
    VideoMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

from src.services.pipeline_service import PipelineResult, PipelineService, pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ReplyText = Callable[[str, str], None]
PushMessages = Callable[[str, list[Any]], None]


def _line_reply_text(access_token: str, reply_token: str, text: str) -> None:
    """Send one text reply through the official LINE SDK."""
    configuration = Configuration(access_token=access_token)
    with ApiClient(configuration) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(
                reply_token=reply_token,
                messages=[TextMessage(text=text)],
            )
        )


def _line_push_messages(access_token: str, target: str, messages: list[Any]) -> None:
    """Push generated media after the webhook response has already returned."""
    configuration = Configuration(access_token=access_token)
    with ApiClient(configuration) as api_client:
        MessagingApi(api_client).push_message(
            PushMessageRequest(to=target, messages=messages)
        )


def _public_output_url(public_base_url: str, public_path: str) -> str:
    encoded_path = quote(public_path.replace("\\", "/"), safe="/")
    return f"{public_base_url.rstrip('/')}/output/{encoded_path}"


def _result_messages(result: PipelineResult, public_base_url: str) -> list[Any]:
    """Build LINE text + image/video messages from one pipeline result."""
    media_url = _public_output_url(public_base_url, result.public_path)
    caption = TextMessage(
        text=f"情緒判斷：{result.emotion}（生成 {result.processing_ms} ms）"
    )
    if result.output_path.suffix.lower() == ".mp4":
        preview_path = result.output_path.with_name(
            f"{result.output_path.stem}_preview.png"
        )
        preview_public_path = str(
            Path(result.public_path).with_name(
                f"{Path(result.public_path).stem}_preview.png"
            )
        ).replace("\\", "/")
        if not preview_path.is_file():
            raise RuntimeError(f"video preview does not exist: {preview_path}")
        media = VideoMessage(
            original_content_url=media_url,
            preview_image_url=_public_output_url(
                public_base_url, preview_public_path
            ),
        )
    else:
        media = ImageMessage(
            original_content_url=media_url,
            preview_image_url=media_url,
        )
    return [caption, media]


async def _process_text_event(
    *,
    text: str,
    user_id: str | None,
    target: str | None,
    reply_token: str,
    history: deque[dict[str, str]],
    history_lock: asyncio.Lock,
    pipeline_service: PipelineService,
    public_base_url: str,
    mask_id: str,
    push_messages: PushMessages,
    reply_text: ReplyText,
) -> None:
    """Analyze and generate after LINE has already received HTTP 200."""
    emotion: str | None = None
    try:
        async with history_lock:
            analysis = await pipeline_service.analyze(
                text,
                list(history),
                user_id=user_id,
                mask_id=mask_id,
            )
            emotion = analysis.emotion
            history.append({"role": "user", "content": text})
            history.append({"role": "assistant", "content": emotion})

        # LINE phase 2 requests a LoRA-generated still image. WebSocket keeps
        # the original automatic image/video modality selection.
        result = await pipeline_service.generate(
            analysis, force_modality="sticker"
        )
        if not public_base_url:
            raise RuntimeError("LINE_PUBLIC_BASE_URL is not configured")
        if not target:
            raise RuntimeError("LINE event does not contain a push target")
        await asyncio.to_thread(
            push_messages,
            target,
            _result_messages(result, public_base_url),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[line][generation][error] {type(exc).__name__}: {exc}")
        fallback = (
            f"情緒判斷：{emotion}，但圖片生成暫時失敗，請稍後再試。"
            if emotion
            else "情緒分析暫時失敗，請稍後再試。"
        )
        try:
            if target:
                await asyncio.to_thread(
                    push_messages, target, [TextMessage(text=fallback)]
                )
            else:
                await asyncio.to_thread(reply_text, reply_token, fallback)
        except Exception as push_exc:  # noqa: BLE001
            print(
                "[line][fallback][error] "
                f"{type(push_exc).__name__}: {push_exc}"
            )


def create_line_router(
    *,
    channel_secret: str | None = None,
    channel_access_token: str | None = None,
    reply_text: ReplyText | None = None,
    push_messages: PushMessages | None = None,
    public_base_url: str | None = None,
    default_mask_id: str | None = None,
    pipeline_service: PipelineService = pipeline,
) -> APIRouter:
    """Create the LINE webhook router, with injectable settings for tests."""
    load_dotenv(PROJECT_ROOT / ".env")
    secret = channel_secret or os.getenv("LINE_CHANNEL_SECRET", "")
    access_token = channel_access_token or os.getenv(
        "LINE_CHANNEL_ACCESS_TOKEN", ""
    )
    base_url = (
        public_base_url
        or os.getenv("LINE_PUBLIC_BASE_URL", "")
        or os.getenv("PUBLIC_BASE_URL", "")
    ).strip().rstrip("/")
    line_mask_id = (
        default_mask_id or os.getenv("LINE_DEFAULT_MASK_ID", "human")
    ).strip()
    parser = WebhookParser(secret) if secret else None
    histories: dict[str, deque[dict[str, str]]] = {}
    history_locks: dict[str, asyncio.Lock] = {}
    pusher = push_messages or (
        lambda target, messages: _line_push_messages(
            access_token, target, messages
        )
    )

    router = APIRouter(prefix="/line", tags=["line"])

    @router.post("/callback")
    async def callback(
        request: Request,
        background_tasks: BackgroundTasks,
        x_line_signature: str | None = Header(default=None),
    ) -> dict[str, bool]:
        if parser is None or not access_token:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="LINE credentials are not configured",
            )
        if not x_line_signature:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing X-Line-Signature header",
            )

        body = (await request.body()).decode("utf-8")
        try:
            events = parser.parse(body, x_line_signature)
        except InvalidSignatureError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid LINE webhook signature",
            ) from exc

        for event in events:
            if isinstance(event, MessageEvent) and isinstance(
                event.message, TextMessageContent
            ):
                sender = reply_text or (
                    lambda reply_token, message: _line_reply_text(
                        access_token, reply_token, message
                    )
                )
                source = event.source
                user_id = getattr(source, "user_id", None)
                target = (
                    getattr(source, "group_id", None)
                    or getattr(source, "room_id", None)
                    or user_id
                )
                conversation_id = target or event.reply_token
                history = histories.setdefault(
                    str(conversation_id), deque(maxlen=12)
                )
                lock = history_locks.setdefault(
                    str(conversation_id), asyncio.Lock()
                )
                background_tasks.add_task(
                    _process_text_event,
                    text=event.message.text,
                    user_id=user_id,
                    target=target,
                    reply_token=event.reply_token,
                    history=history,
                    history_lock=lock,
                    pipeline_service=pipeline_service,
                    public_base_url=base_url,
                    mask_id=line_mask_id,
                    push_messages=pusher,
                    reply_text=sender,
                )

        return {"ok": True}

    return router


router = create_line_router()
