"""FastAPI WebSocket chat interface backed by the transport-neutral pipeline."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from db.feedback_repo import (
    get_intensity_multiplier,
    save_emotion_feedback,
    save_intensity_feedback,
)
from db.lora_repo import get_user_active_lora, get_user_loras, set_user_active_lora
from db.user_repo import verify_login
from src.reasoning.identity_db import identity_db
from src.services.pipeline_service import PipelineAnalysis, PipelineService, pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "output" / "current"
INDEX_PATH = PROJECT_ROOT / "index.html"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


class LoginRequest(BaseModel):
    account: str
    password: str


class SetLoraRequest(BaseModel):
    username: str
    lora_key: str


class EmotionFeedbackRequest(BaseModel):
    username: str
    correct_label: str


class IntensityFeedbackRequest(BaseModel):
    username: str
    emotion: str
    adjusted_intensity: float
    base_intensity: float


class ConnectionManager:
    def __init__(self) -> None:
        self.connections: list[tuple[WebSocket, str]] = []

    async def connect(self, websocket: WebSocket, name: str) -> None:
        await websocket.accept()
        stale = [ws for ws, old_name in self.connections if old_name == name]
        self.connections = [item for item in self.connections if item[1] != name]
        for old in stale:
            try:
                await old.close(code=1000, reason="replaced by a newer connection")
            except Exception:
                pass
        self.connections.append((websocket, name))

    def disconnect(self, websocket: WebSocket) -> None:
        self.connections = [item for item in self.connections if item[0] is not websocket]

    async def broadcast(self, payload: dict) -> None:
        encoded = json.dumps(payload, ensure_ascii=False)
        alive: list[tuple[WebSocket, str]] = []
        for websocket, name in self.connections:
            try:
                await websocket.send_text(encoded)
                alive.append((websocket, name))
            except Exception:
                pass
        self.connections = alive


def _public_analysis_payload(
    analysis: PipelineAnalysis,
    *,
    msg_id: str,
    sender: str,
    display_name: str,
) -> dict:
    return {
        "msg_id": msg_id,
        "sender": sender,
        "text": analysis.text,
        "ai_rationale": analysis.rationale,
        "ai_label": analysis.emotion,
        "active_lora": display_name,
        "mask_file": None,
        "status": "generating",
        "base_intensity": analysis.base_intensity,
    }


def create_app(pipeline_service: PipelineService = pipeline) -> FastAPI:
    app = FastAPI(title="Visual Mask WebSocket Chat")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/output", StaticFiles(directory=str(OUTPUT_ROOT)), name="output")

    manager = ConnectionManager()
    histories: dict[str, deque[dict[str, str]]] = {}
    last_context: dict[str, tuple[str, str, float]] = {}

    @app.get("/")
    async def homepage():
        return FileResponse(INDEX_PATH)

    @app.get("/health")
    async def health():
        return {"ok": True, "transport": "websocket"}

    @app.post("/api/login")
    def login(req: LoginRequest):
        username = verify_login(req.account, req.password)
        if username is None:
            raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
        return {"message": "登入成功", "username": username}

    @app.get("/api/my_loras")
    def my_loras(username: str):
        return {"loras": get_user_loras(username)}

    @app.post("/api/set_lora")
    def set_lora(req: SetLoraRequest):
        if req.lora_key not in identity_db.masks:
            raise HTTPException(status_code=400, detail=f"未知的 lora_key: {req.lora_key}")
        if not set_user_active_lora(req.username, req.lora_key):
            raise HTTPException(status_code=403, detail="此帳號沒有該 LoRA 的使用權限")
        display = identity_db.masks[req.lora_key].display_name or req.lora_key
        return {"message": f"已切換至 {display}（{req.lora_key}）"}

    @app.post("/api/emotion_feedback")
    def emotion_feedback(req: EmotionFeedbackRequest):
        context, wrong_label, _ = last_context.get(
            req.username, ("", "unknown", 0.5)
        )
        save_emotion_feedback(req.username, context, wrong_label, req.correct_label)
        return {"message": f"情緒反饋已記錄（{wrong_label} → {req.correct_label}）"}

    @app.post("/api/intensity_feedback")
    def intensity_feedback(req: IntensityFeedbackRequest):
        save_intensity_feedback(
            req.username,
            req.emotion,
            max(0.0, min(1.0, req.adjusted_intensity)),
            req.base_intensity,
        )
        return {
            "message": (
                f"強度反饋已記錄（{req.emotion}: "
                f"{req.base_intensity:.2f} → {req.adjusted_intensity:.2f}）"
            )
        }

    @app.websocket("/ws/{client_name}")
    async def websocket_endpoint(websocket: WebSocket, client_name: str):
        await manager.connect(websocket, client_name)
        history = histories.setdefault(client_name, deque(maxlen=12))
        try:
            while True:
                text = (await websocket.receive_text()).strip()
                if not text:
                    continue
                msg_id = uuid.uuid4().hex
                mask_id = await asyncio.to_thread(get_user_active_lora, client_name)
                if mask_id not in identity_db.masks:
                    mask_id = "human"
                display_name = identity_db.masks[mask_id].display_name or mask_id

                try:
                    analysis = await pipeline_service.analyze(
                        text,
                        list(history),
                        user_id=client_name,
                        mask_id=mask_id,
                    )
                    history.append({"role": "user", "content": text})
                    history.append({"role": "assistant", "content": analysis.emotion})
                    context = json.dumps(
                        {"history": analysis.history, "text": text},
                        ensure_ascii=False,
                    )
                    last_context[client_name] = (
                        context,
                        analysis.emotion,
                        analysis.base_intensity,
                    )
                    await manager.broadcast(_public_analysis_payload(
                        analysis,
                        msg_id=msg_id,
                        sender=client_name,
                        display_name=display_name,
                    ))

                    multiplier = await asyncio.to_thread(
                        get_intensity_multiplier, client_name, analysis.emotion
                    )
                    result = await pipeline_service.generate(
                        analysis, intensity_multiplier=multiplier
                    )
                    await manager.broadcast({
                        "msg_id": msg_id,
                        "sender": client_name,
                        "text": text,
                        "ai_rationale": analysis.rationale,
                        "ai_label": analysis.emotion,
                        "active_lora": display_name,
                        "mask_file": result.public_path,
                        "status": "done",
                        "base_intensity": result.base_intensity,
                        "effective_intensity": result.effective_intensity,
                        "processing_ms": result.processing_ms,
                    })
                except Exception as exc:
                    await manager.broadcast({
                        "msg_id": msg_id,
                        "sender": client_name,
                        "text": text,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
        except WebSocketDisconnect:
            manager.disconnect(websocket)
            await manager.broadcast({
                "sender": "System",
                "text": f"{client_name} 離開了聊天室。",
            })

    return app


app = create_app()
