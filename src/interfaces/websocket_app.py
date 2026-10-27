"""FastAPI WebSocket chat interface backed by the transport-neutral pipeline."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Awaitable, Callable, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import inference as inference_config
from db.feedback_repo import (
    get_intensity_multiplier,
    save_emotion_feedback,
    save_intensity_feedback,
)
from db.lora_repo import get_user_active_lora, get_user_loras, set_user_active_lora
from db.user_repo import verify_login
from src.reasoning.identity_db import identity_db
from src.services.pipeline_service import PipelineAnalysis, PipelineService, pipeline
from src.services.emotion_service import explain_emotion


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
    correct_label: Literal[
        "neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"
    ]
    msg_id: str | None = None


class ExplainRequest(BaseModel):
    msg_id: str


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
        "emotions": analysis.emotions,
        "compound_name": analysis.compound_name,
    }


def create_app(
    pipeline_service: PipelineService = pipeline,
    explainer: Callable[..., Awaitable[dict]] = explain_emotion,
) -> FastAPI:
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
    # 聊天室共用語境：所有人的發言依時間序記錄，格式「發話者: 內容」，與微調資料的
    # 「角色: 台詞」一致。只放真實發言，不放模型的預測標籤（訓練語境裡從未出現標籤，
    # 放進去會造成訓練/推論不一致，且上一則判錯會連帶影響下一則）。
    room_history: deque[dict[str, str]] = deque(
        maxlen=max(inference_config.ERC_HISTORY_TURNS, 1))
    # 聊天室錄製（VMS_CHAT_RECORD）：每次啟動算一個 session，供之後切成對話建測試集
    record_path = inference_config.CHAT_RECORD_PATH
    record_session = time.strftime("%Y%m%d-%H%M%S")

    def record_message(speaker: str, text: str) -> None:
        if record_path is None:
            return
        record_path.parent.mkdir(parents=True, exist_ok=True)
        with open(record_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"session": record_session, "ts": time.time(),
                                "speaker": speaker, "text": text},
                               ensure_ascii=False) + "\n")

    last_context: dict[str, tuple[str, str, float]] = {}
    feedback_contexts: dict[tuple[str, str], tuple[str, str, float]] = {}
    # 按需解釋：msg_id → 判讀當下的語境快照（任何人都能對任一則訊息按「為什麼？」）。
    # explanations 快取已成功產生的說明，同一則不重複呼叫模型。
    explain_contexts: dict[str, dict] = {}
    explanations: dict[str, str] = {}

    @app.get("/")
    async def homepage():
        return FileResponse(INDEX_PATH)

    @app.get("/health")
    async def health():
        return {
            "ok": True,
            "transport": "websocket",
            "emotion_inference": (
                "remote" if inference_config.EMOTION_API_URL else "local"
            ),
        }

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
            raise HTTPException(status_code=400, detail=f"未知的面具 key: {req.lora_key}")
        if not set_user_active_lora(req.username, req.lora_key):
            raise HTTPException(status_code=403, detail="此帳號沒有該面具的使用權限")
        display = identity_db.masks[req.lora_key].display_name or req.lora_key
        return {"message": f"已切換至 {display}（{req.lora_key}）"}

    @app.post("/api/emotion_feedback")
    def emotion_feedback(req: EmotionFeedbackRequest):
        stored = (
            feedback_contexts.get((req.username, req.msg_id))
            if req.msg_id
            else last_context.get(req.username)
        )
        if stored is None:
            raise HTTPException(status_code=409, detail="找不到要修正的訊息，請重新送出訊息後再試")
        context, wrong_label, _ = stored
        if wrong_label == req.correct_label:
            raise HTTPException(status_code=400, detail="正確情緒不可與原判斷相同")
        save_emotion_feedback(req.username, context, wrong_label, req.correct_label)
        return {"message": f"情緒反饋已記錄（{wrong_label} → {req.correct_label}）"}

    @app.post("/api/explain")
    async def explain(req: ExplainRequest):
        if req.msg_id in explanations:
            return {"msg_id": req.msg_id, "explanation": explanations[req.msg_id],
                    "cached": True}
        ctx = explain_contexts.get(req.msg_id)
        if ctx is None:
            raise HTTPException(status_code=404,
                                detail="找不到這則訊息的分析紀錄（可能已過期或伺服器重啟過）")
        result = await explainer(ctx["text"], ctx["history"], ctx["emotion"],
                                 speaker=ctx["speaker"])
        text = str(result.get("explanation", "")).strip()
        if result.get("source") == "llm" and text:   # 只快取成功的說明，失敗的可重試
            explanations[req.msg_id] = text
            if len(explanations) > 1000:
                explanations.pop(next(iter(explanations)))
        return {"msg_id": req.msg_id, "explanation": text, "cached": False}

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

                # 先取「這句之前」的語境快照，再立刻把這句記進聊天室語境。
                # 收到就記（而非分析完才記），他人在本句分析期間發言時也看得到本句。
                context_turns = (list(room_history)
                                 if inference_config.ERC_HISTORY_TURNS else [])
                room_history.append({"role": client_name, "content": text})
                record_message(client_name, text)

                try:
                    analysis = await pipeline_service.analyze(
                        text,
                        context_turns,
                        user_id=client_name,
                        mask_id=mask_id,
                    )
                    context = json.dumps(
                        {"history": analysis.history, "text": text},
                        ensure_ascii=False,
                    )
                    last_context[client_name] = (
                        context,
                        analysis.emotion,
                        analysis.base_intensity,
                    )
                    feedback_contexts[(client_name, msg_id)] = last_context[client_name]
                    explain_contexts[msg_id] = {
                        "text": text, "history": analysis.history,
                        "emotion": analysis.emotion, "speaker": client_name,
                    }
                    if len(explain_contexts) > 1000:
                        explain_contexts.pop(next(iter(explain_contexts)))
                    if len(feedback_contexts) > 1000:
                        feedback_contexts.pop(next(iter(feedback_contexts)))
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
