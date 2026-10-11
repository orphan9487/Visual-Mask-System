"""Private FastAPI app for dependency-isolated ERC model inference."""

from __future__ import annotations

import asyncio
import os
from typing import Awaitable, Callable, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

from ..reasoning.labels import CANONICAL_EMOTIONS
from ..services.emotion_service import (
    _explain_sync,
    predict_emotion_local,
    status as inference_status,
)


class HistoryItem(BaseModel):
    role: str = Field(min_length=1, max_length=100)
    content: str = Field(min_length=1, max_length=1000)


class CorrectionExample(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    correct_label: Literal[
        "neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"
    ]
    similarity: float | None = Field(default=None, ge=0.0, le=1.0)


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    history: list[HistoryItem] = Field(default_factory=list, max_length=50)
    speaker_prior: str | None = Field(default=None, max_length=2000)
    correction_examples: list[CorrectionExample] = Field(
        default_factory=list, max_length=5
    )


class ExplainRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    history: list[HistoryItem] = Field(default_factory=list, max_length=50)
    emotion: Literal[
        "neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"
    ]
    speaker: str | None = Field(default=None, max_length=100)


Predictor = Callable[..., Awaitable[dict]]
Explainer = Callable[..., Awaitable[dict]]


async def _explain_local(text, history, emotion, speaker=None) -> dict:
    return await asyncio.to_thread(_explain_sync, text, history, emotion, speaker)


def create_app(predictor: Predictor = predict_emotion_local,
               explainer: Explainer = _explain_local) -> FastAPI:
    app = FastAPI(title="Visual Mask Emotion Inference", docs_url=None, redoc_url=None)
    expected_token = os.getenv("VMS_EMOTION_API_TOKEN", "").strip()
    inference_lock = asyncio.Lock()

    def authorize(authorization: str | None = Header(default=None)) -> None:
        if expected_token and authorization != f"Bearer {expected_token}":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid inference token",
            )

    @app.get("/health")
    async def health(_: None = Depends(authorize)) -> dict:
        return {"ok": True, "inference": inference_status()}

    @app.post("/analyze")
    async def analyze(payload: AnalyzeRequest, _: None = Depends(authorize)) -> dict:
        history = [item.model_dump() for item in payload.history]
        corrections = [item.model_dump() for item in payload.correction_examples]
        # Transformer generation on one shared GPU model is not thread-safe.
        async with inference_lock:
            result = await predictor(
                payload.text.strip(),
                history,
                speaker_prior=payload.speaker_prior,
                correction_examples=corrections,
            )
        emotion = str(result.get("emotion", "neutral"))
        if emotion not in CANONICAL_EMOTIONS:
            emotion = "neutral"
        return {
            "emotion": emotion,
            "rationale": str(result.get("rationale", "")),
            "parse_ok": bool(result.get("parse_ok", False)),
            "source": str(result.get("source", "llm")),
        }

    @app.post("/explain")
    async def explain(payload: ExplainRequest, _: None = Depends(authorize)) -> dict:
        history = [item.model_dump() for item in payload.history]
        async with inference_lock:
            result = await explainer(payload.text.strip(), history, payload.emotion,
                                     speaker=payload.speaker)
        return {
            "emotion": payload.emotion,
            "explanation": str(result.get("explanation", "")),
            "source": str(result.get("source", "llm")),
        }

    return app


app = create_app()
