"""Transport-independent orchestration for emotion analysis and generation."""

from __future__ import annotations

import asyncio
import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "current"

Analyzer = Callable[..., Awaitable[dict[str, Any]]]
Generator = Callable[[dict[str, Any]], str]


async def _default_analyzer(**kwargs) -> dict[str, Any]:
    from .emotion_service import produce_visual_instruction

    return await produce_visual_instruction(**kwargs)


def _default_generator(instruction: dict[str, Any]) -> str:
    from ..generation.instruction_runner import generate_from_instruction

    return generate_from_instruction(instruction)


@dataclass(frozen=True)
class PipelineAnalysis:
    """Internal analysis result retained between inference and generation."""

    payload: dict[str, Any]
    text: str
    history: list[dict[str, str]]
    user_id: str | None

    @property
    def emotion(self) -> str:
        return str(self.payload["emotion"])

    @property
    def instruction(self) -> dict[str, Any]:
        return self.payload["instruction"]

    @property
    def base_intensity(self) -> float:
        return float(self.instruction.get("intensity", 0.5))

    @property
    def rationale(self) -> str:
        return str(self.instruction.get("source", {}).get("rationale", ""))

    @property
    def mask_id(self) -> str:
        return str(self.instruction.get("identity", {}).get("mask_id", "human"))

    @property
    def emotions(self) -> list[dict[str, Any]]:
        """呈現用的情緒集合（單一或主/次）；缺欄位時退回單一情緒。"""
        got = self.payload.get("emotions")
        if got:
            return got
        return [{"emotion": self.emotion, "role": "primary", "source": "text"}]

    @property
    def compound_name(self) -> str | None:
        return self.payload.get("compound_name")


@dataclass(frozen=True)
class PipelineResult:
    emotion: str
    mask_id: str
    output_path: Path
    public_path: str
    base_intensity: float
    effective_intensity: float
    processing_ms: int


class PipelineService:
    """Compose reasoning and image generation without knowing the UI transport."""

    def __init__(
        self,
        *,
        analyzer: Analyzer | None = None,
        generator: Generator | None = None,
        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    ) -> None:
        self._analyzer = analyzer or _default_analyzer
        self._generator = generator or _default_generator
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        # Renderers swap LoRA adapters on shared pipelines; serialize GPU work.
        self._generation_lock = asyncio.Lock()

    async def analyze(
        self,
        text: str,
        history: list[dict[str, str]],
        *,
        user_id: str | None = None,
        mask_id: str | None = None,
        with_intent: bool = False,
    ) -> PipelineAnalysis:
        clean_text = (text or "").strip()
        if not clean_text:
            raise ValueError("text must not be empty")
        safe_history = [
            {"role": str(item.get("role", "user")),
             "content": str(item.get("content", ""))[:1000]}
            for item in history[-12:]
            if isinstance(item, dict) and item.get("content")
        ]
        payload = await self._analyzer(
            text=clean_text,
            history=safe_history,
            user_id=user_id,
            mask_id=mask_id,
            with_intent=with_intent,
        )
        if not isinstance(payload.get("instruction"), dict):
            raise RuntimeError("analyzer did not return a VisualInstruction")
        return PipelineAnalysis(payload, clean_text, safe_history, user_id)

    async def generate(
        self,
        analysis: PipelineAnalysis,
        *,
        intensity_multiplier: float = 1.0,
        force_modality: str | None = None,
    ) -> PipelineResult:
        instruction = copy.deepcopy(analysis.instruction)
        base_intensity = analysis.base_intensity
        multiplier = max(0.5, min(1.5, float(intensity_multiplier)))
        effective_intensity = round(max(0.0, min(1.0, base_intensity * multiplier)), 4)
        instruction["intensity"] = effective_intensity

        # Keep modality and renderer hints consistent with the adjusted value.
        if force_modality not in (None, "video", "sticker"):
            raise ValueError("force_modality must be 'video', 'sticker', or None")
        modality = force_modality or (
            "video" if effective_intensity >= 0.6 else "sticker"
        )
        instruction["modality"] = modality
        hints = instruction.setdefault("generation_hints", {})
        hints["num_frames"] = 16 if modality == "video" else 1

        started = time.perf_counter()
        async with self._generation_lock:
            generated = await asyncio.to_thread(self._generator, instruction)
        processing_ms = round((time.perf_counter() - started) * 1000)

        output_path = Path(generated).resolve()
        try:
            public_path = output_path.relative_to(self.output_root).as_posix()
        except ValueError as exc:
            raise RuntimeError(
                f"generator output escaped output/current: {output_path}"
            ) from exc
        if not output_path.is_file():
            raise RuntimeError(f"generator did not create output: {output_path}")

        return PipelineResult(
            emotion=analysis.emotion,
            mask_id=analysis.mask_id,
            output_path=output_path,
            public_path=public_path,
            base_intensity=base_intensity,
            effective_intensity=effective_intensity,
            processing_ms=processing_ms,
        )


pipeline = PipelineService()
