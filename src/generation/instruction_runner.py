"""Bridge VisualInstruction JSON to modular sticker and video renderers.

The reasoning layer owns the instruction schema. Generation-time experiments
can apply local overrides without changing that schema or its source JSON.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.generation.diffusion_engine import DiffusionSynthesisEngine
from src.generation.renderers import (
    AnimatedVideoRenderer,
    GenerationRequest,
    Renderer,
    StaticStickerRenderer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "current" / "instruction_runs"
_SAFE_NAME = re.compile(r"[^a-zA-Z0-9_-]+")


class InstructionValidationError(ValueError):
    """Raised when a VisualInstruction lacks a field required for generation."""


@dataclass(frozen=True)
class GenerationOverrides:
    """Optional generation-only controls for LoRA/emotion experiments.

    These values are intentionally not written back into ``VisualInstruction``:
    the reasoning result remains reproducible and the generation experiment is
    explicit at the call site.
    """

    lora_weight: float | None = None
    seed: int | None = None
    force_modality: str | None = None
    expression_weight: float | None = None


_EMOTION_EXPRESSION = {
    "anger": "angry",
    "disgust": "disgusted",
    "fear": "fearful",
    "joy": "joyful",
    "sadness": "sad",
    "surprise": "surprised",
}


class InstructionGenerationRunner:
    """Route instructions to the renderer suited to their requested modality."""

    def __init__(
        self,
        engine: DiffusionSynthesisEngine | None = None,
        sticker_renderer: Renderer | None = None,
        video_renderer: Renderer | None = None,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ) -> None:
        # Renderers are lazy: a sticker request never loads AnimateDiff, and a
        # video request never loads the static SD1.5 pipeline unnecessarily.
        self._legacy_video_engine = engine
        self._sticker_renderer = sticker_renderer
        self._video_renderer = video_renderer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _renderer_for(self, modality: str) -> Renderer:
        if modality == "sticker":
            if self._sticker_renderer is None:
                self._sticker_renderer = StaticStickerRenderer()
            return self._sticker_renderer
        if self._video_renderer is None:
            self._video_renderer = AnimatedVideoRenderer(self._legacy_video_engine)
            self._legacy_video_engine = None
        return self._video_renderer

    @staticmethod
    def _require(mapping: dict[str, Any], key: str) -> Any:
        value = mapping.get(key)
        if value is None or value == "":
            raise InstructionValidationError(f"VisualInstruction is missing '{key}'.")
        return value

    def _normalise(
        self, vi: dict[str, Any], overrides: GenerationOverrides
    ) -> tuple[dict[str, Any], dict[str, Any], str]:
        if not isinstance(vi, dict):
            raise InstructionValidationError("VisualInstruction must be a dictionary.")

        identity = self._require(vi, "identity")
        hints = self._require(vi, "generation_hints")
        if not isinstance(identity, dict) or not isinstance(hints, dict):
            raise InstructionValidationError("'identity' and 'generation_hints' must be objects.")

        lora_path = Path(self._require(identity, "lora_path"))
        if not lora_path.is_absolute():
            lora_path = PROJECT_ROOT / lora_path
        lora_path = lora_path.resolve()
        if not lora_path.is_file():
            raise InstructionValidationError(f"LoRA file does not exist: {lora_path}")

        modality = overrides.force_modality or vi.get("modality", "sticker")
        if modality not in {"video", "sticker"}:
            raise InstructionValidationError("'modality' must be 'video' or 'sticker'.")

        lora_weight = (
            overrides.lora_weight
            if overrides.lora_weight is not None
            else identity.get("lora_weight", hints.get("lora_weight", 0.8))
        )
        lora_weight = float(lora_weight)
        if not 0.0 <= lora_weight <= 1.5:
            raise InstructionValidationError("'lora_weight' must be between 0.0 and 1.5.")
        if overrides.expression_weight is not None:
            expression_weight = float(overrides.expression_weight)
            if not 0.1 <= expression_weight <= 2.0:
                raise InstructionValidationError(
                    "'expression_weight' must be between 0.1 and 2.0."
                )

        normalised_identity = {
            "path": str(lora_path),
            "adapter_name": f"instruction_{_SAFE_NAME.sub('_', str(self._require(identity, 'mask_id')))}",
            "lora_weight": lora_weight,
        }
        requested_frames = int(hints.get("num_frames", 16 if modality == "video" else 1))
        normalised_hints = {
            "num_frames": requested_frames,
            "num_inference_steps": int(self._require(hints, "num_inference_steps")),
            "guidance_scale": float(self._require(hints, "guidance_scale")),
            "width": int(hints.get("width", 384)),
            "height": int(hints.get("height", 512)),
        }
        if normalised_hints["num_frames"] < 1:
            raise InstructionValidationError("'generation_hints.num_frames' must be at least 1.")
        if (normalised_hints["width"] < 64 or normalised_hints["height"] < 64
                or normalised_hints["width"] % 8 or normalised_hints["height"] % 8):
            raise InstructionValidationError(
                "'generation_hints.width' and 'height' must be multiples of 8 and at least 64."
            )
        return normalised_identity, normalised_hints, modality

    def _apply_expression_override(
        self,
        positive_prompt: str,
        negative_prompt: str,
        emotion: str,
        expression_weight: float | None,
    ) -> tuple[str, str]:
        """Emphasise facial affect locally, without mutating the instruction."""
        if expression_weight is None or emotion == "neutral":
            return positive_prompt, negative_prompt

        expression = _EMOTION_EXPRESSION.get(emotion, emotion)
        positive_prompt = (
            f"{positive_prompt}, (clear, unmistakable {expression} facial expression:"
            f"{expression_weight:.2f})"
        )
        negative_prompt = f"{negative_prompt}, neutral expression, blank expression, emotionless face"
        return positive_prompt, negative_prompt

    def generate_from_instruction(
        self,
        vi: dict[str, Any],
        *,
        lora_weight: float | None = None,
        seed: int | None = None,
        force_modality: str | None = None,
        expression_weight: float | None = None,
    ) -> str:
        """Generate an MP4 for ``video`` or PNG for ``sticker`` and return its path.

        A video run also writes a same-stem PNG preview beside the MP4 for
        clients that need a poster image.
        """
        overrides = GenerationOverrides(
            lora_weight=lora_weight,
            seed=seed,
            force_modality=force_modality,
            expression_weight=(
                None if expression_weight is None else float(expression_weight)
            ),
        )
        identity, hints, modality = self._normalise(vi, overrides)
        prompt = str(self._require(vi, "positive_prompt"))
        negative_prompt = str(self._require(vi, "negative_prompt"))
        emotion = _SAFE_NAME.sub("_", str(vi.get("emotion", "neutral"))) or "neutral"
        prompt, negative_prompt = self._apply_expression_override(
            prompt, negative_prompt, emotion, overrides.expression_weight
        )

        renderer = self._renderer_for(modality)
        rendered = renderer.render(GenerationRequest(
            identity=identity,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=hints["num_frames"],
            num_inference_steps=hints["num_inference_steps"],
            guidance_scale=hints["guidance_scale"],
            width=hints["width"],
            height=hints["height"],
            seed=overrides.seed,
        ))
        frames, seed = rendered.frames, rendered.seed

        # Include a high-resolution timestamp and the effective LoRA weight so
        # controlled runs with the same seed never overwrite one another.
        stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}"
        weight_tag = f"lw{identity['lora_weight']:.2f}".replace(".", "p")
        stem = f"{emotion}_{weight_tag}_{stamp}_{seed}"
        if modality == "sticker":
            path = self.output_dir / f"sticker_{stem}.png"
            frames[0].save(path)
            return str(path)

        from diffusers.utils import export_to_video

        path = self.output_dir / f"video_{stem}.mp4"
        preview_path = self.output_dir / f"video_{stem}_preview.png"
        export_to_video(frames, str(path), fps=8)
        frames[0].save(preview_path)
        return str(path)


_shared_runner: InstructionGenerationRunner | None = None


def generate_from_instruction(
    vi: dict[str, Any],
    *,
    lora_weight: float | None = None,
    seed: int | None = None,
    force_modality: str | None = None,
    expression_weight: float | None = None,
) -> str:
    """Generate from a VisualInstruction using a lazily created shared engine."""
    global _shared_runner
    if _shared_runner is None:
        _shared_runner = InstructionGenerationRunner()
    return _shared_runner.generate_from_instruction(
        vi,
        lora_weight=lora_weight,
        seed=seed,
        force_modality=force_modality,
        expression_weight=expression_weight,
    )
