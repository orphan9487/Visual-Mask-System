"""Shared contracts for generation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationRequest:
    """Normalised, generation-only input shared by all renderers."""

    identity: dict[str, Any]
    prompt: str
    negative_prompt: str
    num_frames: int
    num_inference_steps: int
    guidance_scale: float
    width: int
    height: int
    seed: int | None


@dataclass(frozen=True)
class RenderedFrames:
    """Frames returned by a renderer before the output writer encodes them."""

    frames: list[Any]
    seed: int


class Renderer(ABC):
    """Common interface implemented by static and animated diffusion backends."""

    @abstractmethod
    def render(self, request: GenerationRequest) -> RenderedFrames:
        """Render a request and return PIL frames plus the effective seed."""
