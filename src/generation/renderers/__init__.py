"""Swappable rendering backends for VisualInstruction generation."""

from .animated_video_renderer import AnimatedVideoRenderer
from .base import GenerationRequest, RenderedFrames, Renderer
from .faceid_sticker_renderer import FaceIDStickerRenderer
from .static_sticker_renderer import StaticStickerRenderer

__all__ = [
    "AnimatedVideoRenderer",
    "GenerationRequest",
    "FaceIDStickerRenderer",
    "RenderedFrames",
    "Renderer",
    "StaticStickerRenderer",
]
