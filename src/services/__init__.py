"""Application services that compose perception, reasoning, and generation."""

from .emotion_service import predict_emotion, produce_visual_instruction

__all__ = ["predict_emotion", "produce_visual_instruction"]
