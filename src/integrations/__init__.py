"""Adapters for optional systems maintained outside Visual Mask System."""

from .talking_face import (
    HttpTalkingFaceAdapter,
    TalkingFaceAdapter,
    TalkingFaceError,
    TalkingFaceResult,
    create_talking_face_adapter,
)
from .tts import (
    EdgeTextToSpeechAdapter,
    TextToSpeechAdapter,
    TextToSpeechError,
    TextToSpeechResult,
    create_tts_adapter,
    sanitize_tts_text,
)

__all__ = [
    "HttpTalkingFaceAdapter",
    "TalkingFaceAdapter",
    "TalkingFaceError",
    "TalkingFaceResult",
    "create_talking_face_adapter",
    "EdgeTextToSpeechAdapter",
    "TextToSpeechAdapter",
    "TextToSpeechError",
    "TextToSpeechResult",
    "create_tts_adapter",
    "sanitize_tts_text",
]
