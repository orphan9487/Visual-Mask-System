"""Environment-backed text-to-speech configuration."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass


LINE_TALKING_FACE_ENABLED = (
    os.getenv("LINE_TALKING_FACE_ENABLED", "0").strip() == "1"
)
TTS_VOICE = os.getenv("VMS_TTS_VOICE", "zh-TW-YunJheNeural").strip()
TTS_RATE = os.getenv("VMS_TTS_RATE", "+0%").strip()
TTS_VOLUME = os.getenv("VMS_TTS_VOLUME", "+0%").strip()
TTS_PITCH = os.getenv("VMS_TTS_PITCH", "+0Hz").strip()
TTS_TIMEOUT = float(os.getenv("VMS_TTS_TIMEOUT", "60"))
TTS_MAX_TEXT_CHARS = int(os.getenv("VMS_TTS_MAX_TEXT_CHARS", "500"))
