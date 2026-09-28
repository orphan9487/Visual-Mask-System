"""Environment-backed configuration for emotion inference."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass


MOCK_MODE = os.getenv("VMS_MOCK", "0") == "1"
ERC_MODEL = os.getenv("VMS_MODEL", "breeze2-3b")

_quantization = os.getenv("VMS_QUANT", "4bit").strip().lower()
ERC_QUANT = (
    None
    if _quantization in ("", "none", "fp16", "float16", "no")
    else _quantization
)

# 選用的 ERC LoRA adapter（留空＝跑純底座）。相對路徑以專案根解析。
_lora = os.getenv("VMS_LORA_PATH", "").strip()
if _lora:
    _lora_path = Path(_lora)
    if not _lora_path.is_absolute():
        _lora_path = PROJECT_ROOT / _lora_path
    ERC_LORA_PATH = str(_lora_path)
else:
    ERC_LORA_PATH = None

# Optional dependency-isolated emotion service. Keeping this URL empty runs
# Breeze2 in the WebSocket process exactly as before.
EMOTION_API_URL = os.getenv("VMS_EMOTION_API_URL", "").strip().rstrip("/")
EMOTION_API_TOKEN = os.getenv("VMS_EMOTION_API_TOKEN", "").strip()
EMOTION_API_TIMEOUT = float(os.getenv("VMS_EMOTION_API_TIMEOUT", "120"))
EMOTION_API_FALLBACK_LOCAL = (
    os.getenv("VMS_EMOTION_API_FALLBACK", "local").strip().lower() == "local"
)
