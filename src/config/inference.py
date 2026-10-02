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

# ERC 推理模式：two_stage（先推理鏈再標籤，純底座時較能壓低幻覺）或 single（直接出標籤）。
# 微調 adapter 是用單次格式訓練的，搭配 single 最一致、也快一倍，但不會產生推理鏈文字。
ERC_TWO_STAGE = os.getenv("VMS_ERC_MODE", "two_stage").strip().lower() != "single"

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
