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
