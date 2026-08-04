"""Environment-backed settings for the Talking Face black-box adapter."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass


TALKING_FACE_URL = os.getenv("VMS_TALKING_FACE_URL", "").strip().rstrip("/")
TALKING_FACE_TOKEN = os.getenv("VMS_TALKING_FACE_TOKEN", "").strip()
TALKING_FACE_TIMEOUT = float(os.getenv("VMS_TALKING_FACE_TIMEOUT", "300"))
TALKING_FACE_MAX_OUTPUT_BYTES = (
    int(os.getenv("VMS_TALKING_FACE_MAX_OUTPUT_MB", "100")) * 1024 * 1024
)
TALKING_FACE_ROOT = os.getenv("VMS_TALKING_FACE_ROOT", "").strip()
_work_dir = os.getenv(
    "VMS_TALKING_FACE_WORK_DIR", "output/current/talking_face_sidecar"
).strip()
TALKING_FACE_WORK_DIR = (
    Path(_work_dir) if Path(_work_dir).is_absolute() else PROJECT_ROOT / _work_dir
).resolve()
TALKING_FACE_MAX_INPUT_BYTES = (
    int(os.getenv("VMS_TALKING_FACE_MAX_INPUT_MB", "25")) * 1024 * 1024
)
