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

# 語境視窗：判讀當前訊息時帶入的前文則數。adapter 雖以 6 句訓練，實測給更長前文仍有幫助
# （CPED 測試集：整段前文比 6 句 W-F1 +0.010、增益集中在前文 7–20 句），而前文 >20 句僅佔 6%，
# 故預設 20：取得幾乎全部增益，同時給聊天室語境一個明確上限。
ERC_HISTORY_TURNS = max(0, int(os.getenv("VMS_HISTORY_TURNS", "20")))

# ERC 推理模式：two_stage（先推理鏈再標籤，純底座時較能壓低幻覺）或 single（直接出標籤）。
# 微調 adapter 是用單次格式訓練的，搭配 single 最一致、也快一倍，但不會產生推理鏈文字。
ERC_TWO_STAGE = os.getenv("VMS_ERC_MODE", "two_stage").strip().lower() != "single"

# 先驗校正（僅單次模式、Breeze2）：修正訓練資料降檔 neutral 造成的中性低估。
# 參數檔記錄 shift 與適用的 adapter；VMS_ERC_CALIB_ALPHA 可覆寫強度（0＝關閉，改回生成式判讀）。
ERC_CALIB_PATH = PROJECT_ROOT / os.getenv("VMS_ERC_CALIB_PATH", "configs/erc_calibration.json")
_calib_alpha = os.getenv("VMS_ERC_CALIB_ALPHA", "").strip()
ERC_CALIB_ALPHA = float(_calib_alpha) if _calib_alpha else None   # None＝用參數檔的 alpha

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
