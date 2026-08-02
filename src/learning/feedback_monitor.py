# -*- coding: utf-8 -*-
"""
Feedback Monitor——學習迴圈：監控互動、蒐集使用者回饋。

對應計畫書學習迴圈：Feedback Monitor → M-IT / RLHF（閉環 RLHF 為未來工作，
本模組實作「監控與蒐集」骨架）。做兩件事：
  1. 記錄每次互動（訊息、判讀情緒、送出的視覺指令）→ 供事後分析。
  2. 偵測使用者對系統回饋的反應（讚同/否定/更正）→ 蒐集監督訊號。

關鍵設計：偵測到的「更正」直接輸出為**可餵回微調的 SFT 樣本**
（與 src/training/format_meld.py 同格式），使迴圈在概念上真正閉合：
   Feedback Monitor → 更正資料 → M-IT（LoRA 再訓練）。

研究倫理：本模組會落地使用者訊息。作為研究資料留存前，須取得知情同意
並去識別化；預設寫入本機、不外傳。
"""

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from ..reasoning.labels import coerce_prediction
from ..reasoning.prompts import label_user_prompt, system_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "data" / "learning"

# 讚同 / 否定 的關鍵字與 emoji（中英）
_POSITIVE = ["對", "沒錯", "就是", "正確", "準", "有夠準", "yes", "correct", "👍", "✅", "哈哈對"]
_NEGATIVE = ["不對", "才不是", "亂猜", "錯了", "錯誤", "不是啦", "誰說", "wrong", "no", "👎", "❌"]


@dataclass
class InteractionRecord:
    ts: float
    conv_id: str
    user: str
    utterance: str
    text_emotion: str          # 純文字判讀
    fused_emotion: str         # 多模態融合後（實際送出）
    modality: str
    incongruent: bool = False
    intent: Optional[str] = None


@dataclass
class FeedbackRecord:
    ts: float
    conv_id: str
    user: str
    target_utterance: str      # 被評價的那則訊息
    system_emotion: str        # 系統當時的判讀
    feedback_type: str         # positive / negative / correction
    corrected_emotion: Optional[str] = None
    raw_feedback: str = ""


class FeedbackMonitor:
    def __init__(self, log_dir: Path = LOG_DIR):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.interactions_path = log_dir / "interactions.jsonl"
        self.feedback_path = log_dir / "feedback.jsonl"
        # 每個對話最近一次互動（供「下一則訊息是否為回饋」判斷）
        self._last: dict[str, InteractionRecord] = {}

    # ------------------------------------------------------------------ #
    def log_interaction(self, rec: InteractionRecord):
        self._append(self.interactions_path, asdict(rec))
        self._last[rec.conv_id] = rec

    def last_interaction(self, conv_id: str) -> Optional[InteractionRecord]:
        return self._last.get(conv_id)

    # ------------------------------------------------------------------ #
    def detect_feedback(self, conv_id: str, text: str) -> Optional[FeedbackRecord]:
        """
        判斷這則新訊息是否為對「上一次系統回饋」的評價。
        規則：含讚同詞→positive；含否定詞→negative；
             含情緒詞且與系統判讀不同→correction（更正標籤）。
        """
        prev = self._last.get(conv_id)
        if prev is None:
            return None
        low = text.lower()

        is_pos = any(k.lower() in low for k in _POSITIVE)
        is_neg = any(k.lower() in low for k in _NEGATIVE)
        mentioned = coerce_prediction(text)  # 訊息裡是否點名某情緒

        ftype, corrected = None, None
        if is_neg and mentioned and mentioned != prev.fused_emotion:
            ftype, corrected = "correction", mentioned
        elif is_neg:
            ftype = "negative"
        elif is_pos:
            ftype = "positive"
        elif mentioned and mentioned != prev.fused_emotion:
            # 未含否定詞但點了不同情緒，視為弱更正
            ftype, corrected = "correction", mentioned

        if ftype is None:
            return None
        return FeedbackRecord(
            ts=time.time(), conv_id=conv_id, user=prev.user,
            target_utterance=prev.utterance, system_emotion=prev.fused_emotion,
            feedback_type=ftype, corrected_emotion=corrected, raw_feedback=text[:120],
        )

    def log_feedback(self, rec: FeedbackRecord):
        self._append(self.feedback_path, asdict(rec))

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        n_int = _count_lines(self.interactions_path)
        fb = _read_jsonl(self.feedback_path)
        by_type = {}
        for r in fb:
            by_type[r["feedback_type"]] = by_type.get(r["feedback_type"], 0) + 1
        n_corr = sum(1 for r in fb if r["feedback_type"] == "correction")
        return {"interactions": n_int, "feedback_total": len(fb),
                "by_type": by_type, "corrections_usable_for_training": n_corr}

    def export_training_data(self, out_path: Optional[Path] = None) -> int:
        """
        把「更正」回饋轉為 SFT 樣本（與 format_meld 同格式），供 M-IT 再訓練。
        回傳輸出的樣本數。這是學習迴圈閉合的關鍵一步。
        """
        out_path = out_path or (self.log_dir / "feedback_sft.jsonl")
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for r in _read_jsonl(self.feedback_path):
                if r["feedback_type"] == "correction" and r.get("corrected_emotion"):
                    sample = {
                        "system": system_prompt(),
                        "user": label_user_prompt("(no prior context)", r["target_utterance"]),
                        "assistant": r["corrected_emotion"],
                    }
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    n += 1
        return n

    # ------------------------------------------------------------------ #
    @staticmethod
    def _append(path: Path, obj: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# 單例
feedback_monitor = FeedbackMonitor()
