# -*- coding: utf-8 -*-
"""
Semantic Intent Decoder——推理層：由「情緒」推進到「溝通意圖與成因」。

對應計畫書 B/C 節：情緒辨識回答「處於何種情緒」，本模組進一步逼近
「為何如此表達」——推斷語者的**語用意圖**（speech act）、**情緒成因**
（emotion cause，精神參照 SemEval-2024 Task 3）、**對象**與**真誠/反諷**。

同一情緒在不同意圖下，理想的視覺回饋不同（如「憤怒」之發洩求安慰 vs 調侃），
故此分析可供下游調整回饋策略。

本模組為 training-free（零樣本提示），與 ERC 引擎共用同一 backbone。
輸出採寬鬆的逐行格式（非嚴格 JSON），對小模型較穩健。
"""

import re
from dataclasses import asdict, dataclass
from typing import Optional

from .prompts import context_block, system_prompt  # 沿用共用語境格式化

# 社交訊息常見的溝通意圖（固定集合）。coerce 時對不上者歸 "other"。
INTENT_LABELS = {
    "share_good_news": "分享好消息",
    "vent_frustration": "抒發不滿/發洩",
    "seek_support": "尋求安慰或支持",
    "complain": "抱怨或指責",
    "tease": "調侃或開玩笑",
    "apologize": "道歉",
    "express_care": "表達關心",
    "ask_question": "提問或表達困惑",
    "inform": "陳述或告知",
    "other": "其他",
}
INTENT_LINE = ", ".join(k for k in INTENT_LABELS if k != "other")

_TARGETS = {"self", "other", "situation", "none"}
_SINCERITY = {"sincere", "sarcastic"}


@dataclass
class IntentAnalysis:
    intent: str                # INTENT_LABELS 之鍵
    intent_zh: str
    emotion_cause: str         # 簡短成因
    target: str                # self / other / situation / none
    sincerity: str             # sincere / sarcastic
    parse_ok: bool
    raw: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _coerce_intent(text: str) -> Optional[str]:
    low = (text or "").strip().lower()
    for k in INTENT_LABELS:
        if k != "other" and k in low:
            return k
    # 常見同義詞
    syn = {"vent": "vent_frustration", "support": "seek_support",
           "good news": "share_good_news", "joke": "tease", "joking": "tease",
           "apolog": "apologize", "care": "express_care", "question": "ask_question",
           "complain": "complain", "inform": "inform"}
    for s, k in syn.items():
        if s in low:
            return k
    return None


def _pick(pattern: str, text: str) -> str:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


class SemanticIntentDecoder:
    def __init__(self, backbone, use_context: bool = True):
        self.bk = backbone
        self.use_context = use_context

    def _system(self) -> str:
        base = (
            "You analyze the COMMUNICATIVE INTENT behind a chat message given its context. "
            "Beyond the surface emotion, infer why the speaker says this and what they want. "
            "Watch for sarcasm, where literal words contradict the intent.\n"
            f"Intent must be exactly one of: {INTENT_LINE}, or 'other'."
        )
        return base

    def decode(self, utterance: str, history=None,
               emotion: Optional[str] = None) -> IntentAnalysis:
        ctx = context_block(history, use_context=self.use_context)
        emo_hint = f"\n[Detected emotion] {emotion}" if emotion else ""
        user = (
            f"[Dialogue context]\n{ctx}\n\n"
            f"[Current utterance]\n{utterance}{emo_hint}\n\n"
            "Answer in EXACTLY this format, one item per line, nothing else:\n"
            f"Intent: <one of [{INTENT_LINE}, other]>\n"
            "Cause: <a short phrase: why the speaker feels/says this>\n"
            "Target: <self | other | situation>\n"
            "Sincerity: <sincere | sarcastic>"
        )
        raw = self.bk.chat(self._system(), user, max_new_tokens=90)

        intent = _coerce_intent(_pick(r"Intent:\s*(.+)", raw)) or _coerce_intent(raw)
        cause = _pick(r"Cause:\s*(.+)", raw)
        target = (_pick(r"Target:\s*(\w+)", raw) or "none").lower()
        sincerity = (_pick(r"Sincerity:\s*(\w+)", raw) or "sincere").lower()

        parse_ok = intent is not None
        intent = intent or "other"
        if target not in _TARGETS:
            target = "none"
        if sincerity not in _SINCERITY:
            sincerity = "sincere"

        return IntentAnalysis(
            intent=intent,
            intent_zh=INTENT_LABELS.get(intent, "其他"),
            emotion_cause=cause,
            target=target,
            sincerity=sincerity,
            parse_ok=parse_ok,
            raw=raw[:200],
        )
