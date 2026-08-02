# -*- coding: utf-8 -*-
"""
多模態情緒融合——輸入感知層：融合「文字情緒」與「emoji/貼圖情緒」。

對應計畫書輸入感知層之多模態編碼與融合。設計理念直接源自專題論點：
純文字常流失情緒（如「沒事」字面中性，配 😭 實為悲傷），而 emoji/貼圖
是使用者補回情緒的手段。本模組以可解釋的規則融合兩個模態，
使系統捕捉到單看文字會漏掉的情緒。

融合規則（可解釋，非黑箱）：
  1. 無 emoji 訊號 → 直接採文字情緒。
  2. 文字為中性/低信心，emoji 有明確情緒 → 採 emoji（符號補回了文字漏掉的情緒）。
  3. 文字與 emoji 一致 → 採該情緒、提升信心。
  4. 文字與 emoji 衝突（皆非中性）→ 標記 incongruent（如反諷「呵呵😊」配負面文字），
     保留文字判讀但降低信心並標記，供下游（意圖解碼/回饋策略）處理。
"""

from dataclasses import dataclass, field
from typing import Optional

from .emoji_emotion import EmojiSignal, encode_sticker, encode_text_emojis


@dataclass
class FusedEmotion:
    emotion: str                   # 融合後情緒
    source: str                    # text / emoji / sticker / agreement
    confidence: float              # 0-1
    text_emotion: Optional[str] = None
    emoji_signal: Optional[dict] = None
    incongruent: bool = False      # 文字與符號情緒是否衝突（潛在反諷）
    note: str = ""

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def fuse(text_emotion: str, text_confident: bool,
         emoji_sig: EmojiSignal) -> FusedEmotion:
    """
    融合文字情緒與 emoji/貼圖訊號。
    text_confident：文字情緒是否可信（例如 parse_ok 且非因無把握退回 neutral）。
    """
    esig = {"emotion": emoji_sig.emotion, "strength": emoji_sig.strength,
            "emojis": emoji_sig.emojis} if emoji_sig.has_signal else None

    # 1. 無符號訊號
    if not emoji_sig.has_signal:
        return FusedEmotion(emotion=text_emotion, source="text",
                            confidence=0.7 if text_confident else 0.5,
                            text_emotion=text_emotion, emoji_signal=None,
                            note="無 emoji/貼圖，採文字情緒")

    emo_emotion = emoji_sig.emotion

    # 3. 一致 → 強化
    if text_emotion == emo_emotion:
        return FusedEmotion(emotion=text_emotion, source="agreement",
                            confidence=min(0.95, 0.75 + emoji_sig.strength * 0.2),
                            text_emotion=text_emotion, emoji_signal=esig,
                            note="文字與符號一致，提升信心")

    # 2. 文字中性或低信心，符號有明確情緒 → 符號補回
    if text_emotion == "neutral" or not text_confident:
        return FusedEmotion(emotion=emo_emotion, source="emoji",
                            confidence=emoji_sig.strength,
                            text_emotion=text_emotion, emoji_signal=esig,
                            note="文字中性/低信心，emoji 補回文字漏掉的情緒")

    # 4. 兩者皆非中性且衝突 → 潛在反諷，保留文字但標記
    return FusedEmotion(emotion=text_emotion, source="text",
                        confidence=0.45, text_emotion=text_emotion,
                        emoji_signal=esig, incongruent=True,
                        note="文字與符號情緒衝突（潛在反諷/口是心非），保留文字並標記")


def perceive(text: str, text_emotion: str, text_confident: bool = True,
             sticker_keywords: Optional[list] = None) -> FusedEmotion:
    """
    感知層對外入口：給定訊息文字、文字情緒判讀、（可選）貼圖 keywords，
    回傳融合後的情緒。
    """
    if sticker_keywords:
        sig = encode_sticker(sticker_keywords)
    else:
        sig = encode_text_emojis(text)
    return fuse(text_emotion, text_confident, sig)
