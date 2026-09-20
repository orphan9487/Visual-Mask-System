# -*- coding: utf-8 -*-
"""
Emoji / 貼圖情緒編碼器——輸入感知層的「第二模態」。

專題核心論點為「純文字流失情緒訊號」，而 emoji 與貼圖正是使用者在文字中
「把情緒加回去」的主要手段。本模組將此象形模態編碼為情緒訊號，供與文字
情緒判讀融合（見 multimodal_fusion.py）。

刻意不使用重型視覺模型——emoji 為離散符號、貼圖附帶 metadata，
以查表方式即可穩健編碼，符合資源受限、純文字聊天之實際場景。
"""

from collections import Counter
from dataclasses import dataclass, field

import emoji as emoji_lib

from ..reasoning.labels import CANONICAL_EMOTIONS

# 常見 emoji → canonical 情緒。涵蓋高頻情緒表情；未列入者忽略（不硬猜）。
EMOJI_EMOTION = {
    # joy
    "😂": "joy", "🤣": "joy", "😄": "joy", "😁": "joy", "😊": "joy", "😃": "joy",
    "🥳": "joy", "🎉": "joy", "😆": "joy", "😍": "joy", "🥰": "joy", "❤️": "joy",
    "👍": "joy", "✨": "joy", "😌": "joy", "😻": "joy",
    # sadness
    "😢": "sadness", "😭": "sadness", "😥": "sadness", "😔": "sadness", "😞": "sadness",
    "💔": "sadness", "🥺": "sadness", "😿": "sadness", "😩": "sadness",
    # anger
    "😡": "anger", "😠": "anger", "🤬": "anger", "👿": "anger", "💢": "anger",
    # surprise
    "😲": "surprise", "😮": "surprise", "😯": "surprise", "😱": "surprise",
    "🤯": "surprise", "😳": "surprise", "‼️": "surprise", "❓": "surprise",
    # fear
    "😨": "fear", "😰": "fear", "😧": "fear", "😦": "fear",
    # disgust
    "🤢": "disgust", "🤮": "disgust", "😖": "disgust", "😬": "disgust",
    # neutral
    "😐": "neutral", "😑": "neutral", "🙂": "neutral", "😶": "neutral",
}


@dataclass
class EmojiSignal:
    """從一則訊息抽出的 emoji 情緒訊號。"""
    emotion: str | None            # 主導情緒（None＝無可辨識 emoji）
    strength: float                # 0-1，依可辨識 emoji 數量遞增
    counts: dict = field(default_factory=dict)  # 各情緒的 emoji 次數
    emojis: list = field(default_factory=list)  # 抽到的 emoji 原字元

    @property
    def has_signal(self) -> bool:
        return self.emotion is not None


def extract_emojis(text: str) -> list[str]:
    """抽出訊息中的所有 emoji 字元（含 ZWJ 組合、膚色等由 emoji lib 處理）。"""
    return [tok["emoji"] for tok in emoji_lib.emoji_list(text)]


def encode_text_emojis(text: str) -> EmojiSignal:
    """把文字中的 emoji 編碼成情緒訊號。"""
    emojis = extract_emojis(text)
    votes = Counter()
    for e in emojis:
        emo = EMOJI_EMOTION.get(e)
        # 去除變異選擇符再試一次（如 ❤️ vs ❤）
        if emo is None:
            emo = EMOJI_EMOTION.get(e.replace("️", ""))
        if emo:
            votes[emo] += 1

    if not votes:
        return EmojiSignal(emotion=None, strength=0.0, counts={}, emojis=emojis)

    top_emotion, top_n = votes.most_common(1)[0]
    # 強度：1 個 emoji → 0.6，多個同向遞增至上限 0.95
    strength = min(0.6 + 0.15 * (top_n - 1), 0.95)
    return EmojiSignal(emotion=top_emotion, strength=round(strength, 2),
                       counts=dict(votes), emojis=emojis)


# 貼圖：LINE sticker 事件附 keywords（如 ["happy","smile"]），映射到情緒。
STICKER_KEYWORD_EMOTION = {
    "joy": ["happy", "smile", "laugh", "joy", "love", "excited", "fun", "lol", "haha"],
    "sadness": ["sad", "cry", "tears", "disappointed", "lonely", "sorry"],
    "anger": ["angry", "mad", "rage", "annoyed", "furious"],
    "surprise": ["surprised", "shock", "wow", "omg", "amazed"],
    "fear": ["scared", "afraid", "fear", "worried", "nervous"],
    "disgust": ["disgust", "gross", "yuck", "sick"],
    "neutral": ["neutral", "ok", "normal", "hello", "hi"],
}
_KW_INDEX = {kw: emo for emo, kws in STICKER_KEYWORD_EMOTION.items() for kw in kws}


def encode_sticker(keywords: list[str]) -> EmojiSignal:
    """把 LINE 貼圖 keywords 編碼成情緒訊號。"""
    votes = Counter()
    for kw in keywords or []:
        emo = _KW_INDEX.get(str(kw).strip().lower())
        if emo:
            votes[emo] += 1
    if not votes:
        return EmojiSignal(emotion=None, strength=0.0, counts={}, emojis=[])
    top_emotion, _ = votes.most_common(1)[0]
    # 貼圖為刻意選擇的強情緒訊號，給較高基礎強度
    return EmojiSignal(emotion=top_emotion, strength=0.85,
                       counts=dict(votes), emojis=[])
