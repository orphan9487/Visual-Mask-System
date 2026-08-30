# -*- coding: utf-8 -*-
"""複合情緒：把多模態融合的兩個情緒訊號整理成「存在集合＋主/次」，並查小詞表命名。

設計原則（見情緒輪設計藍圖）：
  - 不動模型、不動訓練：所有輸入都來自 multimodal_fusion 已算好的 FusedEmotion。
  - 單一情緒為多數情況；僅在「文字與 emoji 衝突（incongruent）且 emoji 夠強」時，
    才輸出「主（語境）＋次（表面）」兩個情緒。
  - 複合名走小詞表；查無則不給名（前端退回「A＋B」）。不談比重、不做校準。

情緒字串與 reasoning.labels 的 canonical 一致（joy/sadness/anger/fear/surprise/disgust）。
"""

from __future__ import annotations

# emoji 訊號要多強才算「存在」。emoji 最低強度為 0.6，故 0.3 等同「只要有可辨識 emoji 就算」。
PRESENCE_THRESHOLD = 0.3

# 複合情緒詞表：frozenset({情緒A, 情緒B}) → 複合名。查無 → None。
# 「和諧混合」與「矛盾衝突」共用一張表；哪種由這一對情緒本身決定，程式不需區分。
COMPOUND_NAMES: dict[frozenset, str] = {
    frozenset({"joy", "sadness"}):    "口是心非",   # 矛盾：反諷 / 口是心非（招牌）
    frozenset({"joy", "surprise"}):   "驚喜",
    frozenset({"anger", "disgust"}):  "輕蔑",
    frozenset({"anger", "fear"}):     "焦躁不安",
    frozenset({"sadness", "fear"}):   "絕望",
    frozenset({"sadness", "disgust"}): "悔恨",
    frozenset({"surprise", "fear"}):  "驚駭",
}


def compound_name(a: str, b: str) -> str | None:
    """查兩情緒的複合名；查無回 None。"""
    return COMPOUND_NAMES.get(frozenset({a, b}))


def resolve_emotions(fusion: dict, threshold: float = PRESENCE_THRESHOLD) -> dict:
    """從 fusion（FusedEmotion.to_dict()）整理出呈現用的情緒集合。

    回傳:
      {
        "emotions": [
          {"emotion": <canonical>, "role": "primary"|"secondary", "source": "context"|"surface"|"text"},
          ...
        ],
        "compound_name": <str|None>,
      }
    primary 一律等於 fusion["emotion"]（＝面具與 F1 用的 top-1），確保面板與生成一致。
    """
    fusion = fusion or {}
    primary = fusion.get("emotion") or fusion.get("text_emotion") or "neutral"

    emoji_sig = fusion.get("emoji_signal") or {}
    emoji_emotion = emoji_sig.get("emotion")
    emoji_strength = float(emoji_sig.get("strength") or 0.0)
    incongruent = bool(fusion.get("incongruent"))

    # 複合：文字與 emoji 衝突、emoji 夠強、且兩情緒確實不同
    if (incongruent and emoji_emotion
            and emoji_emotion != primary
            and emoji_strength >= threshold):
        return {
            "emotions": [
                {"emotion": primary, "role": "primary", "source": "context"},   # 語境（文字）
                {"emotion": emoji_emotion, "role": "secondary", "source": "surface"},  # 表面（emoji）
            ],
            "compound_name": compound_name(primary, emoji_emotion),
        }

    # 單一情緒
    return {
        "emotions": [{"emotion": primary, "role": "primary", "source": fusion.get("source", "text")}],
        "compound_name": None,
    }
