# -*- coding: utf-8 -*-
"""
ERC 情緒標籤集與跨資料集映射。

本研究以 7 類情緒為統一標籤空間（對齊 MELD 與 M³ED），並提供各資料集 →
統一空間的映射，使跨語言/跨資料集的訓練與評估能落在同一標籤集上
（做法參照 InstructERC 以 feeling wheel 統一標籤的精神）。
"""

# 統一 7 類情緒（canonical）。順序固定，供 F1 / 混淆矩陣使用。
CANONICAL_EMOTIONS = [
    "neutral",   # 中性
    "joy",       # 喜悅 / 開心
    "sadness",   # 悲傷
    "anger",     # 憤怒
    "surprise",  # 驚訝
    "fear",      # 恐懼
    "disgust",   # 厭惡
]

# 繁體中文顯示名（供報告與 prompt 可讀性）
EMOTION_ZH = {
    "neutral": "中性",
    "joy": "開心",
    "sadness": "悲傷",
    "anger": "憤怒",
    "surprise": "驚訝",
    "fear": "恐懼",
    "disgust": "厭惡",
}

# 各資料集原始標籤 → canonical。全部小寫比對。
DATASET_LABEL_MAP = {
    # MELD (Poria et al., 2019) 原生就是 7 類
    "meld": {
        "neutral": "neutral",
        "joy": "joy",
        "sadness": "sadness",
        "anger": "anger",
        "surprise": "surprise",
        "fear": "fear",
        "disgust": "disgust",
    },
    # M³ED (Zhao et al., 2022) 7 類（happy/sad/angry 等用詞不同）
    "m3ed": {
        "neutral": "neutral",
        "happy": "joy",
        "sad": "sadness",
        "angry": "anger",
        "surprise": "surprise",
        "fear": "fear",
        "disgust": "disgust",
    },
    # IEMOCAP 常用 6 類（無 disgust；excited 併入 joy）
    "iemocap": {
        "neutral": "neutral",
        "happy": "joy",
        "excited": "joy",
        "sad": "sadness",
        "angry": "anger",
        "frustrated": "anger",
        "surprise": "surprise",
        "fear": "fear",
    },
    # Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset（繁體中文/台灣，單句）
    # 原生 8 類；關切語調/疑問語調 不在 canonical 7，不列入映射→normalize 回 None→自動過濾。
    "zh_dialogue": {
        "平淡語氣": "neutral",
        "開心語調": "joy",
        "憤怒語調": "anger",
        "驚奇語調": "surprise",
        "悲傷語調": "sadness",
        "厭惡語調": "disgust",
        # "關切語調" / "疑問語調" 刻意不映射（丟棄，避免硬塞污染）
    },
    # CPED (Chen et al., 2022) 實際 13 類 → 降維到 canonical 7。
    # 曖昧無法乾淨對應者（negative-other / relaxed / grateful）不映射→丟棄，避免污染。
    "cped": {
        "neutral": "neutral",
        "happy": "joy",
        "positive-other": "joy",   # 明確正向
        "depress": "sadness",
        "sadness": "sadness",
        "anger": "anger",
        "worried": "fear",         # 擔憂≈焦慮，歸 fear
        "fear": "fear",
        "astonished": "surprise",
        "disgust": "disgust",
        # 丟棄：negative-other(2395,曖昧負向)、relaxed(2150,價性/喚醒皆模糊)、grateful(31)
    },
}

EMOTION_TO_IDX = {e: i for i, e in enumerate(CANONICAL_EMOTIONS)}


def normalize_label(raw: str, dataset: str) -> str | None:
    """把資料集原始標籤轉成 canonical；無法對應則回 None。"""
    if raw is None:
        return None
    key = str(raw).strip().lower()
    mapping = DATASET_LABEL_MAP.get(dataset.lower(), {})
    return mapping.get(key)


def coerce_prediction(text: str) -> str | None:
    """
    從模型自由輸出中抽出一個 canonical 情緒標籤。
    先找英文 canonical，再找繁中對照詞；都找不到回 None（＝視為幻覺/無效輸出）。
    """
    if not text:
        return None
    low = text.strip().lower()
    # 1) 直接命中 canonical 英文
    for e in CANONICAL_EMOTIONS:
        if e in low:
            return e
    # 2) 常見英文同義詞
    synonyms = {
        "happy": "joy", "happiness": "joy", "glad": "joy",
        "amused": "joy", "amusement": "joy", "excited": "joy",
        "sad": "sadness", "sorrow": "sadness",
        "angry": "anger", "mad": "anger", "furious": "anger",
        "surprised": "surprise", "shocked": "surprise",
        "afraid": "fear", "scared": "fear", "anxious": "fear",
        "disgusted": "disgust",
    }
    for k, v in synonyms.items():
        if k in low:
            return v
    # 3) 繁中對照（正式詞 + 常見口語同義詞）
    for e, zh in EMOTION_ZH.items():
        if zh in text:
            return e
    for zh_syn, e in ZH_EMOTION_SYNONYMS.items():
        if zh_syn in text:
            return e
    return None


# 中文口語情緒同義詞 → canonical（供解析使用者/模型的中文輸出）
ZH_EMOTION_SYNONYMS = {
    "開心": "joy", "高興": "joy", "快樂": "joy", "興奮": "joy", "爽": "joy", "開薰": "joy",
    "難過": "sadness", "傷心": "sadness", "難受": "sadness", "哭": "sadness", "沮喪": "sadness", "低落": "sadness",
    "生氣": "anger", "憤怒": "anger", "火大": "anger", "氣": "anger", "不爽": "anger", "森77": "anger",
    "害怕": "fear", "恐懼": "fear", "怕": "fear", "緊張": "fear", "焦慮": "fear", "擔心": "fear",
    "噁心": "disgust", "厭惡": "disgust", "反感": "disgust", "討厭": "disgust",
    "驚訝": "surprise", "嚇到": "surprise", "意外": "surprise", "震驚": "surprise", "驚": "surprise",
    "中性": "neutral", "平淡": "neutral", "普通": "neutral", "還好": "neutral",
}
