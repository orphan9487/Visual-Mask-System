# -*- coding: utf-8 -*-
"""
ERC 提示詞單一真實來源 (single source of truth)。

訓練與推論**必須**使用完全相同的 prompt，否則微調學到的分布與測試時對不上
——這是 LoRA 微調最常見、也最難察覺的坑。因此把 system / context / label
三段 prompt 抽到這裡，訓練腳本與 erc_engine 都從此匯入，杜絕不一致。
"""

from .labels import CANONICAL_EMOTIONS

LABEL_LINE = ", ".join(CANONICAL_EMOTIONS)


def system_prompt(
    speaker_prior: str | None = None,
    correction_examples: list[dict] | None = None,
) -> str:
    """ERC 系統提示。speaker_prior 只描述長期風格，不預設當前情緒。"""
    base = (
        "You are an expert annotator for Emotion Recognition in Conversation (ERC). "
        "Given the dialogue context and the CURRENT utterance, determine the emotion "
        "of the current utterance. Do NOT assume any fixed or default emotion; judge "
        "strictly from the language and context.\n"
        "Sarcasm/irony is a rhetorical device, NOT an emotion label: if an utterance is "
        "sarcastic, answer with the UNDERLYING felt emotion (usually anger or disgust), "
        "never the word 'irony' or 'sarcasm'.\n"
        f"Your answer MUST be exactly one of these seven and nothing else: {LABEL_LINE}."
    )
    if speaker_prior:
        base += (
            "\n\n[Speaker Behavioral Prior] (long-term style only, NOT the current emotion): "
            + speaker_prior
        )
    if correction_examples:
        lines = []
        for item in correction_examples[:5]:
            text = str(item.get("text", ""))[:500]
            label = str(item.get("correct_label", ""))
            if text and label in CANONICAL_EMOTIONS:
                lines.append(f"- {text!r} was corrected to {label}")
        if lines:
            base += (
                "\n\n[Relevant user corrections] Use these annotations only when "
                "they are linguistically relevant to the current utterance; do not "
                "treat them as a default emotion:\n" + "\n".join(lines)
            )
    return base


def context_block(history, use_context: bool = True) -> str:
    """把對話歷史攤平成文字；use_context=False 或無歷史則回固定佔位字串。"""
    if not use_context or not history:
        return "(no prior context)"
    lines = []
    for m in history:
        if isinstance(m, dict):
            role = m.get("role", "Speaker")
            content = m.get("content", "")
        else:
            role, content = "Speaker", m
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def label_user_prompt(context: str, utterance: str) -> str:
    """單次直出標籤的 user prompt（＝微調訓練樣本的 user 內容）。"""
    return (
        f"[Dialogue context]\n{context}\n\n"
        f"[Current utterance]\n{utterance}\n\n"
        f"Choose exactly ONE label from these seven: [{LABEL_LINE}]. "
        f"Output only that one word — nothing outside the seven."
    )


# --------------------------------------------------------------------------- #
# 按需解釋（判讀改為單次後，M-CoT 的推理鏈改成「使用者要求時才產生」）
# --------------------------------------------------------------------------- #
EXPLAIN_SYSTEM = "你是對話情緒分析助手，負責用繁體中文向使用者解釋一句話被判讀成某種情緒的理由。"


def explain_user_prompt(context: str, speaker: str | None, utterance: str,
                        label: str) -> str:
    """請模型為「已判讀出的標籤」寫 2–3 句理由。這是事後說明，不會改變判讀結果。"""
    from .labels import EMOTION_ZH
    zh = EMOTION_ZH.get(label, label)
    who = f"{speaker}: " if speaker else ""
    return (
        f"[對話前文]\n{context}\n\n"
        f"[當前這句話]\n{who}{utterance}\n\n"
        f"[判讀結果]\n{zh}（{label}）\n\n"
        f"請用繁體中文、2 到 3 句話說明為什麼這句話會被判讀為「{zh}」：指出前文中哪些內容"
        f"影響了語氣、這句話是真心還是反話。直接說明理由，不要重述題目，不要更改判讀結果。"
    )
