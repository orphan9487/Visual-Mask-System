# -*- coding: utf-8 -*-
"""
兩階段 M-CoT 對話情緒辨識引擎 (ERC Engine)。

對應計畫書 B 節「情感與意圖理解模組」：
  - 以 InstructERC (Lei et al., 2023) 生成式路線為基礎，
    導入 Multimodal-CoT [5] 兩階段推理：先產生 Rationale（推理鏈），再據以決定情緒標籤。
  - 支援對話歷史 (Contextual Buffer) 與使用者長期說話風格先驗 (User Behavioral Prior)，
    後者對應 BiosERC / LaERC-S 的 speaker characteristics 設計。
  - use_context 開關供 E 節「有無 Contextual Buffer」消融實驗使用。

相對舊版 reasoning_engine.py 的關鍵修正：
  1. 移除寫死的 "user is OPTIMISTIC" —— 該偏誤會讓模型系統性高估正向情緒。
  2. 情緒不再是自由字串，而是約束到固定 7 類標籤（可評估 Weighted-F1 / Macro-F1）。
  3. 記錄解析失敗率作為「小模型幻覺率」的量化指標（計畫書 F 節困難點）。
"""

from dataclasses import dataclass, field
from typing import Optional

from .labels import (
    CANONICAL_EMOTIONS,
    EMOTION_ZH,
    coerce_prediction,
)
from .prompts import (EXPLAIN_SYSTEM, LABEL_LINE, context_block, explain_user_prompt,
                      label_user_prompt, system_prompt)


@dataclass
class ERCResult:
    predicted: Optional[str]      # canonical 標籤；None = 無法解析（視為幻覺/無效）
    rationale: str                # 第一階段推理鏈
    raw_label_output: str         # 第二階段原始輸出
    parse_ok: bool                # 是否成功解析出合法標籤
    used_context: bool


@dataclass
class ERCStats:
    n: int = 0
    parse_fail: int = 0

    @property
    def hallucination_rate(self) -> float:
        return self.parse_fail / self.n if self.n else 0.0


class ERCEngine:
    def __init__(self, backbone, use_context: bool = True, two_stage: bool = True,
                 label_bias: Optional[list[float]] = None, scorer=None):
        """
        backbone     : src.reasoning.backbone.Backbone 實例
        use_context  : 是否餵入對話歷史（消融開關）
        two_stage    : True = 真兩階段(Rationale→Label)；False = 單次直接出標籤(對照組)
        label_bias   : 單次模式的先驗校正（依 CANONICAL_EMOTIONS 順序的 7 個值）。給定時改以
                       「7 標籤各算 log p + bias 取最大」判讀，而非讓模型生成標籤；
                       用於修正訓練資料降檔 neutral 造成的中性低估（見 configs/erc_calibration.json）。
        scorer       : 打分器（測試注入用）；預設延遲建立 label_scorer.LabelScorer。
        """
        self.bk = backbone
        self.use_context = use_context
        self.two_stage = two_stage
        self.label_bias = list(label_bias) if label_bias is not None else None
        if self.label_bias is not None and len(self.label_bias) != len(CANONICAL_EMOTIONS):
            raise ValueError(f"label_bias 需 {len(CANONICAL_EMOTIONS)} 個值")
        self._scorer = scorer
        self.stats = ERCStats()

    # ------------------------------------------------------------------ #
    # system / context / 單次標籤三段 prompt 皆改用共用模組 prompts.py，
    # 確保與 LoRA 訓練樣本完全一致（見 src/reasoning/prompts.py 說明）。
    def _system_prompt(
        self,
        speaker_prior: Optional[str],
        correction_examples: list[dict] | None = None,
    ) -> str:
        return system_prompt(speaker_prior, correction_examples)

    def _context_block(self, history) -> str:
        return context_block(history, use_context=self.use_context)

    # ------------------------------------------------------------------ #
    def _stage1_rationale(self, context: str, utterance: str) -> str:
        user = (
            f"[Dialogue context]\n{context}\n\n"
            f"[Current utterance]\n{utterance}\n\n"
            "Think step by step (3-4 sentences): How does the context affect the tone? "
            "Is it sincere or sarcastic? What emotion does the speaker most likely feel? "
            "Give ONLY the reasoning, do not output the final label yet."
        )
        return self.bk.chat(self._system_prompt_cache, user, max_new_tokens=220)

    def _stage2_label(self, context: str, utterance: str, rationale: str) -> str:
        user = (
            f"[Dialogue context]\n{context}\n\n"
            f"[Current utterance]\n{utterance}\n\n"
            f"[Reasoning]\n{rationale}\n\n"
            f"Now choose exactly ONE label for the current utterance from these seven: "
            f"[{LABEL_LINE}]. Output only that one word — do NOT answer with 'irony', "
            f"'sarcasm', 'amusement', 'determination', or any word outside the seven."
        )
        return self.bk.chat(self._system_prompt_cache, user, max_new_tokens=10)

    def _single_pass(self, context: str, utterance: str) -> str:
        # 與訓練樣本完全同一個 user prompt（共用 prompts.label_user_prompt）
        user = label_user_prompt(context, utterance)
        if self.label_bias is not None:
            if self._scorer is None:
                from .label_scorer import LabelScorer
                self._scorer = LabelScorer(self.bk)
            scores = self._scorer.score(self._system_prompt_cache, user)
            best = max(range(len(CANONICAL_EMOTIONS)),
                       key=lambda i: scores[i] + self.label_bias[i])
            return CANONICAL_EMOTIONS[best]
        return self.bk.chat(self._system_prompt_cache, user, max_new_tokens=10)

    # ------------------------------------------------------------------ #
    def predict(
        self,
        utterance: str,
        history=None,
        speaker_prior: Optional[str] = None,
        correction_examples: list[dict] | None = None,
    ) -> ERCResult:
        self._system_prompt_cache = self._system_prompt(
            speaker_prior, correction_examples
        )
        context = self._context_block(history)

        if self.two_stage:
            rationale = self._stage1_rationale(context, utterance)
            raw = self._stage2_label(context, utterance, rationale)
        else:
            rationale = ""
            raw = self._single_pass(context, utterance)

        pred = coerce_prediction(raw)
        parse_ok = pred is not None

        self.stats.n += 1
        if not parse_ok:
            self.stats.parse_fail += 1

        return ERCResult(
            predicted=pred,
            rationale=rationale,
            raw_label_output=raw,
            parse_ok=parse_ok,
            used_context=self.use_context,
        )

    def explain(self, utterance: str, label: str, history=None,
                speaker: Optional[str] = None) -> str:
        """
        按需解釋：為已判讀出的 label 產生 2–3 句繁中理由（M-CoT 的推理鏈改為按需提供）。
        以**底座模型**產生（暫停 adapter）：adapter 被微調成只輸出標籤，寫不出完整說明。
        這是事後說明，不是判讀當下的推理過程，僅供參考。
        """
        context = self._context_block(history)
        with self.bk.adapter_disabled():
            return self.bk.chat(EXPLAIN_SYSTEM,
                                explain_user_prompt(context, speaker, utterance, label),
                                max_new_tokens=200).strip()

    # ------------------------------------------------------------------ #
    def to_generation_decision(self, result: ERCResult) -> dict:
        """
        將 ERC 結果轉為生成層可用的決策（供 main.py end-to-end）。
        情緒 → intensity / visual_prompt 的簡單映射；無效預測則退回 neutral。
        """
        emotion = result.predicted or "neutral"
        prompt_map = {
            "anger": ("angry expression, furrowed brows, frowning", 0.9),
            "joy": ("happy smiling expression, bright eyes", 0.8),
            "sadness": ("sad sorrowful expression, downcast eyes", 0.7),
            "surprise": ("surprised expression, wide eyes, raised brows", 0.75),
            "fear": ("fearful anxious expression, tense", 0.7),
            "disgust": ("disgusted expression, wrinkled nose", 0.65),
            "neutral": ("calm neutral expression", 0.4),
        }
        visual, intensity = prompt_map.get(emotion, prompt_map["neutral"])
        return {
            "mask_id": "human",
            "emotion": emotion,
            "emotion_zh": EMOTION_ZH.get(emotion, emotion),
            "intensity": intensity,
            "visual_prompt": visual,
            "rationale": result.rationale,
        }
