# -*- coding: utf-8 -*-
"""
推理層橋接：把文字與對話歷史接到兩階段 M-CoT ERC 引擎。

模型很重（Breeze2-3B 4-bit 載入約 1-2 分鐘），因此：
  - 以單例延遲載入，只在第一次需要時載入一次，之後常駐重用。
  - VMS_MOCK=1 時完全不載入模型，改用關鍵字規則，供介面與管線測試。
"""

import asyncio
import threading

from ..config import inference as config
from ..reasoning.labels import CANONICAL_EMOTIONS

_engine = None
_lock = threading.Lock()

# MOCK 模式用的極簡關鍵字規則（僅供管線驗證，非研究結果）
_MOCK_RULES = [
    ("anger", ["生氣", "氣死", "煩", "討厭", "angry", "mad", "furious", "！！"]),
    ("joy", ["開心", "太好了", "哈哈", "讚", "happy", "great", "congrat", "😂", "🎉"]),
    ("sadness", ["難過", "傷心", "哭", "sad", "sorry", "失敗"]),
    ("surprise", ["驚", "什麼", "天啊", "wow", "really", "？！"]),
    ("fear", ["怕", "恐怖", "緊張", "afraid", "scared"]),
    ("disgust", ["噁", "噁心", "gross", "disgust"]),
]


def _mock_predict(text: str) -> str:
    low = text.lower()
    for emo, kws in _MOCK_RULES:
        if any(k.lower() in low for k in kws):
            return emo
    return "neutral"


def _get_engine():
    """延遲載入 ERC 引擎（執行緒安全，只載入一次）。"""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                from ..reasoning.backbone import Backbone, BackboneConfig
                from ..reasoning.erc_engine import ERCEngine

                lora = config.LORA_PATH or None
                print(f"[erc] 載入模型 {config.ERC_MODEL} (quant={config.ERC_QUANT}"
                      + (f", lora={lora}" if lora else "") + ")…")
                bk = Backbone(BackboneConfig(model=config.ERC_MODEL,
                                             quantization=config.ERC_QUANT,
                                             lora_path=lora))
                _engine = ERCEngine(bk, use_context=True, two_stage=True)
                print("[erc] 模型就緒")
    return _engine


# 過長輸入截斷（避免拖慢/撐爆模型），及模型故障後永久退回關鍵字規則的旗標
MAX_INPUT_CHARS = 500
_engine_broken = False


def _predict_sync(text: str, history: list[dict]) -> dict:
    global _engine_broken
    text = (text or "")[:MAX_INPUT_CHARS]

    # mock 模式，或模型曾載入失敗 → 直接用關鍵字規則
    if config.MOCK_MODE or _engine_broken:
        return {"emotion": _mock_predict(text), "parse_ok": True,
                "rationale": "(keyword rule)", "source": "mock"}

    # 真實推理：任何失敗（OOM/逾時/載入錯誤）都退回關鍵字規則，確保永不當機
    try:
        engine = _get_engine()
        res = engine.predict(text, history=history)
        return {"emotion": res.predicted or "neutral", "rationale": res.rationale,
                "parse_ok": res.parse_ok, "source": "llm"}
    except Exception as e:  # noqa: BLE001
        if _engine is None:                       # 載入階段就失敗 → 之後別再重試
            _engine_broken = True
            print(f"[erc][fatal] 模型載入失敗，永久退回關鍵字規則：{type(e).__name__}: {e}")
        else:
            print(f"[erc][warn] 單次推理失敗，本則退回關鍵字規則：{type(e).__name__}: {e}")
        return {"emotion": _mock_predict(text), "parse_ok": True,
                "rationale": f"(fallback: {type(e).__name__})", "source": "fallback"}


async def predict_emotion_local(text: str, history: list[dict]) -> dict:
    """
    非同步介面：把同步的模型推理丟到執行緒，避免阻塞 event loop。
    """
    return await asyncio.to_thread(_predict_sync, text, history)


async def predict_emotion(text: str, history: list[dict]) -> dict:
    """Use the isolated ERC service when configured, otherwise infer locally."""
    if not config.EMOTION_API_URL:
        return await predict_emotion_local(text, history)

    from .emotion_inference_client import (
        RemoteEmotionInferenceError,
        predict_remote,
    )

    try:
        return await predict_remote(
            text,
            history,
            base_url=config.EMOTION_API_URL,
            token=config.EMOTION_API_TOKEN,
            timeout=config.EMOTION_API_TIMEOUT,
        )
    except RemoteEmotionInferenceError as exc:
        if not config.EMOTION_API_FALLBACK_LOCAL:
            raise
        print(f"[erc][warn] remote inference unavailable; using local fallback: {exc}")
        return await predict_emotion_local(text, history)


_intent_decoder = None


def _get_intent_decoder():
    """延遲建立 Semantic Intent Decoder，與 ERC 共用同一 backbone（mock 模式不支援）。"""
    global _intent_decoder
    if _intent_decoder is None and not config.MOCK_MODE:
        from ..reasoning.intent_decoder import SemanticIntentDecoder
        _intent_decoder = SemanticIntentDecoder(_get_engine().bk)
    return _intent_decoder


def _decode_intent_sync(text: str, history: list[dict], emotion: str) -> dict | None:
    if config.MOCK_MODE or _engine_broken:
        return None
    try:
        dec = _get_intent_decoder()
        return dec.decode(text, history=history, emotion=emotion).to_dict() if dec else None
    except Exception as e:  # noqa: BLE001
        print(f"[intent][warn] 意圖解碼失敗，略過：{type(e).__name__}: {e}")
        return None


async def produce_visual_instruction(text: str, history: list[dict],
                                     user_id: str | None = None,
                                     mask_id: str | None = None,
                                     with_intent: bool = False,
                                     sticker_keywords: list | None = None) -> dict:
    """
    推理層完整輸出：文字 → ERC 情緒判讀 → 多模態感知融合(emoji/貼圖) →
    （可選）意圖解碼 → Visual Instruction Generator。
    多模態融合讓系統捕捉單看文字會漏掉的情緒（如「沒事😭」）。
    """
    import emoji as emoji_lib

    from ..reasoning.visual_instruction import visual_instruction_generator as vig
    from ..perception.multimodal_fusion import perceive

    # 1. 文字模態：剝除 emoji 後只看純文字，使兩模態乾淨分離
    #    （否則 LLM 逕自讀到 emoji，emoji 的貢獻會被藏進文字判讀裡）
    clean_text = emoji_lib.replace_emoji(text, "").strip()
    if clean_text:
        result = await predict_emotion(clean_text, history)
    else:
        result = {"emotion": "neutral", "parse_ok": bool(text.strip()) is False, "rationale": ""}

    # 2. 多模態融合：純文字情緒 + emoji/貼圖情緒（emoji 由原始 text 抽取）
    fused = perceive(text, result["emotion"],
                     text_confident=result.get("parse_ok", True) and bool(clean_text),
                     sticker_keywords=sticker_keywords)
    final_emotion = fused.emotion

    # 3. 可選意圖解碼（以融合後情緒為準）
    intent = None
    if with_intent:
        intent = await asyncio.to_thread(_decode_intent_sync, text, history, final_emotion)

    vi = vig.generate(final_emotion, user_id=user_id, mask_id=mask_id,
                      utterance=text, rationale=result.get("rationale", ""),
                      intent=intent)
    fusion_dict = fused.to_dict()
    from ..reasoning.compound import resolve_emotions
    compound = resolve_emotions(fusion_dict)
    return {"emotion": final_emotion, "text_emotion": result["emotion"],
            "parse_ok": result.get("parse_ok"),
            "fusion": fusion_dict,
            "emotions": compound["emotions"],
            "compound_name": compound["compound_name"],
            "intent": intent, "instruction": vi.to_dict()}


def is_ready() -> bool:
    """能否服務請求（mock 模式不需模型即可服務）。"""
    return config.MOCK_MODE or _engine is not None


def model_loaded() -> bool:
    """真實 LLM 是否已載入（mock 模式下恆為 False，不會誤導）。"""
    return _engine is not None


def status() -> dict:
    return {
        "mode": "mock(關鍵字規則)" if config.MOCK_MODE else "llm(兩階段M-CoT)",
        "configured_model": config.ERC_MODEL,
        "quantization": config.ERC_QUANT,
        "model_loaded": model_loaded(),
        "engine_broken_fallback": _engine_broken,   # True＝模型故障、已退回關鍵字規則
        "can_serve": is_ready(),
    }


def warmup():
    """可選：啟動時先把模型載入，避免第一則訊息等太久。"""
    if not config.MOCK_MODE:
        _get_engine()
