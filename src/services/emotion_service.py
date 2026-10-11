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


def _load_label_bias() -> list[float] | None:
    """讀先驗校正參數，回傳依 CANONICAL_EMOTIONS 順序的 alpha*shift；不適用時回 None。"""
    import json
    from pathlib import Path
    path = Path(config.ERC_CALIB_PATH)
    if not path.exists():
        return None
    cal = json.loads(path.read_text(encoding="utf-8"))
    alpha = cal.get("alpha", 0.0) if config.ERC_CALIB_ALPHA is None else config.ERC_CALIB_ALPHA
    if not alpha:
        return None
    lora_name = Path(config.ERC_LORA_PATH).name if config.ERC_LORA_PATH else None
    if cal.get("adapter") != lora_name:
        # 校正量是針對特定 adapter 的訓練分布算的，套到別的 adapter 會錯
        print(f"[erc][warn] 校正參數適用 {cal.get('adapter')}，目前 adapter 為 {lora_name}，不套用校正")
        return None
    bias = [alpha * cal["shift"][k] for k in CANONICAL_EMOTIONS]
    print(f"[erc] 先驗校正 alpha={alpha}：" + " ".join(f"{k}{b:+.2f}"
                                                  for k, b in zip(CANONICAL_EMOTIONS, bias)))
    return bias


def _get_engine():
    """延遲載入 ERC 引擎（執行緒安全，只載入一次）。"""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                from ..reasoning.backbone import Backbone, BackboneConfig
                from ..reasoning.erc_engine import ERCEngine

                mode = "two_stage" if config.ERC_TWO_STAGE else "single"
                print(f"[erc] 載入模型 {config.ERC_MODEL} (quant={config.ERC_QUANT}"
                      f", lora={config.ERC_LORA_PATH}, mode={mode})…")
                bk = Backbone(BackboneConfig(model=config.ERC_MODEL,
                                             quantization=config.ERC_QUANT,
                                             lora_path=config.ERC_LORA_PATH))
                bias = None if config.ERC_TWO_STAGE or not bk.is_internvl else _load_label_bias()
                _engine = ERCEngine(bk, use_context=True, two_stage=config.ERC_TWO_STAGE,
                                    label_bias=bias)
                print("[erc] 模型就緒")
    return _engine


# 過長輸入截斷（避免拖慢/撐爆模型），及模型故障後永久退回關鍵字規則的旗標
MAX_INPUT_CHARS = 500
_engine_broken = False

# 共用模型的推論鎖：判讀、意圖解碼、按需解釋都用同一個 backbone。解釋時會暫停
# adapter（改變模型狀態），若不加鎖，同時進來的判讀可能跑在「無 adapter」狀態。
_infer_lock = threading.Lock()


def _predict_sync(
    text: str,
    history: list[dict],
    speaker_prior: str | None = None,
    correction_examples: list[dict] | None = None,
) -> dict:
    global _engine_broken
    text = (text or "")[:MAX_INPUT_CHARS]

    # mock 模式，或模型曾載入失敗 → 直接用關鍵字規則
    if config.MOCK_MODE or _engine_broken:
        return {"emotion": _mock_predict(text), "parse_ok": True,
                "rationale": "(keyword rule)", "source": "mock"}

    # 真實推理：任何失敗（OOM/逾時/載入錯誤）都退回關鍵字規則，確保永不當機
    try:
        engine = _get_engine()
        with _infer_lock:
            res = engine.predict(
                text,
                history=history,
                speaker_prior=speaker_prior,
                correction_examples=correction_examples,
            )
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


async def predict_emotion_local(
    text: str,
    history: list[dict],
    speaker_prior: str | None = None,
    correction_examples: list[dict] | None = None,
) -> dict:
    """
    非同步介面：把同步的模型推理丟到執行緒，避免阻塞 event loop。
    """
    return await asyncio.to_thread(
        _predict_sync, text, history, speaker_prior, correction_examples
    )


async def predict_emotion(
    text: str,
    history: list[dict],
    speaker_prior: str | None = None,
    correction_examples: list[dict] | None = None,
) -> dict:
    """Use the isolated ERC service when configured, otherwise infer locally."""
    if not config.EMOTION_API_URL:
        return await predict_emotion_local(
            text, history, speaker_prior, correction_examples
        )

    from .emotion_inference_client import (
        RemoteEmotionInferenceError,
        predict_remote,
    )

    try:
        return await predict_remote(
            text,
            history,
            speaker_prior=speaker_prior,
            correction_examples=correction_examples,
            base_url=config.EMOTION_API_URL,
            token=config.EMOTION_API_TOKEN,
            timeout=config.EMOTION_API_TIMEOUT,
        )
    except RemoteEmotionInferenceError as exc:
        if not config.EMOTION_API_FALLBACK_LOCAL:
            raise
        print(f"[erc][warn] remote inference unavailable; using local fallback: {exc}")
        return await predict_emotion_local(
            text, history, speaker_prior, correction_examples
        )


def _explain_sync(text: str, history: list[dict], emotion: str,
                  speaker: str | None = None) -> dict:
    """按需解釋：為已判讀的 emotion 產生繁中理由（事後說明，不改變判讀）。"""
    text = (text or "")[:MAX_INPUT_CHARS]
    if config.MOCK_MODE or _engine_broken:
        return {"emotion": emotion, "source": "mock",
                "explanation": "（目前為關鍵字規則模式，沒有模型可以產生說明）"}
    try:
        engine = _get_engine()
        with _infer_lock:
            out = engine.explain(text, emotion, history=history, speaker=speaker)
        return {"emotion": emotion, "explanation": out or "（模型沒有產生說明）",
                "source": "llm"}
    except Exception as e:  # noqa: BLE001
        print(f"[explain][warn] 說明產生失敗：{type(e).__name__}: {e}")
        return {"emotion": emotion, "explanation": "（說明產生失敗，請稍後再試）",
                "source": "fallback"}


async def explain_emotion(text: str, history: list[dict], emotion: str,
                          speaker: str | None = None) -> dict:
    """設定了獨立推論服務就交給它，否則本機產生（與 predict_emotion 同一套分派）。"""
    if config.EMOTION_API_URL:
        from .emotion_inference_client import RemoteEmotionInferenceError, explain_remote
        try:
            return await explain_remote(
                text, history, emotion, speaker=speaker,
                base_url=config.EMOTION_API_URL, token=config.EMOTION_API_TOKEN,
                timeout=config.EMOTION_API_TIMEOUT,
            )
        except RemoteEmotionInferenceError as exc:
            if not config.EMOTION_API_FALLBACK_LOCAL:
                raise
            print(f"[explain][warn] remote unavailable; using local fallback: {exc}")
    return await asyncio.to_thread(_explain_sync, text, history, emotion, speaker)


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
        if not dec:
            return None
        with _infer_lock:
            return dec.decode(text, history=history, emotion=emotion).to_dict()
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

    speaker_prior = None
    correction_memory = {"exact_label": None, "examples": []}
    if user_id:
        try:
            from db.feedback_repo import get_emotion_correction_memory
            from db.user_repo import get_user_prior

            if not config.MOCK_MODE:
                correction_memory, speaker_prior = await asyncio.gather(
                    asyncio.to_thread(get_emotion_correction_memory, user_id, text),
                    asyncio.to_thread(get_user_prior, user_id),
                )
            else:
                correction_memory = await asyncio.to_thread(
                    get_emotion_correction_memory, user_id, text
                )
        except Exception as e:  # noqa: BLE001
            print(f"[erc][warn] 讀取使用者修正記憶失敗，略過：{type(e).__name__}: {e}")

    # 1. 文字模態：剝除 emoji 後只看純文字，使兩模態乾淨分離
    #    （否則 LLM 逕自讀到 emoji，emoji 的貢獻會被藏進文字判讀裡）
    clean_text = emoji_lib.replace_emoji(text, "").strip()
    if correction_memory["exact_label"]:
        result = {
            "emotion": correction_memory["exact_label"],
            "parse_ok": True,
            "rationale": "(exact user correction memory)",
            "source": "correction_memory",
        }
    elif clean_text:
        result = await predict_emotion(
            clean_text,
            history,
            speaker_prior=speaker_prior,
            correction_examples=correction_memory["examples"],
        )
    else:
        result = {"emotion": "neutral", "parse_ok": bool(text.strip()) is False, "rationale": ""}

    # 2. 多模態融合：純文字情緒 + emoji/貼圖情緒（emoji 由原始 text 抽取）
    fused = perceive(text, result["emotion"],
                     text_confident=result.get("parse_ok", True) and bool(clean_text),
                     sticker_keywords=sticker_keywords)
    if correction_memory["exact_label"]:
        fused.emotion = correction_memory["exact_label"]
        fused.source = "correction_memory"
        fused.confidence = 1.0
        fused.note = "完全相同的訊息曾由此使用者修正，採用最近一次修正"
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
            "inference_source": result.get("source", "unknown"),
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
