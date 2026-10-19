# -*- coding: utf-8 -*-
"""
初標助手（pre-annotation assistant）——用模型產草稿，供人逐條改。

角色（呼應 docs/標註schema_草案v0.md 第 7 節與資料護城河策略）：
  模型/API 對每一則訊息產出「草稿標註」→ 人**逐條改**（每條過人腦，禁止抽查）。
  本工具只負責把草稿寫進 `_draft`，**絕不寫進 gold**。gold 只能由人裁決產生。

承重牆禁忌，直接落實在設計裡：
  「不能用一個你聲稱會錯的模型，去產生你的 ground truth。」
  因此本助手刻意保守：
    1. 輸出永遠是 `_draft`（不是 gold、也不是 per_annotator）。
    2. **成因（cause）預設不初標**——這是偏誤風險最高、IAA 最脆弱的欄位，
       留給人腦從零判，避免模型的因果臆測被烤進 ground truth。
    3. 模型看不準的（surprise 依情境、無法判讀）一律把 valence 標成 `ambiguous`，
       用逃生口逼人決定，而不是硬給一個好看的答案。

標籤空間轉換：
  競賽 ERC 引擎輸出 7 類 canonical（含 fear/disgust、無 contempt/resignation），
  產品 schema 是另一套。以下映射把引擎輸出接到產品標籤；引擎產不出的
  contempt/resignation 只能靠人在改稿時補上（草稿給不出＝誠實，不硬湊）。

用法：
  # 先用 mock（關鍵字規則，不載入模型）把管線跑通、產出可改的檔案
  python -m src.annotation.pre_annotate --in  data/annotation/_example_pipeline_demo.jsonl \
                                        --out data/annotation/pilot_draft.jsonl --mock

  # 正式初標（載入 Breeze2-3B，兩階段推理才有 rationale 供 sarcasm 啟發式）
  python -m src.annotation.pre_annotate --in data/annotation/pilot_raw.jsonl \
                                        --out data/annotation/pilot_draft.jsonl \
                                        --model breeze2-3b --quant fp16
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from typing import Optional

from . import schema
from .schema import Annotation, Conversation

# --------------------------------------------------------------------------- #
# 1. 競賽 ERC canonical（7 類）→ 產品細層情緒
# --------------------------------------------------------------------------- #
# 引擎產不出 contempt / resignation → 由人在改稿時補；不在此硬湊。
ERC_TO_PRODUCT = {
    "neutral": "neutral",
    "joy": "joy",
    "sadness": "sadness",
    "anger": "anger",
    "surprise": "surprise",
    "fear": "anxiety",      # 產品把 fear 更名為「焦慮擔心」
    "disgust": "contempt",  # 產品砍 disgust、最接近的是「不屑嘲諷」
}


def product_valence(emotion: str) -> str:
    """由產品細層情緒推導 valence；surprise（依情境）/曖昧 → ambiguous（逼人決定）。"""
    return schema.EMOTION_VALENCE.get(emotion, "ambiguous")


# --------------------------------------------------------------------------- #
# 2. 難度標籤的便宜啟發式（草稿用；人會覆核）
# --------------------------------------------------------------------------- #
_CJK = re.compile(r"[一-鿿]")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
_BOPOMOFO = re.compile(r"[ㄅ-ㄩ]")   # 注音 ㄅ–ㄩ（火星文常見）

_NET_SLANG_KW = [
    "ㄏㄏ", "ㄎㄎ", "484", "87", "94", "104", "母湯", "笑死", "顆顆", "呵呵",
    "是在哈囉", "傻眼貓咪", "yyds", "emo", "zzz", "www", "8+9", "郭", "咪納桑",
]


def has_code_switch(text: str) -> bool:
    """中文夾雜 ≥2 字母的英文詞 → 視為語碼混用（台語/客語羅馬拼音也會落此類）。"""
    return bool(_CJK.search(text)) and bool(_LATIN_WORD.search(text))


def has_net_slang(text: str) -> bool:
    if _BOPOMOFO.search(text):
        return True
    low = text.lower()
    return any(k.lower() in low for k in _NET_SLANG_KW)


def draft_difficulty_tags(text: str) -> list[str]:
    """只自動打客觀的詞彙級標籤。

    **sarcasm 刻意不自動初標**：它正是模型的盲點（承重牆主張「模型在反諷上系統性錯」），
    用模型的推理鏈去偵測反諷邏輯自相矛盾，實測也證實不可靠——stage-1 prompt 本身就
    問「is it sarcastic?」，rationale 幾乎必含該詞，關鍵字掃描分不清肯定/否定，
    且真正的殺手反諷（「真好啊」被讀成真心）模型根本沒察覺。故 sarcasm 留給人判。
    distant_cause 由成因距離自動決定，但草稿不初標成因 → 這裡也不打。
    """
    tags = []
    if has_code_switch(text):
        tags.append("code_switch")
    if has_net_slang(text):
        tags.append("net_slang")
    return [t for t in schema.DIFFICULTY_TAGS if t in tags]   # 依正典順序、去重


# --------------------------------------------------------------------------- #
# 3. 判讀後端：mock 關鍵字規則 / 真實 ERC 引擎
# --------------------------------------------------------------------------- #
# mock 規則輸出「ERC canonical」，再走同一套 ERC_TO_PRODUCT，保持與真實路徑一致。
_MOCK_RULES = [
    ("disgust", ["噁", "噁心", "反胃", "呵呵", "顆顆", "傻眼", "無言", "gross"]),
    ("anger", ["生氣", "氣死", "煩", "幹嘛", "白目", "angry", "！！", "怒"]),
    ("joy", ["開心", "太好了", "哈哈", "讚", "爽", "happy", "great", "🎉", "😂"]),
    ("sadness", ["難過", "傷心", "哭", "失敗", "sad", "唉", "😭"]),
    ("surprise", ["驚", "什麼", "天啊", "傻眼", "怎麼會", "wow", "？！"]),
    ("fear", ["怕", "緊張", "擔心", "焦慮", "慘了", "afraid", "scared"]),
]


def _mock_erc(text: str) -> str:
    low = (text or "").lower()
    for emo, kws in _MOCK_RULES:
        if any(k.lower() in low for k in kws):
            return emo
    return "neutral"


class PreAnnotator:
    """對單則訊息產出草稿標註。mock=True 用關鍵字規則；否則載入真實 ERC 引擎。"""

    def __init__(self, mock: bool = False, model: str = "breeze2-3b",
                 quant: Optional[str] = None, use_context: bool = True,
                 two_stage: bool = True):
        self.mock = mock
        self.model = model
        self.quant = quant
        self.use_context = use_context
        self.two_stage = two_stage
        self._engine = None   # 延遲載入

    # -- 真實引擎延遲載入（與 erc_service 同慣例）-------------------------- #
    def _engine_lazy(self):
        if self._engine is None:
            from ..reasoning.backbone import Backbone, BackboneConfig
            from ..reasoning.erc_engine import ERCEngine
            print(f"[pre_annotate] 載入 ERC 引擎 {self.model} (quant={self.quant})…")
            bk = Backbone(BackboneConfig(model=self.model, quantization=self.quant))
            self._engine = ERCEngine(bk, use_context=self.use_context,
                                     two_stage=self.two_stage)
            print("[pre_annotate] 引擎就緒")
        return self._engine

    # -- 產出一則草稿 ----------------------------------------------------- #
    def draft(self, text: str, history: list[dict]) -> tuple[Annotation, dict]:
        if self.mock:
            erc_emotion = _mock_erc(text)
            rationale, parse_ok, used_ctx = "(keyword rule)", True, False
        else:
            res = self._engine_lazy().predict(text, history=history)
            erc_emotion = res.predicted or "neutral"   # 無法解析→退回 neutral（人會改）
            rationale, parse_ok, used_ctx = res.rationale, res.parse_ok, res.used_context

        product_emotion = ERC_TO_PRODUCT.get(erc_emotion, "neutral")
        # 模型無法解析（幻覺）→ 用 ambiguous 逃生口，別硬給 neutral 誤導人
        if not parse_ok:
            product_emotion, valence = "ambiguous", "ambiguous"
        else:
            valence = product_valence(product_emotion)

        ann = Annotation(
            valence=valence,
            emotion_primary=product_emotion,
            difficulty_tags=draft_difficulty_tags(text),
            cause=None,   # 成因刻意不初標（承重牆禁忌）——人從零判
        )
        meta = {
            "backbone": "mock" if self.mock else self.model,
            "erc_emotion": erc_emotion,
            "product_emotion": product_emotion,
            "parse_ok": parse_ok,
            "used_context": used_ctx,
            "rationale": rationale,
        }
        return ann, meta


# --------------------------------------------------------------------------- #
# 4. 對整段對話初標
# --------------------------------------------------------------------------- #


def annotate_conversation(conv: Conversation, pa: PreAnnotator) -> dict:
    """就地把 `_draft`/`_draft_meta` 寫進 conv.annotations，回傳本段的統計。"""
    stat = Counter()
    for i, u in enumerate(conv.utterances):
        # 對話內語境：前面所有訊息（引擎會依 use_context 決定用不用）
        history = [{"role": p.speaker, "content": p.text} for p in conv.utterances[:i]]
        ann, meta = pa.draft(u.text, history)

        # 保留可能已存在的人工標註，只更新草稿
        block = conv.annotations.get(str(u.uid), {})
        block["_draft"] = ann.to_dict()
        block["_draft_meta"] = meta
        block.setdefault("per_annotator", [])   # 供人填
        block.setdefault("gold", None)          # 裁決後才有
        conv.annotations[str(u.uid)] = block

        stat[ann.emotion_primary] += 1
        for t in ann.difficulty_tags:
            stat[f"tag:{t}"] += 1
        if not meta["parse_ok"]:
            stat["parse_fail"] += 1
    return stat


def run(in_path: str, out_path: str, pa: PreAnnotator,
        limit: Optional[int] = None) -> None:
    convs = schema.load_jsonl(in_path)
    if limit:
        convs = convs[:limit]

    total = Counter()
    n_utt = 0
    for c in convs:
        s = annotate_conversation(c, pa)
        total.update(s)
        n_utt += len(c.utterances)

    # 寫出前先自我驗證（草稿理應全數合規；不合規代表 schema/映射有 bug）
    errs = []
    for c in convs:
        errs.extend(schema.validate_conversation(c))
    if errs:
        print(f"[pre_annotate][warn] 草稿有 {len(errs)} 筆未通過 schema 驗證（前 5 筆）：")
        for e in errs[:5]:
            print(f"  ! {e}")

    schema.dump_jsonl(convs, out_path)

    # -- 摘要（給人判斷草稿品質、決定從哪些難例開始改）-------------------- #
    print(f"\n✓ 初標完成：{len(convs)} 段對話、{n_utt} 則訊息 → {out_path}")
    print("  情緒草稿分佈：")
    for emo in schema.EMOTION:
        if total.get(emo):
            print(f"    {emo:<12}{schema.EMOTION_ZH[emo]:<6}{total[emo]}")
    tagline = "  ".join(f"{t}={total.get('tag:'+t, 0)}" for t in schema.DIFFICULTY_TAGS)
    print(f"  難度標籤：{tagline}")
    if total.get("parse_fail"):
        print(f"  ⚠ 模型無法解析（標成 ambiguous）：{total['parse_fail']} 則")
    print("\n  下一步：人逐條改 `_draft` → 填 per_annotator（≥3 人）→ 裁決成 gold。")
    print("  提醒：cause 刻意留空，請人從零判；每一則都要過人腦，禁止抽查。")


def build_cli(argv: Optional[list[str]] = None) -> None:
    # Windows 主控台預設 cp950，無法輸出 ✓/⚠ 等字元會直接崩潰；統一改 UTF-8。
    import sys
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="初標助手：對原始對話 JSONL 產草稿標註（供人逐條改）")
    ap.add_argument("--in", dest="in_path", required=True, help="原始對話 JSONL（conv_id/meta/utterances）")
    ap.add_argument("--out", dest="out_path", required=True, help="輸出含 _draft 的 JSONL")
    ap.add_argument("--mock", action="store_true", help="用關鍵字規則，不載入模型（先驗管線）")
    ap.add_argument("--model", default="breeze2-3b", help="ERC 底座（preset 名或 HF repo）")
    ap.add_argument("--quant", default=None, help="4bit / 8bit / 留空=fp16")
    ap.add_argument("--no-context", action="store_true", help="不餵對話內語境給引擎")
    ap.add_argument("--single", action="store_true", help="單次直出標籤（不產推理鏈；較快、_draft_meta.rationale 會空）")
    ap.add_argument("--limit", type=int, default=None, help="只處理前 N 段對話")
    args = ap.parse_args(argv)

    quant = None if (args.quant or "").lower() in ("", "none", "fp16", "float16") else args.quant
    pa = PreAnnotator(mock=args.mock, model=args.model, quant=quant,
                      use_context=not args.no_context, two_stage=not args.single)
    run(args.in_path, args.out_path, pa, limit=args.limit)


if __name__ == "__main__":
    build_cli()
