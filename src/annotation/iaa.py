# -*- coding: utf-8 -*-
"""
標註者間一致性（Inter-Annotator Agreement, IAA）+ 多數決 gold 輔助。

對應 docs/標註schema_草案v0.md 第 7 節與 pilot 決策點：
  - Valence / 情緒主標：Fleiss' kappa（多標註者）
  - 難度標籤（多選）：逐標籤 binary kappa + 平均 Jaccard
  - 成因：cause_uid 命中率（完全命中 + 允許 ±1 的寬鬆版）+ source 的 kappa

**這是 benchmark 的驗收工具**：pilot 標完 20 條後，用它回答承重牆是否成立——
  valence + 情緒 + 反諷標籤的 IAA 可接受（人類穩定同意）
  ∧ 同一批 case GPT/Claude 明顯錯（另由 Claude 尺量）→ 護城河成立、放大；
  成因 IAA 太低（SemEval 冠軍才 .32–.38）→ v1 砍成因。

輸入：標註 JSONL，每則 annotations[uid]["per_annotator"] 需有 ≥2 位標註者。
  （還沒 pilot 資料時，本工具會明白告訴你「找不到多人標註」，不會硬算。）

用法：
  python -m src.annotation.iaa data/annotation/pilot_labeled.jsonl
  python -m src.annotation.iaa data/annotation/pilot_labeled.jsonl --suggest-gold data/annotation/pilot_gold_suggested.jsonl
  python -m src.annotation.iaa --selftest        # 用 Fleiss 標準例驗證數學

Fleiss' kappa 採「每則標註者數可不同」的一般式（各則 n_i≥2 即可；
n_i 全相等時退化為經典 Fleiss）。
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from . import schema
from .schema import Annotation, Cause, Conversation


# --------------------------------------------------------------------------- #
# 1. Fleiss' kappa（一般式，允許各則標註者數不同）
# --------------------------------------------------------------------------- #


def _fleiss_from_counts(count_rows: list[list[int]]) -> float:
    """
    core：由「每則 × 每類別的計數矩陣」算 Fleiss' kappa。
    count_rows[i][j] = 第 i 則被歸到第 j 類的標註者數。
    允許各則總數 n_i 不同（n_i≥2）；回傳 kappa（無類別變異時回 nan）。
    """
    rows = [r for r in count_rows if sum(r) >= 2]
    N = len(rows)
    if N == 0:
        return float("nan")
    k = len(rows[0])
    col_totals = [0] * k
    total = 0
    P_sum = 0.0
    for r in rows:
        n_i = sum(r)
        total += n_i
        for j in range(k):
            col_totals[j] += r[j]
        P_i = (sum(c * c for c in r) - n_i) / (n_i * (n_i - 1))
        P_sum += P_i
    P_bar = P_sum / N
    p = [col_totals[j] / total for j in range(k)]
    P_e = sum(pj * pj for pj in p)
    denom = 1.0 - P_e
    if denom == 0:                      # 全部落同一類 → kappa 未定義
        return float("nan")
    return (P_bar - P_e) / denom


def fleiss_kappa(ratings: list[list[str]], categories: list[str]) -> float:
    """
    ratings：每則一個內層清單＝各標註者給的類別標籤（長度可不同，≥2 才納入）。
    categories：合法類別（決定計數矩陣的欄）。不在其中的標籤會被忽略（回報前先驗證過就不會有）。
    """
    idx = {c: i for i, c in enumerate(categories)}
    count_rows = []
    for r in ratings:
        row = [0] * len(categories)
        for lab in r:
            if lab in idx:
                row[idx[lab]] += 1
        count_rows.append(row)
    return _fleiss_from_counts(count_rows)


def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977) 分級。"""
    if k != k:                          # nan
        return "未定義（無類別變異）"
    if k < 0:
        return "poor 差（<0）"
    if k < 0.20:
        return "slight 輕微"
    if k < 0.40:
        return "fair 尚可"
    if k < 0.60:
        return "moderate 中等"
    if k < 0.80:
        return "substantial 可觀"
    return "almost perfect 幾近完美"


# --------------------------------------------------------------------------- #
# 2. 難度標籤（多選）：逐標籤 binary kappa + 平均 Jaccard
# --------------------------------------------------------------------------- #


def _mean_pairwise_jaccard(sets_per_item: list[list[set]]) -> float:
    """每則所有標註者兩兩 Jaccard 的平均，再對所有則平均（兩集合皆空記為 1.0）。"""
    micro_num = 0.0
    micro_den = 0
    for annotator_sets in sets_per_item:
        m = len(annotator_sets)
        if m < 2:
            continue
        for a in range(m):
            for b in range(a + 1, m):
                sa, sb = annotator_sets[a], annotator_sets[b]
                union = sa | sb
                j = 1.0 if not union else len(sa & sb) / len(union)
                micro_num += j
                micro_den += 1
    return micro_num / micro_den if micro_den else float("nan")


def tag_agreement(sets_per_item: list[list[set]]) -> dict:
    """
    sets_per_item：每則一個清單，內含各標註者標的難度標籤集合。
    回傳：{ per_tag: {tag: kappa}, mean_jaccard: float, n_items: int }
    每個標籤視為 binary（有/無），對每則各標註者算「有=1/無=0」→ binary Fleiss。
    """
    items = [s for s in sets_per_item if len(s) >= 2]
    result_per_tag = {}
    for tag in schema.DIFFICULTY_TAGS:
        count_rows = []
        for annotator_sets in items:
            has = sum(1 for s in annotator_sets if tag in s)
            no = len(annotator_sets) - has
            count_rows.append([has, no])     # 兩類：有 / 無
        result_per_tag[tag] = _fleiss_from_counts(count_rows)
    return {
        "per_tag": result_per_tag,
        "mean_jaccard": _mean_pairwise_jaccard(items),
        "n_items": len(items),
    }


# --------------------------------------------------------------------------- #
# 3. 成因：source kappa + cause_uid 命中率（完全 / ±1）
# --------------------------------------------------------------------------- #


def cause_agreement(causes_per_item: list[list[Cause]]) -> dict:
    """
    causes_per_item：每則一個清單，內含各標註者標的 Cause（只含有標成因者）。
    - source_kappa：對 CAUSE_SOURCE 三類做 Fleiss（每則 ≥2 個 cause 才納入）
    - uid_exact / uid_within1：僅在 source=in_conversation 且有 cause_uid 的成因間，
      兩兩比對 cause_uid 的一致率（micro：命中對數 / 總對數）
    """
    # -- source kappa --
    src_ratings = [[c.source for c in cs if c and c.source] for cs in causes_per_item]
    source_kappa = fleiss_kappa(src_ratings, schema.CAUSE_SOURCE)

    # -- cause_uid 兩兩命中率 --
    exact_num = within_num = pair_den = 0
    n_uid_items = 0
    for cs in causes_per_item:
        uids = [c.cause_uid for c in cs
                if c and c.source == "in_conversation" and c.cause_uid is not None]
        if len(uids) < 2:
            continue
        n_uid_items += 1
        for a in range(len(uids)):
            for b in range(a + 1, len(uids)):
                pair_den += 1
                if uids[a] == uids[b]:
                    exact_num += 1
                if abs(uids[a] - uids[b]) <= 1:
                    within_num += 1
    return {
        "source_kappa": source_kappa,
        "uid_exact": exact_num / pair_den if pair_den else float("nan"),
        "uid_within1": within_num / pair_den if pair_den else float("nan"),
        "n_source_items": sum(1 for r in src_ratings if len(r) >= 2),
        "n_uid_items": n_uid_items,
    }


# --------------------------------------------------------------------------- #
# 4. 多數決 → gold 建議（人裁決前的機械輔助；不自動當 gold）
# --------------------------------------------------------------------------- #


def majority_vote(per_annotator: list[dict], uid: int) -> tuple[Annotation, dict]:
    """
    由多位標註者多數決出「gold 建議」。tie / 低一致度須人裁決，故寫進 `_gold_suggestion`
    而非 `gold`（沿用 `_draft` 的分離慣例）。
    回傳 (建議 Annotation, agreement dict)；agreement 為各欄與多數一致的比例。
    """
    anns = [Annotation.from_dict(pa) for pa in per_annotator]
    n = len(anns)

    def _mode(values):
        c = Counter(values)
        top, cnt = c.most_common(1)[0]
        return top, cnt / n

    val, val_agr = _mode([a.valence for a in anns])
    emo, emo_agr = _mode([a.emotion_primary for a in anns])

    # 難度標籤：出現在過半標註者 → 納入
    tag_counter = Counter(t for a in anns for t in a.difficulty_tags)
    tags = [t for t in schema.DIFFICULTY_TAGS if tag_counter[t] > n / 2]

    # 成因：多數 source；若 in_conversation 則取多數 cause_uid
    cause = None
    src_votes = [a.cause.source for a in anns if a.cause]
    if src_votes:
        src = Counter(src_votes).most_common(1)[0][0]
        if src == "in_conversation":
            uid_votes = [a.cause.cause_uid for a in anns
                         if a.cause and a.cause.source == "in_conversation"
                         and a.cause.cause_uid is not None]
            cuid = Counter(uid_votes).most_common(1)[0][0] if uid_votes else None
            cause = Cause(source="in_conversation", cause_uid=cuid)
        else:
            cause = Cause(source=src)

    ann = Annotation(valence=val, emotion_primary=emo, difficulty_tags=tags, cause=cause)
    schema.apply_distance(ann, uid)     # 自動算 distance + distant_cause 標籤
    agreement = {"valence": round(val_agr, 3), "emotion_primary": round(emo_agr, 3)}
    return ann, agreement


# --------------------------------------------------------------------------- #
# 5. 彙總報告
# --------------------------------------------------------------------------- #


@dataclass
class IAAReport:
    n_utterances_multi: int = 0          # 有 ≥2 標註者的訊息數
    n_annotators_seen: int = 0
    valence_kappa: float = float("nan")
    emotion_kappa: float = float("nan")
    tag: dict = field(default_factory=dict)
    cause: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _gather(convs: list[Conversation]):
    """收集所有「有 ≥2 標註者」的訊息，回傳各面向的 per-item 清單。"""
    valence_rows, emotion_rows, tag_rows, cause_rows = [], [], [], []
    annotators = set()
    for c in convs:
        for uid_str, block in c.annotations.items():
            pa = block.get("per_annotator") or []
            for entry in pa:
                if entry.get("annotator"):
                    annotators.add(entry["annotator"])
            if len(pa) < 2:
                continue
            anns = [Annotation.from_dict(e) for e in pa]
            valence_rows.append([a.valence for a in anns])
            emotion_rows.append([a.emotion_primary for a in anns])
            tag_rows.append([set(a.difficulty_tags) for a in anns])
            cause_rows.append([a.cause for a in anns if a.cause])
    return valence_rows, emotion_rows, tag_rows, cause_rows, annotators


def compute_iaa(convs: list[Conversation]) -> IAAReport:
    val_rows, emo_rows, tag_rows, cause_rows, annotators = _gather(convs)
    rep = IAAReport()
    rep.n_utterances_multi = len(val_rows)
    rep.n_annotators_seen = len(annotators)

    if rep.n_utterances_multi == 0:
        rep.notes.append(
            "找不到任何「≥2 位標註者」的訊息。IAA 需要多人獨立標註的 per_annotator 資料——"
            "請先完成 pilot（每則 ≥3 人各自標）再跑本工具。")
        return rep

    rep.valence_kappa = fleiss_kappa(val_rows, schema.VALENCE)
    rep.emotion_kappa = fleiss_kappa(emo_rows, schema.EMOTION)
    rep.tag = tag_agreement(tag_rows)
    rep.cause = cause_agreement(cause_rows)
    return rep


def print_report(rep: IAAReport) -> None:
    print("=" * 56)
    print("標註者間一致性（IAA）報告")
    print("=" * 56)
    print(f"納入訊息（≥2 標註者）：{rep.n_utterances_multi}    標註者人數：{rep.n_annotators_seen}")
    if rep.notes:
        for n in rep.notes:
            print(f"\n[!] {n}")
        return

    def line(name, k):
        print(f"  {name:<16}κ = {k:.3f}   {interpret_kappa(k)}")

    print("\n[Tier1 承重牆] Valence")
    line("valence", rep.valence_kappa)
    print("\n[Tier2] 細層情緒主標")
    line("emotion_primary", rep.emotion_kappa)

    print("\n[難度標籤] 逐標籤 binary κ（+ 平均 Jaccard）")
    for tag, k in rep.tag["per_tag"].items():
        line(tag, k)
    print(f"  {'mean Jaccard':<16}{rep.tag['mean_jaccard']:.3f}")

    print("\n[成因]")
    c = rep.cause
    line("source", c["source_kappa"])
    print(f"  {'cause_uid 完全命中':<16}{c['uid_exact']:.3f}   （±1 寬鬆：{c['uid_within1']:.3f}）")
    print(f"  （source 納入 {c['n_source_items']} 則、cause_uid 比對 {c['n_uid_items']} 則）")

    print("\n" + "-" * 56)
    print("pilot 決策參考：valence κ 若達 substantial(≥.6)＝承重牆穩；")
    print("情緒/反諷 κ 達 moderate 以上可用；成因 κ 或命中率太低 → v1 砍成因。")
    print("（一致性只證『人能穩定同意』；還要配 Claude 尺證『模型會錯』才成立護城河。）")


# --------------------------------------------------------------------------- #
# 6. gold 建議輸出
# --------------------------------------------------------------------------- #


def suggest_gold(convs: list[Conversation]) -> int:
    """對每則有 ≥1 標註者的訊息，寫入 `_gold_suggestion`（多數決）+ `_agreement`。
    不動 `gold`（那需人裁決）。回傳寫入的訊息數。"""
    n = 0
    for c in convs:
        for uid_str, block in c.annotations.items():
            pa = block.get("per_annotator") or []
            if not pa:
                continue
            ann, agr = majority_vote(pa, int(uid_str))
            block["_gold_suggestion"] = ann.to_dict()
            block["_agreement"] = agr
            n += 1
    return n


# --------------------------------------------------------------------------- #
# 7. 自我測試（用 Fleiss 標準例驗證數學）
# --------------------------------------------------------------------------- #


def _selftest() -> int:
    # Wikipedia Fleiss' kappa 範例（10 subjects × 5 categories，14 raters）→ κ≈0.210
    wiki = [
        [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
    ]
    k = _fleiss_from_counts(wiki)
    ok1 = abs(k - 0.210) < 0.005
    print(f"[selftest] Fleiss 標準例 κ={k:.4f}（期望≈0.210）→ {'PASS' if ok1 else 'FAIL'}")

    # 完全一致 → κ 應為 1.0（有變異時）
    perfect = fleiss_kappa([["positive", "positive", "positive"],
                            ["negative", "negative", "negative"]], schema.VALENCE)
    ok2 = abs(perfect - 1.0) < 1e-9
    print(f"[selftest] 完全一致（有變異）κ={perfect:.4f}（期望 1.0）→ {'PASS' if ok2 else 'FAIL'}")

    # 變異數不同的標註者數（3 人 vs 2 人）不應崩潰
    mixed = fleiss_kappa([["joy", "joy", "neutral"], ["anger", "anger"]], schema.EMOTION)
    ok3 = mixed == mixed  # not nan
    print(f"[selftest] 變動標註者數 κ={mixed:.4f} → {'PASS' if ok3 else 'FAIL'}")

    # 多數決 + 距離自動算：uid=3 成因指 uid=1 → distance 2 → 自動 distant_cause
    pa = [
        {"annotator": "A", "valence": "negative", "emotion_primary": "contempt",
         "difficulty_tags": ["sarcasm"], "cause": {"source": "in_conversation", "cause_uid": 1}},
        {"annotator": "B", "valence": "negative", "emotion_primary": "anger",
         "difficulty_tags": ["sarcasm"], "cause": {"source": "in_conversation", "cause_uid": 1}},
        {"annotator": "C", "valence": "negative", "emotion_primary": "contempt",
         "difficulty_tags": ["sarcasm"], "cause": {"source": "in_conversation", "cause_uid": 1}},
    ]
    ann, agr = majority_vote(pa, uid=3)
    ok4 = (ann.valence == "negative" and ann.emotion_primary == "contempt"
           and "sarcasm" in ann.difficulty_tags and "distant_cause" in ann.difficulty_tags
           and ann.cause.distance == 2 and abs(agr["emotion_primary"] - 0.667) < 0.01)
    print(f"[selftest] 多數決 gold={ann.emotion_primary} tags={ann.difficulty_tags} "
          f"dist={ann.cause.distance} agr={agr} → {'PASS' if ok4 else 'FAIL'}")

    all_ok = ok1 and ok2 and ok3 and ok4
    print(f"\n[selftest] {'全部通過 ✓' if all_ok else '有失敗 ✗'}")
    return 0 if all_ok else 1


# --------------------------------------------------------------------------- #
# 8. CLI
# --------------------------------------------------------------------------- #


def build_cli(argv: Optional[list[str]] = None) -> int:
    import sys
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="計算標註者間一致性（IAA）")
    ap.add_argument("path", nargs="?", help="標註 JSONL（含 per_annotator）")
    ap.add_argument("--suggest-gold", metavar="OUT",
                    help="以多數決產出 _gold_suggestion 寫到 OUT（不覆蓋 gold）")
    ap.add_argument("--selftest", action="store_true", help="用 Fleiss 標準例驗證數學後結束")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.path:
        ap.error("需要標註 JSONL 路徑（或用 --selftest）")

    convs = schema.load_jsonl(args.path)
    rep = compute_iaa(convs)
    print_report(rep)

    if args.suggest_gold:
        n = suggest_gold(convs)
        schema.dump_jsonl(convs, args.suggest_gold)
        print(f"\n✓ 已對 {n} 則寫入 _gold_suggestion → {args.suggest_gold}")
        print("  提醒：這是多數決建議，非最終 gold。tie 與低一致度須人裁決後才升為 gold。")
    return 0


if __name__ == "__main__":
    raise SystemExit(build_cli())
