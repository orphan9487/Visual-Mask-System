# -*- coding: utf-8 -*-
"""
大模型標註（silver label）的輔助工具。

  show    把待標題目印成精簡文字：每段對話先列原文（截斷），再逐行列出「題號｜發話者：內容」，
          供大模型依《標註準則》閱讀後標註。可用 --start / --count 分批。
  pack    把大模型逐批寫下的標註（每行：題號<TAB>標籤<TAB>反諷 0/1<TAB>信心 1–3）合成
          與標註網頁相同格式的 labels JSON，之後可直接用 chat_testset agree／eval 工具比較。
  sft     把標註轉成與現有微調資料相同格式的訓練樣本（前文取最後 N 句），
          可依信心過濾、限制中性比例。

標籤用英文 7 類（neutral joy sadness anger surprise fear disgust）。
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.annotation.chat_testset import expand_paths, history_for, load_items
from src.reasoning.labels import CANONICAL_EMOTIONS as L


def _clip(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + "…"


def show(args) -> None:
    items = load_items(args.items)
    targets = [it for it in items if not it.get("context_only")]
    batch = {it["id"] for it in targets[args.start:args.start + args.count]}
    pos = {it["id"]: k for k, it in enumerate(targets)}
    shown_conv = set()
    for it in items:
        conv = it["conv"]
        if not any(t["conv"] == conv for t in targets if t["id"] in batch):
            continue
        if conv not in shown_conv:
            shown_conv.add(conv)
            print(f"\n## {conv}")
        if it.get("context_only"):
            print(f"[原文] {it['speaker']}：{_clip(it['text'], args.clip)}")
        elif it["id"] in batch:
            print(f"{it['id']}｜{it['speaker']}：{_clip(it['text'], 200)}")
        elif pos[it["id"]] < args.start:      # 前一批標過、只當語境
            print(f"（前文）{it['speaker']}：{_clip(it['text'], 80)}")
        # 本批之後的句子不顯示：標註時只能看前文
    print(f"\n# 本批 {len(batch)} 則（第 {args.start + 1}–{args.start + len(batch)} 則，共 {len(targets)} 則）")


def read_tsv(paths) -> dict:
    out = {}
    for p in expand_paths(paths):
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            iid, label = parts[0].strip(), parts[1].strip()
            if label not in L:
                raise ValueError(f"{p}: {iid} 的標籤「{label}」不在 7 類中")
            sarc = len(parts) > 2 and parts[2].strip() == "1"
            conf = int(parts[3]) if len(parts) > 3 and parts[3].strip() else 2
            out[iid] = {"label": label, "sarcasm": sarc, "unsure": conf <= 1, "confidence": conf}
    return out


def pack(args) -> None:
    labels = read_tsv(args.tsv)
    items = load_items(args.items)
    targets = [it["id"] for it in items if not it.get("context_only")]
    missing = [t for t in targets if t not in labels]
    extra = [k for k in labels if k not in set(targets)]
    if extra:
        sys.exit(f"[pack] 有 {len(extra)} 個題號不在題目檔中，例如 {extra[:3]}")
    Path(args.out).write_text(json.dumps({"annotator": args.name, "n_items": len(targets),
                                          "labels": labels}, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"[pack] {len(labels)}／{len(targets)} 則 → {args.out}"
          + (f"（尚缺 {len(missing)} 則）" if missing else ""))
    print(f"[pack] 分布 {dict(Counter(v['label'] for v in labels.values()))}　"
          f"反諷 {sum(v['sarcasm'] for v in labels.values())}　低信心 {sum(v['unsure'] for v in labels.values())}")


def sft(args) -> None:
    from src.reasoning.prompts import context_block, label_user_prompt, system_prompt
    items = load_items(args.items)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))["labels"]
    hist = history_for(items)
    rows = []
    for it in items:
        lab = labels.get(it["id"])
        if it.get("context_only") or not lab or lab.get("confidence", 2) < args.min_confidence:
            continue
        # 論壇原文與長推文會讓 prompt 超過訓練的 max_len，答案被截掉；逐句截短前文
        recent = [{**h, "content": _clip(h["content"], args.clip_context)}
                  for h in hist[it["id"]][-args.history_turns:]]
        # 與 data/erc_sft/*.jsonl 相同的扁平格式（system / user / assistant），可直接併入訓練
        rows.append({"system": system_prompt(),
                     "user": label_user_prompt(context_block(recent, use_context=bool(recent)), it["text"]),
                     "assistant": lab["label"],
                     "meta": {"id": it["id"], "source": "llm_silver", "sarcasm": lab.get("sarcasm", False),
                              "confidence": lab.get("confidence", 2)}})
    # 限制中性比例（與保反諷配方同一個理由：中性過多會讓模型把反話判成中性）
    neu = [r for r in rows if r["assistant"] == "neutral"]
    other = [r for r in rows if r["assistant"] != "neutral"]
    cap = int(len(other) * args.max_neutral_ratio / (1 - args.max_neutral_ratio)) if args.max_neutral_ratio < 1 else len(neu)
    random.Random(args.seed).shuffle(neu)
    rows = other + neu[:cap]
    random.Random(args.seed).shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[sft] {len(rows)} 筆 → {args.out}　分布 {dict(Counter(r['assistant'] for r in rows))}"
          f"（中性原有 {len(neu)}，保留 {min(cap, len(neu))}）")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="大模型標註輔助工具")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("items")
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--count", type=int, default=100)
    s.add_argument("--clip", type=int, default=150, help="原文最多顯示幾字")
    s.set_defaults(fn=show)
    p = sub.add_parser("pack")
    p.add_argument("tsv", nargs="+")
    p.add_argument("--items", required=True)
    p.add_argument("--name", default="claude")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=pack)
    f = sub.add_parser("sft")
    f.add_argument("items")
    f.add_argument("--labels", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--history_turns", type=int, default=6, help="與現有微調資料相同：前 6 句")
    f.add_argument("--min_confidence", type=int, default=2, help="只收信心 ≥ 此值（1 低、2 中、3 高）")
    f.add_argument("--clip_context", type=int, default=60,
                   help="每句前文最多保留幾字（避免超過訓練 max_len 384 而截掉答案）")
    f.add_argument("--max_neutral_ratio", type=float, default=0.4)
    f.add_argument("--seed", type=int, default=0)
    f.set_defaults(fn=sft)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
