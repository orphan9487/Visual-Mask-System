# -*- coding: utf-8 -*-
"""
把 MELD train.csv 轉成 LoRA 指令微調樣本 (JSONL)。

每一句 → 一筆樣本：
    system    : ERC 系統提示（prompts.system_prompt，與推論完全相同）
    user      : [對話語境] + [當前句] + Choose exactly ONE label...（prompts.label_user_prompt）
    assistant : 該句的 canonical 情緒標籤（只訓練這個 token）

語境 = 同一對話中「該句之前」的數句（視窗 = --history_turns），與推論時的
Contextual Buffer 行為一致。

用法：
    .venv\\Scripts\\python -m src.training.format_meld --split train --history_turns 6 --limit 4000
"""

import argparse
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.eval_erc import load_cped, load_meld, load_zh_dialogue
from src.reasoning.labels import CANONICAL_EMOTIONS
from src.reasoning.prompts import context_block, label_user_prompt, system_prompt

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "erc_sft"

LOADERS = {"meld": load_meld, "zh": load_zh_dialogue, "cped": load_cped}


def build_samples(dataset: str, split: str, history_turns: int, limit: int | None):
    convs = LOADERS[dataset](split)
    use_ctx = history_turns > 0   # history_turns=0 → 完全無語境（Python [-0:] 會取全部，故明確處理）
    samples = []
    for conv in convs:
        history = []
        for utt in conv:
            gold = utt.get("emotion")
            if gold in CANONICAL_EMOTIONS:
                recent = history[-history_turns:] if use_ctx else []
                ctx = context_block(recent, use_context=use_ctx)
                samples.append({
                    "system": system_prompt(),
                    "user": label_user_prompt(ctx, utt["text"]),
                    "assistant": gold,
                })
            history.append({"role": utt.get("speaker", "Speaker"), "content": utt["text"]})
            if limit and len(samples) >= limit:
                return samples
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="meld", choices=["meld", "zh", "cped"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--history_turns", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    samples = build_samples(args.dataset, args.split, args.history_turns, args.limit)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.dataset}_{args.split}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 分布快照
    from collections import Counter
    dist = Counter(s["assistant"] for s in samples)
    print(f"[format] {len(samples)} 筆樣本 → {out}")
    print(f"[format] 標籤分布：{dict(dist)}")
    print("[format] 樣本示例：")
    print("  system   :", samples[0]["system"][:70], "...")
    print("  user     :", samples[0]["user"][:90].replace("\n", " / "), "...")
    print("  assistant:", samples[0]["assistant"])


if __name__ == "__main__":
    main()
