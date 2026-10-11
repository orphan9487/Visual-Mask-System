# -*- coding: utf-8 -*-
"""
為每句話計算 7 個情緒標籤各自的對數機率（log p(標籤 | prompt)），存成檔案供校正分析。

為什麼要打分而不是直接生成：生成只給出 argmax，看不到模型對其他標籤的信心；
有了 7 個分數，才能做「先驗校正」——訓練資料刻意把 neutral 降到 20%（保反諷配方），
模型學到「中性很少見」，可在分數上依訓練/自然分布之比補償，且不必重訓（反諷能力不受影響）。

打分方式與訓練完全同格式：system+user 併入 user turn、MRPromptV3 組 prompt，
分數 = 標籤 token 加上結尾 <|eot_id|> 的對數機率總和（含 eot 才能區分 "fear" 與其延伸）。
語境取「前 N 句」（預設 6，與訓練 history_turns、聊天室視窗一致）。

用法：
    .venv\\Scripts\\python -m src.evaluation.score_labels --dataset cped80series --split test \\
        --lora models/breeze2_erc_cped80series_sarcasm_lora --out evaluation_results/scores_test.jsonl
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.evaluation.eval_erc import load_cped_resplit, load_m3ed
from src.reasoning.backbone import Backbone, BackboneConfig
from src.reasoning.label_scorer import LabelScorer
from src.reasoning.labels import CANONICAL_EMOTIONS
from src.reasoning.prompts import context_block, label_user_prompt, system_prompt

LOADERS = {
    "cped80series": lambda split: load_cped_resplit(split, by="series"),
    "m3ed": lambda split: load_m3ed(split),
}




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(LOADERS))
    ap.add_argument("--split", default="test")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--history_turns", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="最多幾句（對話以 seed 打亂後依序取）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.90)   # 寧可報 OOM，也不要默默溢出變慢
    convs = LOADERS[args.dataset](args.split)
    if args.limit:
        random.Random(args.seed).shuffle(convs)    # 抽樣時避免只取到開頭幾部劇
    bk = Backbone(BackboneConfig(model="breeze2-3b", lora_path=args.lora))
    scorer = LabelScorer(bk)
    sys_p = system_prompt()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n, t0 = 0, time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        for conv in convs:
            history = []
            for utt in conv:
                gold = utt.get("emotion")
                if gold in CANONICAL_EMOTIONS:
                    recent = history[-args.history_turns:] if args.history_turns > 0 else []
                    ctx = context_block(recent, use_context=args.history_turns > 0)
                    scores = scorer.score(sys_p, label_user_prompt(ctx, utt["text"]))
                    f.write(json.dumps({"text": utt["text"], "gold": gold, "scores": scores},
                                       ensure_ascii=False) + "\n")
                    n += 1
                    if n % 1000 == 0:
                        el = time.time() - t0
                        print(f"[score] {n} 句（{el:.0f}s，{el / n:.2f}s/句）", flush=True)
                history.append({"role": utt.get("speaker", "Speaker"), "content": utt["text"]})
                if args.limit and n >= args.limit:
                    break
            if args.limit and n >= args.limit:
                break
    print(f"[score] 完成 {n} 句 → {args.out}（標籤順序 {CANONICAL_EMOTIONS}）")


if __name__ == "__main__":
    main()
