# -*- coding: utf-8 -*-
"""
混合式二階段試驗：只在第一階段「沒把握」的句子上啟動 M-CoT，看推理鏈能否改善判讀。

做法：
  1. 第一階段＝上線設定（單次打分＋先驗校正，完整語境）的分數，取自
     evaluation_results/calib/scores_test_cped_fullctx.jsonl。
  2. 低信心＝最高與第二高機率差 < --margin（預設 0.10）。從中隨機抽 --n 句。
  3. 每句產生兩份推理鏈：LoRA 開啟（rationale_lora）與關閉（rationale_base）。
  4. 對每份推理鏈，用「LoRA 開／關」兩種模型做第二階段決策，三種決策方式：
       gen   ＝讓模型生成標籤（原版 M-CoT）
       all7  ＝在含推理鏈的 prompt 下對 7 個標籤打分取最高
       top2  ＝同上，但只在第一階段的前兩名之間選
     共 2 × 2 × 3 = 12 種組合；原始資料全部存檔，分析可重跑。
  5. 為避免「在同一批資料上挑最佳組合又報告它」的偏差：以奇偶分成
     挑選組／驗證組，只在挑選組選最佳組合，再於驗證組報告。

第二階段的推理語境取前 20 句（＝聊天室上線設定）。

用法：
    .venv\\Scripts\\python -m src.evaluation.hybrid_cot_trial --n 400
    .venv\\Scripts\\python -m src.evaluation.hybrid_cot_trial --analyze
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.reasoning.labels import CANONICAL_EMOTIONS as L, coerce_prediction

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "evaluation_results/calib/scores_test_cped_fullctx.jsonl"
CALIB = ROOT / "configs/erc_calibration.json"
OUT = ROOT / "evaluation_results/hybrid_cot/trial.jsonl"


def stage1():
    """第一階段（上線設定）的機率、預測與前兩名。"""
    cal = json.loads(CALIB.read_text(encoding="utf-8"))
    bias = cal["alpha"] * np.array([cal["shift"][k] for k in L])
    rows = [json.loads(l) for l in open(SCORES, encoding="utf-8")]
    S = np.array([r["scores"] for r in rows]) + bias
    P = np.exp(S - S.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    order = np.argsort(-P, 1)
    margin = P[np.arange(len(P)), order[:, 0]] - P[np.arange(len(P)), order[:, 1]]
    return rows, P, order, margin


def contexts(history_turns: int):
    """與打分時相同的走訪順序，重建每句的前文（只取前 history_turns 句）。"""
    from src.evaluation.eval_erc import load_cped_resplit
    out = []
    for conv in load_cped_resplit("test", by="series"):
        history = []
        for utt in conv:
            if utt.get("emotion") in L:
                out.append((utt["text"], history[-history_turns:]))
            history.append({"role": utt.get("speaker", "Speaker"), "content": utt["text"]})
    return out


def rationale_prompt(ctx, utt):
    return (f"[Dialogue context]\n{ctx}\n\n[Current utterance]\n{utt}\n\n"
            "Think step by step (3-4 sentences): How does the context affect the tone? "
            "Is it sincere or sarcastic? What emotion does the speaker most likely feel? "
            "Give ONLY the reasoning, do not output the final label yet.")


def decision_prompt(ctx, utt, rationale):
    from src.reasoning.prompts import LABEL_LINE
    return (f"[Dialogue context]\n{ctx}\n\n[Current utterance]\n{utt}\n\n"
            f"[Reasoning]\n{rationale}\n\n"
            f"Now choose exactly ONE label for the current utterance from these seven: "
            f"[{LABEL_LINE}]. Output only that one word — nothing outside the seven.")


def run(args):
    rows, P, order, margin = stage1()
    ctxs = contexts(args.history_turns)
    assert len(ctxs) == len(rows), (len(ctxs), len(rows))
    bad = sum(c[0].strip() != r["text"].strip() for c, r in zip(ctxs, rows))
    assert bad == 0, f"打分檔與資料順序對不上（{bad} 句）"

    low = [i for i in range(len(rows)) if margin[i] < args.margin]
    random.Random(args.seed).shuffle(low)
    picked = low[:args.n]
    done = set()
    if OUT.exists():
        done = {json.loads(l)["idx"] for l in open(OUT, encoding="utf-8")}
    todo = [i for i in picked if i not in done]
    print(f"[trial] 低信心 {len(low)} 句，抽 {len(picked)} 句，已完成 {len(done)}，待跑 {len(todo)}")

    import torch
    from src.reasoning.backbone import Backbone, BackboneConfig
    from src.reasoning.label_scorer import LabelScorer
    from src.reasoning.prompts import context_block, system_prompt
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.90)
    cal = json.loads(CALIB.read_text(encoding="utf-8"))
    bk = Backbone(BackboneConfig(model="breeze2-3b", lora_path=str(ROOT / "models" / cal["adapter"])))
    scorer = LabelScorer(bk)
    sys_p = system_prompt()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(OUT, "a", encoding="utf-8") as f:
        for n, i in enumerate(todo, 1):
            text, hist = ctxs[i]
            ctx = context_block(hist, use_context=bool(hist))
            rec = {"idx": i, "gold": rows[i]["gold"], "text": text,
                   "s1_pred": L[order[i, 0]], "s1_top2": [L[order[i, 0]], L[order[i, 1]]],
                   "margin": float(margin[i])}
            rats = {}
            rats["lora"] = bk.chat(sys_p, rationale_prompt(ctx, text), max_new_tokens=220)
            with bk.adapter_disabled():
                rats["base"] = bk.chat(sys_p, rationale_prompt(ctx, text), max_new_tokens=220)
            rec["rationale"] = rats
            rec["decide"] = {}
            for src, rat in rats.items():
                dp = decision_prompt(ctx, text, rat)
                for mode in ("on", "off"):
                    if mode == "on":
                        gen = bk.chat(sys_p, dp, max_new_tokens=10)
                        sc = scorer.score(sys_p, dp)
                    else:
                        with bk.adapter_disabled():
                            gen = bk.chat(sys_p, dp, max_new_tokens=10)
                            sc = scorer.score(sys_p, dp)
                    rec["decide"][f"{src}/{mode}"] = {"gen_raw": gen, "gen": coerce_prediction(gen),
                                                      "scores": sc}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"[trial] {n}/{len(todo)}（{el / n:.1f}s/句，剩約 {el / n * (len(todo) - n) / 60:.0f} 分）",
                  flush=True)


def paired_ci(a, b, iters=2000, seed=0):
    """b 相對 a 的準確率差（b−a）之 95% bootstrap 信賴區間。"""
    rng = np.random.default_rng(seed)
    d = np.array(b, float) - np.array(a, float)
    vals = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(iters)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def analyze(_args):
    recs = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    print(f"[analyze] {len(recs)} 句低信心樣本")

    def pred(r, cond):
        src, mode, how = cond
        d = r["decide"][f"{src}/{mode}"]
        if how == "gen":
            return d["gen"] or r["s1_pred"]          # 解析失敗 → 退回第一階段答案
        sc = d["scores"]
        cand = L if how == "all7" else r["s1_top2"]
        return max(cand, key=lambda k: sc[L.index(k)])

    conds = [(s, m, h) for s in ("lora", "base") for m in ("on", "off") for h in ("gen", "all7", "top2")]
    sel = [r for r in recs if r["idx"] % 2 == 0]
    val = [r for r in recs if r["idx"] % 2 == 1]
    print(f"挑選組 {len(sel)} 句、驗證組 {len(val)} 句\n")
    print(f"{'推理鏈/決策模型/方式':<20}{'挑選組':>8}{'驗證組':>8}{'全部':>8}{'解析失敗':>8}")
    base_sel = np.mean([r["s1_pred"] == r["gold"] for r in sel])
    base_val = np.mean([r["s1_pred"] == r["gold"] for r in val])
    base_all = np.mean([r["s1_pred"] == r["gold"] for r in recs])
    print(f"{'第一階段（上線設定）':<20}{base_sel:>8.3f}{base_val:>8.3f}{base_all:>8.3f}")
    best, best_acc = None, -1
    for c in conds:
        a_sel = np.mean([pred(r, c) == r["gold"] for r in sel])
        a_val = np.mean([pred(r, c) == r["gold"] for r in val])
        a_all = np.mean([pred(r, c) == r["gold"] for r in recs])
        fail = np.mean([r["decide"][f"{c[0]}/{c[1]}"]["gen"] is None for r in recs]) if c[2] == "gen" else 0
        print(f"{'/'.join(c):<22}{a_sel:>8.3f}{a_val:>8.3f}{a_all:>8.3f}{fail:>8.1%}")
        if a_sel > best_acc:
            best, best_acc = c, a_sel
    a = [r["s1_pred"] == r["gold"] for r in val]
    b = [pred(r, best) == r["gold"] for r in val]
    lo, hi = paired_ci(a, b)
    print(f"\n挑選組最佳：{'/'.join(best)}")
    print(f"驗證組：第一階段 {np.mean(a):.3f} → 混合 {np.mean(b):.3f}，差 {np.mean(b) - np.mean(a):+.3f}"
          f"（95% CI {lo:+.3f} ~ {hi:+.3f}）")
    fixed = sum((not x) and y for x, y in zip(a, b))
    broke = sum(x and (not y) for x, y in zip(a, b))
    print(f"驗證組中：改對 {fixed} 句、改錯 {broke} 句")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--history_turns", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    analyze(args) if args.analyze else run(args)


if __name__ == "__main__":
    main()
