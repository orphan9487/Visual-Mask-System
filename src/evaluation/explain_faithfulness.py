# -*- coding: utf-8 -*-
"""
推理鏈解釋的忠實度測試（反事實刪除）。

問題：按需解釋說「判成悲傷是因為前文提到 X」，X 真的影響了判讀嗎？
做法（CPED 測試集，上線設定：單次打分＋先驗校正＋前 20 句語境）：
  1. 抽 --n 句（前文至少 3 句），第一階段判讀出標籤 ŷ。
  2. 照上線流程（暫停 LoRA）為 ŷ 產生解釋。
  3. 找出解釋引用了哪幾句前文：先比對引號內的字串，比對不到時再看與前文
     是否有 ≥4 字的共同片段（不算出現在目前這句裡的片段）。
  4. 刪除後重新判讀：
       loo    ＝逐句刪除：一次只刪一句前文，得到每句前文的實際重要度（原標籤機率下降量）
       cited  ＝刪掉被引用的前文
       random ＝隨機刪掉同樣句數、未被引用的前文（對照組）
       all    ＝刪掉全部前文（參考上限）
  5. 指標：
       - 引用句 vs 未引用句的平均重要度（成對 bootstrap 95% CI）
       - 最重要的那句前文是否被解釋引用（命中率 vs 隨機猜中的機率）
       - cited 與 random 的改判比例、機率下降量差
     引用句明顯比未引用句重要 → 解釋指到的前文確實是判讀依據（較忠實）。

解釋常以改寫方式提到前文，字串比對會漏掉一部分引用；逐句重要度與前文都存檔，
分析時可用 --min_overlap 調整比對規則重算，不必重跑模型。原始紀錄可續跑。

用法：
    .venv\\Scripts\\python -m src.evaluation.explain_faithfulness --n 200
    .venv\\Scripts\\python -m src.evaluation.explain_faithfulness --analyze
"""

import argparse
import json
import random
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.reasoning.labels import CANONICAL_EMOTIONS as L

ROOT = Path(__file__).resolve().parents[2]
SCORES = ROOT / "evaluation_results/calib/scores_test_cped_fullctx.jsonl"
CALIB = ROOT / "configs/erc_calibration.json"
OUT_DIR = ROOT / "evaluation_results/explain_faithfulness"

QUOTE = re.compile(r"「([^」]{2,})」|『([^』]{2,})』|“([^”]{2,})”|\"([^\"]{2,})\"")
PUNCT = re.compile(r"[\s，。！？、；：,.!?;:…~～「」『』“”\"'（）()\[\]【】—\-]+")


def norm(s: str) -> str:
    return PUNCT.sub("", s or "")


def cited_turns(explanation: str, history: list, utterance: str,
                min_overlap: int = 4) -> tuple[list[int], str]:
    """回傳被解釋引用的前文索引與判定方式（quote／overlap／none）。history 可為 dict 或字串。"""
    utt = norm(utterance)
    turns = [norm(h["content"] if isinstance(h, dict) else h) for h in history]
    quotes = [norm(next(g for g in m.groups() if g)) for m in QUOTE.finditer(explanation)]
    quotes = [q for q in quotes if len(q) >= 2 and q not in utt]   # 引的是目前這句 → 不算前文
    hits = sorted({i for i, t in enumerate(turns) for q in quotes
                   if t and (q in t or (len(t) >= 2 and t in q))})
    if hits:
        return hits, "quote"
    exp = norm(explanation)
    hits = []
    for i, t in enumerate(turns):
        if not t:
            continue
        m = SequenceMatcher(None, exp, t, autojunk=False).find_longest_match(0, len(exp), 0, len(t))
        frag = t[m.b:m.b + m.size]
        if m.size >= min_overlap and frag not in utt:
            hits.append(i)
    return hits, ("overlap" if hits else "none")


def calibrated_probs(scores, bias) -> np.ndarray:
    s = np.array(scores) + bias
    p = np.exp(s - s.max())
    return p / p.sum()


def contexts(history_turns: int):
    from src.evaluation.hybrid_cot_trial import contexts as _ctx
    return _ctx(history_turns)


def run(args):
    out = OUT_DIR / "records.jsonl"
    rows = [json.loads(l) for l in open(SCORES, encoding="utf-8")]
    ctxs = contexts(args.history_turns)
    assert len(ctxs) == len(rows), (len(ctxs), len(rows))
    assert all(c[0].strip() == r["text"].strip() for c, r in zip(ctxs, rows)), "打分檔與資料順序對不上"

    pool = [i for i, (_, h) in enumerate(ctxs) if len(h) >= 3]
    picked = random.Random(args.seed).sample(pool, min(args.n, len(pool)))
    done = set()
    if out.exists():
        done = {json.loads(l)["idx"] for l in open(out, encoding="utf-8")}
    todo = [i for i in picked if i not in done]
    print(f"[faith] 前文 ≥3 句的候選 {len(pool)} 句，抽 {len(picked)}，已完成 {len(done)}，待跑 {len(todo)}")
    if not todo:
        return

    import torch
    from src.reasoning.backbone import Backbone, BackboneConfig
    from src.reasoning.label_scorer import LabelScorer
    from src.reasoning.prompts import (EXPLAIN_SYSTEM, context_block, explain_user_prompt,
                                       label_user_prompt, system_prompt)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.90)
    cal = json.loads(CALIB.read_text(encoding="utf-8"))
    bias = cal["alpha"] * np.array([cal["shift"][k] for k in L])
    bk = Backbone(BackboneConfig(model="breeze2-3b", lora_path=str(ROOT / "models" / cal["adapter"])))
    scorer = LabelScorer(bk)
    sys_p = system_prompt()

    def judge(hist, text):
        sc = scorer.score(sys_p, label_user_prompt(context_block(hist, use_context=bool(hist)), text))
        return calibrated_probs(sc, bias)

    rng = random.Random(args.seed + 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out, "a", encoding="utf-8") as f:
        for n, i in enumerate(todo, 1):
            text, hist = ctxs[i]
            p0 = judge(hist, text)
            y = int(np.argmax(p0))
            with bk.adapter_disabled():
                expl = bk.chat(EXPLAIN_SYSTEM,
                               explain_user_prompt(context_block(hist), None, text, L[y]),
                               max_new_tokens=200).strip()
            cited, how = cited_turns(expl, hist, text)
            rec = {"idx": i, "gold": rows[i]["gold"], "text": text, "n_ctx": len(hist),
                   "history": [h["content"] for h in hist],
                   "pred": L[y], "p_pred": float(p0[y]), "explanation": expl,
                   "cited": cited, "cite_method": how, "cf": {}, "loo": []}
            for k in range(len(hist)):
                p = judge(hist[:k] + hist[k + 1:], text)
                rec["loo"].append({"pred": L[int(np.argmax(p))], "p_orig": float(p[y])})
            variants = {"all": []}
            if cited:
                keep = [k for k in range(len(hist)) if k not in cited]
                variants["cited"] = [hist[k] for k in keep]
                others = [k for k in range(len(hist)) if k not in cited]
                if len(others) >= len(cited):
                    drop = set(rng.sample(others, len(cited)))
                    variants["random"] = [hist[k] for k in range(len(hist)) if k not in drop]
            for name, h in variants.items():
                p = judge(h, text)
                rec["cf"][name] = {"pred": L[int(np.argmax(p))], "p_orig": float(p[y])}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            el = time.time() - t0
            print(f"[faith] {n}/{len(todo)}（{el / n:.1f}s/句，剩約 {el / n * (len(todo) - n) / 60:.0f} 分）",
                  flush=True)


def paired_ci(d, iters=2000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    vals = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(iters)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def importance_stats(recs, min_overlap: int) -> dict:
    """用逐句刪除的重要度檢驗引用：引用句是否比未引用句重要、最重要的句子是否被引用。"""
    diffs, hit, chance = [], [], []
    for r in recs:
        cited, _ = cited_turns(r["explanation"], r["history"], r["text"], min_overlap)
        imp = np.array([r["p_pred"] - x["p_orig"] for x in r["loo"]])
        others = [k for k in range(len(imp)) if k not in cited]
        if not cited or not others:
            continue
        diffs.append(imp[cited].mean() - imp[others].mean())
        hit.append(float(int(np.argmax(imp)) in cited))
        chance.append(len(cited) / len(imp))
    if not diffs:
        return {"n": 0}
    return {"n": len(diffs), "importance_diff": float(np.mean(diffs)),
            "importance_diff_ci": paired_ci(diffs),
            "top1_hit": float(np.mean(hit)), "top1_chance": float(np.mean(chance))}


def analyze(args):
    recs = [json.loads(l) for l in open(OUT_DIR / "records.jsonl", encoding="utf-8")]
    recs = [r for r in recs if "loo" in r]
    n = len(recs)
    cites = [cited_turns(r["explanation"], r["history"], r["text"], args.min_overlap)[1] for r in recs]
    methods = {m: cites.count(m) for m in ("quote", "overlap", "none")}
    print(f"[analyze] {n} 句；解釋引用前文：引號比對 {methods['quote']}、片段比對 {methods['overlap']}、"
          f"沒有指出可比對的前文 {methods['none']}（片段門檻 {args.min_overlap} 字）")

    def flip(r, k):
        return float(r["cf"][k]["pred"] != r["pred"])

    def drop(r, k):
        return r["p_pred"] - r["cf"][k]["p_orig"]

    summary = {"n": n, "min_overlap": args.min_overlap, "cite_methods": methods,
               "accuracy_of_explained_pred": float(np.mean([r["pred"] == r["gold"] for r in recs])),
               "all_context_removed": {"flip_rate": float(np.mean([flip(r, "all") for r in recs])),
                                       "prob_drop": float(np.mean([drop(r, "all") for r in recs]))}}
    paired = [r for r in recs if "cited" in r["cf"] and "random" in r["cf"]]
    if paired:
        fc = np.array([flip(r, "cited") for r in paired])
        fr = np.array([flip(r, "random") for r in paired])
        dc = np.array([drop(r, "cited") for r in paired])
        dr = np.array([drop(r, "random") for r in paired])
        summary["paired"] = {
            "n": len(paired),
            "flip_cited": float(fc.mean()), "flip_random": float(fr.mean()),
            "flip_diff": float((fc - fr).mean()), "flip_diff_ci": paired_ci(fc - fr),
            "drop_cited": float(dc.mean()), "drop_random": float(dr.mean()),
            "drop_diff": float((dc - dr).mean()), "drop_diff_ci": paired_ci(dc - dr),
        }
        s = summary["paired"]
        print(f"\n成對比較 {s['n']} 句（解釋有指出前文、且有足夠未引用前文可當對照）")
        print(f"  改判比例：刪引用 {s['flip_cited']:.3f}  vs  隨機刪 {s['flip_random']:.3f}  "
              f"差 {s['flip_diff']:+.3f}（95% CI {s['flip_diff_ci'][0]:+.3f} ~ {s['flip_diff_ci'][1]:+.3f}）")
        print(f"  原標籤機率下降：刪引用 {s['drop_cited']:.3f}  vs  隨機刪 {s['drop_random']:.3f}  "
              f"差 {s['drop_diff']:+.3f}（95% CI {s['drop_diff_ci'][0]:+.3f} ~ {s['drop_diff_ci'][1]:+.3f}）")
    imp = importance_stats(recs, args.min_overlap)
    summary["importance"] = imp
    if imp["n"]:
        print(f"\n逐句重要度（{imp['n']} 句）：引用句比未引用句的機率下降多 {imp['importance_diff']:+.4f}"
              f"（95% CI {imp['importance_diff_ci'][0]:+.4f} ~ {imp['importance_diff_ci'][1]:+.4f}）")
        print(f"  最重要的前文被解釋引用：{imp['top1_hit']:.3f}（隨機猜中機率 {imp['top1_chance']:.3f}）")
    a = summary["all_context_removed"]
    print(f"\n參考：刪掉全部前文 → 改判 {a['flip_rate']:.3f}，原標籤機率下降 {a['prob_drop']:.3f}")
    print(f"被解釋的判讀本身準確率：{summary['accuracy_of_explained_pred']:.3f}")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(f"[analyze] → {OUT_DIR / 'summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--history_turns", type=int, default=20)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--min_overlap", type=int, default=4, help="分析時：片段比對的最短共同字數")
    args = ap.parse_args()
    analyze(args) if args.analyze else run(args)


if __name__ == "__main__":
    main()
