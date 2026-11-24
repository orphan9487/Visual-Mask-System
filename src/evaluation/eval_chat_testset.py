# -*- coding: utf-8 -*-
"""
在真實台灣聊天測試集上評估部署中的 ERC 模型，並與人類一致度並排比較。

判讀方式與上線完全相同：單次打分（LabelScorer）＋先驗校正（configs/erc_calibration.json）
＋前 20 句語境。另列出未校正版本供對照。

報告內容：
  - 對 gold：W-F1 / Macro-F1 / Acc（bootstrap 95% CI）、多數類基線、各類 P/R/F1、混淆矩陣
  - 反諷子集準確率
  - 模型 vs 各標註者的 Cohen's κ，對照標註者彼此的 κ（＝人類天花板）
  - 與 CPED 測試集結果（W-F1 0.454）比較，量化領域落差

模型分數會快取到 --scores；重跑分析（例如補完裁決後）不必再載入模型。

用法：
    .venv\\Scripts\\python -m src.evaluation.eval_chat_testset \\
        --items data/chat_testset/items.jsonl --gold data/chat_testset/gold.jsonl \\
        --annotators data/chat_testset/labels_a.json data/chat_testset/labels_b.json
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from src.annotation.chat_testset import (bootstrap_ci, cohen_kappa, confusion, expand_paths,
                                         history_for,
                                         interpret_kappa, load_items, load_labels,
                                         print_confusion)
from src.reasoning.labels import CANONICAL_EMOTIONS as L

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CPED_REFERENCE = {"w_f1": 0.454, "macro": 0.373, "acc": 0.469}   # CPED 6 部未見劇、上線設定


def score_items(items, wanted, lora, history_turns, cache_path) -> dict[str, list[float]]:
    """回傳 item id → 7 類對數機率，只算 wanted 裡的題目（語境仍取整段對話）。
    已快取的題目不重算（新增題目時只算新的）。"""
    cache = {}
    if cache_path and Path(cache_path).exists():
        cache = {r["id"]: r["scores"] for r in load_items(cache_path)}
    todo = [it for it in items if it["id"] in wanted and it["id"] not in cache]
    if not todo:
        return cache

    import torch
    from src.reasoning.backbone import Backbone, BackboneConfig
    from src.reasoning.label_scorer import LabelScorer
    from src.reasoning.prompts import context_block, label_user_prompt, system_prompt

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.90)
    bk = Backbone(BackboneConfig(model="breeze2-3b", lora_path=lora))
    scorer = LabelScorer(bk)
    sys_p = system_prompt()
    hist = history_for(items)
    t0 = time.time()
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        for n, it in enumerate(todo, 1):
            recent = hist[it["id"]][-history_turns:] if history_turns > 0 else []
            ctx = context_block(recent, use_context=history_turns > 0)
            cache[it["id"]] = scorer.score(sys_p, label_user_prompt(ctx, it["text"]))
            f.write(json.dumps({"id": it["id"], "scores": cache[it["id"]]}) + "\n")
            if n % 100 == 0:
                print(f"[score] {n}/{len(todo)}（{(time.time() - t0) / n:.2f}s/句）", flush=True)
    return cache


# 反諷的關鍵是「方向」：真心開心與反話的開心字面一樣，判對方向才算看懂語境
VALENCE = {"joy": "正面", "sadness": "負面", "anger": "負面", "fear": "負面", "disgust": "負面",
           "neutral": "中性", "surprise": "驚訝"}


def sarcasm_breakdown(gold, sarc, pred) -> dict:
    """反諷子集與非反諷子集並排：準確率（含 CI）、正負方向正確率、
    被字面騙成「開心」的比例（gold 不是 joy 卻判 joy），以及被判成中性的比例。"""
    gold, sarc, pred = np.array(gold), np.array(sarc, dtype=bool), np.array(pred)
    out = {}
    for name, mask in (("反諷", sarc), ("非反諷", ~sarc)):
        g, p = gold[mask], pred[mask]
        if len(g) == 0:
            continue
        direction = np.array([VALENCE[a] == VALENCE[b] for a, b in zip(g, p)])
        not_joy = g != "joy"
        out[name] = {
            "n": int(len(g)),
            "acc": float(np.mean(g == p)),
            "acc_ci": bootstrap_ci(lambda idx: float(np.mean(g[idx] == p[idx])), len(g)),
            "direction_acc": float(direction.mean()),
            "literal_joy_rate": float(np.mean(p[not_joy] == "joy")) if not_joy.any() else float("nan"),
            "neutral_rate": float(np.mean(p == "neutral")),
            "gold_dist": dict(Counter(g.tolist())),
        }
    return out


def flag_kappa(a: list[bool], b: list[bool]) -> float:
    """兩位標註者對「是否反諷」勾選的 Cohen's κ（二元）。"""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def default_paths(items_path: str) -> tuple[str, str]:
    """分數快取與報告依測試集檔名命名：聊天與論壇題號格式相同，共用快取會互相污染。"""
    p = Path(items_path)
    return (str(p.with_name(f"{p.stem}_model_scores.jsonl")),
            str(p.with_name(f"{p.stem}_eval_report.json")))


def metrics(gold, pred) -> dict:
    # Macro 只平均 gold 裡出現過的類別：小測試集常缺某些類（如 fear），算進去會被當 0 拉低
    present = [k for k in L if k in set(gold)]
    return {"w_f1": f1_score(gold, pred, average="weighted", labels=L, zero_division=0),
            "macro": f1_score(gold, pred, average="macro", labels=present, zero_division=0),
            "acc": accuracy_score(gold, pred)}


def metrics_with_ci(gold, pred) -> dict:
    gold, pred = np.array(gold), np.array(pred)
    out = metrics(gold, pred)
    for key in ("w_f1", "macro", "acc"):
        out[key + "_ci"] = bootstrap_ci(lambda idx: metrics(gold[idx], pred[idx])[key], len(gold))
    return out


def fmt(m) -> str:
    return (f"W-F1 {m['w_f1']:.3f} [{m['w_f1_ci'][0]:.3f}–{m['w_f1_ci'][1]:.3f}]  "
            f"Macro {m['macro']:.3f} [{m['macro_ci'][0]:.3f}–{m['macro_ci'][1]:.3f}]  "
            f"Acc {m['acc']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="data/chat_testset/items.jsonl")
    ap.add_argument("--gold", default="data/chat_testset/gold.jsonl")
    ap.add_argument("--annotators", nargs="*", default=[], help="各標註者的 labels_*.json")
    ap.add_argument("--calib", default=str(PROJECT_ROOT / "configs/erc_calibration.json"))
    ap.add_argument("--lora", default=None, help="預設＝校正檔記錄的 adapter（即上線版本）")
    ap.add_argument("--alpha", type=float, default=None, help="預設＝校正檔的 alpha")
    ap.add_argument("--history_turns", type=int, default=20)
    ap.add_argument("--scores", default=None,
                    help="模型分數快取；預設依 --items 檔名命名（聊天與論壇的題號格式相同，不可共用）")
    ap.add_argument("--out", default=None, help="預設依 --items 檔名命名")
    args = ap.parse_args()
    scores_default, out_default = default_paths(args.items)
    args.scores = args.scores or scores_default
    args.out = args.out or out_default

    cal = json.loads(Path(args.calib).read_text(encoding="utf-8"))
    lora = args.lora or str(PROJECT_ROOT / "models" / cal["adapter"])
    alpha = cal["alpha"] if args.alpha is None else args.alpha
    bias = np.array([alpha * cal["shift"][k] for k in L])
    if Path(lora).name != cal["adapter"]:
        print(f"[warn] 校正參數適用 {cal['adapter']}，評估的是 {Path(lora).name} → 不套校正")
        bias = np.zeros(len(L))

    items = load_items(args.items)
    gold = {g["id"]: g for g in load_items(args.gold)} if Path(args.gold).exists() else {}
    named = [load_labels(p) for p in expand_paths(args.annotators)]
    wanted = set(gold) | set().union(*(set(l) for _, l in named))
    if not wanted:
        sys.exit("[eval] 沒有任何標註（gold 或標註者檔），無從評估")
    scores = score_items(items, wanted, lora, args.history_turns, args.scores)
    scores = {k: v for k, v in scores.items() if k in wanted}

    def predict(ids, use_bias=True):
        S = np.array([scores[i] for i in ids])
        return list(np.array(L)[np.argmax(S + (bias if use_bias else 0), axis=1)])

    report = {"adapter": Path(lora).name, "alpha": alpha, "history_turns": args.history_turns}
    if gold:
        ids = sorted(gold)
        y = [gold[i]["label"] for i in ids]
        print(f"\n=== 對 gold（{len(ids)} 則，{dict(Counter(y))}）===")
        maj = Counter(y).most_common(1)[0][0]
        base = metrics_with_ci(y, [maj] * len(y))
        raw = metrics_with_ci(y, predict(ids, use_bias=False))
        cal_pred = predict(ids)
        calm = metrics_with_ci(y, cal_pred)
        print(f"  多數類基線（全猜 {maj}）：{fmt(base)}")
        print(f"  模型（未校正）          ：{fmt(raw)}")
        print(f"  模型（上線設定，α={alpha}）：{fmt(calm)}")
        print(f"  對照 CPED 未見劇（上線設定）：W-F1 {CPED_REFERENCE['w_f1']:.3f}  "
              f"Macro {CPED_REFERENCE['macro']:.3f}  Acc {CPED_REFERENCE['acc']:.3f}")
        p, r, f, s = precision_recall_fscore_support(y, cal_pred, labels=L, zero_division=0)
        print("\n  各類（上線設定）     P      R      F1   支持數")
        for k, pi, ri, fi, si in zip(L, p, r, f, s):
            print(f"    {k:<10}{pi:>7.2f}{ri:>7.2f}{fi:>7.2f}{si:>7}")
        print_confusion(confusion(y, cal_pred), "gold", "模型")
        flags = [bool(gold[i].get("sarcasm")) for i in ids]
        if any(flags):
            print("\n=== 反諷子集 vs 非反諷（上線設定）===")
            report["sarcasm"] = {}
            for tag, pr in (("上線設定", cal_pred), ("未校正", predict(ids, use_bias=False))):
                rep = sarcasm_breakdown(y, flags, pr)
                report["sarcasm"][tag] = rep
                if tag != "上線設定":
                    continue
                for name, sub in rep.items():
                    print(f"  {name:<4} n={sub['n']:<4} 準確率 {sub['acc']:.3f} "
                          f"[{sub['acc_ci'][0]:.3f}–{sub['acc_ci'][1]:.3f}]  方向正確 {sub['direction_acc']:.3f}  "
                          f"被字面騙成開心 {sub['literal_joy_rate']:.3f}  判成中性 {sub['neutral_rate']:.3f}")
            print(f"  未校正的反諷準確率："
                  f"{report['sarcasm']['未校正'].get('反諷', {}).get('acc', float('nan')):.3f}")
        report.update({"n_gold": len(ids), "majority": base, "raw": raw, "calibrated": calm,
                       "per_class": {k: {"p": float(pi), "r": float(ri), "f1": float(fi),
                                         "support": int(si)}
                                     for k, pi, ri, fi, si in zip(L, p, r, f, s)}})

    if named:
        print("\n=== 人類天花板：模型 vs 標註者 的 Cohen's κ ===")
        rows = []
        for i in range(len(named)):
            for j in range(i + 1, len(named)):
                (na, la), (nb, lb) = named[i], named[j]
                common = sorted(set(la) & set(lb))
                k = cohen_kappa([la[c]["label"] for c in common], [lb[c]["label"] for c in common])
                rows.append({"pair": f"{na} vs {nb}", "n": len(common), "kappa": k})
        for name, lab in named:
            ids = sorted(lab)
            k = cohen_kappa([lab[c]["label"] for c in ids], predict(ids))
            rows.append({"pair": f"模型 vs {name}", "n": len(ids), "kappa": k})
        for r in rows:
            print(f"  {r['pair']:<24} n={r['n']:<5} κ = {r['kappa']:.3f}（{interpret_kappa(r['kappa'])}）")
        report["kappa"] = rows
        # 人對「這句是不是反諷」本身的一致度：一致度低時，反諷準確率也要保守解讀
        sarc_rows = []
        for i in range(len(named)):
            for j in range(i + 1, len(named)):
                (na, la), (nb, lb) = named[i], named[j]
                common = sorted(set(la) & set(lb))
                fa = [bool(la[c].get("sarcasm")) for c in common]
                fb = [bool(lb[c].get("sarcasm")) for c in common]
                k = flag_kappa(fa, fb)
                sarc_rows.append({"pair": f"{na} vs {nb}", "n": len(common), "kappa": k,
                                  "flagged": [sum(fa), sum(fb)]})
                print(f"  反諷勾選 {na} vs {nb}：各勾 {sum(fa)}／{sum(fb)} 則，"
                      f"κ = {k:.3f}（{interpret_kappa(k)}）")
        report["sarcasm_flag_kappa"] = sarc_rows

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=float),
                              encoding="utf-8")
    print(f"\n[eval] 報告 → {args.out}")


if __name__ == "__main__":
    main()
