# -*- coding: utf-8 -*-
"""
先驗校正（label-shift correction）：修正「保反諷配方把 neutral 降到 20%」造成的中性低估。

原理：模型在訓練分布 π_train 下學到 p(y|x)。若真實分布是 π_nat，貝氏校正為
    score'(y) = log p(y|x) + α · [log π_nat(y) − log π_train(y)]
α=1 是理論值；α 在驗證集上微調。π_train / π_nat 都只用訓練資料統計（降檔後 / 降檔前），
**完全不使用測試集資訊**。α 只在驗證集（M3ED，部署的 adapter 未訓練過、且已排除與測試劇
重疊的劇）上挑選；測試集只在最後評估一次。

用法：
    .venv\\Scripts\\python -m src.evaluation.calibrate_erc \\
        --dev evaluation_results/calib/scores_dev_m3ed.jsonl \\
        --test evaluation_results/calib/scores_test_cped.jsonl
"""

import argparse
import json
import sys
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

from src.reasoning.labels import CANONICAL_EMOTIONS as L

VALENCE = {"joy": "pos", "neutral": "neu", "surprise": "amb",
           "sadness": "neg", "anger": "neg", "fear": "neg", "disgust": "neg"}


def load_scores(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return np.array([r["scores"] for r in rows]), np.array([r["gold"] for r in rows])


def label_prior(jsonl):
    c = Counter(json.loads(l)["assistant"] for l in open(jsonl, encoding="utf-8"))
    n = sum(c.values())
    return np.array([c[k] / n for k in L])


def predict(S, bias):
    return np.array(L)[np.argmax(S + bias, axis=1)]


def metrics(gold, pred):
    return {
        "w_f1": f1_score(gold, pred, average="weighted", labels=L, zero_division=0),
        "macro": f1_score(gold, pred, average="macro", labels=L, zero_division=0),
        "acc": accuracy_score(gold, pred),
    }


def tune_per_class(S, gold, iters=3, grid=np.linspace(-3, 3, 61)):
    """參考用：每類各一個偏移，座標上升最大化驗證集 Macro-F1（易過擬合驗證集分布）。"""
    b = np.zeros(len(L))
    for _ in range(iters):
        for j in range(1, len(L)):              # 以 neutral 為基準固定 0
            best = (metrics(gold, predict(S, b))["macro"], b[j])
            for v in grid:
                t = b.copy()
                t[j] = v
                m = metrics(gold, predict(S, t))["macro"]
                if m > best[0]:
                    best = (m, v)
            b[j] = best[1]
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--train_capped", default="data/erc_sft/cped80series_sarcasm_train.jsonl")
    ap.add_argument("--train_natural", default="data/erc_sft/cped80series_train.jsonl")
    ap.add_argument("--out", default="evaluation_results/calib/calibration.json")
    args = ap.parse_args()

    pi_train, pi_nat = label_prior(args.train_capped), label_prior(args.train_natural)
    shift = np.log(pi_nat) - np.log(pi_train)
    print("校正方向 log(π_nat/π_train)：" + "  ".join(f"{k} {v:+.2f}" for k, v in zip(L, shift)))

    Sd, gd = load_scores(args.dev)
    St, gt = load_scores(args.test)

    # 1) 在驗證集上選 α
    print(f"\n[驗證集 M3ED，n={len(gd)}] α 掃描：")
    best_a, best_m = 0.0, -1
    for a in np.round(np.arange(0, 2.01, 0.25), 2):
        m = metrics(gd, predict(Sd, a * shift))
        flag = ""
        if m["macro"] > best_m:
            best_a, best_m, flag = a, m["macro"], " ←"
        print(f"  α={a:<4} W-F1 {m['w_f1']:.3f}  Macro {m['macro']:.3f}  Acc {m['acc']:.3f}{flag}")
    print(f"→ 選定 α={best_a}（驗證集 Macro 最高）")
    b_tuned = tune_per_class(Sd, gd)

    # 2) 測試集只評估一次
    variants = {"未校正 (α=0)": np.zeros(len(L)), "理論值 (α=1)": shift,
                f"驗證集選定 (α={best_a})": best_a * shift, "逐類調參（參考）": b_tuned}
    print(f"\n[測試集 CPED 6 部未見劇，n={len(gt)}]")
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(gt), len(gt)) for _ in range(1000)]
    base_pred = predict(St, variants["未校正 (α=0)"])
    report = {"alpha": best_a, "shift": dict(zip(L, shift.round(4).tolist())),
              "per_class_bias": dict(zip(L, b_tuned.round(3).tolist())), "test": {}}
    for name, b in variants.items():
        pred = predict(St, b)
        m = metrics(gt, pred)
        P, R, F, _ = precision_recall_fscore_support(gt, pred, labels=L, zero_division=0)
        val = np.mean([VALENCE[a] == VALENCE[b_] for a, b_ in zip(gt, pred)])
        line = f"  {name:16} W-F1 {m['w_f1']:.3f}  Macro {m['macro']:.3f}  Acc {m['acc']:.3f}  valence {val:.3f}"
        if name != "未校正 (α=0)":
            dw = [f1_score(gt[i], pred[i], average="weighted", labels=L, zero_division=0)
                  - f1_score(gt[i], base_pred[i], average="weighted", labels=L, zero_division=0)
                  for i in idx[:300]]
            dm = [f1_score(gt[i], pred[i], average="macro", labels=L, zero_division=0)
                  - f1_score(gt[i], base_pred[i], average="macro", labels=L, zero_division=0)
                  for i in idx[:300]]
            line += (f"\n{'':18}vs 未校正：ΔW-F1 [{np.percentile(dw, 2.5):+.3f}, {np.percentile(dw, 97.5):+.3f}]"
                     f"  ΔMacro [{np.percentile(dm, 2.5):+.3f}, {np.percentile(dm, 97.5):+.3f}]")
        print(line)
        dist = Counter(pred)
        print("    " + "  ".join(f"{k}: F1 {F[i]:.2f} R {R[i]:.2f} 預測{dist[k] / len(pred):.0%}"
                                  for i, k in enumerate(L)))
        report["test"][name] = {**{k: round(v, 4) for k, v in m.items()}, "valence_acc": round(val, 4),
                                "f1": dict(zip(L, F.round(4).tolist())),
                                "recall": dict(zip(L, R.round(4).tolist()))}
    gold_dist = Counter(gt)
    print("  實際分布：" + "  ".join(f"{k} {gold_dist[k] / len(gt):.0%}" for k in L))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] → {args.out}")


if __name__ == "__main__":
    main()
