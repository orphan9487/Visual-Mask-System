# -*- coding: utf-8 -*-
"""
資料集檢視器：把評估用的資料集「單獨拉出來看」，不載入任何模型。

用途：了解資料長什麼樣、情緒分布、對話結構，以及原始檔在哪。
這回答「訓練/評估用的 dataset 如何單獨拉出來檢視」。

重要觀念：本專案目前的推理層是 **training-free（零樣本提示）**，
這些資料集只用於「評估」(算 F1)，不用於「訓練」模型。

用法：
    # 看 MELD 的分布與前幾段對話
    .venv\\Scripts\\python -m src.evaluation.inspect_dataset --dataset meld --show 3

    # 只看某個情緒的例子
    .venv\\Scripts\\python -m src.evaluation.inspect_dataset --dataset meld --emotion disgust --show 8
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

# Windows 主控台預設 cp950，遇到 MELD 資料裡的 Windows-1252 字元（如彎引號 \x92）會崩潰。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.eval_erc import load_builtin, load_csv, load_meld
from src.reasoning.labels import CANONICAL_EMOTIONS, EMOTION_ZH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="meld", choices=["builtin", "csv", "meld"])
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--show", type=int, default=3, help="顯示幾段完整對話")
    ap.add_argument("--emotion", default=None, help="只挑含此情緒的句子來看")
    args = ap.parse_args()

    if args.dataset == "builtin":
        convs = load_builtin()
    elif args.dataset == "csv":
        convs = load_csv(args.data_path, "meld")
    else:
        convs = load_meld(args.split)

    n_utt = sum(len(c) for c in convs)
    labelled = [u for c in convs for u in c if u.get("emotion") in CANONICAL_EMOTIONS]

    # 1) 總覽
    print("\n" + "=" * 60)
    print(f"資料集：{args.dataset}（split={args.split}）")
    print(f"對話數：{len(convs)}　總句數：{n_utt}　可對應到7類標籤：{len(labelled)}")
    print("=" * 60)

    # 2) 情緒分布
    dist = Counter(u["emotion"] for u in labelled)
    print("\n[情緒分布]（依 canonical 7 類）")
    for emo in CANONICAL_EMOTIONS:
        c = dist.get(emo, 0)
        bar = "█" * int(c / max(dist.values()) * 30) if dist else ""
        print(f"  {emo:9s}({EMOTION_ZH[emo]}) {c:5d}  {c/len(labelled)*100:4.1f}%  {bar}")

    # 3) 對話樣例（保留語境，這正是 ERC 的重點）
    print(f"\n[對話樣例]（顯示 {args.show} 段，情緒標在句尾）")
    shown = 0
    for i, conv in enumerate(convs):
        if args.emotion and not any(u.get("emotion") == args.emotion for u in conv):
            continue
        print(f"\n--- 對話 #{i} ---")
        for u in conv:
            emo = u.get("emotion")
            tag = f"[{emo}/{EMOTION_ZH.get(emo,'?')}]" if emo in CANONICAL_EMOTIONS else "[—]"
            mark = " ←" if (args.emotion and emo == args.emotion) else ""
            print(f"  {u.get('speaker','?'):>10s}: {u['text']}  {tag}{mark}")
        shown += 1
        if shown >= args.show:
            break

    print("\n提示：把某段對話丟給推理層測試 →")
    print('  .venv\\Scripts\\python -m src.reasoning.try_erc --text "<某句>" --history "<前文||前文>"')


if __name__ == "__main__":
    main()
