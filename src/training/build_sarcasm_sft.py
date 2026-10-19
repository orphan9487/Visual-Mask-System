# -*- coding: utf-8 -*-
"""
「保反諷」資料配方：把手工撰寫的台灣口語反諷/語境翻轉範例，與（中性降檔的）
CPED 訓練樣本合併，產出新的單次格式 SFT jsonl。

動機：純 CPED 單次微調會把模型「態化」——反諷（真好啊→disgust）塌成 neutral/joy。
CPED 逐句表層標註本就不獎勵反諷推理，且中性佔 37% 造成拉平。本配方：
  (1) 中性降檔（cap），降低拉平壓力；
  (2) 注入手工反諷範例並 oversample，直接教「看語境不看字面」；
      關鍵是**對比對**（同一句話不同語境→不同情緒）與**真心版對照**，
      避免模型學成「反諷詞→一律負面」。

用法：
    .venv\\Scripts\\python -m src.training.build_sarcasm_sft \\
        --cped data/erc_sft/cped_train.jsonl --neutral_cap 5000 \\
        --oversample 15 --out data/erc_sft/cped_sarcasm_train.jsonl
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

from src.reasoning.prompts import context_block, label_user_prompt, system_prompt

# (history[list[(role, content)]], utterance, gold_label)
# 每個反諷案例盡量配一個「真心/中性對照」，讓模型學語境相依而非字面。
_H = lambda *turns: [{"role": r, "content": c} for r, c in turns]

SARCASM_EXEMPLARS = [
    # ---- 假讚美：對方做了壞事 → 表面稱讚，實為 anger/disgust ----
    (_H(("對方", "我不小心把你的報告檔刪掉了")), "真是太好了", "anger"),
    (_H(("對方", "我把你整個資料夾都刪光了")), "真好啊，謝謝你喔", "disgust"),
    (_H(("對方", "我剛剛把你昨天做的簡報覆蓋掉了")), "哇，你真的幫了大忙呢", "disgust"),
    (_H(("對方", "我又忘記幫你帶那份文件了")), "你真的很可靠耶", "disgust"),
    (_H(("對方", "這個月的錢我又花光了")), "你理財能力真好", "disgust"),
    (_H(("對方", "我把咖啡打翻在你的鍵盤上了")), "太棒了，真有你的", "anger"),
    (_H(("同事", "報告我還沒開始寫，明天要交")), "哇你效率真高", "disgust"),
    (_H(("對方", "我把你車子的鑰匙弄丟了")), "你真是太厲害了", "anger"),
    # ---- 真心讚美對照：對方做了好事 → 同樣的稱讚，實為 joy ----
    (_H(("對方", "我提早幫你把報告都寫完了")), "真是太好了", "joy"),
    (_H(("對方", "我幫你把資料夾都整理好備份了")), "真好啊，謝謝你喔", "joy"),
    (_H(("對方", "我們這次專案拿到第一名了")), "太棒了，真有你的", "joy"),
    (_H(("對方", "我幫你把明天的簡報都做好了")), "哇，你真的幫了大忙呢", "joy"),
    (_H(("對方", "我這個月存了不少錢")), "你理財能力真好", "joy"),
    (_H(("同事", "報告我昨天就提前交出去了")), "哇你效率真高", "joy"),
    # ---- 諷刺式感謝 → disgust ----
    (_H(("對方", "我把你的祕密跟大家講了")), "真是謝謝你喔", "disgust"),
    (_H(("對方", "我擅自幫你答應了那件事")), "感謝你的幫忙呢", "disgust"),
    (_H(("對方", "我把你要的顏色全部買錯了")), "謝謝你這麼用心", "disgust"),
    (_H(("對方", "我遲到害我們錯過火車了")), "真是多虧了你", "anger"),
    # ---- 諷刺式讚嘆 / 假驚嘆 → disgust ----
    (_H(("對方", "我第三次遲到了")), "哇你真準時耶", "disgust"),
    (_H(("對方", "我這次又忘記你的生日了")), "你記性真好", "disgust"),
    (_H(("對方", "我把音樂開超大聲吵到半夜")), "你真體貼", "disgust"),
    (_H(("對方", "我插隊插到你前面了")), "你真有禮貌", "disgust"),
    # ---- 口是心非：被關心 → 否認，實為 sadness ----
    (_H(("對方", "你還好嗎？剛剛看你在哭")), "沒事啊", "sadness"),
    (_H(("對方", "你看起來很累，怎麼了")), "我很好，別擔心", "sadness"),
    (_H(("對方", "你今天怎麼都不說話")), "沒有啊，我沒事", "sadness"),
    (_H(("對方", "聽說你被當掉了，還好嗎")), "沒關係啦，無所謂", "sadness"),
    (_H(("對方", "你跟他分手了對不對")), "我早就不在意他了", "sadness"),
    (_H(("對方", "你需要我陪你嗎")), "不用了，我一個人可以", "sadness"),
    # ---- 口是心非對照：真的沒事 → neutral ----
    (_H(("對方", "你等下要吃什麼")), "都可以啊，沒差", "neutral"),
    (_H(("對方", "這件事你有意見嗎")), "沒有啊，我沒意見", "neutral"),
    (_H(("對方", "會議改到三點可以嗎")), "可以啊，沒問題", "neutral"),
    # ---- 被激怒的反話 → anger ----
    (_H(("對方", "這麼簡單你也不會，是不是很笨")), "對啦我最笨了", "anger"),
    (_H(("對方", "都是你害的，你知道嗎")), "好啦好啦都我的錯", "anger"),
    (_H(("對方", "你就不能做好一件事嗎")), "是喔，那你來做啊", "anger"),
    (_H(("對方", "我早就跟你說過了吧")), "對，你最厲害，你都對", "anger"),
    (_H(("對方", "你這樣做根本沒用")), "隨便你怎麼說", "anger"),
    # ---- 被動攻擊 / 冷嘲 → disgust ----
    (_H(("對方", "我覺得我的想法比較好")), "是喔，那真是恭喜你了", "disgust"),
    (_H(("對方", "我升遷了，靠的是實力")), "嗯，實力，我懂", "disgust"),
    (_H(("對方", "你怎麼還在用這種舊方法")), "抱歉喔沒你那麼聰明", "disgust"),
    (_H(("對方", "反正你也做不到啦")), "呵呵，是喔", "disgust"),
    # ---- 假熱情 → disgust ----
    (_H(("對方", "這週末又要加班喔")), "哇好期待喔", "disgust"),
    (_H(("對方", "又要開三小時的會了")), "太好了我最愛開會了", "disgust"),
    (_H(("對方", "這個月要再繳一筆錢")), "喔耶又要花錢了", "disgust"),
    # ---- 真心期待對照 → joy ----
    (_H(("對方", "這週末我們去墾丁玩吧")), "哇好期待喔", "joy"),
    (_H(("對方", "下週開始放連假囉")), "太好了，我最愛放假了", "joy"),
    # ---- 語境使中性字面轉負面 → anger / disgust ----
    (_H(("對方", "所以你到底要不要去")), "隨便", "anger"),
    (_H(("對方", "我已經道歉了你還要怎樣")), "沒怎樣啊", "anger"),
    (_H(("對方", "你是不是在生氣")), "我哪有生氣", "anger"),
    (_H(("對方", "這樣安排你ok嗎")), "很好啊，非常好", "disgust"),
    # ---- 真心正面短語 → joy（避免把短肯定學成負面）----
    (_H(("對方", "我幫你多準備了一份")), "太好了謝謝你", "joy"),
    (_H(("對方", "考試通過了喔")), "真的假的太棒了", "joy"),
    (_H(("對方", "我請你喝手搖")), "耶謝謝你人真好", "joy"),
    # ---- 真心中性 → neutral ----
    (_H(("對方", "資料我放桌上了")), "好，我知道了", "neutral"),
    (_H(("對方", "等下記得關燈")), "嗯好", "neutral"),
    (_H(("對方", "明天幾點集合")), "九點吧應該", "neutral"),
    # ---- 難過直述（非反諷，補難過樣本）→ sadness ----
    (_H(("對方", "結果還是沒選上嗎")), "嗯…我盡力了", "sadness"),
    (_H(("對方", "牠昨天晚上走了")), "我還沒辦法接受", "sadness"),
    # ---- 生氣直述 → anger ----
    (_H(("對方", "我就是故意的，不行喔")), "你到底在想什麼", "anger"),
    (_H(("對方", "這種事我才不管")), "你太過分了吧", "anger"),
]


def exemplar_samples():
    out = []
    for hist, utt, label in SARCASM_EXEMPLARS:
        out.append({
            "system": system_prompt(),
            "user": label_user_prompt(context_block(hist, use_context=True), utt),
            "assistant": label,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cped", default="data/erc_sft/cped_train.jsonl")
    ap.add_argument("--neutral_cap", type=int, default=5000,
                    help="CPED 中性樣本上限（降檔以減少拉平；0=不限）")
    ap.add_argument("--oversample", type=int, default=15,
                    help="反諷範例重複倍數")
    ap.add_argument("--out", default="data/erc_sft/cped_sarcasm_train.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    with open(args.cped, encoding="utf-8") as f:
        cped = [json.loads(l) for l in f if l.strip()]

    # 中性降檔
    if args.neutral_cap:
        neutral = [s for s in cped if s["assistant"] == "neutral"]
        others = [s for s in cped if s["assistant"] != "neutral"]
        random.shuffle(neutral)
        cped = others + neutral[:args.neutral_cap]

    exemplars = exemplar_samples() * args.oversample
    combined = cped + exemplars
    random.shuffle(combined)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in combined:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    dist = Counter(s["assistant"] for s in combined)
    print(f"[build] CPED(降檔後) {len(cped)} + 反諷 {len(exemplars)}"
          f"（{len(SARCASM_EXEMPLARS)}唯一×{args.oversample}）= {len(combined)} 筆 → {args.out}")
    print(f"[build] 標籤分布：{dict(dist)}")


if __name__ == "__main__":
    main()
