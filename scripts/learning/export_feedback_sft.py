# -*- coding: utf-8 -*-
"""把使用者 👎 更正（MySQL training_data）匯出成 ERC SFT 樣本。

離線重訓迴圈的第一步。輸出格式與 src/training/format_meld.py 完全相同
（system / user / assistant），可直接餵給 src.training.train_erc_lora。

training_data.context 存的是 {"history": [...], "text": "被判錯的那句"}，
搭配 correct_label 即可還原成一筆「語境＋當前句 → 正確情緒」的樣本。

用法：
    python -m scripts.learning.export_feedback_sft --out data/erc_sft/feedback.jsonl

之後把 feedback.jsonl 與底層資料（如 data/erc_sft/cped_train.jsonl）合併，
再用 train_erc_lora 重訓，避免只用少量更正資料造成過擬合／災難性遺忘。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from db.connection import get_connection
from src.reasoning.labels import CANONICAL_EMOTIONS
from src.reasoning.prompts import context_block, label_user_prompt, system_prompt


def fetch_corrections() -> list[tuple]:
    """讀所有更正紀錄（context_json, wrong_label, correct_label）。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT context, wrong_label, correct_label "
                "FROM training_data ORDER BY correction_time"
            )
            return list(cur.fetchall())
    finally:
        conn.close()


def to_sample(context_json: str, correct: str) -> dict | None:
    """把一筆更正還原成 SFT 樣本；資料不足回 None。"""
    try:
        obj = json.loads(context_json) if context_json else {}
    except (ValueError, TypeError):
        obj = {}
    text = (obj.get("text") or "").strip()
    if not text:
        return None
    history = obj.get("history") or []
    ctx = context_block(history, use_context=bool(history))
    return {
        "system": system_prompt(),
        "user": label_user_prompt(ctx, text),
        "assistant": correct,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/erc_sft/feedback.jsonl")
    args = ap.parse_args()

    rows = fetch_corrections()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_written, skipped_same, skipped_bad = 0, 0, 0
    with open(out, "w", encoding="utf-8") as f:
        for context_json, wrong, correct in rows:
            if not correct or correct not in CANONICAL_EMOTIONS:
                skipped_bad += 1
                continue
            if correct == wrong:            # 沒有真的更正（讚同/未變）
                skipped_same += 1
                continue
            sample = to_sample(context_json, correct)
            if sample is None:
                skipped_bad += 1
                continue
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[export] training_data {len(rows)} 筆 → 有效更正 SFT {n_written} 筆 → {out}")
    print(f"[export] 略過：無更正 {skipped_same}、資料不足/標籤異常 {skipped_bad}")
    if n_written:
        from collections import Counter
        with open(out, encoding="utf-8") as f:
            dist = Counter(json.loads(l)["assistant"] for l in f if l.strip())
        print(f"[export] 更正標籤分布：{dict(dist)}")


if __name__ == "__main__":
    main()
