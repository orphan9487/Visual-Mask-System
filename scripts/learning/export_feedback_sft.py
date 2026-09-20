# -*- coding: utf-8 -*-
"""Export MySQL emotion corrections as deduplicated ERC SFT JSONL.

This command only prepares research/training data. It does not train or load an
image LoRA, and it does not alter the FaceID generation path.

Usage:
    python -m scripts.learning.export_feedback_sft --out data/erc_sft/feedback.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from db.connection import get_connection
from db.feedback_repo import normalize_emotion_label, normalize_utterance
from src.reasoning.prompts import context_block, label_user_prompt, system_prompt


def fetch_corrections() -> list[tuple]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT username, context, wrong_label, correct_label "
                "FROM training_data ORDER BY correction_time DESC"
            )
            return list(cur.fetchall())
    finally:
        conn.close()


def to_sample(username: str, context_json: str, correct_label: str) -> dict | None:
    try:
        obj = json.loads(context_json) if context_json else {}
    except (TypeError, ValueError):
        return None
    text = str(obj.get("text") or "").strip()
    if not text:
        return None
    try:
        label = normalize_emotion_label(correct_label)
    except ValueError:
        return None
    history = obj.get("history") or []
    if not isinstance(history, list):
        history = []
    anonymous_user = hashlib.sha256(username.encode("utf-8")).hexdigest()[:12]
    return {
        "system": system_prompt(),
        "user": label_user_prompt(
            context_block(history, use_context=bool(history)), text
        ),
        "assistant": label,
        "metadata": {"user_id": anonymous_user, "source": "user_correction"},
    }


def export_rows(rows: list[tuple], out_path: Path) -> tuple[int, int]:
    """Write newest valid correction per user/utterance and return written/skipped."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    samples = []
    skipped = 0
    for username, context_json, wrong_label, correct_label in rows:
        try:
            context_obj = json.loads(context_json) if context_json else {}
        except (TypeError, ValueError):
            context_obj = {}
        key = (username, normalize_utterance(str(context_obj.get("text") or "")))
        if not key[1] or key in seen:
            skipped += 1
            continue
        seen.add(key)
        sample = to_sample(username, context_json, correct_label)
        if sample is None:
            skipped += 1
            continue
        try:
            wrong = normalize_emotion_label(wrong_label)
        except ValueError:
            wrong = None
        if wrong == sample["assistant"]:
            skipped += 1
            continue
        samples.append(sample)

    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return len(samples), skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/erc_sft/feedback.jsonl")
    args = parser.parse_args()
    rows = fetch_corrections()
    written, skipped = export_rows(rows, Path(args.out))
    print(f"[export] corrections={len(rows)} written={written} skipped={skipped} out={args.out}")


if __name__ == "__main__":
    main()
