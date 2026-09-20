"""
feedback_repo.py — 使用者反饋資料存取層

情緒反饋：
  save_emotion_feedback()        → 寫入 training_data，觸發 baseline 更新
  _update_user_emotion_baseline() → 依近期修正重建 users.emotion_prior 摘要

強度反饋：
  save_intensity_feedback()      → 寫入 intensity_adjustments
  get_intensity_multiplier()     → 計算歷史平均乘數（0.5–1.5）
"""
import json
import re
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

from db.connection import get_connection
from src.reasoning.labels import CANONICAL_EMOTIONS, coerce_prediction

# ── 情緒標籤正規化 ─────────────────────────────────────────────────────────────
# 與 visual_instruction_generator.py 的 EMOTION_MAP key 一致（子字串匹配）
_EMOTION_KEYS = [
    "calm", "bored", "neutral", "happy", "joyful", "sad",
    "disappoint", "passive", "withdraw", "sarcastic", "anxious",
    "confused", "frustrat", "stressed", "overwhelm", "angry",
    "disgusted", "fearful", "furious", "rage",
]


def _canonical(label: str) -> str:
    """將任意情緒字串正規化為最接近的 EMOTION_MAP key；找不到則回傳小寫原字。"""
    label_lower = label.lower().strip()
    for key in _EMOTION_KEYS:
        if key in label_lower:
            return key
    return label_lower


def normalize_emotion_label(label: str) -> str:
    """Normalize a feedback label into the ERC seven-label space."""
    normalized = coerce_prediction(label)
    if normalized not in CANONICAL_EMOTIONS:
        raise ValueError(
            f"Unsupported emotion label: {label!r}; expected one of "
            + ", ".join(CANONICAL_EMOTIONS)
        )
    return normalized


def normalize_utterance(text: str) -> str:
    """Create a stable comparison key without changing the stored utterance."""
    text = unicodedata.normalize("NFKC", text or "").casefold().strip()
    return re.sub(r"\s+", " ", text)


def get_emotion_correction_memory(
    username: str,
    utterance: str,
    *,
    scan_limit: int = 200,
    example_limit: int = 3,
    similarity_threshold: float = 0.72,
) -> dict:
    """Return an exact correction and nearby corrected examples for one user.

    ``training_data`` remains the single source of truth. Exact matches may be
    applied immediately; similar examples are prompt evidence only and never
    override the model by themselves.
    """
    target = normalize_utterance(utterance)
    if not username or not target:
        return {"exact_label": None, "examples": []}

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT context, correct_label FROM training_data
                   WHERE username = %s ORDER BY correction_time DESC LIMIT %s""",
                (username, max(1, min(int(scan_limit), 1000))),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    exact_label = None
    candidates: list[tuple[float, dict]] = []
    seen: set[str] = set()
    for context_json, raw_label in rows:
        try:
            obj = json.loads(context_json) if context_json else {}
        except (TypeError, ValueError):
            continue
        source_text = str(obj.get("text") or "").strip()
        key = normalize_utterance(source_text)
        if not key or key in seen:
            continue
        seen.add(key)
        try:
            label = normalize_emotion_label(str(raw_label))
        except ValueError:
            continue
        if key == target:
            exact_label = label
            break
        score = SequenceMatcher(None, target, key).ratio()
        if score >= similarity_threshold:
            candidates.append((score, {
                "text": source_text[:500],
                "correct_label": label,
                "similarity": round(score, 3),
            }))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return {
        "exact_label": exact_label,
        "examples": [item[1] for item in candidates[:max(0, example_limit)]],
    }


# ── 情緒反饋 ──────────────────────────────────────────────────────────────────

def save_emotion_feedback(
    username: str,
    context: str,
    wrong_label: str,
    correct_label: str,
) -> None:
    """
    儲存一筆情緒修正紀錄至 training_data，
    並呼叫 _update_user_emotion_baseline() 更新 users.emotion_prior。
    """
    wrong = normalize_emotion_label(wrong_label)
    correct = normalize_emotion_label(correct_label)
    if wrong == correct:
        raise ValueError("correct_label must differ from wrong_label")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO training_data
                   (username, context, wrong_label, correct_label, correction_time)
                   VALUES (%s, %s, %s, %s, NOW())""",
                (
                    username,
                    context,
                    wrong,
                    correct,
                ),
            )
        conn.commit()
    finally:
        conn.close()

    _update_user_emotion_baseline(username)


def _update_user_emotion_baseline(username: str) -> None:
    """
    讀取最近 10 筆 training_data，統計高頻修正對，
    重新生成情緒趨勢摘要並寫回 users.emotion_prior。
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT wrong_label, correct_label FROM training_data
                   WHERE username = %s ORDER BY correction_time DESC LIMIT 10""",
                (username,),
            )
            rows = cur.fetchall()

        if not rows:
            return

        pair_counts = Counter(rows)
        parts = [
            f"'{c}' (often misread as '{w}')"
            for (w, c), _ in pair_counts.most_common(3)
        ]
        summary = "Emotion tendency: frequently " + ", ".join(parts) + "."

        with conn.cursor() as cur:
            cur.execute(
                "SELECT emotion_prior FROM users WHERE username = %s", (username,)
            )
            row = cur.fetchone()
            base = row[0] if row else ""

            # 剔除前次自動生成的摘要段落，保留使用者手寫的基礎描述
            base_parts = [
                p.strip()
                for p in base.split(".")
                if p.strip() and not p.strip().startswith("Emotion tendency:")
            ]
            base_clean = ". ".join(base_parts)
            new_prior = f"{base_clean}. {summary}" if base_clean else summary

            cur.execute(
                "UPDATE users SET emotion_prior = %s WHERE username = %s",
                (new_prior, username),
            )
        conn.commit()
    finally:
        conn.close()


# ── 強度反饋 ──────────────────────────────────────────────────────────────────

_MULTIPLIER_MIN = 0.50   # 強度下限乘數
_MULTIPLIER_MAX = 1.50   # 強度上限乘數（使用者覺得不夠強時最高拉到 1.5）
_MIN_SAMPLES    = 3      # 至少需要 N 筆歷史才啟用乘數調整


def save_intensity_feedback(
    username: str,
    emotion: str,
    adjusted_intensity: float,
    base_intensity: float,
) -> None:
    """儲存一筆使用者手動強度調整至 intensity_adjustments。"""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO intensity_adjustments
                   (username, emotion, adjusted_intensity, base_intensity, timestamp)
                   VALUES (%s, %s, %s, %s, NOW())""",
                (
                    username,
                    _canonical(emotion),
                    round(float(adjusted_intensity), 4),
                    round(float(base_intensity), 4),
                ),
            )
        conn.commit()
    finally:
        conn.close()


def get_intensity_multiplier(username: str, emotion: str) -> float:
    """
    從 intensity_adjustments 計算該使用者對此情緒的平均強度偏好乘數。

    演算法：
      1. 取最近 5 筆 (adjusted / base) 比值
      2. 計算平均比值
      3. 夾在 [_MULTIPLIER_MIN, _MULTIPLIER_MAX] 後回傳
      4. 若歷史不足 _MIN_SAMPLES 筆，直接回傳 1.0（不調整）

    範例：使用者三次把 angry(base=0.90) 往上拉至 0.90→1.0，
          avg_ratio ≈ 1.11，系統下次自動將 0.90×1.11 ≈ 1.0 輸出。
          若一直拉到 1.0（ratio=1.11）*3 次以上 → 乘數穩定在 1.11；
          若拉到底（ratio=1.5）→ 乘數上限夾在 1.5。
    """
    emotion_key = _canonical(emotion)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT adjusted_intensity, base_intensity
                   FROM intensity_adjustments
                   WHERE username = %s AND emotion = %s
                   ORDER BY timestamp DESC LIMIT 5""",
                (username, emotion_key),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if len(rows) < _MIN_SAMPLES:
        return 1.0

    ratios = [adj / base for adj, base in rows if base > 0]
    if not ratios:
        return 1.0

    avg_ratio = sum(ratios) / len(ratios)
    return round(max(_MULTIPLIER_MIN, min(_MULTIPLIER_MAX, avg_ratio)), 3)
