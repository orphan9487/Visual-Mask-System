# -*- coding: utf-8 -*-
"""
繁中對話情緒標註 Schema（產品 benchmark 用）——資料結構 + 驗證 + 距離自動算。

對應 docs/標註schema_草案v0.md。這是「資料護城河」的地基：定義每一則訊息要
標什麼、怎麼存成 JSONL、以及哪些欄位是機器可自動檢核的（valence↔emotion 一致性、
成因距離、distant_cause 自動打標）。

刻意與 src/reasoning/labels.py（競賽 ERC 引擎的 7 類 canonical）**分開**：
產品 benchmark 的細層情緒是另一套（砍 disgust→併 contempt、fear→更名 anxiety、
新增 contempt/resignation）。兩者不可混用，否則會污染承重牆主張
（「通用強模型在繁中真實對話上系統性錯」的量化證據建在這套標籤上）。

承重牆禁忌（寫在這裡提醒每個用到本模組的人）：
  模型可初標，但 gold 的最後一關必須是人逐條改。本模組刻意只提供 `_draft`
  的容器與驗證，不提供「自動把 draft 當 gold」的捷徑——見 pre_annotate.py。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Optional

# --------------------------------------------------------------------------- #
# 1. 標籤空間（草案 v0 第 2–5 節）
# --------------------------------------------------------------------------- #

# Tier 1 — Valence（粗層，承重牆）。必標。
VALENCE = ["positive", "negative", "neutral", "ambiguous"]

# Tier 2 — 細層情緒（單標必填，不設次標）。ambiguous 為逃生口。
EMOTION = [
    "neutral",      # 中性
    "joy",          # 開心          → positive
    "anger",        # 生氣（對事、想要你改、熱）→ negative
    "contempt",     # 不屑嘲諷（對人、居高臨下、冷）→ negative
    "sadness",      # 難過          → negative
    "anxiety",      # 焦慮擔心      → negative
    "surprise",     # 驚訝          → 依情境（valence 不固定）
    "resignation",  # 無奈（認命、擺爛；≠難過）→ negative
    "ambiguous",    # 曖昧（逃生口）→ valence 不固定
]

EMOTION_ZH = {
    "neutral": "中性", "joy": "開心", "anger": "生氣", "contempt": "不屑嘲諷",
    "sadness": "難過", "anxiety": "焦慮擔心", "surprise": "驚訝",
    "resignation": "無奈", "ambiguous": "曖昧",
}

# 細層情緒 → 應有的 valence（供一致性驗證）。
# surprise（依情境）與 ambiguous（逃生口）刻意不列入 → valence 自由，不強制。
EMOTION_VALENCE = {
    "neutral": "neutral",
    "joy": "positive",
    "anger": "negative",
    "contempt": "negative",
    "sadness": "negative",
    "anxiety": "negative",
    "resignation": "negative",
}

# 難度標籤（多選，護城河的量化指標）。草案 v0 第 4 節。
DIFFICULTY_TAGS = ["sarcasm", "code_switch", "net_slang", "distant_cause"]

# 成因來源（草案 v0 第 5 節）。
CAUSE_SOURCE = ["in_conversation", "external", "unclear"]

# distance ≥ 此值 → 自動打 distant_cause 標籤（成因在 ≥2 則之前）。
DISTANT_CAUSE_MIN_DISTANCE = 2


# --------------------------------------------------------------------------- #
# 2. 資料結構
# --------------------------------------------------------------------------- #


@dataclass
class Cause:
    """成因（TECPE 簡化版）。只對非中性訊息標；中性訊息 cause = None。"""
    source: str                       # CAUSE_SOURCE 之一
    cause_uid: Optional[int] = None   # source=in_conversation 時＝「第幾則」（單一最直接因）
    distance: Optional[int] = None    # 自動算 = 當前 uid − cause_uid；不手填
    note: Optional[str] = None        # 成因在對話外時的簡述（去識別化）

    def to_dict(self) -> dict:
        d = {"source": self.source}
        if self.cause_uid is not None:
            d["cause_uid"] = self.cause_uid
        if self.distance is not None:
            d["distance"] = self.distance
        if self.note:
            d["note"] = self.note
        return d

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> Optional["Cause"]:
        if d is None:
            return None
        return cls(
            source=d.get("source"),
            cause_uid=d.get("cause_uid"),
            distance=d.get("distance"),
            note=d.get("note"),
        )


@dataclass
class Annotation:
    """單一判讀（可為某位標註者的、或裁決後的 gold）。"""
    valence: str
    emotion_primary: str
    difficulty_tags: list[str] = field(default_factory=list)
    cause: Optional[Cause] = None

    def to_dict(self) -> dict:
        d = {
            "valence": self.valence,
            "emotion_primary": self.emotion_primary,
            "difficulty_tags": list(self.difficulty_tags),
        }
        d["cause"] = self.cause.to_dict() if self.cause else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        return cls(
            valence=d.get("valence"),
            emotion_primary=d.get("emotion_primary"),
            difficulty_tags=list(d.get("difficulty_tags") or []),
            cause=Cause.from_dict(d.get("cause")),
        )


@dataclass
class Utterance:
    uid: int
    speaker: str      # 匿名代號（S1/S2…）
    text: str         # 去識別化後的內容

    def to_dict(self) -> dict:
        return {"uid": self.uid, "speaker": self.speaker, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> "Utterance":
        return cls(uid=d["uid"], speaker=d.get("speaker", ""), text=d.get("text", ""))


@dataclass
class ConvMeta:
    source: str = "unknown"          # line / discord / …
    consent: bool = False            # 是否取得每位參與者同意
    deidentified: bool = False       # 是否已去識別化
    n_utterances: Optional[int] = None
    speakers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ConvMeta":
        d = d or {}
        return cls(
            source=d.get("source", "unknown"),
            consent=bool(d.get("consent", False)),
            deidentified=bool(d.get("deidentified", False)),
            n_utterances=d.get("n_utterances"),
            speakers=list(d.get("speakers") or []),
        )


@dataclass
class Conversation:
    """一段對話 = JSONL 的一行。

    annotations 以「uid 字串」為 key，每則含：
      - per_annotator: list[{"annotator": id, **Annotation}]   ≥3 人獨立標
      - gold:          Annotation or None                       裁決後
      - _draft:        Annotation or None                       模型初標（供人改，非 gold）
      - _draft_meta:   dict or None                             初標的模型出處/推理鏈
    """
    conv_id: str
    meta: ConvMeta
    utterances: list[Utterance]
    annotations: dict[str, dict] = field(default_factory=dict)

    # -- 便利存取 ------------------------------------------------------------ #
    def uids(self) -> list[int]:
        return [u.uid for u in self.utterances]

    def utterance_by_uid(self, uid: int) -> Optional[Utterance]:
        for u in self.utterances:
            if u.uid == uid:
                return u
        return None

    # -- 序列化 -------------------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "conv_id": self.conv_id,
            "meta": self.meta.to_dict(),
            "utterances": [u.to_dict() for u in self.utterances],
            "annotations": self.annotations,
        }

    def to_jsonl_line(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "Conversation":
        return cls(
            conv_id=d["conv_id"],
            meta=ConvMeta.from_dict(d.get("meta")),
            utterances=[Utterance.from_dict(u) for u in d.get("utterances", [])],
            annotations=d.get("annotations") or {},
        )

    @classmethod
    def from_jsonl_line(cls, line: str) -> "Conversation":
        return cls.from_dict(json.loads(line))


# --------------------------------------------------------------------------- #
# 3. 距離自動算 / distant_cause 自動打標
# --------------------------------------------------------------------------- #


def compute_distance(cause: Optional[Cause], uid: int) -> Optional[int]:
    """當前 uid − cause_uid（僅 in_conversation 且有 cause_uid 時）。"""
    if cause is None or cause.source != "in_conversation" or cause.cause_uid is None:
        return None
    return uid - cause.cause_uid


def apply_distance(ann: Annotation, uid: int) -> Annotation:
    """就地填入 cause.distance，並依距離自動補/移除 distant_cause 標籤。

    distant_cause 是「可自動判定」的難度標籤，故以此函式為單一事實來源，
    避免標註者手動打標時漏打或誤打（草案 v0 第 5 節：distance≥2 自動打標）。
    """
    dist = compute_distance(ann.cause, uid)
    if ann.cause is not None:
        ann.cause.distance = dist

    tags = [t for t in ann.difficulty_tags if t != "distant_cause"]
    if dist is not None and dist >= DISTANT_CAUSE_MIN_DISTANCE:
        tags.append("distant_cause")
    # 去重、保序（依 DIFFICULTY_TAGS 的正典順序）
    ann.difficulty_tags = [t for t in DIFFICULTY_TAGS if t in tags]
    return ann


# --------------------------------------------------------------------------- #
# 4. 驗證
# --------------------------------------------------------------------------- #


def validate_annotation(ann: Annotation, uid: int, valid_uids: Iterable[int]) -> list[str]:
    """驗證單一標註，回傳錯誤訊息清單（空＝合法）。

    valid_uids：本段對話所有 uid，用來檢查 cause_uid 是否指向存在且在當前訊息之前。
    """
    errs: list[str] = []
    valid = set(valid_uids)

    # -- Tier 1 valence --
    if ann.valence not in VALENCE:
        errs.append(f"valence '{ann.valence}' 不在合法集合 {VALENCE}")

    # -- Tier 2 emotion --
    if ann.emotion_primary not in EMOTION:
        errs.append(f"emotion_primary '{ann.emotion_primary}' 不在合法集合 {EMOTION}")

    # -- valence ↔ emotion 一致性（surprise/ambiguous 情緒不強制）--
    expected = EMOTION_VALENCE.get(ann.emotion_primary)
    if expected is not None and ann.valence in VALENCE and ann.valence != expected:
        errs.append(
            f"valence '{ann.valence}' 與 emotion_primary "
            f"'{ann.emotion_primary}' 不一致（該情緒應為 '{expected}'）"
        )

    # -- 難度標籤 --
    for t in ann.difficulty_tags:
        if t not in DIFFICULTY_TAGS:
            errs.append(f"difficulty_tag '{t}' 不在合法集合 {DIFFICULTY_TAGS}")
    if len(set(ann.difficulty_tags)) != len(ann.difficulty_tags):
        errs.append(f"difficulty_tags 有重複：{ann.difficulty_tags}")

    # -- 成因（草案 v0 第 5 節）--
    is_neutral = ann.emotion_primary == "neutral"
    if is_neutral:
        if ann.cause is not None:
            errs.append("中性訊息的 cause 必須為 null")
    else:
        # 非中性：cause 允許為 None（v1 可先不做成因），但一旦給了就要合規
        if ann.cause is not None:
            errs.extend(_validate_cause(ann.cause, uid, valid))

    # -- distant_cause 與距離一致性（若有填距離）--
    dist = compute_distance(ann.cause, uid)
    has_tag = "distant_cause" in ann.difficulty_tags
    if dist is not None:
        should = dist >= DISTANT_CAUSE_MIN_DISTANCE
        if should and not has_tag:
            errs.append(f"距離為 {dist}（≥{DISTANT_CAUSE_MIN_DISTANCE}）但未打 distant_cause 標籤")
        if not should and has_tag:
            errs.append(f"距離為 {dist}（<{DISTANT_CAUSE_MIN_DISTANCE}）卻打了 distant_cause 標籤")
    elif has_tag:
        errs.append("打了 distant_cause 標籤，但沒有可計算距離的 in_conversation 成因")

    return errs


def _validate_cause(cause: Cause, uid: int, valid_uids: set[int]) -> list[str]:
    errs: list[str] = []
    if cause.source not in CAUSE_SOURCE:
        errs.append(f"cause.source '{cause.source}' 不在合法集合 {CAUSE_SOURCE}")

    if cause.source == "in_conversation":
        if cause.cause_uid is None:
            errs.append("cause.source=in_conversation 但缺 cause_uid")
        else:
            if cause.cause_uid not in valid_uids:
                errs.append(f"cause_uid={cause.cause_uid} 不存在於本段對話")
            elif cause.cause_uid > uid:
                errs.append(f"cause_uid={cause.cause_uid} 在當前訊息（uid={uid}）之後，成因不可在未來")
    else:
        # external / unclear：不應指定 cause_uid
        if cause.cause_uid is not None:
            errs.append(f"cause.source={cause.source} 不應有 cause_uid（得到 {cause.cause_uid}）")

    return errs


def validate_conversation(conv: Conversation, require_gold: bool = False) -> list[str]:
    """驗證整段對話（結構 + 每則的 gold/per_annotator）。回傳錯誤清單（空＝合法）。"""
    errs: list[str] = []
    prefix = f"[{conv.conv_id}]"

    # -- 結構 --
    uids = conv.uids()
    if len(uids) != len(set(uids)):
        errs.append(f"{prefix} utterances 的 uid 有重複：{uids}")
    if conv.meta.n_utterances is not None and conv.meta.n_utterances != len(conv.utterances):
        errs.append(f"{prefix} meta.n_utterances={conv.meta.n_utterances} 與實際 {len(conv.utterances)} 則不符")

    valid_uids = set(uids)

    for uid_str, block in conv.annotations.items():
        # key 必須是能對到某則 utterance 的字串 uid
        try:
            uid = int(uid_str)
        except (TypeError, ValueError):
            errs.append(f"{prefix} annotations key '{uid_str}' 不是整數 uid")
            continue
        if uid not in valid_uids:
            errs.append(f"{prefix} annotations 指向不存在的 uid={uid}")
            continue

        gold = block.get("gold")
        if gold:
            errs.extend(f"{prefix} uid={uid} gold: {e}"
                        for e in validate_annotation(Annotation.from_dict(gold), uid, valid_uids))
        elif require_gold:
            errs.append(f"{prefix} uid={uid} 缺 gold（require_gold=True）")

        for pa in block.get("per_annotator") or []:
            who = pa.get("annotator", "?")
            errs.extend(f"{prefix} uid={uid} annotator={who}: {e}"
                        for e in validate_annotation(Annotation.from_dict(pa), uid, valid_uids))

    return errs


# --------------------------------------------------------------------------- #
# 5. JSONL 讀寫 + CLI 檢查
# --------------------------------------------------------------------------- #


def load_jsonl(path: str) -> list[Conversation]:
    convs: list[Conversation] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                convs.append(Conversation.from_jsonl_line(line))
            except Exception as e:  # noqa: BLE001
                raise ValueError(f"{path}:{ln} JSONL 解析失敗：{e}") from e
    return convs


def dump_jsonl(convs: Iterable[Conversation], path: str) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for c in convs:
            f.write(c.to_jsonl_line() + "\n")
            n += 1
    return n


def _cli(argv: Optional[list[str]] = None) -> int:
    import argparse
    import sys

    # Windows 主控台預設 cp950，無法輸出 ✓/✗ 會崩潰；統一改 UTF-8。
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="驗證標註 JSONL 是否符合 schema v0")
    ap.add_argument("path", help="要檢查的 .jsonl（一行一段對話）")
    ap.add_argument("--require-gold", action="store_true", help="要求每則都有 gold")
    args = ap.parse_args(argv)

    convs = load_jsonl(args.path)
    all_errs: list[str] = []
    for c in convs:
        all_errs.extend(validate_conversation(c, require_gold=args.require_gold))

    print(f"讀入 {len(convs)} 段對話。")
    if all_errs:
        print(f"發現 {len(all_errs)} 個問題：")
        for e in all_errs:
            print(f"  ✗ {e}")
        return 1
    print("✓ 全部通過 schema 驗證。")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
