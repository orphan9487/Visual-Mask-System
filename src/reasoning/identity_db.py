# -*- coding: utf-8 -*-
"""
Identity-DB：使用者身分 → 個人化「面具」設定的註冊表。

對應計畫書推理層之 Identity-DB。負責回答「該用誰的視覺面具」：
每位使用者對應一組 LoRA 權重與觸發詞，供生成層載入以維持人設一致性。

本模組屬「推理層」職責——只決定與描述身分，不執行任何影像生成
（生成由隊友負責之生成層消費本模組輸出的設定）。以 JSON 持久化，
可離線編輯、版本控管。
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "identity_db.json"


@dataclass
class MaskIdentity:
    """一個視覺面具的完整設定（支援 LoRA 或免訓練 FaceID）。"""
    mask_id: str                 # 面具識別碼
    trigger: str                 # LoRA 觸發詞
    base_prompt: str             # 身分基底描述（不含情緒）
    lora_path: str               # LoRA 權重相對路徑
    lora_weight: float = 0.8     # LoRA 融合權重（身分↔表情的取捨）
    display_name: str = ""       # 可讀名稱
    identity_mode: str = "lora"  # "lora" | "faceid"
    faceid_reference_paths: list[str] = field(default_factory=list)
    faceid_scale: float = 0.55
    faceid_checkpoint: str = (
        "models/ip_adapter_faceid/ip-adapter-faceid-portrait-v11_sd15.bin"
    )


def _configured_henry_faceid_references() -> list[str]:
    """Read the local Henry enrollment without committing personal photos."""
    configured = os.getenv("VMS_HENRY_FACEID_REFERENCES", "").strip()
    if configured:
        return [item.strip() for item in configured.split(os.pathsep) if item.strip()]
    profile_dir = PROJECT_ROOT / "data" / "faceid_profiles" / "henry"
    supported = {".jpg", ".jpeg", ".png", ".webp"}
    return [
        str(path)
        for path in sorted(profile_dir.glob("*"))
        if path.is_file() and path.suffix.lower() in supported
    ]


# 預設面具庫（沿用專案既有的 person8692 人物 LoRA；日後每位使用者可註冊自己的）
_DEFAULT_MASKS = {
    "human": MaskIdentity(
        mask_id="human",
        trigger="person8692",
        base_prompt="1boy, masculine, a portrait of a person, realistic skin",
        lora_path="models/8692_mask/person8692_v1_baseline-000009.safetensors",
        lora_weight=0.8,
        display_name="預設人物面具",
    ),
    # Legacy alias retained while older clients still send ``human_8692``.
    "human_8692": MaskIdentity(
        mask_id="human_8692",
        trigger="person8692",
        base_prompt="1boy, masculine, a portrait of a person, realistic skin",
        lora_path="models/8692_mask/person8692_v1_baseline-000009.safetensors",
        lora_weight=0.8,
        display_name="Person 8692",
    ),
    "human_5805": MaskIdentity(
        mask_id="human_5805",
        trigger="person5805",
        base_prompt="1girl, feminine, a portrait of a person, realistic skin",
        lora_path="models/5805_mask/person5805_v2_ep9.safetensors",
        lora_weight=0.8,
        display_name="Person 5805",
    ),
    "henry": MaskIdentity(
        mask_id="henry",
        trigger="henrymask",
        base_prompt=(
            "color front portrait, young East Asian man, slender oval face, "
            "almond eyes, thin round glasses, center-parted black hair, natural skin"
        ),
        lora_path="models/henry_mask_lora/henrymask_v3.safetensors",
        lora_weight=0.8,
        display_name="Henry",
        identity_mode="faceid",
        faceid_reference_paths=_configured_henry_faceid_references(),
        faceid_scale=0.55,
    ),
    "ethan": MaskIdentity(
        mask_id="ethan",
        trigger="ethan_mask",
        base_prompt="1boy, masculine, a portrait of Ethan, realistic skin",
        lora_path="models/ethan_mask_lora/pytorch_lora_weights.safetensors",
        lora_weight=0.8,
        display_name="Ethan",
    ),
    "yourname": MaskIdentity(
        mask_id="yourname",
        trigger="yourname_mask",
        base_prompt="1person, a portrait of a person, realistic skin",
        lora_path="models/yourname_mask_lora/pytorch_lora_weights.safetensors",
        lora_weight=0.8,
        display_name="Yourname",
    ),
}

# The restored WebSocket product uses Henry's enrolled FaceID profile by
# default.  LoRA identities remain registered as optional fallback choices.
DEFAULT_MASK_ID = os.getenv("VMS_DEFAULT_MASK_ID", "henry").strip() or "henry"


class IdentityDB:
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self.masks: dict[str, MaskIdentity] = dict(_DEFAULT_MASKS)
        # user_id → mask_id 的綁定
        self.bindings: dict[str, str] = {}
        self._load()

    # ------------------------------------------------------------------ #
    def _load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for m in data.get("masks", []):
                self.masks[m["mask_id"]] = MaskIdentity(**m)
            self.bindings.update(data.get("bindings", {}))

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "masks": [asdict(m) for m in self.masks.values()],
            "bindings": self.bindings,
        }
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    # ------------------------------------------------------------------ #
    def register_mask(self, mask: MaskIdentity, persist: bool = True):
        self.masks[mask.mask_id] = mask
        if persist:
            self.save()

    def bind_user(self, user_id: str, mask_id: str, persist: bool = True):
        """把某使用者綁定到某面具。"""
        if mask_id not in self.masks:
            raise KeyError(f"未知 mask_id：{mask_id}")
        self.bindings[user_id] = mask_id
        if persist:
            self.save()

    def resolve(self, user_id: Optional[str] = None,
                mask_id: Optional[str] = None) -> MaskIdentity:
        """
        解析出該用哪個面具，優先序：
        明確指定的 mask_id > 該 user 的綁定 > 預設面具。
        """
        if mask_id and mask_id in self.masks:
            return self.masks[mask_id]
        if user_id and user_id in self.bindings:
            return self.masks[self.bindings[user_id]]
        return self.masks[DEFAULT_MASK_ID]


# 單例
identity_db = IdentityDB()
