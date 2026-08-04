# -*- coding: utf-8 -*-
"""
Visual Instruction Generator（＋Modality Selector）——推理層的輸出介面。

對應計畫書推理層：Affective Analyzer 產出情緒後，本模組將其轉為一份
**結構化視覺指令 (VisualInstruction)**，作為推理層交付給生成層的「合約」。
生成層（隊友負責）只需消費此 JSON、無須理解上游如何判讀情緒，
兩層即可獨立開發與測試。

設計原則：
  - 語言無關：Scene Description Script 一律以英文產出（計畫方法 A 節：
    生成端與語言無關，繁中挑戰集中於理解層）。
  - 穩定 schema：欄位為兩層之間的公開合約，變更需雙方同步。
  - 純推理層職責：本模組只「描述要生成什麼」，不執行任何影像生成。

VisualInstruction 之 JSON 範例：
{
  "schema_version": "1.0",
  "user_id": "Uxxxx",
  "identity": {"mask_id": "human", "trigger": "person8692",
               "lora_path": "...", "lora_weight": 0.8, "base_prompt": "..."},
  "emotion": "anger", "emotion_zh": "憤怒", "intensity": 0.9,
  "modality": "video",
  "scene_description": "A portrait of person8692 ... angry expression ...",
  "positive_prompt": "person8692, 1boy, ... (angry expression...:1.3), ...",
  "negative_prompt": "blurry eyes, deformed iris, ...",
  "generation_hints": {"num_frames": 16, "num_inference_steps": 25, "guidance_scale": 8.0},
  "source": {"utterance": "...", "rationale": "..."}
}
"""

from dataclasses import asdict, dataclass, field
from typing import Optional

from .identity_db import IdentityDB, MaskIdentity, identity_db
from .labels import EMOTION_ZH

SCHEMA_VERSION = "1.1"   # 1.1：新增選用欄位 intent（語意意圖分析，向下相容）

# 每種情緒的：英文表情描述、喚醒度(intensity, 0-1)。
# 喚醒度同時決定表情強度與 modality（高喚醒→動態影片，低喚醒→靜態貼圖）。
EMOTION_PROFILE = {
    "anger":    ("angry expression, furrowed brows, clenched jaw, intense glare, frowning", 0.90),
    "joy":      ("happy bright smile, sparkling eyes, cheerful expression", 0.80),
    "surprise": ("surprised expression, wide eyes, raised eyebrows, slightly open mouth", 0.80),
    "fear":     (
        "(terrified face:1.5), (wide-open eyes, white sclera:1.7), "
        "raised eyebrows, screaming, dropped jaw",
        0.75,
    ),
    "disgust":  ("disgusted expression, wrinkled nose, curled upper lip", 0.60),
    "sadness":  ("sad expression, downcast teary eyes, drooping mouth, sorrowful", 0.60),
    "neutral":  ("calm relaxed neutral expression", 0.30),
}

DEFAULT_NEGATIVE = ("blurry eyes, deformed iris, hazy, low quality, "
                    "oil painting, cross-eyed, extra fingers, deformed, "
                    "cropped head, cropped face, extreme close-up, close-up, "
                    "face close-up, zoomed in, cut off hair, hands, fingers, "
                    "black and white, monochrome, grayscale, desaturated, "
                    "sepia, muted colors")

COLOR_PHOTO_STYLE = ("realistic color photo, natural skin tones, "
                     "soft indoor lighting")

HENRY_NEGATIVE = ("bald, receding hairline, gray hair, white hair, "
                  "elderly, old man")

# Modality Selector 門檻：喚醒度 ≥ 此值 → 動態影片；否則靜態貼圖。
INTENSITY_VIDEO_THRESHOLD = 0.6


@dataclass
class VisualInstruction:
    user_id: Optional[str]
    identity: dict
    emotion: str
    emotion_zh: str
    intensity: float
    modality: str                 # "video" | "sticker"
    scene_description: str        # 英文 Scene Description Script
    positive_prompt: str
    negative_prompt: str
    generation_hints: dict
    source: dict = field(default_factory=dict)
    # 選用（schema 1.1）：語意意圖分析。生成層可忽略；回饋策略層可用（如反諷→調整表情）。
    intent: Optional[dict] = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


class ModalitySelector:
    """依情緒喚醒度決定輸出型態。獨立成類，便於日後改為更複雜策略。"""

    def __init__(self, threshold: float = INTENSITY_VIDEO_THRESHOLD):
        self.threshold = threshold

    def select(self, emotion: str, intensity: float) -> str:
        return "video" if intensity >= self.threshold else "sticker"


class VisualInstructionGenerator:
    """
    推理層輸出端：ERC 結果（＋身分）→ VisualInstruction。

    輸入為情緒標籤（來自 Affective Analyzer / ERCEngine），
    輸出為生成層可直接消費的結構化指令。
    """

    def __init__(self, db: IdentityDB = identity_db,
                 modality_selector: Optional[ModalitySelector] = None):
        self.db = db
        self.modality = modality_selector or ModalitySelector()

    def _scene_description(self, mask: MaskIdentity, expression: str,
                           modality: str) -> str:
        motion = ("with subtle natural head movement and blinking"
                  if modality == "video" else "as a still portrait")
        return (f"A portrait of {mask.trigger} person showing {expression}, "
                f"{motion}, vivid lighting, masterpiece, high quality, highly detailed.")

    def _positive_prompt(self, mask: MaskIdentity, expression: str,
                         emotion: str) -> str:
        # Fear spends the limited CLIP token budget on its clearest cue: eyes.
        # Other emotions keep the shared expression weighting.
        if emotion == "fear":
            # Put fear before identity so SD1.5 does not dilute the expression.
            return (f"{mask.trigger}, {expression}, {mask.base_prompt}, "
                    f"{COLOR_PHOTO_STYLE}")
        weighted_expression = f"({expression}:1.3)"
        return (f"{mask.trigger}, {mask.base_prompt}, "
                f"{weighted_expression}, {COLOR_PHOTO_STYLE}")

    def generate(self, emotion: str, *, user_id: Optional[str] = None,
                 mask_id: Optional[str] = None, intensity: Optional[float] = None,
                 utterance: str = "", rationale: str = "",
                 intent: Optional[dict] = None) -> VisualInstruction:
        emotion = emotion if emotion in EMOTION_PROFILE else "neutral"
        expression, base_intensity = EMOTION_PROFILE[emotion]
        intensity = base_intensity if intensity is None else float(intensity)

        mask = self.db.resolve(user_id=user_id, mask_id=mask_id)
        modality = self.modality.select(emotion, intensity)

        hints = {
            "num_frames": 16 if modality == "video" else 1,
            "num_inference_steps": 25,
            "guidance_scale": 8.0,
            "lora_weight": mask.lora_weight,
            # Training uses portrait buckets; keep the same portrait framing
            # at inference rather than letting AnimateDiff default to 512x512.
            "width": 384,
            "height": 512,
        }
        if mask.generation_seed is not None:
            hints["seed"] = int(mask.generation_seed)

        negative_prompt = DEFAULT_NEGATIVE
        if mask.mask_id == "henry":
            negative_prompt = f"{negative_prompt}, {HENRY_NEGATIVE}"

        return VisualInstruction(
            user_id=user_id,
            identity={"mask_id": mask.mask_id, "trigger": mask.trigger,
                      "lora_path": mask.lora_path, "lora_weight": mask.lora_weight,
                      "base_prompt": mask.base_prompt},
            emotion=emotion,
            emotion_zh=EMOTION_ZH.get(emotion, emotion),
            intensity=round(intensity, 3),
            modality=modality,
            scene_description=self._scene_description(mask, expression, modality),
            positive_prompt=self._positive_prompt(mask, expression, emotion),
            negative_prompt=negative_prompt,
            generation_hints=hints,
            source={"utterance": utterance, "rationale": rationale[:300]},
            intent=intent,
        )

    def from_erc_result(self, result, **kwargs) -> VisualInstruction:
        """便利介面：直接吃 ERCEngine 的 ERCResult。"""
        return self.generate(
            result.predicted or "neutral",
            utterance=kwargs.pop("utterance", ""),
            rationale=getattr(result, "rationale", ""),
            **kwargs,
        )


# 單例
visual_instruction_generator = VisualInstructionGenerator()
