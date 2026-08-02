"""Render a controlled seven-emotion comparison of Henry LoRA v2 and v3.

Both versions use the same base model, prompts, seed, LoRA inference weight,
and sticker output mode.  Therefore visual differences are attributable to the
trained LoRA rather than sampling choices.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.instruction_runner import InstructionGenerationRunner
from src.reasoning.identity_db import MaskIdentity, identity_db
from src.reasoning.visual_instruction import VisualInstructionGenerator


EMOTIONS = ("anger", "joy", "sadness", "surprise", "fear", "disgust", "neutral")
VERSIONS = ("v2", "v3")
SEED = 42
LORA_WEIGHT = 0.8
EXPRESSION_WEIGHT = 1.5
OUTPUT_ROOT = PROJECT_ROOT / "output" / "current" / "henry_lora_comparison"


def register_version(version: str) -> str:
    mask_id = f"henry_{version}"
    identity_db.register_mask(
        MaskIdentity(
            mask_id=mask_id,
            trigger="henrymask",
            base_prompt="1man, a portrait of a man, round glasses, realistic skin",
            lora_path=f"models/henry_mask_lora/henrymask_{version}.safetensors",
            lora_weight=LORA_WEIGHT,
            display_name=f"Henry {version.upper()} comparison",
        ),
        persist=False,
    )
    return mask_id


def main() -> None:
    generator = VisualInstructionGenerator()
    runner = InstructionGenerationRunner(output_dir=OUTPUT_ROOT)

    for version in VERSIONS:
        mask_id = register_version(version)
        for emotion in EMOTIONS:
            instruction = generator.generate(
                emotion,
                user_id="Uhenry_comparison",
                mask_id=mask_id,
                utterance="Controlled Henry LoRA v2/v3 comparison",
            ).to_dict()
            output_path = runner.generate_from_instruction(
                instruction,
                lora_weight=LORA_WEIGHT,
                seed=SEED,
                force_modality="sticker",
                expression_weight=EXPRESSION_WEIGHT,
            )
            print(f"[done] version={version}, emotion={emotion}: {output_path}", flush=True)


if __name__ == "__main__":
    main()
