"""Render one comparable sticker for each supported emotion with Henry's LoRA.

Run this file directly in VS Code using the ``mask_env`` interpreter.  The
outputs are intentionally kept in their own directory for side-by-side review
and do not replace the LINE demo assets.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.instruction_runner import InstructionGenerationRunner
from src.reasoning.visual_instruction import VisualInstructionGenerator


EMOTIONS = ("anger", "joy", "sadness", "surprise", "fear", "disgust", "neutral")
MASK_ID = "henry"
SEED = 42
LORA_WEIGHT = 0.7
EXPRESSION_WEIGHT = 1.3
OUTPUT_DIR = PROJECT_ROOT / "output" / "current" / "henry_emotion_test"


def main() -> None:
    generator = VisualInstructionGenerator()
    runner = InstructionGenerationRunner(output_dir=OUTPUT_DIR)

    for emotion in EMOTIONS:
        instruction = generator.generate(
            emotion,
            user_id="Uhenry_test",
            mask_id=MASK_ID,
            utterance="Henry LoRA seven-emotion validation",
        ).to_dict()
        output_path = runner.generate_from_instruction(
            instruction,
            lora_weight=LORA_WEIGHT,
            seed=SEED,
            force_modality="sticker",
            expression_weight=EXPRESSION_WEIGHT,
        )
        print(f"[done] emotion={emotion}: {output_path}", flush=True)


if __name__ == "__main__":
    main()
