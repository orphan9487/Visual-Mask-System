"""Generate comparable candidate stickers for the difficult emotions.

Run this file directly in VS Code with the ``mask_env`` interpreter.  It keeps
candidate outputs separate from one-off instruction runs so review is clean.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.instruction_runner import InstructionGenerationRunner
from src.reasoning.visual_instruction import VisualInstructionGenerator


EMOTIONS = ("sadness", "surprise", "fear", "disgust", "neutral")
SEEDS = (42, 314, 777)
MASK_ID = "human_8692"
LORA_WEIGHT = 0.6
EXPRESSION_WEIGHT = 1.5
OUTPUT_DIR = PROJECT_ROOT / "output" / "current" / "emotion_candidates"


def main() -> None:
    generator = VisualInstructionGenerator()
    runner = InstructionGenerationRunner(output_dir=OUTPUT_DIR)

    for emotion in EMOTIONS:
        instruction = generator.generate(
            emotion,
            user_id="Udebug",
            mask_id=MASK_ID,
            utterance="multi-seed emotion candidate test",
        ).to_dict()
        for seed in SEEDS:
            output_path = runner.generate_from_instruction(
                instruction,
                lora_weight=LORA_WEIGHT,
                seed=seed,
                force_modality="sticker",
                expression_weight=EXPRESSION_WEIGHT,
            )
            print(f"[done] emotion={emotion}, seed={seed}: {output_path}", flush=True)


if __name__ == "__main__":
    main()
