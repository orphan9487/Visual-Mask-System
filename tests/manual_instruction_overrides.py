"""Manual VS Code check for generation-only VisualInstruction overrides.

Open this file in VS Code and use ``Run Python File`` while the selected
interpreter is the ``mask_env`` Conda environment.  Edit only the constants in
the configuration block when you want to test a different emotion or mask.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.instruction_runner import InstructionGenerationRunner
from src.reasoning.visual_instruction import VisualInstructionGenerator


# ----- Experiment configuration: change these values, then press Run. -----
EMOTION = "anger"
MASK_ID = "human_8692"
LORA_WEIGHTS = (0.5, 0.6, 0.7)
EXPRESSION_WEIGHT = 1.5
SEED = 42
MODALITY = "sticker"  # "sticker" creates PNG; change to "video" for MP4.


def main() -> None:
    instruction = VisualInstructionGenerator().generate(
        EMOTION,
        user_id="Udebug",
        mask_id=MASK_ID,
        utterance="VS Code generation override test",
    ).to_dict()

    runner = InstructionGenerationRunner()
    print(f"Testing {EMOTION=} {MASK_ID=} {SEED=} {MODALITY=}")
    print(f"LoRA weights: {LORA_WEIGHTS}; expression weight: {EXPRESSION_WEIGHT}")

    for lora_weight in LORA_WEIGHTS:
        output_path = runner.generate_from_instruction(
            instruction,
            lora_weight=lora_weight,
            seed=SEED,
            force_modality=MODALITY,
            expression_weight=EXPRESSION_WEIGHT,
        )
        print(f"[done] lora_weight={lora_weight:.2f} -> {output_path}")


if __name__ == "__main__":
    main()
