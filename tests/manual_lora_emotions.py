"""Manually render several emotions through the current generation pipeline.

Examples:
  python tests/manual_lora_emotions.py --quick
  python tests/manual_lora_emotions.py --emotions anger sadness joy
  python tests/manual_lora_emotions.py --mask human_5805
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ALL_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
QUICK_EMOTIONS = ["anger", "sadness", "joy", "neutral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render LoRA emotion samples")
    parser.add_argument("--mask", default="human_8692")
    parser.add_argument("--emotions", nargs="+")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    emotions = QUICK_EMOTIONS if args.quick else (args.emotions or ALL_EMOTIONS)

    from src.generation.instruction_runner import InstructionGenerationRunner
    from src.reasoning.identity_db import identity_db
    from src.reasoning.visual_instruction import visual_instruction_generator

    if args.mask not in identity_db.masks:
        raise SystemExit(f"Unknown mask '{args.mask}'. Available: {sorted(identity_db.masks)}")

    output_dir = PROJECT_ROOT / "output" / "current" / f"test_{args.mask}"
    runner = InstructionGenerationRunner(output_dir=output_dir)
    print(f"Mask: {args.mask}; emotions: {', '.join(emotions)}")

    for emotion in emotions:
        instruction = visual_instruction_generator.generate(
            emotion,
            mask_id=args.mask,
        ).to_dict()
        output = runner.generate_from_instruction(
            instruction,
            seed=args.seed,
            force_modality="sticker",
        )
        print(f"[{emotion}] {output}")


if __name__ == "__main__":
    main()
