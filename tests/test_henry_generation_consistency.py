from pathlib import Path

from PIL import Image

from src.generation.instruction_runner import InstructionGenerationRunner
from src.generation.renderers.base import RenderedFrames
from src.reasoning.visual_instruction import VisualInstructionGenerator


class CapturingRenderer:
    def __init__(self):
        self.request = None

    def render(self, request):
        self.request = request
        return RenderedFrames(
            frames=[Image.new("RGB", (request.width, request.height), "white")],
            seed=request.seed,
        )


def test_henry_instruction_uses_fixed_identity_seed():
    instruction = VisualInstructionGenerator().generate(
        "sadness", mask_id="henry"
    ).to_dict()

    assert instruction["generation_hints"]["seed"] == 314159
    assert "sad expression" in instruction["positive_prompt"]


def test_non_henry_instruction_remains_random():
    instruction = VisualInstructionGenerator().generate(
        "neutral", mask_id="human"
    ).to_dict()

    assert "seed" not in instruction["generation_hints"]


def test_runner_uses_instruction_seed_and_allows_override(tmp_path: Path):
    lora = tmp_path / "test.safetensors"
    lora.touch()
    renderer = CapturingRenderer()
    runner = InstructionGenerationRunner(
        sticker_renderer=renderer,
        output_dir=tmp_path / "output",
    )
    instruction = {
        "identity": {
            "mask_id": "henry",
            "lora_path": str(lora),
            "lora_weight": 0.8,
        },
        "emotion": "neutral",
        "modality": "sticker",
        "positive_prompt": "henrymask, neutral face",
        "negative_prompt": "blurry",
        "generation_hints": {
            "num_frames": 1,
            "num_inference_steps": 1,
            "guidance_scale": 1.0,
            "width": 64,
            "height": 64,
            "seed": 314159,
        },
    }

    runner.generate_from_instruction(instruction)
    assert renderer.request.seed == 314159

    runner.generate_from_instruction(instruction, seed=7)
    assert renderer.request.seed == 7
