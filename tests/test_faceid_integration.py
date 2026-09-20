"""FaceID routing tests; pretrained models are not loaded."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.instruction_runner import InstructionGenerationRunner
from src.generation.renderers.base import GenerationRequest, RenderedFrames, Renderer
from src.reasoning.visual_instruction import VisualInstructionGenerator


class FakeFaceIDRenderer(Renderer):
    def __init__(self) -> None:
        self.request: GenerationRequest | None = None

    def render(self, request: GenerationRequest) -> RenderedFrames:
        self.request = request
        return RenderedFrames([Image.new("RGB", (64, 64))], request.seed or 42)


class FaceIDIntegrationTests(unittest.TestCase):
    def test_faceid_prompt_keeps_full_head_inside_frame(self):
        instruction = VisualInstructionGenerator().generate(
            "anger",
            mask_id="henry",
        )

        self.assertIn("centered head-and-shoulders portrait", instruction.positive_prompt)
        self.assertIn("full head visible", instruction.positive_prompt)
        self.assertIn("space above hair", instruction.positive_prompt)
        self.assertIn("cropped head", instruction.negative_prompt)
        self.assertIn("extreme close-up", instruction.negative_prompt)
        self.assertIn("cut off hair", instruction.negative_prompt)

    def test_faceid_does_not_require_lora_and_routes_to_faceid_renderer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "face.jpg"
            checkpoint = root / "faceid.bin"
            reference.write_bytes(b"reference")
            checkpoint.write_bytes(b"checkpoint")
            renderer = FakeFaceIDRenderer()
            runner = InstructionGenerationRunner(
                faceid_renderer=renderer,
                output_dir=root / "output",
            )
            result = runner.generate_from_instruction(
                {
                    "identity": {
                        "mask_id": "alice",
                        "mode": "faceid",
                        "faceid_reference_paths": [str(reference)],
                        "faceid_checkpoint": str(checkpoint),
                        "faceid_scale": 0.55,
                    },
                    "emotion": "joy",
                    "modality": "sticker",
                    "positive_prompt": "joyful portrait",
                    "negative_prompt": "deformed",
                    "generation_hints": {
                        "num_frames": 1,
                        "num_inference_steps": 30,
                        "guidance_scale": 7.0,
                        "width": 512,
                        "height": 512,
                    },
                },
                seed=123,
            )

            self.assertTrue(Path(result).is_file())
            self.assertIsNotNone(renderer.request)
            assert renderer.request is not None
            self.assertEqual(renderer.request.identity["mode"], "faceid")
            self.assertEqual(renderer.request.identity["scale"], 0.55)
            self.assertEqual(renderer.request.seed, 123)


if __name__ == "__main__":
    unittest.main()
