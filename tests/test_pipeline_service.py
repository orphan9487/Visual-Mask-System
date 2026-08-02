"""Unit tests for the transport-independent pipeline; no model is loaded."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.services.pipeline_service import PipelineAnalysis, PipelineService


def analysis_payload(mask_id: str = "human") -> dict:
    return {
        "emotion": "anger",
        "instruction": {
            "identity": {"mask_id": mask_id},
            "emotion": "anger",
            "intensity": 0.9,
            "modality": "video",
            "generation_hints": {"num_frames": 16},
            "source": {"rationale": "unit-test rationale"},
        },
    }


class PipelineServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_analyze_passes_identity_and_sanitizes_history(self):
        received = {}

        async def analyzer(**kwargs):
            received.update(kwargs)
            return analysis_payload(kwargs["mask_id"])

        with tempfile.TemporaryDirectory() as temp_dir:
            service = PipelineService(
                analyzer=analyzer,
                generator=lambda _: "unused",
                output_root=temp_dir,
            )
            result = await service.analyze(
                "  hello  ",
                [{"role": "user", "content": "old"}],
                user_id="alice",
                mask_id="henry",
            )

        self.assertEqual(result.text, "hello")
        self.assertEqual(result.mask_id, "henry")
        self.assertEqual(received["user_id"], "alice")
        self.assertEqual(received["mask_id"], "henry")
        self.assertEqual(received["history"][0]["content"], "old")

    async def test_generate_adjusts_intensity_and_returns_public_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            captured = {}

            def generator(instruction):
                captured.update(instruction)
                path = root / "instruction_runs" / "result.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"image")
                return str(path)

            service = PipelineService(generator=generator, output_root=root)
            analysis = PipelineAnalysis(
                analysis_payload(), "hello", [], "alice"
            )
            result = await service.generate(analysis, intensity_multiplier=0.5)

        self.assertEqual(result.effective_intensity, 0.45)
        self.assertEqual(result.public_path, "instruction_runs/result.png")
        self.assertEqual(captured["modality"], "sticker")
        self.assertEqual(captured["generation_hints"]["num_frames"], 1)

    async def test_generation_lock_serializes_shared_renderer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            guard = threading.Lock()
            active = 0
            maximum = 0
            counter = 0

            def generator(_instruction):
                nonlocal active, maximum, counter
                with guard:
                    active += 1
                    maximum = max(maximum, active)
                    counter += 1
                    number = counter
                time.sleep(0.03)
                path = root / f"result-{number}.png"
                path.write_bytes(b"image")
                with guard:
                    active -= 1
                return str(path)

            service = PipelineService(generator=generator, output_root=root)
            analysis = PipelineAnalysis(
                analysis_payload(), "hello", [], "alice"
            )
            await asyncio.gather(
                service.generate(analysis),
                service.generate(analysis),
            )

        self.assertEqual(maximum, 1)

    async def test_rejects_generator_output_outside_public_root(self):
        with tempfile.TemporaryDirectory() as output_dir, tempfile.TemporaryDirectory() as other:
            escaped = Path(other) / "escaped.png"
            escaped.write_bytes(b"image")
            service = PipelineService(
                generator=lambda _: str(escaped), output_root=output_dir
            )
            analysis = PipelineAnalysis(
                analysis_payload(), "hello", [], "alice"
            )
            with self.assertRaisesRegex(RuntimeError, "escaped output/current"):
                await service.generate(analysis)


if __name__ == "__main__":
    unittest.main()
