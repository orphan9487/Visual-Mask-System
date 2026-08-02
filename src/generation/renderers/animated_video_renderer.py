"""AnimateDiff backend used only for video output."""

from __future__ import annotations

from src.generation.diffusion_engine import DiffusionSynthesisEngine
from src.generation.renderers.base import GenerationRequest, RenderedFrames, Renderer


class AnimatedVideoRenderer(Renderer):
    """Adapter around the existing AnimateDiff implementation."""

    def __init__(self, engine: DiffusionSynthesisEngine | None = None) -> None:
        self.engine = engine or DiffusionSynthesisEngine()

    def render(self, request: GenerationRequest) -> RenderedFrames:
        self.engine.load_lora(request.identity)
        frames, seed = self.engine.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_frames=max(request.num_frames, 16),
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            seed=request.seed,
        )
        return RenderedFrames(frames=frames, seed=seed)
