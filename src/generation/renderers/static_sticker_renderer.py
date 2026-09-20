"""High-detail SD1.5 backend used for still sticker output."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline

from src.generation.diffusion_engine import SD_MODEL_ID, VAE_DIR
from src.generation.renderers.base import GenerationRequest, RenderedFrames, Renderer


class StaticStickerRenderer(Renderer):
    """Generate a single PNG-quality frame without AnimateDiff motion blur."""

    def __init__(self) -> None:
        print("[Diffusion] Loading SD1.5 static sticker pipeline...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        ).to("cuda")

        if VAE_DIR.is_dir():
            print("[Diffusion] Loading local fp32 VAE...")
            self.pipe.vae = AutoencoderKL.from_pretrained(
                VAE_DIR.as_posix(), torch_dtype=torch.float32
            ).to("cuda")

        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="linear",
            clip_sample=False,
            timestep_spacing="linspace",
            steps_offset=1,
        )
        # The runner can serve a sticker and then a video in one process.  Keep
        # this otherwise-large static pipeline off GPU between denoising steps
        # so it can coexist with the lazily loaded AnimateDiff renderer.
        self.pipe.enable_model_cpu_offload()
        print("[Diffusion] Static sticker pipeline ready.")

    def _load_lora(self, identity: dict) -> None:
        lora_path = Path(identity["path"])
        if not lora_path.is_file():
            raise FileNotFoundError(f"LoRA file does not exist: {lora_path}")
        self.pipe.unload_lora_weights()
        self.pipe.load_lora_weights(
            str(lora_path.parent),
            weight_name=lora_path.name,
            adapter_name=str(identity["adapter_name"]),
        )
        self.pipe.set_adapters(
            [str(identity["adapter_name"])],
            adapter_weights=[float(identity.get("lora_weight", 0.8))],
        )

    def render(self, request: GenerationRequest) -> RenderedFrames:
        self._load_lora(request.identity)
        seed = request.seed if request.seed is not None else random.randint(0, 1_000_000)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        original_decode = self.pipe.vae.decode

        def fp32_decode(latents, **kwargs):
            return original_decode(latents.to(dtype=torch.float32), **kwargs)

        self.pipe.vae.decode = fp32_decode
        try:
            with torch.inference_mode():
                image = self.pipe(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    width=request.width,
                    height=request.height,
                    generator=generator,
                ).images[0]
        finally:
            self.pipe.vae.decode = original_decode
        return RenderedFrames(frames=[image], seed=seed)
