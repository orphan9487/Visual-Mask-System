"""High-detail SD1.5 backend used for still sticker output."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline

from src.generation.diffusion_engine import SD_MODEL_ID, VAE_DIR
from src.generation.renderers.base import GenerationRequest, RenderedFrames, Renderer


class StaticStickerRenderer(Renderer):
    """Generate a single PNG-quality frame without AnimateDiff motion blur."""

    MAX_REJECTED_IMAGE_RETRIES = 2

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

    @staticmethod
    def _safety_checker_rejected(result: Any) -> bool:
        detected = getattr(result, "nsfw_content_detected", None)
        if detected is None:
            return False
        if isinstance(detected, (list, tuple)):
            return any(bool(value) for value in detected)
        return bool(detected)

    @staticmethod
    def _is_black_image(image: Any) -> bool:
        """Treat an all-black/near-black safety placeholder as invalid output."""
        try:
            extrema = image.convert("RGB").getextrema()
        except (AttributeError, TypeError, ValueError):
            return False
        return bool(extrema) and all(channel_max <= 1 for _, channel_max in extrema)

    def render(self, request: GenerationRequest) -> RenderedFrames:
        self._load_lora(request.identity)
        seed = request.seed if request.seed is not None else random.randint(0, 1_000_000)

        original_decode = self.pipe.vae.decode
        decode_dtype = next(
            self.pipe.vae.post_quant_conv.parameters()
        ).dtype

        def dtype_safe_decode(latents, **kwargs):
            return original_decode(latents.to(dtype=decode_dtype), **kwargs)

        self.pipe.vae.decode = dtype_safe_decode
        try:
            for attempt in range(self.MAX_REJECTED_IMAGE_RETRIES + 1):
                if attempt:
                    previous_seed = seed
                    while seed == previous_seed:
                        seed = random.randint(0, 1_000_000)
                generator = torch.Generator(device="cuda").manual_seed(seed)
                with torch.inference_mode():
                    result = self.pipe(
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        num_inference_steps=request.num_inference_steps,
                        guidance_scale=request.guidance_scale,
                        width=request.width,
                        height=request.height,
                        generator=generator,
                    )
                image = result.images[0]
                rejected_by_safety = self._safety_checker_rejected(result)
                black_image = self._is_black_image(image)
                if not rejected_by_safety and not black_image:
                    return RenderedFrames(frames=[image], seed=seed)

                reasons = []
                if rejected_by_safety:
                    reasons.append("safety checker")
                if black_image:
                    reasons.append("black image")
                retry_status = (
                    f"retry {attempt + 1}/{self.MAX_REJECTED_IMAGE_RETRIES}"
                    if attempt < self.MAX_REJECTED_IMAGE_RETRIES
                    else "no retries left"
                )
                print(
                    "[Diffusion] Rejected static image "
                    f"(seed={seed}, reason={'+'.join(reasons)}); {retry_status}.",
                    flush=True,
                )

            raise RuntimeError(
                "SD1.5 returned a safety-rejected or black image after "
                f"{self.MAX_REJECTED_IMAGE_RETRIES + 1} attempts."
            )
        finally:
            self.pipe.vae.decode = original_decode
