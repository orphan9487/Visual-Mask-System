"""AnimateDiff + Stable Diffusion 1.5 engine used by the generation layer.

This is the single production implementation for personalised LoRA generation.
Both the VisualInstruction runner and temporary legacy entry points use this
module, so generation behaviour cannot silently diverge between architectures.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from diffusers import AnimateDiffPipeline, AutoencoderKL, DDIMScheduler, MotionAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SD_MODEL_ID = "runwayml/stable-diffusion-v1-5"
ANIMATEDIFF_ID = "guoyww/animatediff-motion-adapter-v1-5-2"
VAE_DIR = PROJECT_ROOT / "models" / "sd-vae-mse"
MOTION_ADAPTER_DIR = PROJECT_ROOT / "models" / "motion_adapter"


class DiffusionSynthesisEngine:
    """Load one identity LoRA and generate AnimateDiff image frames."""

    def __init__(self) -> None:
        print("[Diffusion] Loading AnimateDiff pipeline...")
        dtype = torch.bfloat16
        # Prefer the checked-in local adapter and cached SD model.  Generation
        # must not make a network metadata request every time VS Code runs it.
        motion_source = (
            MOTION_ADAPTER_DIR.as_posix()
            if MOTION_ADAPTER_DIR.is_dir()
            else ANIMATEDIFF_ID
        )
        motion_adapter = MotionAdapter.from_pretrained(
            motion_source,
            torch_dtype=dtype,
            local_files_only=True,
        )
        self.pipe = AnimateDiffPipeline.from_pretrained(
            SD_MODEL_ID,
            motion_adapter=motion_adapter,
            torch_dtype=dtype,
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
        self.pipe.enable_model_cpu_offload()
        print("[Diffusion] Pipeline ready.")

    def load_lora(self, identity: dict) -> None:
        """Replace the active identity LoRA with the supplied configuration."""
        lora_path = Path(identity["path"])
        if not lora_path.is_file():
            raise FileNotFoundError(f"LoRA file does not exist: {lora_path}")

        adapter_name = str(identity["adapter_name"])
        adapter_weight = float(identity.get("lora_weight", 0.8))
        self.pipe.unload_lora_weights()
        self.pipe.load_lora_weights(
            str(lora_path.parent),
            weight_name=lora_path.name,
            adapter_name=adapter_name,
        )
        self.pipe.set_adapters([adapter_name], adapter_weights=[adapter_weight])

    def generate(
        self,
        prompt: str,
        negative_prompt: str = (
            "multiple people, two people, two faces, multiple faces, group, collage, "
            "split image, hands, fingers, hand, arm, arms, holding, blurry eyes, "
            "deformed iris, hazy, low quality, oil painting, cross-eyed"
        ),
        num_frames: int = 16,
        num_inference_steps: int = 25,
        guidance_scale: float = 8.0,
        width: int = 384,
        height: int = 512,
        seed: int | None = None,
    ) -> tuple[list, int]:
        """Return generated PIL frames and the seed used for reproducibility."""
        seed = seed if seed is not None else random.randint(0, 1_000_000)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        original_decode = self.pipe.vae.decode

        def fp32_decode(z, **kwargs):
            return original_decode(z.to(dtype=torch.float32), **kwargs)

        self.pipe.vae.decode = fp32_decode
        try:
            with torch.inference_mode():
                output = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=num_frames,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator,
                )
        finally:
            self.pipe.vae.decode = original_decode

        return output.frames[0], seed
