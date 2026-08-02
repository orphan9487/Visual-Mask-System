"""Controlled identity tests for Henry v3.

Experiment A compares a static SD1.5 image with AnimateDiff under identical
prompts and seeds.  Experiment B uses static SD1.5 to isolate the influence of
LoRA inference weight on identity and expression without motion effects.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline

from src.generation.diffusion_engine import DiffusionSynthesisEngine, VAE_DIR
from src.reasoning.visual_instruction import VisualInstructionGenerator


ROOT = PROJECT_ROOT / "output" / "current" / "henry_identity_controls"
MODEL_ID = "runwayml/stable-diffusion-v1-5"
EMOTIONS = ("neutral", "anger")
SEED = 42
WIDTH, HEIGHT = 384, 512
STEPS, CFG = 25, 8.0
LORA_PATH = PROJECT_ROOT / "models" / "henry_mask_lora" / "henrymask_v3.safetensors"
WEIGHTS = (0.5, 0.7, 0.9, 1.0)


def instruction_for(emotion: str) -> dict:
    return VisualInstructionGenerator().generate(
        emotion,
        user_id="Uhenry_control",
        mask_id="henry",
        utterance="Controlled identity evaluation",
    ).to_dict()


def render_animatediff() -> None:
    engine = DiffusionSynthesisEngine()
    ROOT.mkdir(parents=True, exist_ok=True)
    engine.load_lora({"path": str(LORA_PATH.resolve()), "adapter_name": "henry_v3_ad", "lora_weight": 0.7})
    for emotion in EMOTIONS:
        vi = instruction_for(emotion)
        frames, _ = engine.generate(
            prompt=vi["positive_prompt"], negative_prompt=vi["negative_prompt"],
            num_frames=16, num_inference_steps=STEPS, guidance_scale=CFG,
            width=WIDTH, height=HEIGHT, seed=SEED,
        )
        path = ROOT / f"animatediff_{emotion}_lw0p70.png"
        frames[0].save(path)
        print(f"[done] {path}", flush=True)
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def static_pipe() -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    if VAE_DIR.is_dir():
        pipe.vae = AutoencoderKL.from_pretrained(
            VAE_DIR.as_posix(), torch_dtype=torch.float32
        ).to("cuda")
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        original_decode = pipe.vae.decode

        def fp32_decode(latents, **kwargs):
            return original_decode(latents.to(dtype=torch.float32), **kwargs)

        pipe.vae.decode = fp32_decode
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        beta_start=0.00085, beta_end=0.012, beta_schedule="linear",
        clip_sample=False, timestep_spacing="linspace", steps_offset=1,
    )
    return pipe


def render_static(pipe: StableDiffusionPipeline, emotion: str, weight: float, name: str) -> None:
    vi = instruction_for(emotion)
    pipe.unload_lora_weights()
    pipe.load_lora_weights(str(LORA_PATH.parent), weight_name=LORA_PATH.name, adapter_name="henry_v3_static")
    pipe.set_adapters(["henry_v3_static"], adapter_weights=[weight])
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    with torch.inference_mode():
        image = pipe(
            prompt=vi["positive_prompt"], negative_prompt=vi["negative_prompt"],
            width=WIDTH, height=HEIGHT, num_inference_steps=STEPS,
            guidance_scale=CFG, generator=generator,
        ).images[0]
    path = ROOT / name
    image.save(path)
    print(f"[done] {path}", flush=True)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    render_animatediff()
    pipe = static_pipe()
    for emotion in EMOTIONS:
        render_static(pipe, emotion, 0.7, f"static_{emotion}_lw0p70.png")
    for weight in WEIGHTS:
        for emotion in EMOTIONS:
            render_static(pipe, emotion, weight, f"weight_{emotion}_lw{weight:.2f}.png")


if __name__ == "__main__":
    main()
