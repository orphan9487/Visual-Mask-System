"""Offline Henry test: LivePortrait expression image -> SD1.5 img2img."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionImg2ImgPipeline
from src.generation.diffusion_engine import SD_MODEL_ID, VAE_DIR
from src.integrations.liveportrait_expression import generate_from_expression_profile

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--emotion', default='anger')
    parser.add_argument('--strength', type=float, default=0.22)
    args = parser.parse_args()
    source = Path(generate_from_expression_profile({'emotion': args.emotion, 'identity': {}}))
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=torch.float16, local_files_only=True, safety_checker=None, feature_extractor=None, requires_safety_checker=False).to('cuda')
    if VAE_DIR.is_dir(): pipe.vae = AutoencoderKL.from_pretrained(str(VAE_DIR), torch_dtype=torch.float16, local_files_only=True).to('cuda')
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    image = Image.open(source).convert('RGB').resize((512, 512))
    result = pipe(prompt=f"realistic portrait, {args.emotion} facial expression, natural skin", negative_prompt='blurry, deformed, changed identity, extra face, cropped face', image=image, strength=args.strength, guidance_scale=5.5, num_inference_steps=25, generator=torch.Generator('cuda').manual_seed(314159)).images[0]
    output = ROOT / 'output' / 'current' / 'instruction_runs' / f'hybrid_{args.emotion}_strength{args.strength:.2f}.png'
    output.parent.mkdir(parents=True, exist_ok=True); result.save(output); print(output)

if __name__ == '__main__': main()
