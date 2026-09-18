"""Isolated FaceID 0.55 + face-cropped Canny ControlNet experiment."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
)
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LINE_REPO = PROJECT_ROOT.parent / "Visual-Mask-System-line-integration"
IP_ADAPTER_REPO = LINE_REPO / "external" / "IP-Adapter"
CONTROLNET_DIR = PROJECT_ROOT / "models" / "controlnet" / "sd-controlnet-canny"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_henry_faceid import (  # noqa: E402
    DEFAULT_REFERENCES,
    FACEID_CHECKPOINT,
    VAE_DIR,
    analyze_generated_face,
    load_face_embeddings,
)
from test_henry_faceid_img2img import square_face_crop  # noqa: E402


SOURCES = {
    "sadness": Path(
        r"C:\Users\User\OneDrive\桌面\henry_train\PXL_20260527_083739326.RAW-01.jpg"
    ),
    "anger": PROJECT_ROOT
    / "evaluation_results"
    / "henry_faceid_targeted_emotions_v6"
    / "henry_anger_faceid_0.55_lora_none_seed_314159.png",
    "disgust": Path(
        r"C:\Users\User\OneDrive\桌面\henry_train\PXL_20260527_083728188.RAW-01.jpg"
    ),
}

PROMPTS = {
    "sadness": (
        "sad face, raised inner eyebrows, downcast eyes, downturned mouth, "
        "front-facing close portrait photo, young East Asian man, slender oval face, "
        "round brown glasses, center-parted black hair, natural skin color"
    ),
    "anger": (
        "angry scowl, lowered eyebrows drawn together, glaring eyes, pressed lips, "
        "front-facing close portrait photo, young East Asian man, slender oval face, "
        "round brown glasses, center-parted black hair, natural skin color"
    ),
    "disgust": (
        "disgusted face, wrinkled nose, raised upper lip, asymmetric squint, "
        "front-facing close portrait photo, young East Asian man, slender oval face, "
        "round brown glasses, center-parted black hair, natural skin color"
    ),
}

NEGATIVE_PROMPT = (
    "round wide face, oversized eyes, wide-set eyes, protruding nose, side view, "
    "hands, multiple faces, deformed, blurry, low quality, painting, anime, "
    "monochrome, grayscale, sepia"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-scales", nargs="+", type=float, default=[0.35, 0.55])
    parser.add_argument("--faceid-scale", type=float, default=0.55)
    parser.add_argument("--control-guidance-end", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=334105)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=8.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "evaluation_results"
        / "henry_faceid_canny_expressions_v1",
    )
    return parser.parse_args()


def make_canny(source: Image.Image) -> Image.Image:
    rgb = np.asarray(source.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return Image.fromarray(np.repeat(edges[:, :, None], 3, axis=2))


def build_pipeline() -> StableDiffusionControlNetPipeline:
    dtype = torch.float16
    controlnet = ControlNetModel.from_pretrained(
        str(CONTROLNET_DIR), torch_dtype=dtype, local_files_only=True
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        local_files_only=True,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    if VAE_DIR.is_dir():
        pipe.vae = AutoencoderKL.from_pretrained(
            str(VAE_DIR), torch_dtype=dtype, local_files_only=True
        )
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    pipe.enable_vae_slicing()
    return pipe


def main() -> None:
    args = parse_args()
    required = [
        CONTROLNET_DIR / "config.json",
        CONTROLNET_DIR / "diffusion_pytorch_model.safetensors",
        FACEID_CHECKPOINT,
        *DEFAULT_REFERENCES,
        *SOURCES.values(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing experiment input(s):\n" + "\n".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    control_dir = args.output_dir / "control_maps"
    control_dir.mkdir(exist_ok=True)

    faceid_embeds, detections, face_app = load_face_embeddings(
        [path.resolve() for path in DEFAULT_REFERENCES]
    )
    reference_mean = faceid_embeds.squeeze(0).mean(dim=0).numpy()
    reference_mean /= np.linalg.norm(reference_mean)

    controls: dict[str, Image.Image] = {}
    for emotion, source_path in SOURCES.items():
        source_crop = square_face_crop(face_app, source_path)
        source_crop.save(control_dir / f"{emotion}_source_crop.png")
        control = make_canny(source_crop)
        control.save(control_dir / f"{emotion}_canny.png")
        controls[emotion] = control

    sys.path.insert(0, str(IP_ADAPTER_REPO))
    from ip_adapter.attention_processor import CNAttnProcessor2_0
    from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID

    pipe = build_pipeline()
    adapter = IPAdapterFaceID(
        pipe,
        str(FACEID_CHECKPOINT),
        "cuda",
        num_tokens=16,
        n_cond=5,
        torch_dtype=torch.float16,
    )
    adapter.pipe.controlnet.set_attn_processor(CNAttnProcessor2_0(num_tokens=80))

    generated: list[dict] = []
    for emotion, control in controls.items():
        for control_scale in args.control_scales:
            image = adapter.generate(
                prompt=PROMPTS[emotion],
                negative_prompt=NEGATIVE_PROMPT,
                faceid_embeds=faceid_embeds,
                scale=args.faceid_scale,
                num_samples=1,
                width=512,
                height=512,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                image=control,
                controlnet_conditioning_scale=control_scale,
                control_guidance_start=0.0,
                control_guidance_end=args.control_guidance_end,
            )[0]
            output_path = args.output_dir / (
                f"henry_{emotion}_faceid_{args.faceid_scale:.2f}_"
                f"canny_{control_scale:.2f}.png"
            )
            image.save(output_path)
            similarity, _ = analyze_generated_face(face_app, image, reference_mean)
            generated.append(
                {
                    "emotion": emotion,
                    "source": str(SOURCES[emotion]),
                    "faceid_scale": args.faceid_scale,
                    "control_scale": control_scale,
                    "control_guidance_end": args.control_guidance_end,
                    "seed": args.seed,
                    "identity_similarity": similarity,
                    "path": str(output_path),
                }
            )
            print(f"Saved {output_path} (identity={similarity})", flush=True)

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().astimezone().isoformat(),
                "controlnet": str(CONTROLNET_DIR),
                "faceid_checkpoint": str(FACEID_CHECKPOINT),
                "reference_detections": detections,
                "generated": generated,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {manifest_path}", flush=True)

    del adapter, pipe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
