"""Isolated FaceID 0.55 + cropped expression img2img experiment."""

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
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionImg2ImgPipeline
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LINE_REPO = PROJECT_ROOT.parent / "Visual-Mask-System-line-integration"
IP_ADAPTER_REPO = LINE_REPO / "external" / "IP-Adapter"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_henry_faceid import (  # noqa: E402
    DEFAULT_REFERENCES,
    FACEID_CHECKPOINT,
    VAE_DIR,
    analyze_generated_face,
    load_face_embeddings,
)


SOURCE_VARIANTS = {
    "sadness_original": (
        "sadness",
        Path(
            r"C:\Users\User\OneDrive\桌面\henry_train\PXL_20260527_083739326.RAW-01.jpg"
        ),
    ),
    "sadness_expressive": (
        "sadness",
        PROJECT_ROOT
        / "evaluation_results"
        / "henry_faceid_targeted_emotions_v6"
        / "henry_sadness_faceid_0.55_lora_none_seed_314159.png",
    ),
    "anger_original": (
        "anger",
        Path(
            r"C:\Users\User\OneDrive\桌面\henry_train\PXL_20260527_083742033.RAW-01.jpg"
        ),
    ),
    "anger_expressive": (
        "anger",
        PROJECT_ROOT
        / "evaluation_results"
        / "henry_faceid_targeted_emotions_v6"
        / "henry_anger_faceid_0.55_lora_none_seed_314159.png",
    ),
    "disgust_original": (
        "disgust",
        Path(
            r"C:\Users\User\OneDrive\桌面\henry_train\PXL_20260527_083728188.RAW-01.jpg"
        ),
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
    "hands, fingers, multiple faces, deformed, blurry, low quality, painting, anime, "
    "monochrome, grayscale, sepia"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.35, 0.50, 0.65])
    parser.add_argument("--faceid-scale", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=334105)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=8.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "evaluation_results"
        / "henry_faceid_img2img_expressions_v1",
    )
    return parser.parse_args()


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def square_face_crop(face_app, path: Path) -> Image.Image:
    bgr = read_bgr(path)
    faces = face_app.get(bgr)
    if not faces:
        raise RuntimeError(f"No face found in expression source: {path}")
    face = max(
        faces,
        key=lambda item: float(
            (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
        ),
    )
    x1, y1, x2, y2 = [float(value) for value in face.bbox]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * 1.65
    left = max(0, round(center_x - side / 2))
    top = max(0, round(center_y - side / 2))
    right = min(bgr.shape[1], round(center_x + side / 2))
    bottom = min(bgr.shape[0], round(center_y + side / 2))
    rgb = cv2.cvtColor(bgr[top:bottom, left:right], cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).resize((512, 512), Image.Resampling.LANCZOS)


def build_pipeline() -> StableDiffusionImg2ImgPipeline:
    dtype = torch.float16
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
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
    missing = [str(path) for _, path in SOURCE_VARIANTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source image(s):\n" + "\n".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = args.output_dir / "source_crops"
    crop_dir.mkdir(exist_ok=True)

    faceid_embeds, detections, face_app = load_face_embeddings(
        [path.resolve() for path in DEFAULT_REFERENCES]
    )
    reference_mean = faceid_embeds.squeeze(0).mean(dim=0).numpy()
    reference_mean /= np.linalg.norm(reference_mean)

    source_crops: dict[str, Image.Image] = {}
    for variant, (_, source_path) in SOURCE_VARIANTS.items():
        crop = square_face_crop(face_app, source_path)
        crop_path = crop_dir / f"{variant}.png"
        crop.save(crop_path)
        source_crops[variant] = crop
        print(f"Saved {crop_path}", flush=True)

    sys.path.insert(0, str(IP_ADAPTER_REPO))
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

    generated: list[dict] = []
    for variant, (emotion, source_path) in SOURCE_VARIANTS.items():
        for strength in args.strengths:
            image = adapter.generate(
                prompt=PROMPTS[emotion],
                negative_prompt=NEGATIVE_PROMPT,
                faceid_embeds=faceid_embeds,
                scale=args.faceid_scale,
                num_samples=1,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                image=source_crops[variant],
                strength=strength,
            )[0]
            output_path = args.output_dir / (
                f"henry_{variant}_faceid_{args.faceid_scale:.2f}_"
                f"strength_{strength:.2f}.png"
            )
            image.save(output_path)
            similarity, _ = analyze_generated_face(face_app, image, reference_mean)
            generated.append(
                {
                    "variant": variant,
                    "emotion": emotion,
                    "source": str(source_path),
                    "strength": strength,
                    "faceid_scale": args.faceid_scale,
                    "seed": args.seed,
                    "identity_similarity": similarity,
                    "path": str(output_path),
                }
            )
            print(f"Saved {output_path} (identity={similarity})", flush=True)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "faceid_checkpoint": str(FACEID_CHECKPOINT),
        "reference_detections": detections,
        "generated": generated,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {manifest_path}", flush=True)

    del adapter, pipe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
