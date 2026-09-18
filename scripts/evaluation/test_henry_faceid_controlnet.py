"""Isolated FaceID 0.55 + OpenPose facial-landmark experiment.

This script does not import or modify the LINE/FastAPI renderer. It uses
Henry's own expression photos as structural references, renders face-only
OpenPose maps, then compares several ControlNet strengths.
"""

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
from controlnet_aux import OpenposeDetector
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
CONTROLNET_DIR = PROJECT_ROOT / "models" / "controlnet" / "sd-controlnet-openpose"
ANNOTATOR_CACHE = PROJECT_ROOT / "models" / "controlnet" / "annotator_cache"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_henry_faceid import (  # noqa: E402
    DEFAULT_REFERENCES,
    FACEID_CHECKPOINT,
    INSIGHTFACE_ROOT,
    VAE_DIR,
    analyze_generated_face,
    load_face_embeddings,
)


EXPRESSION_SOURCES = {
    "sadness": Path(
        r"C:\Users\User\OneDrive\桌面\henry_train\PXL_20260527_083739326.RAW-01.jpg"
    ),
    "anger": Path(
        r"C:\Users\User\OneDrive\桌面\henry_train\PXL_20260527_083742033.RAW-01.jpg"
    ),
    "disgust": Path(
        r"C:\Users\User\OneDrive\桌面\henry_train\PXL_20260527_083728188.RAW-01.jpg"
    ),
}

PROMPTS = {
    "sadness": (
        "sad expression, raised inner eyebrows, downcast eyes, downturned mouth, "
        "front-facing portrait photo, young East Asian man, slender oval face, "
        "thin round brown glasses, center-parted black hair, natural skin"
    ),
    "anger": (
        "angry scowl, lowered eyebrows drawn together, glaring eyes, pressed lips, "
        "front-facing portrait photo, young East Asian man, slender oval face, "
        "thin round brown glasses, center-parted black hair, natural skin"
    ),
    "disgust": (
        "disgusted expression, wrinkled nose, raised upper lip, asymmetric squint, "
        "front-facing portrait photo, young East Asian man, slender oval face, "
        "thin round brown glasses, center-parted black hair, natural skin"
    ),
}

NEGATIVE_PROMPT = (
    "round wide face, oversized eyes, wide-set eyes, protruding nose, side view, "
    "hands touching face, multiple faces, deformed, cross-eyed, blurry, low quality, "
    "painting, anime, monochrome, grayscale, sepia"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emotions",
        nargs="+",
        choices=list(EXPRESSION_SOURCES),
        default=list(EXPRESSION_SOURCES),
    )
    parser.add_argument(
        "--control-scales", nargs="+", type=float, default=[0.35, 0.55, 0.75]
    )
    parser.add_argument("--faceid-scale", type=float, default=0.55)
    parser.add_argument("--control-guidance-end", type=float, default=0.75)
    parser.add_argument(
        "--landmark-size",
        type=int,
        default=300,
        help="Longest side of the normalized face-landmark region on a 512 canvas.",
    )
    parser.add_argument("--seed", type=int, default=334105)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=8.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "evaluation_results"
        / "henry_faceid_controlnet_expressions_v1",
    )
    return parser.parse_args()


def read_rgb(path: Path) -> Image.Image:
    bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read image: {path}")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def normalize_landmark_map(control_map: Image.Image, target_size: int) -> Image.Image:
    """Enlarge face-only landmarks into a centered headshot-sized condition."""
    array = np.asarray(control_map.convert("RGB"))
    foreground = np.any(array > 8, axis=2)
    ys, xs = np.where(foreground)
    if not len(xs):
        raise RuntimeError("OpenPose produced an empty face-landmark map")
    pad = 6
    left = max(0, int(xs.min()) - pad)
    top = max(0, int(ys.min()) - pad)
    right = min(array.shape[1], int(xs.max()) + pad + 1)
    bottom = min(array.shape[0], int(ys.max()) + pad + 1)
    crop = Image.fromarray(array[top:bottom, left:right])
    scale = target_size / max(crop.width, crop.height)
    resized = crop.resize(
        (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
        Image.Resampling.NEAREST,
    )
    canvas = Image.new("RGB", (512, 512), "black")
    # Put the landmark center slightly above the canvas center, matching a headshot.
    paste_x = (512 - resized.width) // 2
    paste_y = 225 - resized.height // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def validate_inputs(emotions: list[str]) -> None:
    required = [CONTROLNET_DIR / "config.json", CONTROLNET_DIR / "diffusion_pytorch_model.safetensors"]
    required.extend(EXPRESSION_SOURCES[emotion] for emotion in emotions)
    required.extend(DEFAULT_REFERENCES)
    required.append(FACEID_CHECKPOINT)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing experiment input(s):\n" + "\n".join(missing))


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
    validate_inputs(args.emotions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    control_dir = args.output_dir / "control_maps"
    control_dir.mkdir(exist_ok=True)

    detector = OpenposeDetector.from_pretrained(
        "lllyasviel/ControlNet",
        cache_dir=str(ANNOTATOR_CACHE),
        local_files_only=True,
    )
    control_maps: dict[str, Image.Image] = {}
    for emotion in args.emotions:
        source = read_rgb(EXPRESSION_SOURCES[emotion])
        raw_control_map = detector(
            source,
            detect_resolution=512,
            image_resolution=512,
            include_body=False,
            include_hand=False,
            include_face=True,
        )
        raw_path = control_dir / f"{emotion}_face_landmarks_raw.png"
        raw_control_map.save(raw_path)
        control_map = normalize_landmark_map(raw_control_map, args.landmark_size)
        control_path = control_dir / f"{emotion}_face_landmarks.png"
        control_map.save(control_path)
        control_maps[emotion] = control_map
        print(f"Saved {control_path}", flush=True)

    faceid_embeds, detections, face_app = load_face_embeddings(
        [path.resolve() for path in DEFAULT_REFERENCES]
    )
    reference_mean = faceid_embeds.squeeze(0).mean(dim=0).numpy()
    reference_mean /= np.linalg.norm(reference_mean)

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
    # FaceID Portrait v11 appends 16 tokens for each of five references.
    # Keep those 80 identity tokens out of ControlNet's text cross-attention.
    adapter.pipe.controlnet.set_attn_processor(CNAttnProcessor2_0(num_tokens=80))

    generated: list[dict] = []
    for emotion in args.emotions:
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
                image=control_maps[emotion],
                controlnet_conditioning_scale=control_scale,
                control_guidance_start=0.0,
                control_guidance_end=args.control_guidance_end,
            )[0]
            output_path = args.output_dir / (
                f"henry_{emotion}_faceid_{args.faceid_scale:.2f}_"
                f"control_{control_scale:.2f}.png"
            )
            image.save(output_path)
            similarity, _ = analyze_generated_face(face_app, image, reference_mean)
            generated.append(
                {
                    "emotion": emotion,
                    "faceid_scale": args.faceid_scale,
                    "control_scale": control_scale,
                    "control_guidance_end": args.control_guidance_end,
                    "seed": args.seed,
                    "identity_similarity": similarity,
                    "path": str(output_path),
                }
            )
            print(f"Saved {output_path} (identity={similarity})", flush=True)

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "controlnet": str(CONTROLNET_DIR),
        "faceid_checkpoint": str(FACEID_CHECKPOINT),
        "expression_sources": {
            emotion: str(EXPRESSION_SOURCES[emotion]) for emotion in args.emotions
        },
        "reference_detections": detections,
        "generated": generated,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {manifest_path}", flush=True)

    del adapter, pipe, detector
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
