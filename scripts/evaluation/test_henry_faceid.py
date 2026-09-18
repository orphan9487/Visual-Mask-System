"""Offline Henry identity experiment using IP-Adapter FaceID Portrait v11.

This script is deliberately isolated from the LINE/FastAPI production path.
It extracts five ArcFace embeddings once, loads the existing cached SD1.5
pipeline, and renders a fixed-seed scale grid for visual comparison.
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
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline
from insightface.app import FaceAnalysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LINE_REPO = PROJECT_ROOT.parent / "Visual-Mask-System-line-integration"
IP_ADAPTER_REPO = LINE_REPO / "external" / "IP-Adapter"
FACEID_CHECKPOINT = (
    PROJECT_ROOT
    / "models"
    / "ip_adapter_faceid"
    / "ip-adapter-faceid-portrait-v11_sd15.bin"
)
FACE_EMOTION_MODEL_DIR = PROJECT_ROOT / "models" / "face_emotion_vit"
HENRY_LORA_CHECKPOINT = (
    PROJECT_ROOT / "models" / "henry_mask_lora" / "henrymask_v4-000003.safetensors"
)
INSIGHTFACE_ROOT = PROJECT_ROOT / "models" / "insightface"
VAE_DIR = PROJECT_ROOT / "models" / "sd-vae-mse"

DEFAULT_REFERENCES = (
    Path(r"C:\Users\User\OneDrive\桌面\henry_train\PXL_20240923_005023366.jpg"),
    Path(r"C:\Users\User\OneDrive\桌面\henry_train\PXL_20260527_083700214.RAW-01.jpg"),
    Path(r"C:\Users\User\OneDrive\桌面\henry_train\PXL_20260527_083739326.RAW-01.jpg"),
    Path(r"C:\Users\User\OneDrive\桌面\henry_train\PXL_20260527_083742033.RAW-01.jpg"),
    Path(r"C:\Users\User\OneDrive\桌面\henry_train\PXL_20260527_083743196.RAW-01.jpg"),
)

DEFAULT_PROMPT = (
    "front-facing portrait photo, young East Asian man looking at camera, "
    "slender oval face, almond eyes, thin round brown glasses, center-parted black hair, "
    "natural skin, closed mouth, neutral expression, soft light, plain background"
)
DEFAULT_NEGATIVE_PROMPT = (
    "round wide face, oversized wide-set eyes, protruding nose, pitch-black features, "
    "harsh shadows, side view, tilted head, squinting, broad smile, open mouth, "
    "multiple faces, deformed, cross-eyed, blurry, low quality, painting, anime"
)

EMOTION_PROMPTS = {
    "neutral": "calm neutral expression, closed mouth",
    "joy": "joyful happy face, genuine smile, raised cheeks, bright eyes",
    "sadness": (
        "sad sorrowful unhappy face, sad expression, inner eyebrows raised, "
        "watery downcast eyes, mouth corners down"
    ),
    "anger": (
        "angry furious face, angry expression, eyebrows pulled down and together, "
        "glaring eyes, tight lips, clenched jaw"
    ),
    "surprise": (
        "shocked surprised face, surprised expression, eyebrows lifted high, "
        "eyes wide open, mouth open"
    ),
    "fear": (
        "terrified frightened fearful face, fear expression, eyebrows high and together, "
        "wide eyes, visible sclera, tense open mouth"
    ),
    "disgust": (
        "disgusted repulsed face, disgust expression, deeply wrinkled nose, "
        "upper lip raised, uneven narrowed eyes"
    ),
}

# Stronger, anatomy-focused alternatives for expressions that SD1.5 tends to
# collapse back to neutral when five FaceID references are active.  Keep these
# separate from the original baseline prompts so previous runs remain exactly
# reproducible and opt in with ``--strong-expression``.
STRONG_EMOTION_PROMPTS = {
    "sadness": (
        "(clearly sorrowful sad expression:1.35), inner eyebrows raised and pinched, "
        "drooping upper eyelids, downcast moist eyes, lower lip slightly raised, "
        "mouth corners visibly pulled down"
    ),
    "anger": (
        "(clear angry scowl:1.40), eyebrows deeply lowered and drawn together, "
        "vertical brow furrows, narrowed glaring eyes, tense lower eyelids, "
        "lips firmly pressed, tense jaw"
    ),
    "disgust": (
        "(clear disgusted grimace:1.40), nose strongly wrinkled, upper lip raised "
        "on one side, cheeks raised, lower lip pushed forward, eyes slightly narrowed"
    ),
}

STRONG_EMOTION_NEGATIVE_SUFFIX = {
    "sadness": "neutral face, relaxed eyebrows, straight mouth, smile, happy",
    "anger": "neutral face, relaxed eyebrows, straight mouth, smile, happy, sad",
    "disgust": "neutral face, smooth nose, relaxed upper lip, smile, happy",
}

EMOTION_NEGATIVE_SUFFIX = {
    "neutral": "broad smile, open mouth",
    "joy": "sad, angry, neutral expression",
    "sadness": "smile, happy, angry, neutral expression",
    "anger": "smile, happy, sad, neutral expression",
    "surprise": "smile, neutral expression, squinting",
    "fear": "smile, calm, neutral expression, squinting",
    "disgust": "smile, calm, neutral expression",
}

EMOTION_CLASSIFIER_LABEL = {
    "neutral": "neutral",
    "joy": "happy",
    "sadness": "sad",
    "anger": "angry",
    "surprise": "surprise",
    "fear": "fear",
    "disgust": "disgust",
}

EMOTION_IDENTITY_PROMPT = (
    "{expression}, front-facing symmetrical portrait, young East Asian man "
    "looking at camera, slender oval face, almond eyes, thin round brown glasses, "
    "center-parted black hair, realistic color photo, natural skin tones, plain background"
)

# Short profile for the strong-expression experiment.  The expression anatomy
# stays at the front of CLIP's 77-token window while the essential Henry traits
# still fit; decorative quality/background phrases are intentionally omitted.
STRONG_EMOTION_IDENTITY_PROMPT = (
    "{expression}, front portrait of a young East Asian man, slender oval face, "
    "almond eyes, thin round brown glasses, center-parted black hair, natural photo"
)

EMOTION_NEGATIVE_PROMPT = (
    "round wide face, oversized eyes, wide-set eyes, protruding nose, side view, "
    "tilted head, multiple faces, deformed, cross-eyed, blurry, low quality, "
    "painting, anime, monochrome, grayscale, sepia"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--references",
        nargs=5,
        type=Path,
        default=list(DEFAULT_REFERENCES),
        metavar="IMAGE",
        help="Exactly five Henry reference images.",
    )
    parser.add_argument("--scales", nargs="+", type=float, default=[0.6, 0.8, 1.0])
    parser.add_argument(
        "--lora-weights",
        nargs="*",
        type=float,
        default=[],
        help="Optional Henry v4 Epoch 3 weights. LoRA is fused before FaceID is attached.",
    )
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--emotion-grid",
        action="store_true",
        help="Generate all seven system emotions with identity-safe prompts.",
    )
    parser.add_argument(
        "--strong-expression",
        action="store_true",
        help=(
            "Use anatomy-focused prompts for sadness, anger, and disgust while "
            "leaving the reproducible seven-emotion baseline unchanged."
        ),
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        choices=list(EMOTION_PROMPTS),
        default=list(EMOTION_PROMPTS),
        help="Subset used with --emotion-grid.",
    )
    parser.add_argument(
        "--candidates-per-emotion",
        type=int,
        default=1,
        help="Generate this many deterministic seed candidates for each emotion.",
    )
    parser.add_argument(
        "--emotion-ranker",
        action="store_true",
        help="Rank candidates with the local seven-class facial emotion ViT.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "henry_faceid_portrait_v11",
    )
    return parser.parse_args()


def validate_inputs(references: list[Path]) -> None:
    missing = [str(path) for path in references if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing reference image(s):\n" + "\n".join(missing))
    if not IP_ADAPTER_REPO.is_dir():
        raise FileNotFoundError(f"Official IP-Adapter checkout not found: {IP_ADAPTER_REPO}")
    if not FACEID_CHECKPOINT.is_file():
        raise FileNotFoundError(f"FaceID checkpoint not found: {FACEID_CHECKPOINT}")


def load_face_embeddings(
    references: list[Path],
) -> tuple[torch.Tensor, list[dict], FaceAnalysis]:
    app = FaceAnalysis(
        name="buffalo_l",
        root=str(INSIGHTFACE_ROOT),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    embeddings: list[torch.Tensor] = []
    detections: list[dict] = []
    for path in references:
        # cv2.imread cannot reliably open non-ASCII Windows paths.  Reading
        # bytes with NumPy and decoding them keeps Chinese directory names safe.
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV could not read reference image: {path}")
        faces = app.get(image)
        if not faces:
            raise RuntimeError(f"InsightFace found no face in: {path}")
        face = max(
            faces,
            key=lambda item: float(
                (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
            ),
        )
        embeddings.append(
            torch.from_numpy(face.normed_embedding).unsqueeze(0).unsqueeze(0)
        )
        detections.append(
            {
                "path": str(path),
                "faces_detected": len(faces),
                "selected_bbox": [round(float(value), 2) for value in face.bbox],
                "detection_score": round(float(face.det_score), 6),
            }
        )

    return torch.cat(embeddings, dim=1), detections, app


def analyze_generated_face(
    app: FaceAnalysis, image, reference_mean: np.ndarray
) -> tuple[float | None, object]:
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = app.get(bgr)
    if not faces:
        return None, image
    face = max(
        faces,
        key=lambda item: float(
            (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
        ),
    )
    similarity = round(float(np.dot(face.normed_embedding, reference_mean)), 6)
    x1, y1, x2, y2 = [float(value) for value in face.bbox]
    pad_x = (x2 - x1) * 0.18
    pad_y = (y2 - y1) * 0.18
    left = max(0, round(x1 - pad_x))
    top = max(0, round(y1 - pad_y))
    right = min(image.width, round(x2 + pad_x))
    bottom = min(image.height, round(y2 + pad_y))
    return similarity, image.crop((left, top, right, bottom))


def build_pipeline(lora_weight: float | None = None) -> StableDiffusionPipeline:
    dtype = torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(
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
    if lora_weight is not None:
        if not HENRY_LORA_CHECKPOINT.is_file():
            raise FileNotFoundError(f"Henry LoRA not found: {HENRY_LORA_CHECKPOINT}")
        adapter_name = "henry_epoch3_experiment"
        pipe.load_lora_weights(
            str(HENRY_LORA_CHECKPOINT.parent),
            weight_name=HENRY_LORA_CHECKPOINT.name,
            adapter_name=adapter_name,
        )
        # Portrait v11 installs custom attention processors. Fuse first so the
        # LoRA delta remains in the base weights after FaceID attaches itself.
        pipe.fuse_lora(
            lora_scale=lora_weight,
            safe_fusing=True,
            adapter_names=[adapter_name],
        )
        pipe.unload_lora_weights()
    return pipe


def main() -> None:
    args = parse_args()
    references = [path.resolve() for path in args.references]
    validate_inputs(references)
    if args.candidates_per_emotion < 1:
        raise ValueError("--candidates-per-emotion must be at least 1")

    sys.path.insert(0, str(IP_ADAPTER_REPO))
    from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID

    args.output_dir.mkdir(parents=True, exist_ok=True)
    faceid_embeds, detections, face_app = load_face_embeddings(references)
    reference_mean = faceid_embeds.squeeze(0).mean(dim=0).numpy()
    reference_mean /= np.linalg.norm(reference_mean)
    emotion_ranker = None
    if args.emotion_ranker:
        if not FACE_EMOTION_MODEL_DIR.is_dir():
            raise FileNotFoundError(
                f"Facial emotion model not found: {FACE_EMOTION_MODEL_DIR}"
            )
        from transformers import pipeline as transformers_pipeline

        # Keep the evaluator on CPU so SD1.5, FaceID, and InsightFace retain GPU room.
        emotion_ranker = transformers_pipeline(
            "image-classification",
            model=str(FACE_EMOTION_MODEL_DIR),
            device=-1,
        )
    generated: list[dict] = []
    lora_weights: list[float | None] = args.lora_weights or [None]
    emotion_prompts = dict(EMOTION_PROMPTS)
    emotion_negative_suffix = dict(EMOTION_NEGATIVE_SUFFIX)
    if args.strong_expression:
        emotion_prompts.update(STRONG_EMOTION_PROMPTS)
        emotion_negative_suffix.update(STRONG_EMOTION_NEGATIVE_SUFFIX)

    identity_prompt = (
        STRONG_EMOTION_IDENTITY_PROMPT
        if args.strong_expression
        else EMOTION_IDENTITY_PROMPT
    )
    prompt_variants = (
        [
            (
                emotion,
                identity_prompt.format(expression=expression),
                f"{EMOTION_NEGATIVE_PROMPT}, {emotion_negative_suffix[emotion]}",
            )
            for emotion, expression in emotion_prompts.items()
            if emotion in args.emotions
        ]
        if args.emotion_grid
        else [("custom", args.prompt, args.negative_prompt)]
    )
    for lora_weight in lora_weights:
        pipe = build_pipeline(lora_weight)
        adapter = IPAdapterFaceID(
            pipe,
            str(FACEID_CHECKPOINT),
            "cuda",
            num_tokens=16,
            n_cond=5,
            torch_dtype=torch.float16,
        )
        for scale in args.scales:
            for emotion, base_prompt, negative_prompt in prompt_variants:
                prompt = (
                    base_prompt if lora_weight is None else f"henrymask, {base_prompt}"
                )
                candidate_records: list[dict] = []
                candidate_images = []
                for candidate_index in range(args.candidates_per_emotion):
                    candidate_seed = args.seed + candidate_index * 9973
                    images = adapter.generate(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        faceid_embeds=faceid_embeds,
                        scale=scale,
                        num_samples=1,
                        width=args.width,
                        height=args.height,
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        seed=candidate_seed,
                    )
                    lora_tag = (
                        "none" if lora_weight is None else f"{lora_weight:.2f}"
                    )
                    output_path = args.output_dir / (
                        f"henry_{emotion}_faceid_{scale:.2f}_lora_{lora_tag}_"
                        f"seed_{candidate_seed}.png"
                    )
                    image = images[0]
                    image.save(output_path)
                    similarity, face_crop = analyze_generated_face(
                        face_app, image, reference_mean
                    )
                    emotion_scores: dict[str, float] = {}
                    target_emotion_score = None
                    if emotion_ranker is not None and emotion != "custom":
                        predictions = emotion_ranker(face_crop, top_k=None)
                        emotion_scores = {
                            str(item["label"]).lower(): round(float(item["score"]), 6)
                            for item in predictions
                        }
                        target_emotion_score = emotion_scores.get(
                            EMOTION_CLASSIFIER_LABEL[emotion], 0.0
                        )
                    identity_component = max(similarity or 0.0, 0.0)
                    target_component = target_emotion_score or 0.0
                    selection_score = round(
                        0.35 * identity_component + 0.65 * target_component, 6
                    )
                    record = {
                        "emotion": emotion,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "scale": scale,
                        "lora_weight": lora_weight,
                        "seed": candidate_seed,
                        "identity_similarity": similarity,
                        "target_emotion_score": target_emotion_score,
                        "emotion_scores": emotion_scores,
                        "selection_score": selection_score,
                        "selected": False,
                        "path": str(output_path),
                    }
                    candidate_records.append(record)
                    candidate_images.append(image)
                    print(
                        f"Saved {output_path} (identity={similarity}, "
                        f"emotion={target_emotion_score}, combined={selection_score})",
                        flush=True,
                    )

                eligible = [
                    (index, record)
                    for index, record in enumerate(candidate_records)
                    if (record["identity_similarity"] or 0.0) >= 0.60
                ]
                pool = eligible or list(enumerate(candidate_records))
                best_index, best_record = max(
                    pool,
                    key=lambda item: (
                        item[1]["selection_score"],
                        item[1]["identity_similarity"] or 0.0,
                    ),
                )
                best_record["selected"] = True
                selected_path = args.output_dir / (
                    f"selected_henry_{emotion}_faceid_{scale:.2f}.png"
                )
                candidate_images[best_index].save(selected_path)
                best_record["selected_path"] = str(selected_path)
                generated.extend(candidate_records)
                print(f"Selected {selected_path}", flush=True)
        del adapter, pipe
        gc.collect()
        torch.cuda.empty_cache()

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "checkpoint": str(FACEID_CHECKPOINT),
        "lora_checkpoint": str(HENRY_LORA_CHECKPOINT),
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "strong_expression": args.strong_expression,
        "size": [args.width, args.height],
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "detections": detections,
        "generated": generated,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
