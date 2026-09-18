"""FaceID 0.55 with face-only connected landmarks through Canny ControlNet.

Unlike the earlier OpenPose experiment, this script does not send sparse face
points to an OpenPose ControlNet (which can interpret them as people/body
poses).  It connects only the jaw, brows, eyes, nose, and lips and sends that
clean geometry to the Canny model.  The production LINE and LoRA paths are not
imported or modified.
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
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LINE_REPO = PROJECT_ROOT.parent / "Visual-Mask-System-line-integration"
IP_ADAPTER_REPO = LINE_REPO / "external" / "IP-Adapter"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_henry_faceid import (  # noqa: E402
    DEFAULT_REFERENCES,
    FACEID_CHECKPOINT,
    STRONG_EMOTION_IDENTITY_PROMPT,
    STRONG_EMOTION_NEGATIVE_SUFFIX,
    STRONG_EMOTION_PROMPTS,
    EMOTION_NEGATIVE_PROMPT,
    analyze_generated_face,
    load_face_embeddings,
)
from test_henry_faceid_canny import SOURCES, build_pipeline  # noqa: E402
from test_henry_faceid_controlnet import ANNOTATOR_CACHE, read_rgb  # noqa: E402


# Standard OpenPose 70-point face topology.  Pupils (68, 69) remain dots.
FACE_CHAINS = (
    tuple(range(0, 17)),
    tuple(range(17, 22)),
    tuple(range(22, 27)),
    tuple(range(27, 31)),
    tuple(range(31, 36)),
    (36, 37, 38, 39, 40, 41, 36),
    (42, 43, 44, 45, 46, 47, 42),
    (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 48),
    (60, 61, 62, 63, 64, 65, 66, 67, 60),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emotions", nargs="+", choices=sorted(SOURCES), default=sorted(SOURCES)
    )
    parser.add_argument("--faceid-scale", type=float, default=0.55)
    parser.add_argument("--control-scales", nargs="+", type=float, default=[0.35, 0.55])
    parser.add_argument(
        "--expression-gains",
        nargs="+",
        type=float,
        default=[1.0, 1.35],
        help="1.0 preserves detected landmarks; values above 1 amplify key features.",
    )
    parser.add_argument("--control-guidance-end", type=float, default=0.70)
    parser.add_argument("--landmark-size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=334105)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=8.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT
        / "evaluation_results"
        / "henry_faceid_landmark_canny_v1",
    )
    return parser.parse_args()


def _amplify_expression(points: np.ndarray, emotion: str, gain: float) -> np.ndarray:
    """Apply small, explicit landmark shifts without changing head shape."""
    result = points.copy()
    amount = max(0.0, gain - 1.0)
    if not amount:
        return result
    face_w = float(np.nanmax(points[:, 0]) - np.nanmin(points[:, 0]))
    face_h = float(np.nanmax(points[:, 1]) - np.nanmin(points[:, 1]))
    if emotion == "sadness":
        # Lift inner brows and lower the mouth corners.
        result[[20, 21, 22, 23], 1] -= 0.035 * face_h * amount
        result[[48, 54], 1] += 0.055 * face_h * amount
    elif emotion == "anger":
        # Pull inner brows down and toward the nose to create a clear V/scowl.
        result[[20, 21, 22, 23], 1] += 0.045 * face_h * amount
        result[[20, 21], 0] += 0.025 * face_w * amount
        result[[22, 23], 0] -= 0.025 * face_w * amount
    elif emotion == "disgust":
        # Raise one side of the upper lip and nose for an asymmetric sneer.
        result[[49, 50, 51, 52, 53], 1] -= 0.055 * face_h * amount
        result[[31, 32, 33], 1] -= 0.025 * face_h * amount
        result[[48, 49, 50], 1] -= 0.035 * face_h * amount
    return result


def make_landmark_edges(
    detector: OpenposeDetector,
    source: Image.Image,
    emotion: str,
    target_size: int,
    expression_gain: float,
) -> Image.Image:
    resized = source.convert("RGB")
    resized.thumbnail((768, 768), Image.Resampling.LANCZOS)
    poses = detector.detect_poses(
        np.asarray(resized), include_hand=False, include_face=True
    )
    poses = [pose for pose in poses if pose.face and len(pose.face) >= 68]
    if not poses:
        raise RuntimeError(f"OpenPose found no usable face for {emotion}")
    pose = max(
        poses,
        key=lambda item: sum(
            point is not None and point.x > 0 and point.y > 0 for point in item.face
        ),
    )
    points = np.array(
        [
            (point.x, point.y) if point is not None else (np.nan, np.nan)
            for point in pose.face
        ],
        dtype=np.float32,
    )
    points = _amplify_expression(points, emotion, expression_gain)

    valid = np.isfinite(points).all(axis=1) & (points[:, 0] > 0) & (points[:, 1] > 0)
    if valid.sum() < 60:
        raise RuntimeError(f"Only {valid.sum()} valid face landmarks for {emotion}")
    minimum = points[valid].min(axis=0)
    maximum = points[valid].max(axis=0)
    center = (minimum + maximum) / 2
    scale = target_size / float((maximum - minimum).max())
    canvas_points = (points - center) * scale + np.array([256.0, 225.0])

    canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    for chain in FACE_CHAINS:
        chain_points = [
            canvas_points[index]
            for index in chain
            if index < len(canvas_points) and valid[index]
        ]
        if len(chain_points) >= 2:
            cv2.polylines(
                canvas,
                [np.rint(chain_points).astype(np.int32)],
                isClosed=False,
                color=(255, 255, 255),
                thickness=3,
                lineType=cv2.LINE_AA,
            )
    for index in (68, 69):
        if index < len(canvas_points) and valid[index]:
            cv2.circle(
                canvas,
                tuple(np.rint(canvas_points[index]).astype(int)),
                3,
                (255, 255, 255),
                -1,
                cv2.LINE_AA,
            )
    return Image.fromarray(canvas)


def main() -> None:
    args = parse_args()
    required = [FACEID_CHECKPOINT, *DEFAULT_REFERENCES]
    required.extend(SOURCES[emotion] for emotion in args.emotions)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing experiment input(s):\n" + "\n".join(missing))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    control_dir = args.output_dir / "control_maps"
    control_dir.mkdir(exist_ok=True)
    detector = OpenposeDetector.from_pretrained(
        "lllyasviel/ControlNet",
        cache_dir=str(ANNOTATOR_CACHE),
        local_files_only=True,
    )
    detector.to("cuda")

    faceid_embeds, detections, face_app = load_face_embeddings(
        [path.resolve() for path in DEFAULT_REFERENCES]
    )
    reference_mean = faceid_embeds.squeeze(0).mean(dim=0).numpy()
    reference_mean /= np.linalg.norm(reference_mean)

    controls: dict[tuple[str, float], Image.Image] = {}
    for emotion in args.emotions:
        source = read_rgb(SOURCES[emotion])
        for gain in args.expression_gains:
            control = make_landmark_edges(
                detector, source, emotion, args.landmark_size, gain
            )
            path = control_dir / f"{emotion}_landmarks_gain_{gain:.2f}.png"
            control.save(path)
            controls[(emotion, gain)] = control
            print(f"Saved {path}", flush=True)
    del detector
    gc.collect()
    torch.cuda.empty_cache()

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
    for (emotion, gain), control in controls.items():
        prompt = STRONG_EMOTION_IDENTITY_PROMPT.format(
            expression=STRONG_EMOTION_PROMPTS[emotion]
        )
        negative = (
            f"{EMOTION_NEGATIVE_PROMPT}, "
            f"{STRONG_EMOTION_NEGATIVE_SUFFIX[emotion]}"
        )
        for control_scale in args.control_scales:
            image = adapter.generate(
                prompt=prompt,
                negative_prompt=negative,
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
                f"landmark_gain_{gain:.2f}_control_{control_scale:.2f}.png"
            )
            image.save(output_path)
            similarity, _ = analyze_generated_face(face_app, image, reference_mean)
            generated.append(
                {
                    "emotion": emotion,
                    "expression_gain": gain,
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
