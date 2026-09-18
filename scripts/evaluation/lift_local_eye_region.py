"""Lift one eye and its glasses region with a smooth local displacement field.

The operation is deterministic and evaluation-only.  Unlike global roll
correction, it leaves the shoulders, clothing, background, nose, and mouth in
place.  ``subject-right`` means the person's right eye (image-left).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INSIGHTFACE_ROOT = PROJECT_ROOT / "models" / "insightface"
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "evaluation_results"
    / "henry_liveportrait_eye_alignment_ab"
    / "faceid_neutral_eyes_aligned.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eye", choices=("subject-right", "subject-left"), default="subject-right")
    parser.add_argument("--lift-px", type=float, default=3.0)
    parser.add_argument("--radius-x", type=float, default=62.0)
    parser.add_argument("--radius-y", type=float, default=38.0)
    parser.add_argument(
        "--center-y-offset",
        type=float,
        default=-2.0,
        help="Move the deformation center upward to include the frame and brow.",
    )
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not read: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix or ".png", image)
    if not success:
        raise RuntimeError(f"OpenCV could not encode: {path}")
    encoded.tofile(path)


def largest_face(app: FaceAnalysis, image: np.ndarray):
    faces = app.get(image)
    if not faces:
        raise RuntimeError("InsightFace did not detect a face")
    return max(
        faces,
        key=lambda face: float(
            (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
        ),
    )


def measured_eyes(face) -> dict[str, object]:
    eyes = np.asarray(face.kps[:2], dtype=np.float64)
    eyes = eyes[np.argsort(eyes[:, 0])]
    image_left, image_right = eyes
    delta = image_right - image_left
    return {
        "image_left_eye": image_left.tolist(),
        "image_right_eye": image_right.tolist(),
        "vertical_delta_px": float(delta[1]),
        "angle_degrees": float(np.degrees(np.arctan2(delta[1], delta[0]))),
    }


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source image not found: {source}")
    if args.lift_px <= 0 or args.radius_x <= 0 or args.radius_y <= 0:
        raise ValueError("Lift and radius values must be greater than zero")

    app = FaceAnalysis(
        name="buffalo_l",
        root=str(INSIGHTFACE_ROOT),
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    image = read_image(source)
    face = largest_face(app, image)
    eyes = np.asarray(face.kps[:2], dtype=np.float32)
    eyes = eyes[np.argsort(eyes[:, 0])]
    # A frontal portrait mirrors anatomy: the person's right eye is image-left.
    target = eyes[0] if args.eye == "subject-right" else eyes[1]
    center_x = float(target[0])
    center_y = float(target[1] + args.center_y_offset)

    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    distance = (
        ((grid_x - center_x) / args.radius_x) ** 2
        + ((grid_y - center_y) / args.radius_y) ** 2
    )
    weight = np.exp(-2.0 * distance).astype(np.float32)

    # To move source content upward, each destination pixel samples slightly
    # lower in the original image.  The Gaussian field makes the boundary
    # continuous and keeps the bridge/temple connection from forming a seam.
    map_x = grid_x
    map_y = grid_y + np.float32(args.lift_px) * weight
    corrected = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    write_image(output, corrected)

    after_face = largest_face(app, corrected)
    report = {
        "source": str(source),
        "output": str(output),
        "eye": args.eye,
        "lift_px": args.lift_px,
        "radius": [args.radius_x, args.radius_y],
        "center": [center_x, center_y],
        "before": measured_eyes(face),
        "after_redetected": measured_eyes(after_face),
        "global_rotation_applied": False,
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
