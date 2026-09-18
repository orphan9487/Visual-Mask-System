"""Level a FaceID portrait using InsightFace eye landmarks.

This utility performs only a global in-place rotation around the midpoint of
the detected eyes.  It does not reshape individual facial features, making it
suitable for a controlled before/after LivePortrait experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "evaluation_results"
    / "henry_faceid_seven_emotions_v1"
    / "selected_henry_neutral_faceid_0.55.png"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "evaluation_results"
    / "henry_liveportrait_eye_alignment_ab"
    / "faceid_neutral_eyes_aligned.png"
)
INSIGHTFACE_ROOT = PROJECT_ROOT / "models" / "insightface"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--max-angle",
        type=float,
        default=5.0,
        help="Refuse unexpectedly large corrections (degrees).",
    )
    parser.add_argument(
        "--measure-only",
        action="store_true",
        help="Report eye geometry without writing a corrected image.",
    )
    parser.add_argument(
        "--extra-angle",
        type=float,
        default=0.0,
        help="Additional visual roll compensation after landmark leveling.",
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


def eye_geometry(face) -> dict[str, object]:
    left_eye = np.asarray(face.kps[0], dtype=np.float64)
    right_eye = np.asarray(face.kps[1], dtype=np.float64)
    delta = right_eye - left_eye
    return {
        "left_eye": left_eye.tolist(),
        "right_eye": right_eye.tolist(),
        "vertical_delta_px": float(delta[1]),
        "horizontal_delta_px": float(delta[0]),
        "angle_degrees": float(np.degrees(np.arctan2(delta[1], delta[0]))),
    }


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source image not found: {source}")

    app = FaceAnalysis(
        name="buffalo_l",
        root=str(INSIGHTFACE_ROOT),
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    image = read_image(source)
    before_face = largest_face(app, image)
    before = eye_geometry(before_face)
    measured_correction = float(before["angle_degrees"])
    if args.measure_only:
        print(
            json.dumps(
                {"source": str(source), "measurement": before},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    correction = measured_correction + args.extra_angle
    if abs(correction) > args.max_angle:
        raise ValueError(
            f"Detected eye angle {correction:.3f} exceeds --max-angle {args.max_angle}"
        )

    eyes = np.asarray(before_face.kps[:2], dtype=np.float64)
    center = tuple(np.mean(eyes, axis=0).tolist())
    matrix = cv2.getRotationMatrix2D(center, correction, 1.0)
    height, width = image.shape[:2]
    aligned = cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    write_image(output, aligned)

    after_face = largest_face(app, aligned)
    after = eye_geometry(after_face)
    report = {
        "source": str(source),
        "output": str(output),
        "method": "global_rotation_about_eye_midpoint",
        "landmark_rotation_degrees": measured_correction,
        "extra_visual_compensation_degrees": args.extra_angle,
        "rotation_applied_degrees": correction,
        "before": before,
        "after_redetected": after,
        "facial_reshaping_applied": False,
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
