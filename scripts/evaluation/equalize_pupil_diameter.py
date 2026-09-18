"""Equalize one pupil without moving the iris, eyelids, or surrounding face.

The correction uses a radial remap inside a single iris.  It expands the dark
pupil while compressing the surrounding iris ring, keeping the iris boundary
fixed.  This is intended for controlled, offline LivePortrait cache cleanup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--center-x", type=float, required=True)
    parser.add_argument("--center-y", type=float, required=True)
    parser.add_argument("--current-radius", type=float, required=True)
    parser.add_argument("--target-radius", type=float, required=True)
    parser.add_argument("--iris-radius", type=float, required=True)
    parser.add_argument("--feather", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.input)
    if not 0 < args.current_radius < args.target_radius < args.iris_radius:
        raise ValueError(
            "Expected 0 < current-radius < target-radius < iris-radius"
        )

    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dx = xx - args.center_x
    dy = yy - args.center_y
    radius = np.sqrt(dx * dx + dy * dy)

    source_radius = radius.copy()
    pupil = radius <= args.target_radius
    source_radius[pupil] = (
        radius[pupil] * args.current_radius / args.target_radius
    )

    iris_ring = (radius > args.target_radius) & (radius < args.iris_radius)
    source_radius[iris_ring] = args.current_radius + (
        (radius[iris_ring] - args.target_radius)
        * (args.iris_radius - args.current_radius)
        / (args.iris_radius - args.target_radius)
    )

    safe_radius = np.maximum(radius, 1e-6)
    map_x = args.center_x + dx * source_radius / safe_radius
    map_y = args.center_y + dy * source_radius / safe_radius
    remapped = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    inner = max(args.iris_radius - args.feather, 0.0)
    alpha = np.clip((args.iris_radius - radius) / max(args.feather, 1e-6), 0, 1)
    alpha[radius <= inner] = 1.0
    alpha = alpha[..., None]
    result = np.rint(remapped * alpha + image * (1.0 - alpha)).astype(np.uint8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), result):
        raise OSError(f"Could not write {args.output}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
