"""Rank FaceID candidates by identity similarity and eye-line levelness."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from insightface.app import FaceAnalysis

from align_faceid_source_eyes import (
    INSIGHTFACE_ROOT,
    eye_geometry,
    largest_face,
    read_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--identity-window", type=float, default=0.04)
    parser.add_argument("--minimum-identity", type=float, default=0.60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_dir = args.candidate_dir.resolve()
    manifest_path = candidate_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidates = [
        item
        for item in manifest.get("generated", [])
        if item.get("emotion") == "neutral" and item.get("lora_weight") is None
    ]
    if not candidates:
        raise RuntimeError("No neutral FaceID candidates found in manifest")

    app = FaceAnalysis(
        name="buffalo_l",
        root=str(INSIGHTFACE_ROOT),
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    ranked = []
    for candidate in candidates:
        path = Path(candidate["path"]).resolve()
        try:
            geometry = eye_geometry(largest_face(app, read_image(path)))
            detection_error = None
            eye_angle = float(geometry["angle_degrees"])
            vertical_delta = float(geometry["vertical_delta_px"])
        except RuntimeError as error:
            detection_error = str(error)
            eye_angle = None
            vertical_delta = None
        ranked.append(
            {
                "path": str(path),
                "seed": candidate["seed"],
                "identity_similarity": candidate["identity_similarity"],
                "eye_angle_degrees": eye_angle,
                "absolute_eye_angle_degrees": (
                    abs(eye_angle) if eye_angle is not None else None
                ),
                "vertical_eye_delta_px": vertical_delta,
                "detection_error": detection_error,
                "eligible": False,
                "selected": False,
            }
        )

    valid_identities = [
        float(item["identity_similarity"])
        for item in ranked
        if item["identity_similarity"] is not None
        and item["absolute_eye_angle_degrees"] is not None
    ]
    if not valid_identities:
        raise RuntimeError("No candidate has an identity similarity score")
    maximum_identity = max(valid_identities)
    identity_floor = max(args.minimum_identity, maximum_identity - args.identity_window)
    eligible = [
        item
        for item in ranked
        if (item["identity_similarity"] or 0.0) >= identity_floor
        and item["absolute_eye_angle_degrees"] is not None
    ]
    if not eligible:
        raise RuntimeError("No candidate passed the identity threshold")
    for item in eligible:
        item["eligible"] = True

    selected = min(
        eligible,
        key=lambda item: (
            item["absolute_eye_angle_degrees"],
            -float(item["identity_similarity"]),
        ),
    )
    selected["selected"] = True
    ranked.sort(
        key=lambda item: (
            not item["eligible"],
            (
                item["absolute_eye_angle_degrees"]
                if item["absolute_eye_angle_degrees"] is not None
                else float("inf")
            ),
            -float(item["identity_similarity"] or 0.0),
        )
    )

    selected_path = candidate_dir / "selected_neutral_identity_eye_level.png"
    shutil.copy2(selected["path"], selected_path)
    report = {
        "identity_window": args.identity_window,
        "minimum_identity": args.minimum_identity,
        "maximum_identity": maximum_identity,
        "identity_floor": identity_floor,
        "selected_path": str(selected_path),
        "selected": selected,
        "candidates": ranked,
    }
    report_path = candidate_dir / "eye_level_ranking.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
