"""Measure eye-line angles before and after source-level alignment."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from insightface.app import FaceAnalysis

from align_faceid_source_eyes import (
    INSIGHTFACE_ROOT,
    PROJECT_ROOT,
    eye_geometry,
    largest_face,
    read_image,
)


PROFILES = {
    "neutral": 0.8,
    "joy": 0.8,
    "sadness": 1.2,
    "anger": 1.2,
    "surprise": 0.8,
    "fear": 1.0,
    "disgust": 1.0,
}
BASELINE_ROOT = PROJECT_ROOT / "evaluation_results" / "henry_liveportrait_emotion_library_v1"
ALIGNED_ROOT = (
    PROJECT_ROOT
    / "evaluation_results"
    / "henry_liveportrait_eye_alignment_ab"
    / "aligned"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "evaluation_results"
    / "henry_liveportrait_eye_alignment_ab"
    / "comparison_report.json"
)


def scale_name(multiplier: float) -> str:
    return f"scale_{multiplier:.2f}".replace(".", "p")


def find_single_result(directory: Path) -> Path:
    matches = [
        path
        for path in directory.glob("*.jpg")
        if not path.name.endswith("_concat.jpg")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one result in {directory}, found {len(matches)}")
    return matches[0]


def measure(app: FaceAnalysis, path: Path) -> dict[str, object]:
    geometry = eye_geometry(largest_face(app, read_image(path)))
    return {"path": str(path.resolve()), **geometry}


def main() -> int:
    app = FaceAnalysis(
        name="buffalo_l",
        root=str(INSIGHTFACE_ROOT),
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=-1, det_size=(640, 640))

    results: dict[str, dict] = {}
    for emotion, multiplier in PROFILES.items():
        baseline = find_single_result(
            BASELINE_ROOT / scale_name(multiplier) / emotion
        )
        aligned = find_single_result(ALIGNED_ROOT / emotion)
        before = measure(app, baseline)
        after = measure(app, aligned)
        results[emotion] = {
            "multiplier": multiplier,
            "before": before,
            "after": after,
            "absolute_angle_before": abs(float(before["angle_degrees"])),
            "absolute_angle_after": abs(float(after["angle_degrees"])),
            "absolute_angle_change": (
                abs(float(after["angle_degrees"]))
                - abs(float(before["angle_degrees"]))
            ),
        }

    mean_before = sum(item["absolute_angle_before"] for item in results.values()) / len(results)
    mean_after = sum(item["absolute_angle_after"] for item in results.values()) / len(results)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "InsightFace five-point eye centers",
        "mean_absolute_angle_before": mean_before,
        "mean_absolute_angle_after": mean_after,
        "mean_absolute_angle_change": mean_after - mean_before,
        "results": results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
