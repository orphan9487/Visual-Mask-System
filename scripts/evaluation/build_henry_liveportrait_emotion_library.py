"""Build Henry's seven-emotion LivePortrait template and strength grid.

The selected images come from the current 15-photo Henry training set.  This
script is evaluation-only: it calls ``apply_liveportrait_expression.py`` and
does not import or modify the LINE/FastAPI production pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADAPTER = Path(__file__).with_name("apply_liveportrait_expression.py")
DEFAULT_DATASET_DIR = Path.home() / "OneDrive" / "桌面" / "henry_train"
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "evaluation_results"
    / "henry_faceid_seven_emotions_v1"
    / "selected_henry_neutral_faceid_0.55.png"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "evaluation_results" / "henry_liveportrait_emotion_library_v1"
)

# These are deliberately explicit so the experiment is reproducible.  The
# mapping is based on visible facial anatomy, not filename order.
DRIVER_FILES = {
    "neutral": "PXL_20240923_005023366.jpg",
    "joy": "PXL_20251127_173818658.jpg",
    "sadness": "PXL_20260527_083739326.RAW-01.jpg",
    "anger": "PXL_20260527_083742033.RAW-01.jpg",
    "surprise": "PXL_20260527_083654253.RAW-01.jpg",
    "fear": "PXL_20260527_083700214.RAW-01.jpg",
    "disgust": "PXL_20260527_083728188.RAW-01.jpg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--emotions",
        nargs="+",
        choices=list(DRIVER_FILES),
        default=list(DRIVER_FILES),
    )
    parser.add_argument("--multipliers", nargs="+", type=float, default=[0.8, 1.0, 1.2])
    return parser.parse_args()


def scale_name(multiplier: float) -> str:
    return f"scale_{multiplier:.2f}".replace(".", "p")


def result_images(output_root: Path, emotion: str) -> list[str]:
    emotion_dir = output_root / emotion
    return sorted(
        str(path.resolve())
        for path in emotion_dir.glob("*.jpg")
        if not path.name.endswith("_concat.jpg")
    )


def run_adapter(
    source: Path,
    driver: Path,
    emotion: str,
    multiplier: float,
    output_root: Path,
) -> None:
    command = [
        sys.executable,
        str(ADAPTER),
        "--source",
        str(source),
        "--driver",
        str(driver),
        "--emotion",
        emotion,
        "--driving-multiplier",
        str(multiplier),
        "--output-dir",
        str(output_root),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not source.is_file():
        raise FileNotFoundError(f"FaceID source image not found: {source}")
    if not ADAPTER.is_file():
        raise FileNotFoundError(f"Expression adapter not found: {ADAPTER}")
    if any(value <= 0 for value in args.multipliers):
        raise ValueError("All multipliers must be greater than zero")

    drivers = {emotion: dataset_dir / DRIVER_FILES[emotion] for emotion in args.emotions}
    missing = [str(path) for path in drivers.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing driving images:\n" + "\n".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    requested_scales = list(dict.fromkeys(args.multipliers))
    preparation_scale = 1.0 if 1.0 in requested_scales else requested_scales[0]
    ordered_scales = [preparation_scale] + [
        scale for scale in requested_scales if scale != preparation_scale
    ]

    records: dict[str, dict] = {}
    for emotion, original_driver in drivers.items():
        records[emotion] = {
            "driver_original": str(original_driver),
            "driver_selection_reason": {
                "neutral": "relaxed frontal face",
                "joy": "raised cheeks and visible smile",
                "sadness": "inner brows raised with compressed downturned mouth",
                "anger": "brows lowered and drawn together",
                "surprise": "rounded open mouth and lifted brows",
                "fear": "wide eyes with strongly lifted brows",
                "disgust": "asymmetric upper lip and nose contraction",
            }[emotion],
            "runs": {},
        }

        template_path: Path | None = None
        for index, multiplier in enumerate(ordered_scales):
            output_root = output_dir / scale_name(multiplier)
            driver = original_driver if index == 0 else template_path
            if driver is None:
                raise RuntimeError(f"Template was not created for {emotion}")

            print(f"\n[{emotion}] expression strength {multiplier:.2f}", flush=True)
            run_adapter(source, driver, emotion, multiplier, output_root)

            if index == 0:
                template_path = output_root / "inputs" / f"{emotion}_driver.pkl"
                if not template_path.is_file():
                    raise RuntimeError(f"LivePortrait template was not created: {template_path}")

            records[emotion]["runs"][str(multiplier)] = {
                "output_root": str(output_root),
                "images": result_images(output_root, emotion),
            }

        records[emotion]["template"] = str(template_path.resolve())

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "offline seven-emotion LivePortrait driver selection",
        "source_faceid_image": str(source),
        "animation_region": "exp",
        "multipliers": requested_scales,
        "emotions": records,
        "production_pipeline_modified": False,
    }
    manifest_path = output_dir / "library_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nLibrary manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
