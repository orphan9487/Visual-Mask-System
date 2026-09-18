"""Apply a reference expression to a FaceID portrait with LivePortrait.

This is an offline evaluation adapter.  It deliberately does not import or
modify the LINE/FastAPI image pipeline.  Identity is supplied by an existing
FaceID image; LivePortrait only transfers expression motion from the driving
image.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIVEPORTRAIT_DIR = PROJECT_ROOT / "external" / "LivePortrait"
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "evaluation_results"
    / "henry_faceid_seven_emotions_v1"
    / "selected_henry_neutral_faceid_0.55.png"
)
EMOTIONS = ("neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--emotion", choices=EMOTIONS, required=True)
    parser.add_argument("--driving-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation_results" / "henry_faceid_liveportrait",
    )
    parser.add_argument("--liveportrait-dir", type=Path, default=DEFAULT_LIVEPORTRAIT_DIR)
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="Disable FP16 if the generated image contains black regions.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.source.is_file():
        raise FileNotFoundError(f"Source portrait not found: {args.source}")
    if not args.driver.is_file():
        raise FileNotFoundError(f"Driving image/template not found: {args.driver}")
    if args.driving_multiplier <= 0:
        raise ValueError("--driving-multiplier must be greater than zero")
    if not (args.liveportrait_dir / "inference.py").is_file():
        raise FileNotFoundError(
            f"Official LivePortrait checkout not found: {args.liveportrait_dir}"
        )


def resolve_python(liveportrait_dir: Path) -> Path:
    candidates = (
        liveportrait_dir / ".venv" / "Scripts" / "python.exe",
        liveportrait_dir / ".venv" / "bin" / "python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "LivePortrait virtual-environment Python was not found under "
        f"{liveportrait_dir / '.venv'}"
    )


def prepare_driver(driver: Path, emotion: str, input_dir: Path) -> Path:
    """Copy an image driver to an ASCII path that OpenCV can read on Windows."""
    if driver.suffix.lower() == ".pkl":
        return driver.resolve()

    input_dir.mkdir(parents=True, exist_ok=True)
    safe_driver = input_dir / f"{emotion}_driver{driver.suffix.lower()}"
    if driver.resolve() != safe_driver.resolve():
        shutil.copy2(driver, safe_driver)
    return safe_driver.resolve()


def main() -> int:
    args = parse_args()
    validate_args(args)

    run_dir = args.output_dir.resolve() / args.emotion
    input_dir = args.output_dir.resolve() / "inputs"
    run_dir.mkdir(parents=True, exist_ok=True)
    safe_driver = prepare_driver(args.driver, args.emotion, input_dir)
    python = resolve_python(args.liveportrait_dir.resolve())

    command = [
        str(python),
        "inference.py",
        "--source",
        str(args.source.resolve()),
        "--driving",
        str(safe_driver),
        "--output-dir",
        str(run_dir),
        "--animation-region",
        "exp",
        "--driving-option",
        "expression-friendly",
        "--driving-multiplier",
        str(args.driving_multiplier),
        "--no-flag-do-torch-compile",
        (
            "--no-flag-use-half-precision"
            if args.full_precision
            else "--flag-use-half-precision"
        ),
    ]

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    subprocess.run(
        command,
        cwd=args.liveportrait_dir.resolve(),
        env=environment,
        check=True,
    )

    outputs = sorted(
        str(path.resolve())
        for path in run_dir.glob(f"{args.source.stem}--{safe_driver.stem}*.jpg")
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "faceid_identity_liveportrait_expression_only",
        "source": str(args.source.resolve()),
        "driver_original": str(args.driver.resolve()),
        "driver_safe_copy": str(safe_driver),
        "emotion": args.emotion,
        "animation_region": "exp",
        "driving_option": "expression-friendly",
        "driving_multiplier": args.driving_multiplier,
        "half_precision": not args.full_precision,
        "liveportrait_dir": str(args.liveportrait_dir.resolve()),
        "outputs": outputs,
        "production_pipeline_modified": False,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Expression result directory: {run_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
