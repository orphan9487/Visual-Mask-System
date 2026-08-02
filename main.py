"""Command-line smoke entry for the transport-independent Visual Mask pipeline."""

from __future__ import annotations

import argparse
import asyncio

from src.services.pipeline_service import pipeline


async def run(text: str, user_id: str, mask_id: str | None) -> None:
    analysis = await pipeline.analyze(
        text,
        history=[],
        user_id=user_id,
        mask_id=mask_id,
    )
    print(
        f"emotion={analysis.emotion} intensity={analysis.base_intensity:.2f} "
        f"mask={analysis.mask_id}"
    )
    result = await pipeline.generate(analysis)
    print(f"output={result.output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Visual Mask inference")
    parser.add_argument("text", nargs="?", default="我今天真的很開心！")
    parser.add_argument("--user-id", default="cli-user")
    parser.add_argument("--mask-id", default=None)
    args = parser.parse_args()
    asyncio.run(run(args.text, args.user_id, args.mask_id))


if __name__ == "__main__":
    main()
