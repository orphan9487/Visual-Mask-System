"""Compose a generated mask image, TTS audio, and Talking Face MP4."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from src.integrations.talking_face import (
    TalkingFaceAdapter,
    create_talking_face_adapter,
)
from src.integrations.tts import TextToSpeechAdapter, create_tts_adapter
from src.services.pipeline_service import DEFAULT_OUTPUT_ROOT, PipelineResult


class TalkingFacePipelineError(RuntimeError):
    """Raised when the optional image-to-video stage cannot complete."""


class TalkingFacePipelineService:
    def __init__(
        self,
        *,
        tts: TextToSpeechAdapter,
        talking_face: TalkingFaceAdapter,
        output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    ) -> None:
        self.tts = tts
        self.talking_face = talking_face
        self.output_root = Path(output_root).resolve()
        self.jobs_root = self.output_root / "talking_face_jobs"

    async def generate(
        self, text: str, image_result: PipelineResult
    ) -> PipelineResult:
        image_path = image_result.output_path.resolve()
        if not image_path.is_file() or image_path.suffix.lower() != ".png":
            raise TalkingFacePipelineError(
                f"Talking Face requires a generated PNG: {image_path}"
            )
        if not await self.talking_face.health():
            raise TalkingFacePipelineError("Talking Face Sidecar is not ready")

        self.jobs_root.mkdir(parents=True, exist_ok=True)
        job_dir = Path(tempfile.mkdtemp(prefix="job-", dir=self.jobs_root))
        audio_path = job_dir / "speech.mp3"
        video_path = image_path.with_name(f"talking_face_{image_path.stem}.mp4")
        preview_path = video_path.with_name(f"{video_path.stem}_preview.png")
        try:
            tts_result = await self.tts.synthesize(
                text, audio_path, emotion=image_result.emotion
            )
            face_result = await self.talking_face.generate(
                image_path, tts_result.output_path, video_path
            )
            if face_result.output_path.resolve() != video_path.resolve():
                raise TalkingFacePipelineError(
                    "Talking Face adapter returned an unexpected output path"
                )
            await asyncio.to_thread(shutil.copyfile, image_path, preview_path)
            public_path = video_path.relative_to(self.output_root).as_posix()
            return PipelineResult(
                emotion=image_result.emotion,
                mask_id=image_result.mask_id,
                output_path=video_path,
                public_path=public_path,
                base_intensity=image_result.base_intensity,
                effective_intensity=image_result.effective_intensity,
                processing_ms=(
                    image_result.processing_ms
                    + tts_result.processing_ms
                    + face_result.processing_ms
                ),
            )
        except Exception as exc:  # noqa: BLE001
            video_path.unlink(missing_ok=True)
            preview_path.unlink(missing_ok=True)
            if isinstance(exc, TalkingFacePipelineError):
                raise
            raise TalkingFacePipelineError(str(exc)) from exc
        finally:
            await asyncio.to_thread(shutil.rmtree, job_dir, True)


def create_talking_face_pipeline() -> TalkingFacePipelineService:
    return TalkingFacePipelineService(
        tts=create_tts_adapter(),
        talking_face=create_talking_face_adapter(),
    )
