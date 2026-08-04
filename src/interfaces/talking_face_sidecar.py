"""FastAPI Sidecar exposing the partner Wav2Lip code as a black-box service."""

from __future__ import annotations

import asyncio
import secrets
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Protocol

from fastapi import FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.responses import Response

from src.config import talking_face as config
from src.integrations.talking_face_partner import PartnerWav2LipBackend


class TalkingFaceBackend(Protocol):
    async def generate(
        self, face_path: Path, audio_path: Path, output_path: Path
    ) -> None: ...

    async def close(self) -> None: ...


async def _save_upload(upload: UploadFile, path: Path, limit: int) -> None:
    size = 0
    try:
        with path.open("wb") as destination:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"{upload.filename or 'upload'} exceeds input limit",
                    )
                destination.write(chunk)
    finally:
        await upload.close()
    if size == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{upload.filename or 'upload'} is empty",
        )


def _suffix(upload: UploadFile, allowed: set[str], label: str) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Unsupported {label} file type",
        )
    return suffix


def _authorise(authorization: str | None, token: str) -> None:
    if not token:
        return
    expected = f"Bearer {token}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Talking Face token",
        )


def create_app(
    backend: TalkingFaceBackend | None = None,
    *,
    token: str | None = None,
    work_dir: str | Path | None = None,
    max_input_bytes: int | None = None,
) -> FastAPI:
    configured_token = config.TALKING_FACE_TOKEN if token is None else token
    jobs_root = Path(work_dir or config.TALKING_FACE_WORK_DIR).resolve()
    input_limit = max_input_bytes or config.TALKING_FACE_MAX_INPUT_BYTES
    generation_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_backend = backend
        if active_backend is None:
            if not config.TALKING_FACE_ROOT:
                raise RuntimeError("VMS_TALKING_FACE_ROOT is not configured")
            active_backend = await PartnerWav2LipBackend.load(
                config.TALKING_FACE_ROOT
            )
        jobs_root.mkdir(parents=True, exist_ok=True)
        app.state.backend = active_backend
        try:
            yield
        finally:
            await active_backend.close()

    app = FastAPI(title="Visual Mask Talking Face Sidecar", lifespan=lifespan)

    @app.get("/health")
    async def health(authorization: str | None = Header(default=None)) -> dict:
        _authorise(authorization, configured_token)
        return {"ok": True, "backend": "partner-wav2lip"}

    @app.post("/generate")
    async def generate(
        face: UploadFile = File(...),
        audio: UploadFile = File(...),
        authorization: str | None = Header(default=None),
    ) -> Response:
        _authorise(authorization, configured_token)
        face_suffix = _suffix(face, {".png", ".jpg", ".jpeg"}, "face")
        audio_suffix = _suffix(audio, {".wav", ".mp3", ".m4a", ".mp4"}, "audio")
        job_dir = Path(tempfile.mkdtemp(prefix="job-", dir=jobs_root))
        face_path = job_dir / f"face{face_suffix}"
        audio_path = job_dir / f"audio{audio_suffix}"
        output_path = job_dir / "result.mp4"
        try:
            await _save_upload(face, face_path, input_limit)
            await _save_upload(audio, audio_path, input_limit)
            async with generation_lock:
                await app.state.backend.generate(
                    face_path, audio_path, output_path
                )
            video = await asyncio.to_thread(output_path.read_bytes)
            if len(video) < 12 or video[4:8] != b"ftyp":
                raise RuntimeError("Partner system returned invalid MP4 data")
            if len(video) > config.TALKING_FACE_MAX_OUTPUT_BYTES:
                raise RuntimeError("Partner system output exceeded configured limit")
            return Response(content=video, media_type="video/mp4")
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            print(
                f"[talking-face-sidecar][error] {type(exc).__name__}: {exc}",
                flush=True,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc)[:300],
            ) from exc
        finally:
            await asyncio.to_thread(shutil.rmtree, job_dir, True)

    return app


app = create_app()
