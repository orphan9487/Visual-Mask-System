"""HTTP adapter for a separately deployed Talking Face black box."""

from __future__ import annotations

import asyncio
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from src.config import talking_face as config


class TalkingFaceError(RuntimeError):
    """Raised when the external Talking Face service cannot produce a video."""


@dataclass(frozen=True)
class TalkingFaceResult:
    output_path: Path
    processing_ms: int
    content_type: str


class TalkingFaceAdapter(Protocol):
    """Boundary owned by Visual Mask System, independent of Wav2Lip internals."""

    async def health(self) -> bool:
        """Return whether the configured black-box service is ready."""

    async def generate(
        self,
        face_path: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
    ) -> TalkingFaceResult:
        """Generate one lip-synced MP4 from a face image and speech audio."""


class HttpTalkingFaceAdapter:
    """Upload inputs to POST /generate and persist the returned MP4 atomically.

    Contract:
      - multipart field ``face``: PNG or JPEG image
      - multipart field ``audio``: WAV, MP3, M4A, or MP4 audio
      - response: ``video/mp4`` bytes
      - optional Bearer token on both ``/health`` and ``/generate``
    """

    _FACE_SUFFIXES = {".png", ".jpg", ".jpeg"}
    _AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".mp4"}

    def __init__(
        self,
        base_url: str,
        *,
        token: str = "",
        timeout: float = 300.0,
        max_output_bytes: int = 100 * 1024 * 1024,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        clean_url = base_url.strip().rstrip("/")
        if not clean_url:
            raise ValueError("Talking Face base URL must not be empty")
        if timeout <= 0:
            raise ValueError("Talking Face timeout must be positive")
        if max_output_bytes <= 0:
            raise ValueError("Talking Face output limit must be positive")
        self.base_url = clean_url
        self.token = token.strip()
        self.timeout = float(timeout)
        self.max_output_bytes = int(max_output_bytes)
        self._client = client

    @property
    def _headers(self) -> dict[str, str]:
        return (
            {"Authorization": f"Bearer {self.token}"}
            if self.token
            else {}
        )

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        url = f"{self.base_url}{path}"
        try:
            if self._client is not None:
                return await self._client.request(method, url, **kwargs)
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=False,
            ) as client:
                return await client.request(method, url, **kwargs)
        except (httpx.HTTPError, TimeoutError) as exc:
            raise TalkingFaceError(f"Talking Face request failed: {exc}") from exc

    async def health(self) -> bool:
        try:
            response = await self._request("GET", "/health", headers=self._headers)
        except TalkingFaceError:
            return False
        return response.status_code == 200

    @classmethod
    def _validate_input(
        cls, path: str | Path, *, allowed_suffixes: set[str], label: str
    ) -> Path:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise TalkingFaceError(f"{label} file does not exist: {resolved}")
        if resolved.suffix.lower() not in allowed_suffixes:
            allowed = ", ".join(sorted(allowed_suffixes))
            raise TalkingFaceError(f"{label} must use one of: {allowed}")
        if resolved.stat().st_size <= 0:
            raise TalkingFaceError(f"{label} file is empty: {resolved}")
        return resolved

    async def generate(
        self,
        face_path: str | Path,
        audio_path: str | Path,
        output_path: str | Path,
    ) -> TalkingFaceResult:
        face = self._validate_input(
            face_path, allowed_suffixes=self._FACE_SUFFIXES, label="face"
        )
        audio = self._validate_input(
            audio_path, allowed_suffixes=self._AUDIO_SUFFIXES, label="audio"
        )
        output = Path(output_path).resolve()
        if output.suffix.lower() != ".mp4":
            raise TalkingFaceError("Talking Face output path must end in .mp4")

        started = time.perf_counter()
        face_type = mimetypes.guess_type(face.name)[0] or "application/octet-stream"
        audio_type = mimetypes.guess_type(audio.name)[0] or "application/octet-stream"
        try:
            with face.open("rb") as face_file, audio.open("rb") as audio_file:
                response = await self._request(
                    "POST",
                    "/generate",
                    headers=self._headers,
                    files={
                        "face": (face.name, face_file, face_type),
                        "audio": (audio.name, audio_file, audio_type),
                    },
                )
        except OSError as exc:
            raise TalkingFaceError(f"Could not read Talking Face input: {exc}") from exc

        if response.status_code != 200:
            detail = response.text[:300].strip()
            raise TalkingFaceError(
                f"Talking Face returned HTTP {response.status_code}: {detail}"
            )
        content_type = response.headers.get("content-type", "").split(";", 1)[0]
        if content_type != "video/mp4":
            raise TalkingFaceError(
                f"Talking Face returned unexpected content type: {content_type or 'missing'}"
            )
        video = response.content
        if len(video) < 12 or video[4:8] != b"ftyp":
            raise TalkingFaceError("Talking Face returned invalid MP4 data")
        if len(video) > self.max_output_bytes:
            raise TalkingFaceError(
                f"Talking Face output exceeded {self.max_output_bytes} bytes"
            )

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(f"{output.suffix}.part")
        try:
            await asyncio.to_thread(temporary.write_bytes, video)
            await asyncio.to_thread(os.replace, temporary, output)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise TalkingFaceError(f"Could not save Talking Face output: {exc}") from exc

        return TalkingFaceResult(
            output_path=output,
            processing_ms=round((time.perf_counter() - started) * 1000),
            content_type=content_type,
        )


def create_talking_face_adapter() -> HttpTalkingFaceAdapter:
    """Create the production adapter from environment-backed configuration."""
    if not config.TALKING_FACE_URL:
        raise TalkingFaceError("VMS_TALKING_FACE_URL is not configured")
    return HttpTalkingFaceAdapter(
        config.TALKING_FACE_URL,
        token=config.TALKING_FACE_TOKEN,
        timeout=config.TALKING_FACE_TIMEOUT,
        max_output_bytes=config.TALKING_FACE_MAX_OUTPUT_BYTES,
    )
