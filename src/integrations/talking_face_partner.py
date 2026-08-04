"""Sidecar-only loader for the unmodified partner Wav2Lip system."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


class PartnerSystemError(RuntimeError):
    """Raised when the partner package is incomplete or generation fails."""


class PartnerWav2LipBackend:
    """Load and call ``api_server.process_wav2lip`` without editing that repo.

    This backend must run in its own process because the partner application
    relies on its working directory and module-level GPU globals.
    """

    REQUIRED_FILES = (
        "api_server.py",
        "checkpoints/wav2lip_gan.pth",
        "face_detection/detection/sfd/s3fd.pth",
    )

    def __init__(self, module: ModuleType, lifespan_context: Any) -> None:
        self._module = module
        self._lifespan_context = lifespan_context
        self._closed = False

    @classmethod
    async def load(cls, root: str | Path) -> "PartnerWav2LipBackend":
        partner_root = Path(root).resolve()
        missing = [
            relative
            for relative in cls.REQUIRED_FILES
            if not (partner_root / relative).is_file()
        ]
        if missing:
            raise PartnerSystemError(
                "Talking Face partner package is missing: " + ", ".join(missing)
            )

        # This is a dedicated Sidecar process, so keeping the partner root as
        # cwd and first on sys.path cannot affect the main LINE/WebSocket app.
        os.chdir(partner_root)
        root_text = str(partner_root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)

        module_path = partner_root / "api_server.py"
        spec = importlib.util.spec_from_file_location(
            "vms_partner_talking_face_api", module_path
        )
        if spec is None or spec.loader is None:
            raise PartnerSystemError(f"Could not load partner module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            lifespan_context = module.lifespan(module.app)
            await lifespan_context.__aenter__()
        except Exception as exc:  # noqa: BLE001
            raise PartnerSystemError(
                f"Talking Face model startup failed: {type(exc).__name__}: {exc}"
            ) from exc
        return cls(module, lifespan_context)

    async def generate(
        self, face_path: Path, audio_path: Path, output_path: Path
    ) -> None:
        if self._closed:
            raise PartnerSystemError("Talking Face backend is closed")
        try:
            success, message = await asyncio.to_thread(
                self._module.process_wav2lip,
                str(face_path),
                str(audio_path),
                str(output_path),
            )
        except Exception as exc:  # noqa: BLE001
            raise PartnerSystemError(
                f"Talking Face inference failed: {type(exc).__name__}: {exc}"
            ) from exc
        if not success:
            raise PartnerSystemError(str(message))
        if not output_path.is_file():
            raise PartnerSystemError(
                f"Partner system reported success without output: {output_path}"
            )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._lifespan_context.__aexit__(None, None, None)
