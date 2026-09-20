"""SD1.5 + IP-Adapter FaceID Portrait v11 still-image renderer."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

from src.generation.diffusion_engine import SD_MODEL_ID, VAE_DIR
from src.generation.renderers.base import GenerationRequest, RenderedFrames, Renderer


PROJECT_ROOT = Path(__file__).resolve().parents[3]
IP_ADAPTER_ROOT = PROJECT_ROOT / "external" / "IP-Adapter"
INSIGHTFACE_ROOT = PROJECT_ROOT / "models" / "insightface"


class FaceIDStickerRenderer(Renderer):
    """Generate identity-preserving PNGs without per-user LoRA training."""

    def __init__(self) -> None:
        self._face_analyzer: Any | None = None
        self._adapter: Any | None = None
        self._adapter_key: tuple[str, int] | None = None
        self._embedding_key: tuple[tuple[str, int], ...] | None = None
        self._embeddings: Any | None = None

    @staticmethod
    def _read_face_embedding(path: Path, analyzer: Any):
        import cv2
        import numpy as np
        import torch

        # np.fromfile supports non-ASCII Windows paths more reliably than imread.
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"OpenCV could not read FaceID reference: {path}")
        faces = analyzer.get(image)
        if not faces:
            raise RuntimeError(f"InsightFace found no face in: {path}")
        face = max(
            faces,
            key=lambda item: float(
                (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
            ),
        )
        return torch.from_numpy(face.normed_embedding).unsqueeze(0).unsqueeze(0)

    def _analyzer(self):
        if self._face_analyzer is None:
            try:
                from insightface.app import FaceAnalysis
            except ImportError as exc:
                raise RuntimeError(
                    "FaceID dependencies are missing; install requirements-faceid.txt"
                ) from exc
            self._face_analyzer = FaceAnalysis(
                name="buffalo_l",
                root=str(INSIGHTFACE_ROOT),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        return self._face_analyzer

    def _build_adapter(self, checkpoint: Path, reference_count: int):
        import torch
        from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline

        if not IP_ADAPTER_ROOT.is_dir():
            raise FileNotFoundError(
                f"Vendored IP-Adapter module not found: {IP_ADAPTER_ROOT}"
            )
        root_text = str(IP_ADAPTER_ROOT)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        from ip_adapter.ip_adapter_faceid_separate import IPAdapterFaceID

        dtype = torch.float16
        pipe = StableDiffusionPipeline.from_pretrained(
            SD_MODEL_ID,
            torch_dtype=dtype,
            local_files_only=True,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        if VAE_DIR.is_dir():
            pipe.vae = AutoencoderKL.from_pretrained(
                str(VAE_DIR), torch_dtype=dtype, local_files_only=True
            )
        pipe.scheduler = DDIMScheduler.from_config(
            pipe.scheduler.config,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        return IPAdapterFaceID(
            pipe,
            str(checkpoint),
            "cuda",
            num_tokens=16,
            n_cond=reference_count,
            torch_dtype=dtype,
        )

    def _adapter_for(self, checkpoint: Path, reference_count: int):
        key = (str(checkpoint), reference_count)
        if self._adapter is None or self._adapter_key != key:
            self._adapter = self._build_adapter(checkpoint, reference_count)
            self._adapter_key = key
        return self._adapter

    def render(self, request: GenerationRequest) -> RenderedFrames:
        import torch

        references = [Path(value) for value in request.identity["reference_paths"]]
        if not references:
            raise ValueError("FaceID requires at least one reference image")
        embedding_key = tuple(
            (str(path.resolve()), path.stat().st_mtime_ns) for path in references
        )
        if self._embeddings is None or self._embedding_key != embedding_key:
            analyzer = self._analyzer()
            embeddings = [
                self._read_face_embedding(path, analyzer) for path in references
            ]
            self._embeddings = torch.cat(embeddings, dim=1)
            self._embedding_key = embedding_key
        faceid_embeds = self._embeddings

        checkpoint = Path(request.identity["checkpoint"])
        adapter = self._adapter_for(checkpoint, len(references))
        seed = request.seed if request.seed is not None else random.randint(0, 1_000_000)
        images = adapter.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            faceid_embeds=faceid_embeds,
            scale=float(request.identity.get("scale", 0.55)),
            num_samples=1,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            seed=seed,
        )
        return RenderedFrames(frames=[images[0]], seed=seed)
