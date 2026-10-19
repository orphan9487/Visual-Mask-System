"""Validate and store photos reserved for optional per-user LoRA training."""
from __future__ import annotations

import io
import json
import uuid
from pathlib import Path
from PIL import Image, ImageOps

from db.training_repo import add_training_photo, update_training_photo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = PROJECT_ROOT / "data" / "user_lora_training"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


def store_training_photo(*, user_id: int, image_bytes: bytes, content_type: str, mask_name: str, crop_json: str) -> dict:
    mask_name = mask_name.strip()
    if not mask_name or len(mask_name) > 100:
        raise ValueError("面具名稱必須為 1～100 個字元")
    if content_type not in ALLOWED_TYPES:
        raise ValueError("請上傳 JPG、PNG 或 WebP")
    if not image_bytes or len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("每張照片必須介於 1 byte 與 8 MB 之間")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise ValueError("無法讀取圖片") from exc
    image = apply_training_crop(image, crop_json)
    if min(image.size) < 256:
        raise ValueError("照片尺寸太小，短邊至少需要 256 像素")
    directory = TRAINING_ROOT / str(user_id) / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"photo-{uuid.uuid4().hex}.jpg"
    image.save(path, "JPEG", quality=95)
    photo_id = add_training_photo(user_id, str(path), mask_name)
    try:
        processed = _prepare_face_crop(path, user_id)
    except ValueError:
        update_training_photo(photo_id, status="rejected")
        raise
    update_training_photo(photo_id, path=str(processed), status="ready")
    return {"id": photo_id, "mask_name": mask_name, "path": str(processed), "status": "ready"}


def apply_training_crop(image: Image.Image, crop_json: str) -> Image.Image:
    """Crop oriented source pixels before persistence and face detection."""
    try:
        crop=json.loads(crop_json)
        values=[crop[k] for k in ('x','y','width','height')]
        if any(type(v) is not int for v in values): raise ValueError()
        x,y,w,h=values
        if x<0 or y<0 or w<256 or h<256 or x+w>image.width or y+h>image.height: raise ValueError()
    except (ValueError, TypeError, KeyError):
        raise ValueError('裁切範圍不合法，請保留短邊至少 256 像素並重新裁切')
    return image.crop((x,y,x+w,y+h))


def _prepare_face_crop(path: Path, user_id: int) -> Path:
    """Keep one face with a small margin for the future LoRA dataset."""
    try:
        import cv2
        import numpy as np
        from insightface.app import FaceAnalysis
    except ImportError:
        # Keep upload usable on machines without FaceID dependencies; the
        # training worker can re-run detection later in its own environment.
        return path
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    analyzer = FaceAnalysis(name="buffalo_l", root=str(PROJECT_ROOT / "models" / "insightface"), providers=["CPUExecutionProvider"])
    analyzer.prepare(ctx_id=-1, det_size=(640, 640))
    faces = analyzer.get(image)
    offset = 0
    if not faces:
        # Very large faces can exceed SCRFD's effective scale range. Padding
        # changes detection scale without weakening the confidence threshold.
        offset = max(1, int(max(image.shape[:2]) * 0.25))
        padded = cv2.copyMakeBorder(image, offset, offset, offset, offset,
                                   cv2.BORDER_CONSTANT, value=(127,127,127))
        faces = analyzer.get(padded)
    if len(faces) == 0:
        raise ValueError('裁切範圍內未偵測到人臉，請保留完整臉部並重新上傳')
    if len(faces) > 1:
        raise ValueError(f'裁切範圍內偵測到 {len(faces)} 張人臉，請排除背景人物後重新上傳')
    x1, y1, x2, y2 = [int(v) - offset for v in faces[0].bbox]
    margin = int(max(x2 - x1, y2 - y1) * 0.35)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
    x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
    crop = image[y1:y2, x1:x2]
    output_dir = TRAINING_ROOT / str(user_id) / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / path.name
    if not cv2.imwrite(str(output), crop):
        raise ValueError("無法儲存處理後的照片")
    return output
