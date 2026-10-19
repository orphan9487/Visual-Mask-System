"""Validate, crop, and store one private FaceID enrollment image per user."""

from __future__ import annotations

import io
import json
import uuid
from pathlib import Path

from PIL import Image, ImageOps

from db.profile_repo import save_face_profile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ROOT = PROJECT_ROOT / "data" / "faceid_profiles" / "users"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


def _crop_box(image: Image.Image, crop_json: str | None) -> tuple[int, int, int, int]:
    if not crop_json:
        side = min(image.width, image.height)
        return ((image.width - side) // 2, (image.height - side) // 2,
                (image.width + side) // 2, (image.height + side) // 2)
    crop = json.loads(crop_json)
    x, y = int(crop["x"]), int(crop["y"])
    width, height = int(crop["width"]), int(crop["height"])
    if width < 80 or height < 80:
        raise ValueError("裁切範圍太小，請保留完整臉部")
    if x < 0 or y < 0 or x + width > image.width or y + height > image.height:
        raise ValueError("裁切範圍超出圖片邊界")
    return x, y, x + width, y + height


def _validate_single_face(path: Path) -> None:
    """Use the same InsightFace family as FaceID generation when available."""
    try:
        import cv2
        import numpy as np
        from insightface.app import FaceAnalysis
    except ImportError:
        # Development machines may not install FaceID dependencies yet. The
        # renderer will still validate before generation; production should
        # always install requirements-faceid.txt.
        return
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    app = FaceAnalysis(name="buffalo_l", root=str(PROJECT_ROOT / "models" / "insightface"), providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    faces = app.get(image)
    if len(faces) == 0:
        # Tight crops can put large faces outside the detector's effective
        # scale range. Retry at a smaller relative scale without lowering its
        # confidence threshold or modifying the stored crop.
        padding = max(1, int(max(image.shape[:2]) * 0.25))
        padded = cv2.copyMakeBorder(image, padding, padding, padding, padding,
                                   cv2.BORDER_CONSTANT, value=(255, 255, 255))
        faces = app.get(padded)
    if len(faces) == 0:
        raise ValueError("裁切範圍內未偵測到清楚人臉，請保留完整頭部、增加留邊，並避免模糊或遮擋")
    if len(faces) > 1:
        raise ValueError(f"裁切範圍內偵測到 {len(faces)} 張人臉，請縮小或移動裁切框，只保留一人")


def enroll_face(*, user_id: int, username: str, image_bytes: bytes, crop_json: str | None) -> dict:
    if not image_bytes or len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("圖片必須介於 1 byte 與 8 MB 之間")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise ValueError("無法讀取圖片，請上傳 JPG、PNG 或 WebP") from exc
    box = _crop_box(image, crop_json)
    cropped = image.crop(box)
    user_dir = PROFILE_ROOT / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    path = user_dir / f"reference-{uuid.uuid4().hex}.jpg"
    cropped.save(path, format="JPEG", quality=95)
    try:
        _validate_single_face(path)
        mask_id = save_face_profile(user_id, username, str(path), {
            "x": box[0], "y": box[1], "width": box[2] - box[0], "height": box[3] - box[1],
        })
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return {"mask_id": mask_id, "reference_path": str(path)}
