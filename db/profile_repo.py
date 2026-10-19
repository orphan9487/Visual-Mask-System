"""Persistent per-user FaceID enrollment metadata."""

from __future__ import annotations

import json

from db.connection import get_connection


def save_face_profile(user_id: int, username: str, reference_path: str, crop: dict) -> str:
    mask_id = f"faceid_user_{user_id}"
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM face_profiles WHERE user_id = %s", (user_id,))
            cur.execute(
                "INSERT INTO face_profiles (user_id, mask_id, reference_path, crop_json, status) "
                "VALUES (%s, %s, %s, %s, 'ready')",
                (user_id, mask_id, reference_path, json.dumps(crop, ensure_ascii=False)),
            )
            cur.execute("UPDATE users SET face_profile_status = 'ready', default_mask_id = %s WHERE id = %s", (mask_id, user_id))
            cur.execute("UPDATE user_loras SET is_active = 0 WHERE username = %s", (username,))
            cur.execute(
                "INSERT INTO user_loras (username, lora_key, display_name, is_active) VALUES (%s, %s, %s, 1) "
                "ON DUPLICATE KEY UPDATE display_name = VALUES(display_name), is_active = 1",
                (username, mask_id, f"{username} 的 FaceID 面具"),
            )
        conn.commit()
        return mask_id
    finally:
        conn.close()


def get_face_profile(mask_id: str) -> dict | None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT fp.mask_id, fp.reference_path, u.username, fp.user_id FROM face_profiles fp "
                "JOIN users u ON u.id = fp.user_id WHERE fp.mask_id = %s AND fp.status = 'ready'",
                (mask_id,),
            )
            row = cur.fetchone()
        return None if not row else {"mask_id": row[0], "reference_path": row[1], "username": row[2], "user_id": row[3]}
    finally:
        conn.close()
