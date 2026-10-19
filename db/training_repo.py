"""Persistence for user images collected for optional SD1.5 LoRA training."""
from __future__ import annotations

from db.connection import get_connection


def list_training_photos(user_id: int) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, mask_name, path, status, created_at FROM user_training_photos "
                "WHERE user_id = %s ORDER BY id",
                (user_id,),
            )
            return [
                {"id": r[0], "mask_name": r[1], "path": r[2], "status": r[3], "created_at": r[4].isoformat()}
                for r in cur.fetchall()
            ]


def add_training_photo(user_id: int, path: str, mask_name: str) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO user_training_photos (user_id, mask_name, path, status) "
                "VALUES (%s, %s, %s, 'uploaded')",
                (user_id, mask_name, path),
            )
            conn.commit()
            return int(cur.lastrowid)


def update_training_photo(photo_id: int, *, path: str | None = None, status: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            if path is None:
                cur.execute("UPDATE user_training_photos SET status = %s WHERE id = %s", (status, photo_id))
            else:
                cur.execute("UPDATE user_training_photos SET path = %s, status = %s WHERE id = %s", (path, status, photo_id))
            conn.commit()


def update_training_photo(photo_id: int, *, path: str | None = None, status: str) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            if path is None:
                cur.execute("UPDATE user_training_photos SET status = %s WHERE id = %s", (status, photo_id))
            else:
                cur.execute("UPDATE user_training_photos SET path = %s, status = %s WHERE id = %s", (path, status, photo_id))
            conn.commit()
