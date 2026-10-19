"""Direct chat rooms and durable message history."""

from __future__ import annotations

import uuid

from db.connection import get_connection


def list_chat_candidates(user_id: int) -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, username, face_profile_status FROM users WHERE id <> %s ORDER BY username", (user_id,))
            rows = cur.fetchall()
        return [{"id": r[0], "username": r[1], "ready": r[2] == "ready"} for r in rows]
    finally:
        conn.close()


def get_or_create_direct_room(user_id: int, peer_id: int) -> dict:
    if user_id == peer_id:
        raise ValueError("不能與自己建立聊天室")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT r.id FROM chat_rooms r JOIN chat_room_members a ON a.room_id = r.id "
                "JOIN chat_room_members b ON b.room_id = r.id WHERE r.kind = 'direct' "
                "AND a.user_id = %s AND b.user_id = %s LIMIT 1", (user_id, peer_id)
            )
            row = cur.fetchone()
            room_id = row[0] if row else uuid.uuid4().hex
            if not row:
                cur.execute("INSERT INTO chat_rooms (id, kind) VALUES (%s, 'direct')", (room_id,))
                cur.executemany("INSERT INTO chat_room_members (room_id, user_id) VALUES (%s, %s)", [(room_id, user_id), (room_id, peer_id)])
            cur.execute("SELECT username FROM users WHERE id = %s", (peer_id,))
            peer = cur.fetchone()
        conn.commit()
        return {"id": room_id, "peer_id": peer_id, "peer_name": peer[0] if peer else "未知使用者"}
    finally:
        conn.close()


def list_rooms(user_id: int) -> list[dict]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT r.id, u.id, u.username FROM chat_rooms r "
                "JOIN chat_room_members mine ON mine.room_id = r.id AND mine.user_id = %s "
                "JOIN chat_room_members other ON other.room_id = r.id AND other.user_id <> %s "
                "JOIN users u ON u.id = other.user_id WHERE r.kind = 'direct' ORDER BY r.created_at DESC",
                (user_id, user_id),
            )
            rows = cur.fetchall()
        return [{"id": r[0], "peer_id": r[1], "peer_name": r[2]} for r in rows]
    finally:
        conn.close()


def is_room_member(room_id: str, user_id: int) -> bool:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM chat_room_members WHERE room_id = %s AND user_id = %s", (room_id, user_id))
            return bool(cur.fetchone())
    finally:
        conn.close()


def room_members(room_id: str) -> list[int]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM chat_room_members WHERE room_id = %s", (room_id,))
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def save_message(room_id: str, sender_id: int, text: str, emotion: str | None = None, mask_file: str | None = None, message_id: str | None = None) -> str:
    message_id = message_id or uuid.uuid4().hex
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO chat_messages (id, room_id, sender_id, text, emotion, mask_file) VALUES (%s, %s, %s, %s, %s, %s)",
                        (message_id, room_id, sender_id, text, emotion, mask_file))
        conn.commit()
        return message_id
    finally:
        conn.close()


def update_message_generation(message_id: str, emotion: str | None, mask_file: str | None) -> None:
    """Attach completed generation data to a message already saved at send time."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_messages SET emotion = %s, mask_file = %s WHERE id = %s",
                (emotion, mask_file, message_id),
            )
        conn.commit()
    finally:
        conn.close()


def list_messages(room_id: str, limit: int = 100) -> list[dict]:
    """Return the newest durable messages in chronological order for one room."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT m.id, u.username, m.text, m.emotion, m.mask_file, m.created_at "
                "FROM chat_messages m JOIN users u ON u.id = m.sender_id "
                "WHERE m.room_id = %s ORDER BY m.created_at DESC, m.id DESC LIMIT %s",
                (room_id, max(1, min(int(limit), 200))),
            )
            rows = cur.fetchall()
        rows = tuple(reversed(rows))
        return [
            {
                "msg_id": row[0], "sender": row[1], "text": row[2],
                "ai_label": row[3], "mask_file": row[4],
                "created_at": row[5].isoformat() if row[5] else None,
            }
            for row in rows
        ]
    finally:
        conn.close()
