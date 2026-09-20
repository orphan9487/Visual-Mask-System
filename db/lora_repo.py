from db.connection import get_connection
from src.reasoning.identity_db import DEFAULT_MASK_ID


def get_user_active_lora(username: str, default: str | None = None) -> str:
    """回傳目前 active 的身分 key；查無或 DB 離線時使用 FaceID 預設值。"""
    default = default or DEFAULT_MASK_ID
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT lora_key FROM user_loras WHERE username = %s AND is_active = 1",
                (username,),
            )
            row = cur.fetchone()
        conn.close()
        return row[0] if row else default
    except Exception as e:
        print(f"[lora_repo] get_user_active_lora failed: {e}")
        return default


def get_user_loras(username: str) -> list[dict]:
    """回傳使用者可選的身分清單（FaceID 或 LoRA，含 active 狀態）。"""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT lora_key, display_name, is_active FROM user_loras WHERE username = %s",
                (username,),
            )
            rows = cur.fetchall()
        conn.close()
        return [{"key": r[0], "display_name": r[1], "is_active": bool(r[2])} for r in rows]
    except Exception as e:
        print(f"[lora_repo] get_user_loras failed: {e}")
        return []


def set_user_active_lora(username: str, lora_key: str) -> bool:
    """
    切換 active 身分（資料表名稱為相容舊版而保留 user_loras）。
    回傳 True 成功；False 代表該使用者沒有此 lora_key 的權限。
    """
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM user_loras WHERE username = %s AND lora_key = %s",
                (username, lora_key),
            )
            if not cur.fetchone():
                conn.close()
                return False
            cur.execute("UPDATE user_loras SET is_active = 0 WHERE username = %s", (username,))
            cur.execute(
                "UPDATE user_loras SET is_active = 1 WHERE username = %s AND lora_key = %s",
                (username, lora_key),
            )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[lora_repo] set_user_active_lora failed: {e}")
        return False
