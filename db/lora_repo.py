from db.connection import get_connection


def get_user_active_lora(username: str, default: str = "human_8692") -> str:
    """回傳目前 active 的 lora_key；查無則給 default。"""
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
        print(f"❌ [lora_repo] get_user_active_lora 錯誤: {e}")
        return default


def get_user_loras(username: str) -> list[dict]:
    """回傳使用者的所有 LoRA 清單（含 is_active 狀態）。"""
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
        print(f"❌ [lora_repo] get_user_loras 錯誤: {e}")
        return []


def set_user_active_lora(username: str, lora_key: str) -> bool:
    """
    切換 active LoRA。
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
        print(f"❌ [lora_repo] set_user_active_lora 錯誤: {e}")
        return False
