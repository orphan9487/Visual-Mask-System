import hashlib
from db.connection import get_connection


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def get_user_prior(username: str) -> str:
    """回傳使用者的 emotion_prior；查無則給預設值。"""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT emotion_prior FROM users WHERE username = %s", (username,))
            row = cur.fetchone()
        conn.close()
        return row[0] if row else "A normal, friendly user."
    except Exception as e:
        print(f"❌ [user_repo] get_user_prior 錯誤: {e}")
        return "A normal, friendly user."


def verify_login(account: str, password: str) -> str | None:
    """
    驗證帳號密碼，成功回傳 username（顯示名稱），失敗回傳 None。
    """
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT username, password_hash FROM users WHERE account = %s", (account,)
            )
            row = cur.fetchone()
        conn.close()
    except Exception as e:
        print(f"❌ [user_repo] verify_login 錯誤: {e}")
        return None

    if not row:
        return None
    if _hash(password) != row[1]:
        return None
    return row[0]  # username
