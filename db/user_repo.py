import base64
import hashlib
import hmac
import os
import secrets
from db.connection import get_connection


_ITERATIONS = 600_000


def hash_password(password: str) -> str:
    """PBKDF2 password storage: no plaintext or reversible password data."""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        _ITERATIONS,
        base64.urlsafe_b64encode(salt).decode(),
        base64.urlsafe_b64encode(digest).decode(),
    )


def _verify(password: str, stored: str) -> bool:
    if stored.startswith("pbkdf2_sha256$"):
        try:
            _, rounds, salt, digest = stored.split("$", 3)
            actual = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), base64.urlsafe_b64decode(salt), int(rounds)
            )
            return hmac.compare_digest(actual, base64.urlsafe_b64decode(digest))
        except (ValueError, TypeError):
            return False
    # Allow the seeded demonstration users to migrate when they next sign in.
    return hmac.compare_digest(hashlib.sha256(password.encode()).hexdigest(), stored)


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


def verify_login(account: str, password: str) -> dict | None:
    """
    驗證帳號密碼，成功回傳 username（顯示名稱），失敗回傳 None。
    """
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, account, username, password_hash, face_profile_status, default_mask_id "
                "FROM users WHERE account = %s", (account,)
            )
            row = cur.fetchone()
        conn.close()
    except Exception as e:
        print(f"❌ [user_repo] verify_login 錯誤: {e}")
        return None

    if not row:
        return None
    if not _verify(password, row[3]):
        return None
    if not row[3].startswith("pbkdf2_sha256$"):
        try:
            conn = get_connection()
            with conn.cursor() as cur:
                cur.execute("UPDATE users SET password_hash = %s WHERE id = %s", (hash_password(password), row[0]))
            conn.commit(); conn.close()
        except Exception:
            pass
    return {"id": row[0], "account": row[1], "username": row[2],
            "face_profile_status": row[4] or "pending", "default_mask_id": row[5]}


def register_user(account: str, username: str, password: str) -> dict:
    if not 3 <= len(account) <= 50 or not account.replace("_", "").isalnum():
        raise ValueError("帳號須為 3-50 個英數字或底線")
    if not 2 <= len(username) <= 50:
        raise ValueError("顯示名稱須為 2-50 個字元")
    if len(password) < 8:
        raise ValueError("密碼至少需要 8 個字元")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (account, username, password_hash, emotion_prior, face_profile_status) "
                "VALUES (%s, %s, %s, %s, 'pending')",
                (account, username, hash_password(password), "A normal, friendly user."),
            )
            user_id = cur.lastrowid
        conn.commit()
        return {"id": user_id, "account": account, "username": username,
                "face_profile_status": "pending", "default_mask_id": None}
    except Exception as exc:
        conn.rollback()
        if "Duplicate" in str(exc) or "duplicate" in str(exc):
            raise ValueError("帳號已被使用") from exc
        raise
    finally:
        conn.close()


def get_user(user_id: int) -> dict | None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, account, username, face_profile_status, default_mask_id FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
        return None if not row else {"id": row[0], "account": row[1], "username": row[2],
                                     "face_profile_status": row[3] or "pending", "default_mask_id": row[4]}
    finally:
        conn.close()
