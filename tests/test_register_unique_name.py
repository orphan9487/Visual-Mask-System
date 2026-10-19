"""顯示名稱必須唯一：學習資料以顯示名稱存取，重名會互相汙染。"""

import unittest
from unittest.mock import patch

from db import user_repo


class _Cursor:
    def __init__(self, existing):
        self.existing, self.lastrowid, self.inserted = existing, 5, False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def execute(self, sql, params):
        self.last = (sql, params)
        if sql.startswith("INSERT"):
            self.inserted = True
    def fetchone(self):
        return (1,) if self.last[0].startswith("SELECT") and self.last[1][0] in self.existing else None


class _Conn:
    def __init__(self, cur): self.cur = cur
    def cursor(self): return self.cur
    def commit(self): pass
    def rollback(self): pass
    def close(self): pass


class RegisterUniqueNameTests(unittest.TestCase):
    def test_duplicate_display_name_is_rejected(self):
        cur = _Cursor({"Alice"})
        with patch.object(user_repo, "get_connection", lambda: _Conn(cur)):
            with self.assertRaisesRegex(ValueError, "顯示名稱已被使用"):
                user_repo.register_user("attacker1", "Alice", "password123")
        self.assertFalse(cur.inserted)

    def test_new_display_name_is_registered(self):
        cur = _Cursor({"Alice"})
        with patch.object(user_repo, "get_connection", lambda: _Conn(cur)):
            user = user_repo.register_user("bob_01", "Bob", "password123")
        self.assertTrue(cur.inserted)
        self.assertEqual(user["username"], "Bob")


if __name__ == "__main__":
    unittest.main()
