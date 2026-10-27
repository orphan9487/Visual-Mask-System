"""欄位驗證錯誤要回傳一句中文字串（前端直接顯示 detail，不能是物件，否則出現 [object Object]）。"""

import io
import unittest

from fastapi.testclient import TestClient
from PIL import Image

from src.interfaces.multipage_app import create_app, current


class ValidationErrorTests(unittest.TestCase):
    def setUp(self):
        app = create_app()
        app.dependency_overrides[current] = lambda: {"id": 99, "username": "T"}
        self.client = TestClient(app)

    def test_empty_mask_name_gives_readable_message(self):
        stream = io.BytesIO()
        Image.new("RGB", (400, 560)).save(stream, format="JPEG")
        r = self.client.post("/api/masks", data={"name": "", "mode": "natural", "crop_json": "{}"},
                             files={"image": ("a.jpg", stream.getvalue(), "image/jpeg")})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.json()["detail"], "請填寫：面具名稱")

    def test_missing_register_fields_listed_in_chinese(self):
        r = self.client.post("/api/register", json={"account": "abc"})
        self.assertEqual(r.status_code, 422)
        self.assertIsInstance(r.json()["detail"], str)
        self.assertIn("密碼", r.json()["detail"])
        self.assertIn("顯示名稱", r.json()["detail"])


if __name__ == "__main__":
    unittest.main()
