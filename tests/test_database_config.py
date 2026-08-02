"""Tests for environment-backed database settings; no MySQL server is needed."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.database import DatabaseSettings


class DatabaseSettingsTests(unittest.TestCase):
    def test_reads_connection_values_from_environment(self):
        values = {
            "VMS_DB_HOST": "db.internal",
            "VMS_DB_PORT": "3307",
            "VMS_DB_USER": "visual-mask",
            "VMS_DB_PASSWORD": "secret",
            "VMS_DB_NAME": "visual_mask_test",
        }
        with patch.dict(os.environ, values, clear=False):
            settings = DatabaseSettings.from_env()

        self.assertEqual(settings.host, "db.internal")
        self.assertEqual(settings.port, 3307)
        self.assertEqual(settings.user, "visual-mask")
        self.assertEqual(settings.password, "secret")
        self.assertEqual(settings.database, "visual_mask_test")

    def test_rejects_unsafe_database_name(self):
        with patch.dict(os.environ, {"VMS_DB_NAME": "db; DROP DATABASE db"}, clear=False):
            with self.assertRaisesRegex(ValueError, "letters, numbers"):
                DatabaseSettings.from_env()


if __name__ == "__main__":
    unittest.main()
