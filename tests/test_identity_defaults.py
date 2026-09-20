"""Identity defaults for the restored WebSocket + FaceID product."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import db.lora_repo as identity_repo
from src.reasoning.identity_db import DEFAULT_MASK_ID, identity_db


class IdentityDefaultTests(unittest.TestCase):
    def test_default_identity_is_enrolled_henry_faceid(self):
        identity = identity_db.resolve()
        self.assertEqual(DEFAULT_MASK_ID, "henry")
        self.assertEqual(identity.mask_id, "henry")
        self.assertEqual(identity.identity_mode, "faceid")

    def test_database_failure_falls_back_to_faceid(self):
        with patch.object(identity_repo, "get_connection", side_effect=RuntimeError("offline")):
            self.assertEqual(identity_repo.get_user_active_lora("Alice"), "henry")


if __name__ == "__main__":
    unittest.main()
