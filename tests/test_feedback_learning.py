from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from db.feedback_repo import (
    get_emotion_correction_memory,
    normalize_emotion_label,
    normalize_utterance,
)
from scripts.learning.export_feedback_sft import export_rows
from src.services import emotion_service


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, *_args):
        return None

    def fetchall(self):
        return self.rows


class _Connection:
    def __init__(self, rows):
        self.rows = rows

    def cursor(self):
        return _Cursor(self.rows)

    def close(self):
        return None


class FeedbackLearningTests(unittest.TestCase):
    def test_label_and_utterance_normalization(self):
        self.assertEqual(normalize_emotion_label("生氣"), "anger")
        self.assertEqual(normalize_utterance("  操你媽　"), "操你媽")
        with self.assertRaises(ValueError):
            normalize_emotion_label("confused")

    def test_exact_match_wins_and_similar_examples_are_returned(self):
        rows = [
            (json.dumps({"text": "操你媽"}, ensure_ascii=False), "anger"),
            (json.dumps({"text": "真的很煩耶"}, ensure_ascii=False), "anger"),
        ]
        with patch("db.feedback_repo.get_connection", return_value=_Connection(rows)):
            memory = get_emotion_correction_memory("Alice", " 操你媽 ")
        self.assertEqual(memory["exact_label"], "anger")

    def test_export_deduplicates_and_anonymizes(self):
        context = json.dumps({"history": [], "text": "操你媽"}, ensure_ascii=False)
        rows = [
            ("Alice", context, "neutral", "anger"),
            ("Alice", context, "neutral", "anger"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "feedback.jsonl"
            written, skipped = export_rows(rows, output)
            payload = json.loads(output.read_text(encoding="utf-8").strip())
        self.assertEqual((written, skipped), (1, 1))
        self.assertEqual(payload["assistant"], "anger")
        self.assertNotEqual(payload["metadata"]["user_id"], "Alice")


class CorrectionMemoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_correction_bypasses_model_and_keeps_faceid_identity(self):
        with (
            patch.object(emotion_service.config, "MOCK_MODE", True),
            patch(
                "db.feedback_repo.get_emotion_correction_memory",
                return_value={"exact_label": "anger", "examples": []},
            ),
            patch.object(
                emotion_service, "predict_emotion", new=AsyncMock()
            ) as predict,
        ):
            result = await emotion_service.produce_visual_instruction(
                "操你媽",
                [],
                user_id="Alice",
                mask_id="henry",
            )

        predict.assert_not_awaited()
        self.assertEqual(result["emotion"], "anger")
        self.assertEqual(result["inference_source"], "correction_memory")
        self.assertEqual(result["instruction"]["identity"]["mask_id"], "henry")


if __name__ == "__main__":
    unittest.main()
