"""Static guards for the standalone WebSocket chat frontend."""

from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FrontendMarkupTests(unittest.TestCase):
    def test_generated_media_preserves_the_complete_frame(self):
        html = (PROJECT_ROOT / "index.html").read_text(encoding="utf-8")

        self.assertIn(".mask-area{margin-top:10px;max-width:100%", html)
        self.assertNotIn(".mask-area{margin-top:10px;width:512px", html)
        self.assertIn("width:auto;max-width:100%;height:auto;aspect-ratio:1 / 1", html)
        self.assertIn("object-fit:contain", html)


if __name__ == "__main__":
    unittest.main()
