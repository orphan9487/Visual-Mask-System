"""先驗校正：ERCEngine 單次模式的打分判讀，以及校正參數的載入與防呆。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reasoning.erc_engine import ERCEngine
from src.reasoning.labels import CANONICAL_EMOTIONS as L
from src.services import emotion_service


class FakeBackbone:
    is_internvl = True

    def __init__(self, reply="anger"):
        self.reply = reply
        self.chat_calls = 0

    def chat(self, system, user, max_new_tokens=None):
        self.chat_calls += 1
        return self.reply


class FakeScorer:
    """joy 以些微差距勝過 neutral。"""
    def score(self, system, user):
        s = {k: -5.0 for k in L}
        s["joy"], s["neutral"] = -1.0, -1.2
        return [s[k] for k in L]


class ErcEngineCalibrationTests(unittest.TestCase):
    def test_bias_flips_borderline_prediction_to_neutral(self):
        bias = [0.0] * len(L)
        bias[L.index("neutral")] = 0.5
        bk = FakeBackbone()
        eng = ERCEngine(bk, two_stage=False, label_bias=bias, scorer=FakeScorer())
        self.assertEqual(eng.predict("好").predicted, "neutral")
        self.assertEqual(bk.chat_calls, 0)            # 走打分，不生成

    def test_zero_bias_keeps_argmax(self):
        eng = ERCEngine(FakeBackbone(), two_stage=False, label_bias=[0.0] * len(L),
                        scorer=FakeScorer())
        self.assertEqual(eng.predict("好").predicted, "joy")

    def test_without_bias_uses_generation(self):
        bk = FakeBackbone(reply="anger")
        eng = ERCEngine(bk, two_stage=False)
        self.assertEqual(eng.predict("好").predicted, "anger")
        self.assertEqual(bk.chat_calls, 1)

    def test_wrong_bias_length_rejected(self):
        with self.assertRaises(ValueError):
            ERCEngine(FakeBackbone(), two_stage=False, label_bias=[0.0, 1.0])


class LoadLabelBiasTests(unittest.TestCase):
    def _write(self, d, adapter="adp", alpha=0.5):
        p = Path(d) / "cal.json"
        p.write_text(json.dumps({"adapter": adapter, "alpha": alpha,
                                 "shift": {k: (1.0 if k == "neutral" else -0.5) for k in L}}),
                     encoding="utf-8")
        return p

    def test_applies_alpha_times_shift_for_matching_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            with (patch.object(emotion_service.config, "ERC_CALIB_PATH", self._write(d)),
                  patch.object(emotion_service.config, "ERC_CALIB_ALPHA", None),
                  patch.object(emotion_service.config, "ERC_LORA_PATH", str(Path(d) / "adp"))):
                bias = emotion_service._load_label_bias()
        self.assertEqual(bias[L.index("neutral")], 0.5)
        self.assertEqual(bias[L.index("joy")], -0.25)

    def test_refuses_mismatched_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            with (patch.object(emotion_service.config, "ERC_CALIB_PATH", self._write(d)),
                  patch.object(emotion_service.config, "ERC_CALIB_ALPHA", None),
                  patch.object(emotion_service.config, "ERC_LORA_PATH", str(Path(d) / "other"))):
                self.assertIsNone(emotion_service._load_label_bias())

    def test_alpha_zero_disables(self):
        with tempfile.TemporaryDirectory() as d:
            with (patch.object(emotion_service.config, "ERC_CALIB_PATH", self._write(d)),
                  patch.object(emotion_service.config, "ERC_CALIB_ALPHA", 0.0),
                  patch.object(emotion_service.config, "ERC_LORA_PATH", str(Path(d) / "adp"))):
                self.assertIsNone(emotion_service._load_label_bias())

    def test_repo_calibration_file_matches_deployed_adapter(self):
        cal = json.loads((PROJECT_ROOT / "configs" / "erc_calibration.json").read_text(encoding="utf-8"))
        self.assertEqual(set(cal["shift"]), set(L))
        self.assertGreater(cal["shift"]["neutral"], 0)   # neutral 被降檔 → 應往上補


if __name__ == "__main__":
    unittest.main()
