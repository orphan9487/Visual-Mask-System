"""推理鏈解釋忠實度測試：引用比對與逐句重要度分析（不載入模型）。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.explain_faithfulness import cited_turns, importance_stats

HIST = ["我的報告檔不見了", "好像是我刪的，sad", "你明天幾點到"]


class CitedTurnsTests(unittest.TestCase):

    def test_quote_match(self):
        self.assertEqual(cited_turns("前文提到「sad」，暗示悲傷。", HIST, "真好啊"), ([1], "quote"))

    def test_overlap_match_without_quotes(self):
        self.assertEqual(cited_turns("因為對方說好像是我刪的，所以生氣。", HIST, "真好啊"), ([1], "overlap"))

    def test_quoting_current_utterance_is_not_context(self):
        self.assertEqual(cited_turns("「真好啊」是反話。", HIST, "真好啊"), ([], "none"))

    def test_overlap_threshold(self):
        expl = "提到報告檔，所以難過。"                     # 共同片段「報告檔」只有 3 字
        self.assertEqual(cited_turns(expl, HIST, "嗯", min_overlap=4)[1], "none")
        self.assertEqual(cited_turns(expl, HIST, "嗯", min_overlap=3), ([0], "overlap"))


class ImportanceTests(unittest.TestCase):

    def _rec(self, drops, expl):
        return {"explanation": expl, "history": HIST, "text": "真好啊", "p_pred": 0.9,
                "loo": [{"pred": "anger", "p_orig": 0.9 - d} for d in drops]}

    def test_cited_turn_most_important(self):
        recs = [self._rec([0.0, 0.5, 0.1], "前文提到「sad」")] * 3
        s = importance_stats(recs, 4)
        self.assertEqual(s["n"], 3)
        self.assertAlmostEqual(s["importance_diff"], 0.5 - 0.05)
        self.assertEqual(s["top1_hit"], 1.0)
        self.assertAlmostEqual(s["top1_chance"], 1 / 3)

    def test_no_citation_is_skipped(self):
        self.assertEqual(importance_stats([self._rec([0.1, 0.2, 0.3], "沒有引用")], 4), {"n": 0})


if __name__ == "__main__":
    unittest.main()
