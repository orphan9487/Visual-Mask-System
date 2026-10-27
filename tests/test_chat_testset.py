"""真實聊天測試集工具：語料解析、匿名化、一致度、gold 合併。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.annotation import chat_testset as ct


class TranscriptTests(unittest.TestCase):
    def test_parse_splits_conversations_and_handles_formats(self):
        raw = ("小明：你把我的資料庫刪光了\n"
               "14:03 小華: 真好啊\n"
               "接著換行的內容\n"
               "\n"
               "小明\t晚安\n"
               "---\n"
               "阿土：嗨\n")
        convs = ct.parse_transcript(raw)
        self.assertEqual(len(convs), 3)
        self.assertEqual(convs[0][1], {"speaker": "小華", "text": "真好啊 接著換行的內容"})
        self.assertEqual(convs[1][0]["speaker"], "小明")

    def test_anonymize_and_mask_pii(self):
        conv = [{"speaker": "小明", "text": "打給我 0912-345-678"},
                {"speaker": "小華", "text": "寄到 a.b@example.com"},
                {"speaker": "小明", "text": "看 https://x.tw/abc"}]
        items = ct.build_items([conv, [{"speaker": "x", "text": "太短"}]])
        self.assertEqual([it["speaker"] for it in items], ["A", "B", "A"])   # 單句對話被排除
        self.assertEqual(items[0]["text"], "打給我 [電話]")
        self.assertEqual(items[1]["text"], "寄到 [email]")
        self.assertEqual(items[2]["text"], "看 [網址]")
        self.assertEqual(items[2]["id"], "c001-t002")

    def test_forum_context_only_and_mentions_anonymized(self):
        conv = [{"speaker": "eulbos", "text": "【標題】內文", "context_only": True},
                {"speaker": "VoV", "text": "笑死"},
                {"speaker": "beef68", "text": "樓上VoV說得對"}]
        items = ct.build_items([conv], min_turns=2, sources=["ptt/Boy-Girl"])
        self.assertTrue(items[0]["context_only"])
        self.assertNotIn("context_only", items[1])
        self.assertEqual(items[2]["text"], "樓上B說得對")          # 內文提到的帳號也匿名
        self.assertEqual({it["source"] for it in items}, {"ptt/Boy-Girl"})
        # 可標註句只有 2 則；min_turns=3 時整段不收（原文不算）
        self.assertEqual(ct.build_items([conv], min_turns=3), [])

    def test_recording_split_by_gap(self):
        rows = [{"session": "s1", "ts": 0, "speaker": "a", "text": "1"},
                {"session": "s1", "ts": 60, "speaker": "b", "text": "2"},
                {"session": "s1", "ts": 60 + 31 * 60, "speaker": "a", "text": "3"},
                {"session": "s2", "ts": 5, "speaker": "a", "text": "4"}]
        convs = ct.split_recording(rows, gap_minutes=30)
        self.assertEqual([len(c) for c in convs], [2, 1, 1])

    def test_history_only_includes_earlier_turns(self):
        items = ct.build_items([[{"speaker": "a", "text": "1"}, {"speaker": "b", "text": "2"},
                                 {"speaker": "a", "text": "3"}]])
        hist = ct.history_for(items)
        self.assertEqual(hist["c001-t000"], [])
        self.assertEqual(hist["c001-t002"], [{"role": "A", "content": "1"},
                                             {"role": "B", "content": "2"}])


class AgreementTests(unittest.TestCase):
    def test_cohen_kappa_known_values(self):
        self.assertAlmostEqual(ct.cohen_kappa(["joy", "anger"] * 5, ["joy", "anger"] * 5), 1.0)
        # 經典例：po=0.7, pe=0.5 → κ=0.4
        a = ["joy"] * 5 + ["anger"] * 5
        b = ["joy"] * 4 + ["anger"] + ["joy"] * 2 + ["anger"] * 3
        self.assertAlmostEqual(ct.cohen_kappa(a, b), 0.4)

    def test_merge_gold_agree_adjudicate_and_exclude(self):
        a = ("a", {"1": {"label": "joy", "sarcasm": False},
                   "2": {"label": "anger", "sarcasm": True},
                   "3": {"label": "neutral"},
                   "4": {"label": "fear"}})
        b = ("b", {"1": {"label": "joy"},
                   "2": {"label": "disgust", "sarcasm": True},
                   "3": {"label": "sadness"}})
        gold, stats = ct.merge_gold([a, b], {"2": {"final": "anger"}})
        by_id = {g["id"]: g for g in gold}
        self.assertEqual(by_id["1"], {"id": "1", "label": "joy", "sarcasm": False, "source": "agree"})
        self.assertEqual(by_id["2"]["label"], "anger")
        self.assertTrue(by_id["2"]["sarcasm"])
        self.assertNotIn("3", by_id)                       # 分歧未裁決 → 排除
        self.assertEqual(stats["只有一人標"], 1)


if __name__ == "__main__":
    unittest.main()
