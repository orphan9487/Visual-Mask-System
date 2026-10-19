import unittest

from scripts.evaluation.run_lora_crop_experiment import head_box


class HeadBoxTests(unittest.TestCase):
    def test_square_and_head_margin(self):
        left, top, right, bottom = head_box((100, 100, 200, 300))
        self.assertEqual(right-left, bottom-top)
        self.assertEqual(right-left, 300)
        self.assertLess(left, 100)
        self.assertLess(top, 100)
        self.assertGreater(right, 200)
        self.assertGreater(bottom, 300)

    def test_allows_padding_instead_of_clipping(self):
        left, top, right, bottom = head_box((0, 0, 100, 100))
        self.assertLess(left, 0)
        self.assertLess(top, 0)
        self.assertEqual(right-left, bottom-top)


if __name__ == '__main__':
    unittest.main()
