import unittest

from scripts.evaluate import lcs_f1


class EvaluationTests(unittest.TestCase):
    def test_lcs_f1_uses_precision_and_recall(self):
        self.assertAlmostEqual(lcs_f1('高血压管理', '高血压'), 2 * 1 * (3 / 5) / (1 + 3 / 5))

    def test_lcs_f1_handles_empty_text(self):
        self.assertEqual(lcs_f1('', '参考'), 0.0)


if __name__ == '__main__':
    unittest.main()
