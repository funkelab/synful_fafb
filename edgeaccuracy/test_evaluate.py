import unittest

import logging
import numpy as np

from evaluate import eval_pairwise_connection

logging.basicConfig(level=logging.INFO)


class TestEvaluate(unittest.TestCase):
    def test_basics(self):
        gt_con = np.array([(0, 0, 0), (10, 10, 6), (10, 10, 10)])
        pred_con = np.array([(0, 0, 0), (0, 0, 0), (10, 10, 10)])

        results = eval_pairwise_connection(gt_con, pred_con, connection_thr=5)

        self.assertEqual(results['accuracy'], 0.5)
        self.assertEqual(results['precision'], 1.0)
        self.assertEqual(results['recall'], 0.5)

        results = eval_pairwise_connection(gt_con, pred_con, connection_thr=8)

        self.assertAlmostEqual(results['accuracy'], 4 / 6.)
        self.assertAlmostEqual(results['precision'], 1.0)
        self.assertAlmostEqual(results['recall'], 3 / 5.)


if __name__ == '__main__':
    unittest.main()
