# from __future__ import absolute_import
import unittest
from tempfile import TemporaryDirectory
import os

from synful import synapse
import json
import logging

from evaluate_fafb import EvaluateFafb

logging.basicConfig(level=logging.INFO)
logging.getLogger('synful.evaluation').setLevel(logging.INFO)


class TestEvaluateFafb(unittest.TestCase):
    def test_basics(self):
        # pred1 --> gt1 match !
        # pred2 ---> no match, gt2 --> no match
        # pred3 --> gt3, match, but only if matching threshold is short enough
        syn_pred1 = synapse.Synapse(id=1, location_pre=(1, 2, 3),
                                    location_post=(10, 10, 0), id_skel_pre=1,
                                    id_skel_post=10, score=10)
        syn_pred2 = synapse.Synapse(id=2, location_pre=(3, 4, 5),
                                    location_post=(12, 14, 0), id_skel_pre=2,
                                    id_skel_post=11, score=10)
        syn_pred3 = synapse.Synapse(id=3, location_pre=(100, 100, 0),
                                    location_post=(30, 30, 0), id_skel_pre=1,
                                    id_skel_post=11, score=10)

        syn_gt1 = synapse.Synapse(id=1, location_pre=(0, 0, 0),
                                  location_post=(12, 12, 0), id_skel_pre=1,
                                  id_skel_post=10, score=10)
        syn_gt2 = synapse.Synapse(id=2, location_pre=(0, 0, 0),
                                  location_post=(10, 10, 0), id_skel_pre=1,
                                  id_skel_post=5, score=10)
        syn_gt3 = synapse.Synapse(id=3, location_pre=(10, 10, 0),
                                  location_post=(100, 100, 0), id_skel_pre=1,
                                  id_skel_post=11, score=10)
        with TemporaryDirectory() as d:
            predfile = os.path.join(d, 'pred_synapses.json')
            with open(predfile, 'w') as f:
                json.dump(
                    [syn_pred1.__dict__, syn_pred2.__dict__, syn_pred3.__dict__], f)
            gtfile = os.path.join(d, 'gt_synapses.json')
            with open(gtfile, 'w') as f:
                json.dump([syn_gt1.__dict__, syn_gt2.__dict__, syn_gt3.__dict__], f)

            eval = EvaluateFafb(predfile, gtfile)
            results = eval.get_cremi_score(0, skel_ids=[1, 2])
            self.assertAlmostEqual(results['fscore'], 2 / 3.)

            eval = EvaluateFafb(predfile, gtfile, matching_threshold=50)
            results = eval.get_cremi_score(0, skel_ids=[1, 2])
            self.assertAlmostEqual(results['fscore'], 1 / 3.)




if __name__ == '__main__':
    unittest.main()
