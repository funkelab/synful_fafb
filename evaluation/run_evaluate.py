import json
import logging
import sys

from evaluate_fafb import EvaluateFafb

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config_file = sys.argv[1]

    with open(config_file, 'r') as f:
        config = json.load(f)

    score_thresholds = config['score_thresholds']
    del config['score_thresholds']
    evaluate = EvaluateFafb(**config)
    evaluate.evaluate_synapse_complete(score_thresholds)
