"""
Command-line interface for evaluation of the RESCAL-based matchmakers.
"""
import argparse
import edn_format
import evaluation
import logging
import numpy as np
import sys
from scipy.io import mmread

DEFAULT_CONFIG = {
    "evaluation": {
        "folds": 5,
        "top-k": 10
    },
    "matchmaker": {
        "type": "rescal",
        "conv": 0.001,
        "init": "nvecs",
        "rank": 50,
        "regularization": {
            "lambdaA": 0,
            "lambdaR": 0,
            "lambdaV": 0
        }
    }
}

def merge(d1, d2):
    """
    Deep merge dictionaries `d1` and `d2`.
    """
    return {
	k: merge(d1[k], d2[k])
        if k in d1 and isinstance(d1[k], dict) and isinstance(d2[k], dict) else d2[k]
        for k in d2
    }

def parse_matrix_market(path):
    return mmread(path).tocsr()

def parse_config(path):
    with open(path, "r") as f:
        return merge(DEFAULT_CONFIG, edn_format.loads(f.read()))

def filter_entities(predicate, lines):
    return np.array([idx for idx, line in enumerate(lines) if predicate(line)], dtype = np.int32)

def parse_headers(path):
    with open(path, "r") as f:
        lines = f.readlines()
        # FIXME: Abusing IRI opacity for quick'n'dirty identification of bidders. 
        return {
            "bidders": filter_entities(lambda line: "/business-entity/" in line, lines),
        }

if __name__ == "__main__":
    # Logging configuration
    logging.basicConfig(level = logging.WARNING, stream = sys.stdout,
                        format = "%(asctime)s %(name)s %(levelname)-8s %(message)s",
                        datefmt = "%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser(description = "Evaluation runner for the RESCAL-based matchmakers")
    parser.add_argument("-g", "--ground-truth",
                        required = True, type = parse_matrix_market,
                        help = "Matrix with the ground truth relation")
    parser.add_argument("-s", "--slices", 
                        nargs = "*", type = parse_matrix_market,
                        help = "Matrices for tensor slices")
    parser.add_argument("--headers",
                        required = True, type = parse_headers,
                        help = "Line-separated IRIs of entities, where line number is the entity's index.")
    parser.add_argument("-c", "--config",
                        type = parse_config, default = DEFAULT_CONFIG,
                        help = "EDN configuration")
    args = parser.parse_args()
    evaluation.run(args)
