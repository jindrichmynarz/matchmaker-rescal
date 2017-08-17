#!/usr/bin/env python3
"""
Command-line interface for evaluation of the RESCAL-based matchmakers.
"""
import argparse
import collections
import edn_format
import evaluation
import logging
import numpy as np
import os
import pkg_resources
import sys
from scipy.io import mmread

def merge_key(k, d1, d2):
    """
    Merge objects `d1` and `d2` for given key `k`.
    """
    is_dict = lambda x: isinstance(x, collections.Mapping)
    all_dicts = lambda *x: all(map(is_dict, x))
    if k in d2:
        return merge(d1[k], d2[k]) if all_dicts(d1[k], d2[k]) else d2[k]
    else:
        return d1[k]

def merge(d1, d2):
    """
    Deep merge dictionaries `d1` and `d2`. Values in `d2` are preferred.
    """
    return {k: merge_key(k, d1, d2) for k in d1}

def parse_matrix_market(path):
    return mmread(path).tocsr()

def load_edn(path):
    with open(path, "r") as f:
        return edn_format.loads(f.read())

def parse_config(path):
    default_config = pkg_resources.resource_filename("resources", "config.edn")
    return merge(load_edn(default_config), load_edn(path))

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
    parser.add_argument("-c", "--config",
                        type = parse_config, default = {},
                        help = "EDN configuration")
    args = parser.parse_args()
    evaluation.run(args)
