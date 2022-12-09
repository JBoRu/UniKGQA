import os
import argparse
import random

import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import multiprocessing
from SPARQLWrapper.SPARQLExceptions import EndPointInternalError
import sys
sys.path.append("..")
from KnowledgeBase.KG_api import KnowledgeGraph
from KnowledgeBase.sparql_executor import *


def construct_contrastive_pos_neg_paths(sample):
    paths = sample["Paths"]
    if len(paths) == 0:
        return []

    precision_score_list = set([p[2] for p in paths])
    ordered_pre_score_list = sorted(precision_score_list, reverse=True)

    first_score = ordered_pre_score_list[0]
    second_score = ordered_pre_score_list[1] if len(ordered_pre_score_list) >= 2 else 0.0
    third_score = ordered_pre_score_list[2] if len(ordered_pre_score_list) >= 3 else 0.0

    if filter_order == 1:
        real_min_pre = first_score
    elif filter_order == 2:
        real_min_pre = second_score
    elif filter_order == 3:
        real_min_pre = third_score
    elif filter_order == -1:
        real_min_pre = min_precision
    else:
        real_min_pre = min(first_score, min_precision)
    filtered_paths = [p for p in paths if p[2] >= real_min_pre]

    return filtered_paths


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True,
                        help='the input data path used for extracting the shortest paths')
    parser.add_argument('--output_path', required=True,
                        help='the output data path used for extracting the shortest paths')
    parser.add_argument('--min_precision', default=0.01, type=float,
                        help='the min precision used to filter the searched shortest paths')
    parser.add_argument('--filter_order', default=0, type=int,
                        help='the topk searched paths')
    args = parser.parse_args()

    print("Start constructing contrastive training data.")
    return args


if __name__ == '__main__':
    args = _parse_args()
    min_precision = args.min_precision
    filter_order = args.filter_order

    inp_path = args.input_path
    out_path = args.output_path
    print('Input %s to Output %s' % (inp_path, out_path))

    with open(inp_path, "r") as f:
        all_data = f.readlines()
        all_data = [json.loads(l) for l in all_data]

    valid_count = 0
    with open(out_path, "w") as f:
        path_count = []
        for data in all_data:
            valid_paths = construct_contrastive_pos_neg_paths(data)
            path_count.append(len(valid_paths))
            if len(valid_paths) > 0:
                data["Paths"] = valid_paths
                f.write(json.dumps(data)+"\n")
                valid_count += 1
        print("valid sample: %d/%d-%.2f path count: [%d, %d, %.2f]" % (valid_count, len(all_data), valid_count/(len(all_data)+0.0),
                                                                    min(path_count), max(path_count), float(np.mean(path_count))))