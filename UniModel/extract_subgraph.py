import math
import random
import time
from copy import deepcopy

import json
import os
import numpy as np
import argparse
import sys
from line_profiler import LineProfiler
import multiprocessing

sys.path.append("..")
from KnowledgeBase.KG_api import KnowledgeGraph
from KnowledgeBase.KG_api import KnowledgeGraphSparse
from KnowledgeBase.sparql_executor import *
import multiprocessing as mp

def get_subgraph(idx, args, datasets, output_path, max_hop):
    cur_op = output_path + "_" + str(idx)
    with open(cur_op, "w") as f:
        for data in tqdm(datasets, total=len(datasets), desc="PID: %d" % (os.getpid())):
            qid = data["ID"]
            if qid in alrea_qid:
                continue
            split = data["Split"]
            parse = data["Parse"]
            # get the topic entities and answer entities
            tpe = parse["TopicEntityMid"]  # only one topic entity mid in default
            if tpe is None:
                if split == 'test':
                    print("Qid: %s does not have any topic entity. We skip this parse of example." % qid)
                continue
            cpes = set(parse["GoldEntityMid"]) - {tpe}

            triples = kg.get_subgraph_within_khop(tpe, cpes, max_hop)

            if len(triples) != 0:
                new_data = {"ID":qid, "Subgraph":triples}
                f.write(json.dumps(new_data) + "\n")
            else:
                print("Qid:%s doesn't extract any valid subgraph!" % (qid))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', required=True, type=str)
    parser.add_argument('--ori_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--max_num_processes', default=1, type=int)
    parser.add_argument('--max_hop', default=2, type=int, help='retrieve the topk score paths')
    parser.add_argument('--max_num_triples', default=1500, type=int, help='retrieve the topk score paths')
    parser.add_argument('--dense_kg_source', default="virtuoso", help='the KG source (ex. virtuoso, triples, ckpt)')
    parser.add_argument('--dense_kg_source_path', default=None, help='the KG source path for triples or ckpt types')
    parser.add_argument('--sparse_kg_source_path', default=None, help='the sparse triples file')
    parser.add_argument('--sparse_ent_type_path', default=None, help='the file of entities type of sparse triples')
    parser.add_argument('--sparse_ent2id_path', default=None, help='the sparse ent2id file')
    parser.add_argument('--sparse_rel2id_path', default=None, help='the sparse rel2id file')


    args = parser.parse_args()

    print("Start retrieving the subgraph.")
    return args


if __name__ == '__main__':
    args = _parse_args()
    kg = KnowledgeGraph(args.dense_kg_source, (args.sparse_kg_source_path, args.sparse_ent_type_path),
                        args.sparse_ent2id_path, args.sparse_rel2id_path)

    num_process = args.max_num_processes
    overwrite_flag = args.overwrite
    max_hop = args.max_hop
    print("Parent pid:", os.getpid())
    ori_path = args.ori_path
    out_path = args.output_path
    print('Input %s to Output %s' % (ori_path, out_path))

    alrea_qid = []
    if not overwrite_flag and os.path.exists(out_path):
        with open(out_path, "r") as f:
            already_lines = f.readlines()
            already_lines = [json.loads(l) for l in already_lines]
            alrea_qid = [l['id'] for l in already_lines]
    alrea_count = len(alrea_qid)
    print("There are %d samples have been already processed." % (alrea_count))

    with open(ori_path, "r") as f:
        all_lines = f.readlines()
        ori_dataset = [json.loads(l) for l in all_lines]
    print('Load %d original data from %s' % (len(ori_dataset), ori_path))

    if len(alrea_qid) > 0:
        ready_data = []
        for sample in ori_dataset:
            if sample['ID'] in alrea_qid:
                ready_data.append(sample)
    else:
        ready_data = ori_dataset

    print("Start process %d new samples." % (len(ready_data)))
    split_index = len(ready_data) // num_process + 1

    # p = mp.Pool(num_process)
    # for idx in range(num_process):
    #     select_data = ready_data[idx * split_index: (idx + 1) * split_index]
    #     p.apply_async(get_subgraph, args=(idx, args, select_data, out_path, max_hop))
    # p.close()
    # p.join()
    get_subgraph(0, args, ready_data, out_path, 4)

    # used for deug
    # lp = LineProfiler()
    # lp.add_function(get_subgraph)
    # lp.add_function(KnowledgeGraph.deduce_relation_leaves_and_nodes_by_path)
    # lp.add_function(KnowledgeGraphSparse.deduce_relation_leaves_and_nodes_by_path)
    # lp.add_function(KnowledgeGraphSparse.deduce_next_triples_by_path_wo_reverse)
    # lp_wrapper = lp(extract_abstract_sg_from_kb)
    # results = lp_wrapper(0, 0, args, ori_dataset[0:5], inp_dataset_dict, alrea_qid, out_path)
    # lp.print_stats()
    # exit(0)

    # all_split_paths = []
    # all_split_idx = []
    # for idx in range(num_process):
    #     sp = out_path + "_" + str(idx)
    #     all_split_paths.append(sp)
    #     all_split_idx.append(idx)
    # print("Loading %s split sub files based on %s" % (all_split_idx, out_path))
    #
    # all_abs_sg_data = []
    # for path in all_split_paths:
    #     with open(path, "r") as f:
    #         split_abs_sg_data = f.readlines()
    #         # split_abs_sg_data = [json.loads(l) for l in split_abs_sg_data]
    #         all_abs_sg_data.extend(split_abs_sg_data)
    # print("Totally load %d abstract subgraph data." % (len(all_abs_sg_data)))
    #
    # with open(out_path, "w") as f:
    #     for data in all_abs_sg_data:
    #         # f.write(json.dumps(data)+"\n")
    #         f.write(data.strip("\n")+"\n")