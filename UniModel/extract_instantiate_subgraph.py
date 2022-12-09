import json
import argparse
import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_qid_path', required=True,
                        help='the input data path used for extracting the shortest paths')
    parser.add_argument('--ins_sg_path', required=True,
                        help='the input data path used for extracting the shortest paths')
    parser.add_argument('--output_path', required=True,
                        help='the output data path used for extracting the shortest paths')
    parser.add_argument('--split_list', nargs="+")
    parser.add_argument('--num_process', type=int)
    parser.add_argument('--overwrite', action="store_true")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()

    overwrite_flag = args.overwrite
    ins_sg_path = args.ins_sg_path
    all_split_paths = []
    all_split_idx = []

    for i in range(args.num_process):
        sp = ins_sg_path + "_" + str(i)
        all_split_paths.append(sp)
        all_split_idx.append(i)
    print("Loading %s split subfiles based on %s" % (all_split_idx, ins_sg_path))

    all_ins_sg_data = []
    for path in all_split_paths:
        with open(path, "r") as f:
            split_ins_sg_data = f.readlines()
            split_ins_sg_data = [json.loads(l) for l in split_ins_sg_data]
            all_ins_sg_data.extend(split_ins_sg_data)
    print("Totally load %d abstract subgraph data." % (len(all_ins_sg_data)))

    if not overwrite_flag and os.path.exists(ins_sg_path):
        with open(ins_sg_path, "a+") as f:
            for ins_sg in all_ins_sg_data:
                f.write(json.dumps(ins_sg)+"\n")
    else:
        with open(ins_sg_path, "w") as f:
            for ins_sg in all_ins_sg_data:
                f.write(json.dumps(ins_sg)+"\n")


    # with open(ins_sg_path, "r") as f:
    #     all_ins_sg_data = f.readlines()
    #     all_ins_sg_data = [json.loads(l) for l in all_ins_sg_data]
    #     print("Totally load %d inst subgraph data." % (len(all_ins_sg_data)))

    for split in args.split_list:
        print("Starting extract %s set."%(split))
        out_path = args.output_path.replace("SPLIT", split)
        split_qid_path = args.split_qid_path.replace("SPLIT", split)
        split_qids = np.load(split_qid_path)
        # valid_qids = set()
        all_samples = []
        for abs_sg in tqdm(all_ins_sg_data, total=len(all_ins_sg_data)):
            qid = abs_sg["qid"]
            # valid_qids.add(qid)
            if qid in split_qids:
                all_samples.append(abs_sg)
        # print("Not retrieved qid: ", set(split_qids.tolist())-valid_qids)

        total_num_data = len(split_qids)
        candidates_info = []
        length_of_retrieval_rels = []
        len_of_ret_ent = []
        len_of_ret_tri = []
        avg_answer_recall = []
        avg_rel_recall = []
        print("Starting to aggregate the results of each process!")
        for sample in all_samples:
            # if sample['ans_recall'] < 1:
            #     print("Qid: %s ans recall < 1."%sample['qid'])
            avg_answer_recall.append(sample['ans_recall'])
            avg_rel_recall.append(sample['rel_recall'])
            len_of_ret_ent.append(len(sample["subgraph"][0]))
            len_of_ret_tri.append(len(sample["subgraph"][1]))

        print("Starting write to files!")
        with open(out_path, "w") as f:
            for sample in all_samples:
                f.write(json.dumps(sample) + "\n")

        print("Length of subgraph entities: min:%d max:%d mean:%d" % (min(len_of_ret_ent), max(len_of_ret_ent),
                                                                        np.mean(len_of_ret_ent)))
        print("Length of subgraph triples: min:%d max:%d mean:%d" % (min(len_of_ret_tri), max(len_of_ret_tri),
                                                                      np.mean(len_of_ret_tri)))
        print("Valid subgraph: %d/%d-%.4f"%(len(all_samples), total_num_data, len(all_samples)/total_num_data))
        print("Average answer recall: %.4f" % (np.mean(avg_answer_recall)))
        print("Average relation recall: %.4f" % (np.mean(avg_rel_recall)))
        print("Save the retrieval relations for each query to %s" % (out_path))

