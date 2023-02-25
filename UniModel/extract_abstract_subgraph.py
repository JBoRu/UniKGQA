import json
import argparse
import glob
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm


def extract_shortest_paths(short_paths, min_precision, filter_order):
    paths = short_paths["Paths"]
    if len(paths) == 0:
        return []

    precision_score_list = set([p[2] for p in paths])
    ordered_pre_score_list = sorted(precision_score_list, reverse=True)
    first_score = ordered_pre_score_list[0]
    second_score = ordered_pre_score_list[1] if len(ordered_pre_score_list) >= 2 else first_score
    third_score = ordered_pre_score_list[2] if len(ordered_pre_score_list) >= 3 else second_score

    if filter_order == 0:
        real_min_pre = min_precision
    elif filter_order == 1:
        real_min_pre = first_score
    elif filter_order == 2:
        real_min_pre = second_score
    elif filter_order == 3:
        real_min_pre = third_score
    else:
        if first_score < min_precision:
            real_min_pre = first_score
        else:
            real_min_pre = min_precision

    selected_paths = []
    for p_s in paths:
        _, path, precision = p_s
        if precision >= real_min_pre:
            selected_paths.append(path)

    return selected_paths


def compute_gold_const_rel_recall(retri_const_rels, gold_const_rels):
    gold_const_rels = set(gold_const_rels)
    rel_recall = len(set(retri_const_rels) & gold_const_rels) / (len(gold_const_rels) + 0.0)
    return rel_recall


def compute_gold_rel_str_recall(tris_per_hop, gold_rel_structure):
    last_ans = [0]
    all_gold_rels = set()
    all_shot_rels = set()
    max_hop = len(gold_rel_structure)
    for hop_id, tris in tris_per_hop.items():
        hop_id = int(hop_id)
        if hop_id < max_hop:
            gold_rels = gold_rel_structure[str(hop_id)]
            all_gold_rels.update(gold_rels)
            shot_rels = []
            cur_ans = []
            for tri in tris:
                h, r, t = tri
                if h in last_ans and r in gold_rels:
                    if r not in shot_rels:
                        shot_rels.append(r)
                    if t not in cur_ans:
                        cur_ans.append(t)
            last_ans = deepcopy(cur_ans)
            all_shot_rels.update(shot_rels)
    if len(all_gold_rels) == 0:
        print("Error:", tris_per_hop, gold_rel_structure)
        return 0.0
    rel_recall = len(all_gold_rels & all_shot_rels) / (len(all_gold_rels) + 0.0)
    return rel_recall


def get_abstract_answers_by_paths(weak_gold_paths, tris_per_hop):
    from copy import deepcopy

    result_ans = set()
    for path in weak_gold_paths:
        last_ans = [0]
        len_path = len(path)
        for hi, tris in tris_per_hop.items():
            hi = int(hi)
            if hi < len_path:
                cur_ans = []
                gold_rel = path[hi]
                for tri in tris:
                    h, r, t = tri
                    if h in last_ans and r == gold_rel:
                        if t not in cur_ans:
                            cur_ans.append(t)
                last_ans = deepcopy(cur_ans)
        if len(last_ans) == 1 and 0 in last_ans:
            continue
        result_ans.update(last_ans)
    return list(result_ans)


def get_answers_from_shortest_paths(ori_datasets_dict, all_samples, shortest_paths_dict, min_path_precision,
                                    filter_order):
    new_samples = []
    for sample in all_samples:
        sample_id = sample['ID']

        if split != "test" and sample_id not in shortest_paths_dict:
            continue
        elif split == "test" and sample_id not in shortest_paths_dict:
            short_paths = None
        else:
            short_paths = shortest_paths_dict[sample_id]

        ori_data = ori_datasets_dict[sample_id]
        parse = ori_data["Parse"]
        if "GoldSqlStructure" in ori_data:
            gold_rel_structure = ori_data["GoldSqlStructure"]
        else:
            gold_rel_structure = defaultdict(list)
        gold_const_rels = parse['GoldConstRels'] if 'GoldConstRels' in parse else []

        if short_paths is None:
            weak_gold_paths = []
            print("Qid: %s doesn't have any shortest path between qa." % (sample_id))
        else:
            weak_super_labels = extract_shortest_paths(short_paths, min_path_precision, filter_order)
            weak_gold_paths = weak_super_labels

        tris_per_hop = sample['tris_per_hop']
        all_retri_const_rels = sample['all_retri_const_rels']

        if weak_gold_paths:
            abs_answers = get_abstract_answers_by_paths(weak_gold_paths, tris_per_hop)
        else:
            abs_answers = []

        answers = []
        for a in abs_answers:
            answers.append({'kb_id': a, 'text': a})
        if len(answers) == 0:
            print("Qid:%a doesn't have any abs answers."%(sample_id))

        if len(gold_rel_structure) != 0:
            rel_struct_rec = compute_gold_rel_str_recall(tris_per_hop, gold_rel_structure)
        elif len(gold_const_rels) != 0:
            rel_struct_rec = compute_gold_const_rel_recall(all_retri_const_rels, gold_const_rels)
        else:
            rel_struct_rec = -1.0

        entities_set = set()
        triples_set = set()
        for hop_id, tris in tris_per_hop.items():
            tris = [tuple(tri) for tri in tris]
            triples_set.update(set(tris))
            for tri in tris:
                h, r, t = tri
                entities_set.add(h)
                entities_set.add(t)

        sample_new = {'ID': sample_id, 'question': sample['question'], 'entities': sample['entities'],
                      'const_entities': sample['const_entities'], 'answers': answers,
                      'rel_struct_rec': round(rel_struct_rec, 4),
                      'subgraph': {"tuples": list(triples_set), "entities": list(entities_set)}}
        new_samples.append(sample_new)
    return new_samples


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_qid_path', required=True,
                        help='the input data path used for extracting the shortest paths')
    parser.add_argument('--abs_sg_path', required=True,
                        help='the input data path used for extracting the shortest paths')
    parser.add_argument('--output_path', required=True,
                        help='the output data path used for extracting the shortest paths')
    parser.add_argument('--all_output_path', required=True)
    parser.add_argument('--ori_path', required=True)
    parser.add_argument('--all_shortest_path', required=True)
    parser.add_argument('--split_list', nargs="+")
    parser.add_argument('--num_process', type=int)
    parser.add_argument('--min_path_precision', default=0.1, type=float)
    parser.add_argument('--filter_order', default=0, type=int, help='the minimal cosine similarity')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()

    abs_sg_path = args.abs_sg_path
    all_output_path = args.all_output_path
    all_shortest_path = args.all_shortest_path
    ori_path = args.ori_path
    min_path_precision = args.min_path_precision
    filter_order = args.filter_order

    # load original datasets
    with open(ori_path, "r") as f:
        all_lines = f.readlines()
        ori_dataset = [json.loads(l) for l in all_lines]
    ori_dataset_dict = {l["ID"]: l for l in ori_dataset}
    print('Load original data from %s' % ori_path)

    # load all split sub files
    all_split_paths = []
    all_split_idx = []
    for i in range(args.num_process):
        sp = abs_sg_path + "_" + str(i)
        all_split_paths.append(sp)
        all_split_idx.append(i)
    print("Loading %s split sub files based on %s" % (all_split_idx, abs_sg_path))
    all_abs_sg_data = []
    for path in all_split_paths:
        with open(path, "r") as f:
            split_abs_sg_data_lines = f.readlines()
            split_abs_sg_data = [json.loads(l) for l in split_abs_sg_data_lines]
            all_abs_sg_data.extend(split_abs_sg_data)
    print("Totally load %d abstract subgraph data." % (len(all_abs_sg_data)))

    # load all shortest path
    with open(all_shortest_path, "r") as f:
        all_lines = f.readlines()
        shortest_paths_dataset = [json.loads(l) for l in all_lines]
        shortest_paths_dict = {l["ID"]: l for l in shortest_paths_dataset}
    print('Load shortest paths from %s' % all_shortest_path)

    for split in args.split_list:
        print("Starting extract %s set." % (split))
        out_path = args.output_path.replace("SPLIT", split)
        split_qid_path = args.split_qid_path.replace("SPLIT", split)
        split_qids = np.load(split_qid_path)
        all_samples = []
        for abs_sg in tqdm(all_abs_sg_data, total=len(all_abs_sg_data)):
            q_id = abs_sg["ID"]
            if q_id in split_qids:
                all_samples.append(abs_sg)
        new_all_samples = get_answers_from_shortest_paths(ori_dataset_dict, all_samples, shortest_paths_dict,
                                                          min_path_precision, filter_order)
        total_num_data = len(split_qids)
        ans_valid_num_data = 0.0
        rel_valid_num_data = 0.0
        rel_struct_acc = []
        rel_struct_recall = []
        print("Starting to aggregate the results of each process!")
        for sample in new_all_samples:
            if len(sample["answers"]) > 0:
                ans_valid_num_data += 1
            if sample["rel_struct_rec"] != -1:
                rel_valid_num_data += 1
                rel_struct_recall.append(sample["rel_struct_rec"])
            if sample["rel_struct_rec"] == 1:
                rel_struct_acc.append(1)

        valid_ratio = ans_valid_num_data / total_num_data
        if rel_valid_num_data == 0:
            rel_struct_acc = 0
        else:
            rel_struct_acc = sum(rel_struct_acc) / rel_valid_num_data
        rel_struct_recall = np.mean(rel_struct_recall)
        len_of_ret_ent = []
        len_of_ret_tri = []

        print("Starting write to files!")
        with open(out_path, "w") as f:
            for sample in new_all_samples:
                len_of_ret_ent.append(len(sample["subgraph"]["entities"]))
                len_of_ret_tri.append(len(sample["subgraph"]["tuples"]))
                f.write(json.dumps(sample) + "\n")

        print("Length of subgraph entities: min:%d max:%d mean:%d" % (min(len_of_ret_ent), max(len_of_ret_ent),
                                                                      np.mean(len_of_ret_ent)))
        print("Length of subgraph triples: min:%d max:%d mean:%d" % (min(len_of_ret_tri), max(len_of_ret_tri),
                                                                     np.mean(len_of_ret_tri)))
        print("Extracted abstract subgraph: %.4f-(%d/%d)" % (len(all_samples) / (total_num_data + 0.0),
                                                             len(all_samples), total_num_data))
        print("Answer valid ratio: %.4f-(%d/%d)" % (valid_ratio, ans_valid_num_data, total_num_data))
        print("Relation structure acc:%.4f recall:%.4f valid:%.4f-%d/%d " % (rel_struct_acc, rel_struct_recall,
                                                                             rel_valid_num_data / (
                                                                                         total_num_data + 0.0),
                                                                             rel_valid_num_data, total_num_data))
        print("Save the retrieval relations for each query to %s" % (out_path))
