import argparse
import os
import json
import numpy as np
import multiprocessing
import sys
sys.path.append("..")
from KnowledgeBase.KG_api import KnowledgeGraph
from KnowledgeBase.sparql_executor import *
from utils import *
from line_profiler import LineProfiler

import sys

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def extract_const_rels(sql, const_ents):
    if "#MANUAL SPARQL" in sql:
        return []
    lines = sql.split("\n")
    lines = [l for l in lines if "FILTER" not in l]
    shot_lines = []
    for ent in const_ents:
        for l in lines:
            if ent in l:
                shot_lines.append(l)
    const_rels = []
    for l in shot_lines:
        eles = l.strip().split(" ")
        assert len(eles) == 4, eles
        if eles[1] not in const_rels:
            assert eles[1].startswith("ns:"), sql
            const_rels.append(eles[1][3:])
    return const_rels

def get_paths_from_tpe_to_ans_with_specific_hop(split, source, target, specific_hop, extra_hop_flag=False):
    all_paths_one_pair = []
    paths = kg.get_paths("sparse", source, target, specific_hop, "None")
    if len(paths) != 0:
        for p in paths:
            p = [source] + p
            if p not in all_paths_one_pair:
                all_paths_one_pair.append(p)
    return all_paths_one_pair


def get_paths_from_tpe_to_ans(split, source, target, max_hop, extra_hop_flag=False):
    all_paths_one_pair = []
    for hop in range(1, max_hop+1):
        paths = kg.get_paths("sparse", source, target, hop, "None")
        if len(paths) != 0:
            for p in paths:
                p = [source] + p
                if p not in all_paths_one_pair:
                    all_paths_one_pair.append(p)

            if split == "train" and task_name == "kgc":
                continue

            if extra_hop_flag and (hop + 1) <= max_hop:
                paths = kg.get_paths("sparse", source, target, hop+1, "None")
                if len(paths) != 0:
                    for p in paths:
                        p = [source] + p
                        if p not in all_paths_one_pair:
                            all_paths_one_pair.append(p)
            break
        else:
            continue
    return all_paths_one_pair


def get_paths_from_tpe_to_ans_for_kgc(split, source, target, max_hop, extra_hop_flag, exclude_rel):
    all_paths_one_pair = []
    for hop in range(1, max_hop+1):
        paths = kg.get_paths_kgc("sparse", split, source, target, hop, exclude_rel)
        if len(paths) != 0:
            for p in paths:
                p = [source] + p
                if p not in all_paths_one_pair:
                    all_paths_one_pair.append(p)

            if extra_hop_flag and (hop + 1) <= max_hop:
                paths = kg.get_paths_kgc("sparse", split, source, target, hop+1, exclude_rel)
                if len(paths) != 0:
                    for p in paths:
                        p = [source] + p
                        if p not in all_paths_one_pair:
                            all_paths_one_pair.append(p)
            break
        else:
            continue
    return all_paths_one_pair


def get_paths_from_tpe_to_cpe(source, target, min_hop, max_hop, task_name):
    all_paths_one_pair = []
    if task_name == "cwq":
        paths = kg.get_limit_paths(source, target, max(min_hop, 2), min(max_hop+2, 5))
    elif task_name == "webqsp":
        paths = kg.get_limit_paths(source, target, max(min_hop, 2), min(max_hop+1, 4))
    elif task_name == "kgc":
        return all_paths_one_pair
    else:
        print("Not implement the task %s!"%(task_name))
        paths = []
    if len(paths) != 0:
        for p in paths:
            p = [source] + p
            if len(set(p)) == len(p) and p not in all_paths_one_pair:
                all_paths_one_pair.append(p)
    return all_paths_one_pair


def get_paths_from_tpe_to_cpes(source, targets, task_name, paths_for_ans):
    valid_paths_for_tpe_ans, valid_paths_for_tpe_cpe, valid_const_rels = [], [], set()
    for path in paths_for_ans:
        paths_from_tpe_to_cpe = kg.get_paths_from_tpe_to_cpes(source, targets, path[1:], task_name)
        if len(paths_from_tpe_to_cpe) != 0:
            flag = 0
            for ptc in paths_from_tpe_to_cpe:
                ptc = [source] + ptc
                if len(set(ptc)) == len(ptc) and ptc not in valid_paths_for_tpe_cpe:
                    valid_paths_for_tpe_cpe.append(ptc)
                    const_rels = set(ptc)-set(path)
                    for rel in const_rels:
                        valid_const_rels.add(rel)
                    flag = 1
            if flag:
                valid_paths_for_tpe_ans.append(path)
    if len(valid_paths_for_tpe_ans) == 0:
        valid_paths_for_tpe_ans = paths_for_ans
    return valid_paths_for_tpe_ans, valid_paths_for_tpe_cpe, valid_const_rels


def extract_union_paths(paths4ans, paths4cpe):
    new_paths4ans = []
    new_paths4cpe = []
    const_rels = []
    for p0 in paths4ans:
        for p1 in paths4cpe:
            flag = False
            common_len = 0
            for i in zip(p0[1:], p1[1:]):
                if len(set(i)) == 1:
                    flag = True
                else:
                    break
                common_len += 1
            if flag: # must have common prefix
                # must have common prefix and common length is a trick for KBQA dataset!
                if common_len == len(p0)-1: # const rels after ?x
                    if common_len == len(p1)-1-1:
                        const_rels_tmp = list(set(p1) - set(p0))
                        if len(const_rels_tmp) == 1:
                            if p0 not in new_paths4ans:
                                new_paths4ans.append(p0)
                            if p1 not in new_paths4cpe:
                                new_paths4cpe.append(p1)
                            for const_rel in const_rels_tmp:
                                if const_rel not in const_rels:
                                    const_rels.append(const_rel)
                        # else:
                        #     print(p0, p1)
                    elif common_len == len(p1)-1-2:
                        const_rels_tmp = list(set(p1) - set(p0))
                        if len(const_rels_tmp) == 2:
                            if p0 not in new_paths4ans:
                                new_paths4ans.append(p0)
                            if p1 not in new_paths4cpe:
                                new_paths4cpe.append(p1)
                            for const_rel in const_rels_tmp:
                                if const_rel not in const_rels:
                                    const_rels.append(const_rel)
                        # else:
                        #     print(p0, p1)
                elif common_len == len(p0)-2:  # const rels before ?x
                    const_rels_tmp = list(set(p1) - set(p0))
                    if len(const_rels_tmp) == 1:
                        if p0 not in new_paths4ans:
                            new_paths4ans.append(p0)
                        if p1 not in new_paths4cpe:
                            new_paths4cpe.append(p1)
                        for const_rel in const_rels_tmp:
                            if const_rel not in const_rels:
                                const_rels.append(const_rel)
                    # else:
                    #     print(p0, p1)
    return new_paths4ans, new_paths4cpe, const_rels


def construct_pos_and_neg_rels(sample):
    num_que_no_paths = 0  # record the number of question example where don't have any shortest paths within 2 hops

    # iterate every sample
    qid = sample["ID"]
    # if qid not in ["WebQTrn-354.P0", "WebQTrn-1592.P0", "WebQTrn-3690.P0", "WebQTrn-226.P0"]:
    #     return None
    question = sample["ProcessedQuestion"] if "ProcessedQuestion" in sample else sample["RawQuestion"]
    parse = sample["Parse"]
    tpe = parse["TopicEntityMid"]
    links = parse["GoldEntityMid"]
    if tpe is None:
        print("Qid: %s does not have any topic entity!" % (qid))
        return None
    if use_masked_question and 'SimplifiedQuestion' in parse:
        question = parse["SimplifiedQuestion"]
    # get the constraint relations
    const_ents = set(links) - {tpe}
    if len(const_ents) == 0:
        gold_const_rels = []
    else:
        if "GoldConstRels" in parse:
            gold_const_rels = parse['GoldConstRels']
        else:
            gold_const_rels = extract_const_rels(parse["Sparql"], const_ents)

    # get the answers
    answers = parse["Answers"]
    answers = [a["AnswerArgument"] for a in answers]  # get the mid of each answer
    if len(answers) == 0:
        print("Qid: %s does not have any answer entity!" % (qid))
        return None

    # all_paths_one_pair = []
    paths_for_tpe_ans = []
    paths_for_tpe_cpe = []

    t = tpe.replace('?x', '')
    split = sample['Split']
    for ans in answers:
        if task_name == "kgc":
            exclude_rel = parse['InferentialChain'][0]
            paths = get_paths_from_tpe_to_ans_for_kgc(split, t, ans, max_hop, extra_hop_flag, exclude_rel)
        elif task_name == 'metaqa':
            paths = get_paths_from_tpe_to_ans_with_specific_hop(split, t, ans, max_hop, extra_hop_flag)
        else:
            paths = get_paths_from_tpe_to_ans(split, t, ans, max_hop, extra_hop_flag)
        for path in paths:
            if path not in paths_for_tpe_ans:
                paths_for_tpe_ans.append(path)
    if len(paths_for_tpe_ans) == 0:
        num_que_no_paths += 1
        print("Qid:%s doesn't extract any paths within %d hop between qa!" % (qid, max_hop))
        return None
    if task_name != "kgc":
        # consume time
        valid_paths_for_tpe_ans, valid_paths_for_tpe_cpe, valid_const_rels = get_paths_from_tpe_to_cpes(t, const_ents, task_name, paths_for_tpe_ans)
        flag = 0 if len(valid_const_rels) == 0 else 1
    else:
        valid_paths_for_tpe_ans = paths_for_tpe_ans
        valid_const_rels = []
        flag = 0

    paths_with_score_one_pair = []
    # iterate every searched paths
    for path in valid_paths_for_tpe_ans:
        can_ans = kg.get_tails_with_path("sparse", path)
        if len(can_ans) == 0:
            # print(path)
            print("Retrieved one shortest path but can not deduce candidate answers.")
            continue
        # compute the precision of this paths
        precision = compute_precision(can_ans, answers)
        paths_with_score_one_pair.append((question, path[1:], precision))
    ordered_paths_with_score = sorted(paths_with_score_one_pair, key=lambda x: x[2], reverse=True)

    if len(ordered_paths_with_score) == 0:
        print("Qid: %s can't retrieve any valid path that can deduce candidate answers!" % (qid))
        return None

    pos_relations = []
    for path_score in ordered_paths_with_score:
        _, path, _ = path_score
        for r in path:
            if r not in pos_relations:
                pos_relations.append(r)
    for cr in valid_const_rels:
        if cr not in pos_relations:
            pos_relations.append(cr)

    if len(gold_const_rels) != 0:
        const_rels_recall = len(set(gold_const_rels) & set(valid_const_rels)) / len(set(gold_const_rels))
    else:
        const_rels_recall = 1.0

    # get other meta information and dump to file
    new_sample = {"ID": qid, "TopicEntityMid": tpe, "ConstEntityMid": list(const_ents), "Answers": answers,
                  "ConstRelations": list(valid_const_rels), "ConstRelationsRecall": round(const_rels_recall, 3),
                  "PositiveRelations": list(pos_relations), "Paths": ordered_paths_with_score, "Flag": flag}

    return new_sample


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', required=True, type=str)
    parser.add_argument('--input_path', required=True,
                        help='input path of the original data')
    parser.add_argument('--output_path', required=True,
                        help='output path of shortest path data')
    parser.add_argument('--ids_path', default=None, type=str,
                        help='the qids of each split set')
    parser.add_argument('--exclude_path', default=None, type=str,
                        help='the qids of no shortest path sample')
    parser.add_argument('--log_path', default=None, type=str,
                        help='the path of logging.')
    parser.add_argument('--use_masked_question', action="store_true",
                        help='whether mask the string of topic entity in original question')
    parser.add_argument('--extra_hop_flag', action="store_true",
                        help='whether retrieve extra hop')
    parser.add_argument('--overwrite', action="store_true",
                        help='whether to overwrite the already saved files')
    parser.add_argument('--max_num_processes', default=1, type=int)
    parser.add_argument('--max_hop', default=2, type=int,
                        help='the max search hop of the shortest paths')
    parser.add_argument('--dense_kg_source', default="virtuoso", help='the KG source (ex. virtuoso, triples, ckpt)')
    parser.add_argument('--dense_kg_source_path', default=None, help='the KG source path for triples or ckpt types')
    parser.add_argument('--sparse_kg_source_path', default=None, help='the sparse triples file')
    parser.add_argument('--sparse_ent_type_path', default=None, help='the file of entities type of sparse triples')
    parser.add_argument('--sparse_ent2id_path', default=None, help='the sparse ent2id file')
    parser.add_argument('--sparse_rel2id_path', default=None, help='the sparse rel2id file')
    args = parser.parse_args()

    print("Start extracting weak supervision paths to train the retriever.")

    return args


if __name__ == '__main__':
    args = _parse_args()
    log_path = args.log_path
    # sys.stdout = Logger(log_path, sys.stdout)
    # sys.stderr = Logger(log_path, sys.stderr)  # redirect std err, if necessary

    kg = KnowledgeGraph(args.dense_kg_source, (args.sparse_kg_source_path, args.sparse_ent_type_path), args.sparse_ent2id_path, args.sparse_rel2id_path)

    num_process = args.max_num_processes
    chunk_size = 1
    max_hop = args.max_hop
    use_masked_question = args.use_masked_question
    extra_hop_flag = args.extra_hop_flag
    task_name = args.task_name
    overwrite_flag = args.overwrite

    inp = args.input_path
    oup = args.output_path
    no_paths_ids_p = args.exclude_path
    print('Input %s to Output %s' % (inp, oup))
    with open(inp, "r") as f:
        all_lines = f.readlines()
        all_data = [json.loads(l) for l in all_lines]

    already_processed_qids = []
    all_ids = []
    already_processed_qids_data_dict = {}
    if not overwrite_flag and os.path.exists(oup):
        with open(oup, "r") as f:
            processed_lines = f.readlines()
            processed_samples = [json.loads(line) for line in processed_lines]
        for sample in processed_samples:
            already_processed_qids.append(sample['ID'])
            already_processed_qids_data_dict[sample['ID']] = sample
        print("There are %d processed samples."%(len(already_processed_qids)))
        if no_paths_ids_p is not None and os.path.exists(no_paths_ids_p):
            with open(no_paths_ids_p, "r") as f:
                all_ids = f.readlines()
                all_ids = [l.strip().strip("\n") for l in all_ids]
            print("There are %d no paths samples already processed." % (len(all_ids)))

    retrieved_data_ids = []
    ids_const_rel_recall = defaultdict()
    if overwrite_flag:
        mode = "w"
    else:
        mode = "a+"
    count = 0
    if not overwrite_flag:
        print("Exclude already processed samples.")
        ready_data = []
        for data in all_data:
            qid = data["ID"]
            if qid in already_processed_qids:
                retrieved_data_ids.append(qid)
                count += 1
                ids_const_rel_recall[qid] = already_processed_qids_data_dict[qid]['ConstRelationsRecall']
            elif qid in all_ids:
                continue
            else:
                ready_data.append(data)
    else:
        ready_data = all_data
    print("Start process %d new samples." % (len(ready_data)))
    with open(oup, mode) as fout:
        with multiprocessing.Pool(num_process) as p:
            new_samples = p.imap_unordered(construct_pos_and_neg_rels, ready_data, chunksize=chunk_size)
            for sample in tqdm(new_samples, total=len(ready_data)):
                if sample is not None:
                    retrieved_data_ids.append(sample['ID'])
                    fout.write(json.dumps(sample) + "\n")
                    count += 1
                    ids_const_rel_recall[sample['ID']] = sample['ConstRelationsRecall']
            print("Extract %d / %d samples" % (count, len(all_data)))

        # used for deug
        # lp = LineProfiler()
        # lp.add_function(get_paths_from_tpe_to_ans)
        # lp.add_function(get_paths_from_tpe_to_cpes)
        # lp.add_function(kg.get_paths_from_tpe_to_cpes)
        # lp.add_function(kg.sparse_kg.get_triples_along_relation_path)
        # lp.add_function(kg.sparse_kg.path_join)
        # lp_wrapper = lp(construct_pos_and_neg_rels)
        # for sample in tqdm(ready_data[:5], total=len(ready_data)):
        #     result = lp_wrapper(sample)
        #     if result is not None:
        #         retrieved_data_ids.append(result['ID'])
        #         # fout.write(json.dumps(result) + "\n")
        #         count += 1
        #         ids_const_rel_recall[result['ID']] = result['ConstRelationsRecall']
        # lp.print_stats()
        # print("Extract %d / %d samples" % (count, len(all_data)))


    print("Starting aggregate results.")
    ids_path = args.ids_path
    for split in ['train', 'dev', 'test']:
        p = ids_path.replace('SPLIT', split)
        all_ids = np.load(p)
        count = 0.0
        const_rel = []
        for ids in all_ids:
            if ids in retrieved_data_ids:
                count += 1
                const_rel.append(ids_const_rel_recall[ids])
        print("%s: Samples with paths-%d/%d-%.2f Const rels recall-%.3f" %
              (split, count, len(all_ids), count / len(all_ids), float(np.mean(const_rel))))