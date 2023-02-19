import argparse
import numpy as np
import multiprocessing
import sys

sys.path.append("..")
from KnowledgeBase.KG_api import KnowledgeGraph
from KnowledgeBase.sparql_executor import *
from utils import *


def get_paths_from_tpe_to_ans_with_specific_hop(source, target, specific_hop):
    all_paths_one_pair = []
    paths = kg.get_paths(source, target, specific_hop)
    if len(paths) != 0:
        for p in paths:
            p = [source] + p
            if p not in all_paths_one_pair:
                all_paths_one_pair.append(p)
    return all_paths_one_pair


def get_paths_from_tpe_to_ans(source, target, max_hop, extra_hop_flag=False):
    all_paths_one_pair = []
    for hop in range(1, max_hop + 1):
        paths = kg.get_paths(source, target, hop)
        if len(paths) != 0:
            for p in paths:
                p = [source] + p
                if p not in all_paths_one_pair:
                    all_paths_one_pair.append(p)

            if extra_hop_flag and (hop + 1) <= max_hop:
                paths = kg.get_paths(source, target, hop + 1)
                if len(paths) != 0:
                    for p in paths:
                        p = [source] + p
                        if p not in all_paths_one_pair:
                            all_paths_one_pair.append(p)
            break
        else:
            continue
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
                    const_rels = set(ptc) - set(path)
                    for rel in const_rels:
                        valid_const_rels.add(rel)
                    flag = 1
            if flag:
                valid_paths_for_tpe_ans.append(path)
    if len(valid_paths_for_tpe_ans) == 0:
        valid_paths_for_tpe_ans = paths_for_ans
    return valid_paths_for_tpe_ans, valid_paths_for_tpe_cpe, valid_const_rels


def construct_pos_and_neg_rels(sample):
    num_que_no_paths = 0  # record the number of question example where don't have any shortest paths within 2 hops

    # iterate every sample
    qid = sample["ID"]
    question = sample["ProcessedQuestion"] if "ProcessedQuestion" in sample else sample["RawQuestion"]
    parse = sample["Parse"]
    tpe = parse["TopicEntityMid"]
    links = parse["GoldEntityMid"]
    if tpe is None:
        print("Qid: %s does not have any topic entity!" % (qid))
        return None
    # get the constraint relations
    const_ents = set(links) - {tpe}
    if len(const_ents) == 0:
        gold_const_rels = []
    elif "GoldConstRels" in parse:
        gold_const_rels = parse['GoldConstRels']
    else:
        gold_const_rels = []

    # get the answers
    answers = parse["Answers"]
    answers = [a["AnswerArgument"] for a in answers]  # get the mid of each answer
    if len(answers) == 0:
        print("Qid: %s does not have any answer entity!" % (qid))
        return None

    paths_for_tpe_ans = []

    t = tpe.replace('?x', '')
    split = sample['Split']
    for ans in answers:
        if task_name == 'metaqa':
            paths = get_paths_from_tpe_to_ans_with_specific_hop(t, ans, max_hop)
        else:
            paths = get_paths_from_tpe_to_ans(t, ans, max_hop, extra_hop_flag)
        for path in paths:
            if path not in paths_for_tpe_ans:
                paths_for_tpe_ans.append(path)

    if len(paths_for_tpe_ans) == 0:
        num_que_no_paths += 1
        print("Qid:%s doesn't extract any paths within %d hop between qa!" % (qid, max_hop))
        return None
    if task_name == "metaqa":
        valid_paths_for_tpe_ans = paths_for_tpe_ans
        valid_const_rels = []
        flag = 0
    else:
        # consume time!!!
        valid_paths_for_tpe_ans, valid_paths_for_tpe_cpe, valid_const_rels = \
            get_paths_from_tpe_to_cpes(t, const_ents, task_name, paths_for_tpe_ans)
        flag = 0 if len(valid_const_rels) == 0 else 1

    paths_with_score_one_pair = []
    # iterate every searched paths
    for path in valid_paths_for_tpe_ans:
        can_ans = kg.get_tails_with_path("sparse", path)
        if len(can_ans) == 0:
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
                  "PositiveRelations": list(pos_relations), "Paths": ordered_paths_with_score, "ConstRelFlag": flag}

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
    parser.add_argument('--extra_hop_flag', action="store_true",
                        help='whether retrieve extra hop')
    parser.add_argument('--max_num_processes', default=1, type=int)
    parser.add_argument('--max_hop', default=2, type=int,
                        help='the max search hop of the shortest paths')
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--sparse_kg_source_path', default=None, help='the sparse triples file')
    parser.add_argument('--sparse_ent_type_path', default=None, help='the file of entities type of sparse triples')
    parser.add_argument('--sparse_ent2id_path', default=None, help='the sparse ent2id file')
    parser.add_argument('--sparse_rel2id_path', default=None, help='the sparse rel2id file')
    args = parser.parse_args()

    print("Start extracting weak supervision paths to train the retriever.")

    return args


if __name__ == '__main__':
    args = _parse_args()

    kg = KnowledgeGraph(args.sparse_kg_source_path, args.sparse_ent_type_path, args.sparse_ent2id_path,
                        args.sparse_rel2id_path)

    num_process = args.max_num_processes
    chunk_size = 1
    max_hop = args.max_hop
    extra_hop_flag = args.extra_hop_flag
    task_name = args.task_name

    inp = args.input_path
    oup = args.output_path
    print('Input %s to Output %s' % (inp, oup))
    with open(inp, "r") as f:
        all_lines = f.readlines()
        if args.debug: # for debug mode, we only load 100 samples.
            all_data = [json.loads(l) for l in all_lines[0:100]]
        else:
            all_data = [json.loads(l) for l in all_lines]

    retrieved_data_ids = []
    ids_const_rel_recall = defaultdict()
    count = 0
    print("Start process %d samples." % (len(all_data)))
    with open(oup, 'w') as fout:
        if args.debug:
            for data in all_data:
                new_sample = construct_pos_and_neg_rels(data)
                if new_sample is not None:
                    retrieved_data_ids.append(new_sample['ID'])
                    fout.write(json.dumps(new_sample) + "\n")
                    count += 1
                    ids_const_rel_recall[new_sample['ID']] = new_sample['ConstRelationsRecall']
            print("Extract %d / %d samples" % (count, len(all_data)))
        else:
            with multiprocessing.Pool(num_process) as p:
                new_samples = p.imap_unordered(construct_pos_and_neg_rels, all_data, chunksize=chunk_size)
                for sample in tqdm(new_samples, total=len(all_data)):
                    if sample is not None:
                        retrieved_data_ids.append(sample['ID'])
                        fout.write(json.dumps(sample) + "\n")
                        count += 1
                        ids_const_rel_recall[sample['ID']] = sample['ConstRelationsRecall']
                print("Extract %d / %d samples" % (count, len(all_data)))
            p.join()

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
        if len(const_rel) == 0:
            const_rel.append(0)
        print("%s: Samples with paths-%d/%d-%.2f Const relation recall-%.3f" %
              (split, count, len(all_ids), count / len(all_ids), float(np.mean(const_rel))))
