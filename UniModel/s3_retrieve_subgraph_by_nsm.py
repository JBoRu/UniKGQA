import math
from collections import defaultdict
from copy import deepcopy
from typing import Tuple, List

import logging
from func_timeout import func_set_timeout, FunctionTimedOut

import json
import numpy as np

import argparse
from tqdm import tqdm
import multiprocessing
import sys
sys.path.append("..")
from KnowledgeBase.KG_api import KnowledgeGraph
from KnowledgeBase.sparql_executor import *
from utils import *
import pickle


def prune_abs_sg(tpe, cpes, subgraph_triples, decended_ent_global_idx, paths, num_beams):
    # tpe = abs_subgraph["entities"][0]
    # cpes = abs_subgraph["const_entities"]
    # subgraph_triples = abs_subgraph["subgraph"]["tuples"]
    abs_sg_h_to_rt = defaultdict(lambda: defaultdict(set))
    for tri in subgraph_triples:
        h, r, t = tri
        abs_sg_h_to_rt[h][r].add(t)

    topk_paths = []
    for ent_id in decended_ent_global_idx:
        if ent_id == 0: # we skip topic entity
            continue
        for path in paths:
            assert len(path) >= 3 and path[0] == tpe
            for idx, ent_ in enumerate(path[::2]):
                if ent_ == ent_id:
                    if path[0:2*idx+1] not in topk_paths:
                        topk_paths.append(path[0:2*idx+1])
                        break
                else:
                    continue
    # topk_paths = topk_paths[:num_beams]
    topk_triples = set()
    for path in topk_paths[:num_beams]:
        for hop, idx in enumerate(range(0, len(path) - 1, 2)):
            h, r, t = path[idx: 2 * (idx + 1) + 1][0], path[idx: 2 * (idx + 1) + 1][1], path[idx: 2 * (idx + 1) + 1][2]
            topk_triples.add((h, r, t))

    if len(cpes) == 0:
        topk_paths = topk_paths[:num_beams]
    new_topk_paths = []
    new_topk_paths_const = []
    for path in topk_paths:
        flag = False
        path_rels_and_const = defaultdict(lambda: defaultdict(list))
        # path = tuple(path)
        # if path not in path_to_const_rels:
        #     const_rels = []
        # else:
        #     const_rels = path_to_const_rels[path]
        assert len(path) % 2 == 1
        for i in range(0, len(path), 2):  # 0, 2, 4
            if i < (len(path) - 1):
                tri = path[i: i+3]
                h, r, t = tri
                path_rels_and_const[i]["chain"].append(r)
                const_rels = abs_sg_h_to_rt[h]
            else:
                final_t = path[i]
                const_rels = abs_sg_h_to_rt[final_t]
            for cont_rel, tails in const_rels.items():
                if len(set(tails) & set(cpes)) > 0:
                    flag = True
                    path_rels_and_const[i]["const"].append(cont_rel)
        if flag:
            new_topk_paths_const.append(path_rels_and_const)
        else:
            new_topk_paths.append(path_rels_and_const)

    final_topk_paths = []
    if len(new_topk_paths_const) != 0:
        final_topk_paths.extend(new_topk_paths_const)
    else:
        final_topk_paths.extend(new_topk_paths)
    final_topk_paths = final_topk_paths[:num_beams]

    return final_topk_paths, topk_triples

@func_set_timeout(60)
def extract_paths_from_sg(tpe, cpes, abs_sg_triples):
    last_tails = [tpe]
    last_paths = [[tpe]]
    results_paths = []
    results_paths_for_const_rels = defaultdict(set)
    last_sg_triples = deepcopy(abs_sg_triples)

    # abs_sg_h_to_rt = defaultdict(lambda: defaultdict(set))
    # for tri in abs_sg_triples:
    #     h, r, t = tri
    #     abs_sg_h_to_rt[h][r].add(t)

    while len(last_sg_triples):
        choosed_paths = []
        cur_sg_triples = []
        cur_tails = []
        cur_paths = []
        for tri in last_sg_triples:
            h, r, t = tri
            if h in last_tails:
                for path in last_paths:
                    if path[-1] == h:
                        path_tmp = deepcopy(path)
                        path_tmp.extend([r, t])
                        if t not in cur_tails:
                            cur_tails.append(t)
                        if path_tmp not in cur_paths:
                            cur_paths.append(path_tmp)
                        choosed_paths.append(path)
            else:
                if tri not in cur_sg_triples:
                    cur_sg_triples.append(tri)

        if len(set(cur_sg_triples) - set(last_sg_triples)) == 0 and \
                len(set(last_sg_triples) - set(cur_sg_triples)) == 0:
            # exist some no reachable triples
            break

        for path in last_paths:
            if path not in choosed_paths:
                if path not in results_paths:
                    results_paths.append(path)

        last_tails = deepcopy(cur_tails)
        last_paths = deepcopy(cur_paths)
        last_sg_triples = deepcopy(cur_sg_triples)

    for path in last_paths:
        if path not in results_paths:
            results_paths.append(path)

    # for path in results_paths:
    #     path = tuple(path)
    #     for idx in range(0, len(path), 2): # 0 2 4
    #         node = path[idx]
    #         rt = abs_sg_h_to_rt[node]
    #         for r, t in rt.items():
    #             if len(set(t) & set(cpes)) > 0:
    #                 results_paths_for_const_rels[path].add((idx, r))

    # return results_paths, results_paths_for_const_rels
    return results_paths


def get_subgraph_from_abs_sg(abs_sg_data, id2rel):
    subgraph = abs_sg_data['subgraph']
    tpe = abs_sg_data["entities"]
    if len(tpe) != 0:
        tpe = tpe[0]
    else:
        raise "Don't have any topic entities!"
    cpes = abs_sg_data['const_entities']
    triples_set, entities_set = subgraph["tuples"], subgraph["entities"]
    new_triples_set = set([(tri[0], id2rel[tri[1]], tri[2]) for tri in triples_set])
    # new_entities_set = set()
    # for tri in triples_set:
    #     h, r, t = tri
        # h_ = int(id2ent[h])
        # r_ = id2rel[r]
        # t_ = int(id2ent[t])
        # new_triples_set.add((h_, r_, t_))
        # new_entities_set.add(h_)
        # new_entities_set.add(t_)
        # new_triples_set.add((h, r_, t))
    # for ent in entities_set:
    #     ent_ = int(id2ent[ent])
    #     new_entities_set.add(ent_)
    paths = extract_paths_from_sg(tpe, cpes, new_triples_set)

    return paths, new_triples_set, entities_set


def reason_sg_from_kb(data, question, id2rel, abs_sg_data, num_return_paths, sg_retri_model):
    # subgraph = abs_sg_data["subgraph"]
    initial_paths, initial_sg_triples_set, initial_sg_entities_set = get_subgraph_from_abs_sg(abs_sg_data, id2rel)


    score_answers, globalid2loc = reason_ans_over_sg(data, sg_retri_model, question, abs_sg_data["entities"], initial_sg_triples_set,
                                                     initial_sg_entities_set)
    score_answers = score_answers.cpu().numpy()
    localid2glob = {l: g for g, l in globalid2loc.items()}
    decended_ent_local_idx = np.argsort(-score_answers)  # descend
    decended_ent_global_idx = [localid2glob[i] for i in decended_ent_local_idx]
    topk_path_rels_and_const, topk_triples = prune_abs_sg(tpe=abs_sg_data["entities"][0], cpes=abs_sg_data["const_entities"],
                                                          subgraph_triples=initial_sg_triples_set,
                                                          decended_ent_global_idx=decended_ent_global_idx,
                                                          paths=initial_paths, num_beams=num_return_paths)

    return topk_path_rels_and_const, topk_triples


def reason_ans_over_sg(data, model, question, topic_entity, next_hop_subgraph, next_hop_sg_entities):
    new_sample = {"ID": data["ID"], "question": question, "entities": topic_entity,
                  "answers": None, "subgraph": {"tuples": list(next_hop_subgraph),
                                                "entities": list(next_hop_sg_entities)}}
    pred_dist, globalid2loc = model.evaluate(data=new_sample, test_batch_size=1, mode="retrieve", write_info=False)
    return pred_dist, globalid2loc


def extract_rel_from_retri_sg_tris(topk_paths):
    relations = set()
    for path in topk_paths:
        for hop_id, rel_const in path.items():
            relations.update(rel_const["chain"])
            relations.update(rel_const["const"])
    return relations


def creat_nsm_model(args):
    import sys
    path = './nsm/'
    sys.path.append(path)
    from NSM.train.trainer_nsm import Trainer_KBQA

    nsm_checkpoint_path = args.model_path
    nsm_args_path = args.arg_path
    with open(nsm_args_path, "rb") as f:
        nsm_args = pickle.load(f)

    # if args.task_name == "webqsp":
    #     nsm_args["data_folder"] = "data/webqsp/"
    #     nsm_args["checkpoint_dir"] = "retriever_ckpt/webqsp_nsm_retri_v0/"
    #     nsm_args["model_path"] = "retriever/results/webqsp_rel_retri_1/"
    #     nsm_args["relation_model_path"] = "retriever/results/webqsp_rel_retri_1/"
    # elif args.task_name == "minerva":
    #     nsm_args["data_folder"] = "data/WN18RR/minerva/"
    #     nsm_args["checkpoint_dir"] = "retriever_ckpt/WN18RR_nsm_retri_v0/"
    #     nsm_args["model_path"] = "retriever/results/WN18RR_rel_retri_0/"
    # elif args.task_name == "cwq":
    #     nsm_args["data_folder"] = "data/cwq/"
    #     nsm_args["checkpoint_dir"] = "retriever_ckpt/cwq_nsm_retri/"
    #     nsm_args["model_path"] = "retriever/results/cwq_rel_retri_0/"

    # nsm_args["data_cache"] = None #"data/cwq/abs_r0_sg_for_retri.cache"
    # nsm_args["data_cache"] = "data/cwq/new_abs_r0_sg_for_retri.cache"
    # nsm_args["overwrite_cache"] = False
    # nsm_args["init_same_as_plm"] = False
    # nsm_args["retriever_ckpt_path"] = None
    # nsm_args["local_rank"] = -1
    # nsm_args["fix_all_plm"] = False
    # nsm_args["fixed_plm_for_query_encoding"] = False
    # nsm_args["fixed_plm_for_relation_encoding"] = False
    # nsm_args["encode_relation_separate"] = False
    nsm_args["overwrite_cache"] = False
    nsm_args["data_cache"] = args.input_data_cache_abs_path
    trainer = Trainer_KBQA(args=nsm_args)
    trainer.load_ckpt(nsm_checkpoint_path)
    return trainer


def infer_relations_from_kb(idx, device, args, datasets, abs_sg_data_dict, id2rel, id2ent, output_path):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)

    print("Start PID %d for processing %d-%d" % (os.getpid(), idx*split_index, (idx+1)*split_index))

    sg_retri_model = creat_nsm_model(args)
    kg = KnowledgeGraph(args.sparse_kg_source_path, args.sparse_ent_type_path,
                        args.sparse_ent2id_path, args.sparse_rel2id_path)

    final_topk = args.final_topk
    max_deduced_triples = args.max_deduced_triples

    cur_op = output_path + "_" + str(idx)
    with open(cur_op, "w") as f:
        for data in tqdm(datasets, total=len(datasets), desc="PID: %d"%(os.getpid())):
            qid = data['ID']
            split = data['Split']
            if qid in abs_sg_data_dict:
                abs_sg_data = abs_sg_data_dict[qid]
            else:
                if split == "test":
                    print("Qid:%s doesn't have any abstract subgraph. We skip this test example"%(qid))
                continue

            answers = []
            gold_relations_list = []
            parse = data['Parse']
            if parse["Answers"] is not None:
                for ans in parse['Answers']:
                    answers.append(ans['AnswerArgument'])
            if 'GoldSqlStructure' in data:
                gold_relations = data['GoldSqlStructure']
                for k, v in gold_relations.items():
                    for rel in v:
                        if rel not in gold_relations_list:
                            gold_relations_list.append(rel)

            answers = list(set(answers))
            if "test" not in split and len(answers) == 0:
                print("Qid: %s does not have any gold answers, we skip this training example."%(qid))
                continue

            query = data["ProcessedQuestion"]
            tpe = parse['TopicEntityMid']
            gold_entities = parse['GoldEntityMid']
            cpes = set(gold_entities) - {tpe}

            try:
                if len(abs_sg_data["subgraph"]["entities"]) == 0:
                    if split == "test":
                        print("No entities in abstract subgraph.We skip this example.")
                    continue
                topk_path_rels_and_const, topk_triples = reason_sg_from_kb(data, query, id2rel=id2rel,
                                                                           abs_sg_data=abs_sg_data,
                                                                           num_return_paths=final_topk,
                                                                           sg_retri_model=sg_retri_model)
            except Exception as e:
                logging.exception(e)
                print("Qid: %s meet some error, we skip this example." % (qid))
                continue

            relations_set = extract_rel_from_retri_sg_tris(topk_path_rels_and_const)

            subgraph = kg.deduce_subgraph_by_abstract_sg(tpe, max_deduced_triples, topk_path_rels_and_const)  # (list[nodes],list[triplets])

            subgraph_ent, inst_triples = subgraph
            relations_set_from_final = set()
            for tri in inst_triples:
                h, r, t = tri
                relations_set_from_final.add(r)
            if len(answers) == 0:
                ans_rec = 1.0
            else:
                ans_rec = len(set(subgraph_ent) & set(answers)) / (len(answers)+0.0)

            if len(gold_relations_list) == 0:
                rel_rec = 1.0
            else:
                rel_rec = len(relations_set_from_final & set(gold_relations_list)) / (len(gold_relations_list)+0.0)

            data_new = {'qid': qid, 'query': query, 'topic_entities': [tpe], 'const_entities':list(cpes), "ans_recall": round(ans_rec, 4),
                        "rel_recall": round(rel_rec, 4), 'retrieval_relations': list(relations_set), 'subgraph': subgraph}

            f.write(json.dumps(data_new)+"\n")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--task_name', required=True)
    parser.add_argument('--ori_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--arg_path', default=None)
    parser.add_argument('--relation2id_path', default=None, type=str)
    parser.add_argument('--entity2id_path', default=None, type=str)
    parser.add_argument('--input_data_cache_abs_path', default=None, type=str)
    parser.add_argument('--split_list', default=["train", "dev", "test"], nargs="+")
    parser.add_argument('--device', default=[0, 1, 2, 3, 4, 5, 6, 7], nargs="+", help='the gpu device')
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--num_pro_each_device', default=3, type=int)
    parser.add_argument('--max_num_processes', default=1, type=int)
    parser.add_argument('--final_topk', default=10, type=int, help='final num of  retrieved paths')
    parser.add_argument('--max_hop', default=4, type=int, help='max hop of paths')
    parser.add_argument('--max_deduced_triples', default=200, type=int, help='max triples for one path')
    parser.add_argument('--sparse_kg_source_path', default=None, help='the sparse triples file')
    parser.add_argument('--sparse_ent_type_path', default=None, help='the file of entities type of sparse triples')
    parser.add_argument('--sparse_ent2id_path', default=None, help='the sparse ent2id file')
    parser.add_argument('--sparse_rel2id_path', default=None, help='the sparse rel2id file')
    args = parser.parse_args()

    print("Start retrieving the subgraph.")
    return args


if __name__ == '__main__':
    args = _parse_args()

    overwrite_flag = args.overwrite
    num_process = args.max_num_processes
    chunk_size = 1

    rel2id = load_dict(args.relation2id_path)
    id2rel = {v: k for k, v in rel2id.items()}
    ent2id = load_dict(args.entity2id_path)
    id2ent = {v: k for k, v in ent2id.items()}

    ori_path = args.ori_path
    out_path = args.output_path
    print('Input %s to Output %s' % (ori_path, out_path))
    with open(ori_path, "r") as f:
        all_lines = f.readlines()
        ori_dataset = [json.loads(l) for l in all_lines]
    print('Load original data from %s' % ori_path)

    inp_path = args.input_path
    with open(inp_path, "r") as f:
        all_lines = f.readlines()
        if args.debug:
            all_sg_lines = [json.loads(l) for l in all_lines[0:100]]
        else:
            all_sg_lines = [json.loads(l) for l in all_lines]
    abs_sg_data_dict = {d["ID"]: d for d in all_sg_lines}
    print('Load abstract subgraphs from %s' % inp_path)

    already_ids = []
    if not overwrite_flag and os.path.exists(out_path):
        with open(out_path, "r") as f:
            processed_samples = f.readlines()
            processed_samples = [json.loads(sample) for sample in processed_samples]
            already_ids = [sample['qid'] for sample in processed_samples]

        ready_data = []
        for data in ori_dataset:
            if data['ID'] not in already_ids:
                ready_data.append(data)

        print("There are %d new samples." % (len(ready_data)))
    else:
        ready_data = ori_dataset

    device_list = args.device
    device_list = sum([device_list]*args.num_pro_each_device, [])
    num_process = min(len(device_list), num_process)
    print("Using %d process and %s gpu"%(num_process, device_list))
    split_index = len(ready_data) // num_process + 1
    new_samples, ans_recall, rel_recall = [], [], []

    if args.debug:
        # deug
        infer_relations_from_kb(0, 6, args, ready_data, abs_sg_data_dict, id2rel, id2ent, out_path)
    else:
        p = multiprocessing.Pool(num_process)
        result_for_process = []
        for cuda, idx in zip(device_list, range(num_process)):
            select_data = ready_data[idx * split_index: (idx + 1) * split_index]
            p.apply_async(infer_relations_from_kb,
                          args=(idx, cuda, args, select_data, abs_sg_data_dict, id2rel, id2ent, out_path))
        p.close()
        p.join()
        print("All child process over!")
