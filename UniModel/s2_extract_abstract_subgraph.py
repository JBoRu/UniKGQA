import math
import random
import time
from copy import deepcopy

import json
import os

import logging
import numpy as np
import argparse
import sys
from line_profiler import LineProfiler

sys.path.append("..")
from KnowledgeBase.KG_api import KnowledgeGraph
from KnowledgeBase.KG_api import KnowledgeGraphSparse
from KnowledgeBase.sparql_executor import *
import multiprocessing as mp


def get_texts_embeddings(texts, tokenizer, model, rel_emb_cache):
    import torch

    ids2emb = {}
    new_texts = []
    new_ids = []

    for idx, text in enumerate(texts):
        if text in rel_emb_cache:
            ids2emb[idx] = rel_emb_cache[text]
        else:
            new_texts.append(text)
            new_ids.append(idx)

    max_bs = 300
    total_new = len(new_texts)
    steps = math.ceil(total_new / max_bs)
    new_embeddings = []
    for i in range(steps):
        texts_batch = new_texts[i * max_bs: (i + 1) * max_bs]
        inputs = tokenizer(texts_batch, padding=True, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            new_embeddings.append(embeddings)
    if len(new_embeddings) > 0:
        new_embeddings_cpu = torch.cat(new_embeddings, dim=0).to('cpu')

        for text_idx, emb_idx in zip(new_ids, range(new_embeddings_cpu.shape[0])):
            emb = new_embeddings_cpu[emb_idx, :]  # hid_dim
            text = texts[text_idx]
            rel_emb_cache[text] = emb
            ids2emb[text_idx] = emb

    total_embeddings = []
    for i in range(len(texts)):
        total_embeddings.append(ids2emb[i])
    total_embeddings = torch.stack(total_embeddings, dim=0)  # (bs, hid)
    total_embeddings = total_embeddings.to(model.device)
    return total_embeddings


def score_relations(query, cur_relations, tokenizer, model, rel_emb_cache, theta: float = 0.05):
    import torch
    cur_relations = list(cur_relations)
    all_relation_list = cur_relations
    query = query.replace("[MASK]", tokenizer.mask_token)
    query_lined_list = [query]
    q_emb = get_texts_embeddings(query_lined_list, tokenizer, model, rel_emb_cache).unsqueeze(1)  # (1,1,hid)
    target_emb = get_texts_embeddings(all_relation_list, tokenizer, model, rel_emb_cache).unsqueeze(0)  # (1,bs,hid)
    sim_score = torch.cosine_similarity(q_emb, target_emb, dim=2)  # (1,bs)
    sim_score = sim_score.squeeze(dim=0) / theta  # (bs)
    results = []
    filtered_results = []
    for idx, rel in enumerate(cur_relations):
        score = sim_score[idx]
        if score >= 0.0:
            results.append((rel, score))
            filtered_results.append((rel, score))
        else:
            results.append((rel, score))
    return results, filtered_results


def filter_rels(reserve_rels, retrieved_const_rels, cur_paths, tris_per_hop, hop_id):
    filtered_cur_paths = []
    filtered_tris_per_hop = defaultdict(set)
    for p in cur_paths:
        if p[hop_id * 2 + 1] in reserve_rels:
            if p[hop_id * 2 + 1] not in retrieved_const_rels:
                filtered_cur_paths.append(p)
    for hid, tris in tris_per_hop.items():
        if hid != hop_id:
            filtered_tris_per_hop[hid].update(tris)
        else:
            for tri in tris:
                if tri[1] in reserve_rels:
                    filtered_tris_per_hop[hid].add(tri)
    return filtered_cur_paths, filtered_tris_per_hop


def extract_shortest_paths(short_paths):
    paths = short_paths["Paths"]
    if len(paths) == 0:
        return []
    selected_paths = [p_s[1] for p_s in paths]
    selected_rels_per_hop = defaultdict(set)
    for path in selected_paths:
        for idx, rel in enumerate(path):
            selected_rels_per_hop[idx].add(rel)

    return selected_rels_per_hop


def compute_gold_const_rel_recall(retri_const_rels, gold_const_rels):
    gold_const_rels = set(gold_const_rels)
    rel_recall = len(retri_const_rels & gold_const_rels) / (len(gold_const_rels) + 0.0)
    return rel_recall


def compute_gold_rel_str_recall(tris_per_hop, gold_rel_structure):
    last_ans = [0]
    all_gold_rels = set()
    all_shot_rels = set()
    max_hop = len(gold_rel_structure)
    for hop_id, tris in tris_per_hop.items():
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
    # if rel_recall == 1 and len(gold_rel_structure) >= 2:
    #     print("2hop: %.3f: %s"%(rel_recall, gold_rel_structure))
    # elif len(gold_rel_structure) >= 2:
    #     print("2hop: %.3f: %s"%(rel_recall, gold_rel_structure))
    # elif rel_recall != 1 and len(gold_rel_structure) == 1:
    #     print("1hop: %.3f: %s" % (rel_recall, gold_rel_structure))
    return rel_recall


def get_subgraph(task_name, qid, question, tpe, cpes, split, short_paths, topk, filter_score,
                 not_filter_rels, max_hop, max_num_triples, tokenizer, model, kg):
    if short_paths is None:
        weak_gold_rels_per_hop = defaultdict(set)
        print("Qid: %s doesn't have any shortest path between qa." % (qid))
    else:
        weak_gold_rels_per_hop = extract_shortest_paths(short_paths)

    cpes = set(cpes)

    tris_per_hop, reserved_rel_and_score, all_retri_const_rels, abs_cpes_id = \
        kg.get_subgraph_within_khop(qid, task_name, split, question, weak_gold_rels_per_hop,
                                    tpe, cpes, max_hop, tokenizer, model, topk, filter_score, not_filter_rels)

    num_selected_tris = sum([len(v) for k, v in tris_per_hop.items()])
    num_extra_tris = max_num_triples - num_selected_tris

    if num_extra_tris < 0:
        ori_num_selected_tris = num_selected_tris
        reserved_rels_sorted = sorted(reserved_rel_and_score.items(), key=lambda kv: (kv[1], kv[0]))
        for rel_score in reserved_rels_sorted:  # from lower to larger
            if num_extra_tris > 0:
                break
            filter_rel, _ = rel_score
            new_tris_per_hop = defaultdict(set)
            last_filter_heads = []
            for k, v in tris_per_hop.items():
                cur_filter_heads = []
                for tri in v:
                    h, r, t = tri
                    if h in last_filter_heads or r == filter_rel:
                        cur_filter_heads.append(t)
                    else:
                        new_tris_per_hop[k].add(tri)
                last_filter_heads = deepcopy(cur_filter_heads)
            tris_per_hop = deepcopy(new_tris_per_hop)
            num_selected_tris = sum([len(v) for k, v in tris_per_hop.items()])
            num_extra_tris = max_num_triples - num_selected_tris
        print("Qid-[%s] abs subgraph from %d to %d" % (qid, ori_num_selected_tris, num_selected_tris))

    tris_per_hop_list = {}
    for k, v in tris_per_hop.items():
        tris_per_hop_list[k] = list(v)

    data_new = {'ID': qid, 'question': question, 'entities': [0], 'const_entities': list(abs_cpes_id),
                'tris_per_hop': tris_per_hop_list, 'all_retri_const_rels': list(all_retri_const_rels)}

    return data_new


def extract_abstract_sg_from_kb(idx, device, args, datasets, shortest_paths_dict, alrea_qid, output_path):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    from transformers import AutoModel, AutoTokenizer

    print("Start PID %d for processing %d-%d" % (os.getpid(), idx * split_index, (idx + 1) * split_index))

    kg = KnowledgeGraph(args.dense_kg_source, (args.sparse_kg_source_path, args.sparse_ent_type_path),
                        args.sparse_ent2id_path, args.sparse_rel2id_path)
    if args.not_filter_rels:
        tokenizer = None
        model = None
    else:
        retrieval_model_ckpt = args.model_path
        tokenizer = AutoTokenizer.from_pretrained(retrieval_model_ckpt)
        model = AutoModel.from_pretrained(retrieval_model_ckpt)
        model.cuda()

    cur_op = output_path + "_" + str(idx)
    with open(cur_op, "w") as f:
        for data in tqdm(datasets, total=len(datasets), desc="PID: %d" % (os.getpid())):
            qid = data["ID"]
            if qid in alrea_qid:
                continue
            # if qid not in ['test_8882']:
            #     continue
            split = data["Split"]

            if qid not in shortest_paths_dict:
                short_paths = None
            else:
                short_paths = shortest_paths_dict[qid]

            parse = data["Parse"]

            # get the topic entities and answer entities
            tpe = parse["TopicEntityMid"]  # only one topic entity mid in default
            if tpe is None:
                print("Qid: %s does not have any topic entity. We skip this parse of example." % qid)
                continue
            cpes = set(parse["GoldEntityMid"]) - {tpe}

            if args.use_masked_question and "SimplifiedQuestion" in parse:
                question = parse["SimplifiedQuestion"]
            else:
                question = data["ProcessedQuestion"]
            try:
                data_new = get_subgraph(task_name=args.task_name, qid=qid, question=question, tpe=tpe, cpes=cpes,
                                        split=split, topk=args.topk, short_paths=short_paths,
                                        filter_score=args.filter_score, not_filter_rels=args.not_filter_rels,
                                        max_hop=args.max_hop,
                                        max_num_triples=args.max_num_triples,
                                        tokenizer=tokenizer, model=model, kg=kg)
            except Exception as e:
                print("Error in %s" % (qid))
                logging.exception(e)
                continue
            if data_new is not None:
                f.write(json.dumps(data_new) + "\n")
            else:
                print("Qid:%s doesn't extract any valid subgraph!" % (qid))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', required=True, type=str)
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--ori_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--use_masked_question', action="store_true",
                        help='whether mask the string of topic entity in original question')
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--not_filter_rels', action="store_true")
    parser.add_argument('--device', default=[0, 1, 2, 3, 4, 5, 6, 7], nargs="+", help='the gpu device')
    parser.add_argument('--num_pro_each_device', default=3, type=int)
    parser.add_argument('--max_num_processes', default=1, type=int)
    parser.add_argument('--topk', default=10, type=int, help='retrieve the topk score paths')
    parser.add_argument('--filter_score', default=0.0, type=float, help='the minimal cosine similarity')
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

    num_process = args.max_num_processes
    chunk_size = 1
    print("Parent pid:", os.getpid())
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
        inp_dataset = [json.loads(l) for l in all_lines]
        inp_dataset_dict = {l["ID"]: l for l in inp_dataset}
    print('Load shortest paths from %s' % inp_path)

    alrea_qid = []
    if not args.overwrite and os.path.exists(out_path):
        with open(out_path, "r") as f:
            already_lines = f.readlines()
            already_lines = [json.loads(l) for l in already_lines]
            alrea_qid = [l['ID'] for l in already_lines]
    alrea_count = len(alrea_qid)
    print("There are %d samples have been already processed." % (alrea_count))

    if len(alrea_qid) > 0:
        ready_data = []
        for data in ori_dataset:
            if data['ID'] not in alrea_qid:
                ready_data.append(data)
    else:
        ready_data = ori_dataset
    print("There are %d samples need to be processed." % (len(ready_data)))

    device_list = sum([args.device] * args.num_pro_each_device, [])
    num_process = min(len(device_list), num_process)
    print("Using %d process and %s gpu" % (num_process, device_list))
    split_index = len(ready_data) // num_process + 1

    p = mp.Pool(num_process)
    for cuda, idx in zip(device_list, range(num_process)):
        select_data = ready_data[idx * split_index: (idx + 1) * split_index]
        p.apply_async(extract_abstract_sg_from_kb,
                      args=(idx, cuda, args, select_data, inp_dataset_dict, alrea_qid, out_path))
    p.close()
    p.join()
    print("All of the child processes over!")
    print("Starting aggregate all samples to one file!")

    # used for time analysis
    # lp = LineProfiler()
    # lp.add_function(get_subgraph)
    # lp.add_function(KnowledgeGraph.deduce_relation_leaves_and_nodes_by_path)
    # lp.add_function(KnowledgeGraphSparse.deduce_relation_leaves_and_nodes_by_path)
    # lp.add_function(KnowledgeGraphSparse.deduce_next_triples_by_path_wo_reverse)
    # lp_wrapper = lp(extract_abstract_sg_from_kb)
    # results = lp_wrapper(0, 0, args, ori_dataset[0:5], inp_dataset_dict, alrea_qid, out_path)
    # lp.print_stats()
    # exit(0)

    # debug
    # extract_abstract_sg_from_kb(0, 0, args, ori_dataset, inp_dataset_dict, alrea_qid, out_path)
