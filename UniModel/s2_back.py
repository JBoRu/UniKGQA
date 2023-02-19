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
            emb = new_embeddings_cpu[emb_idx, :] # hid_dim
            text = texts[text_idx]
            rel_emb_cache[text] = emb
            ids2emb[text_idx] = emb

    total_embeddings = []
    for i in range(len(texts)):
        total_embeddings.append(ids2emb[i])
    total_embeddings = torch.stack(total_embeddings, dim=0) # (bs, hid)
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


def get_abstract_answers_by_per_hop(weak_gold_rels_per_hop, tris_per_hop):
    from copy import deepcopy

    last_ans = [0]
    max_len_path = len(weak_gold_rels_per_hop)
    for hi, tris in tris_per_hop.items():
        if hi < max_len_path:
            cur_ans = []
            gold_rels = weak_gold_rels_per_hop[hi]
            for tri in tris:
                h, r, t = tri
                if h in last_ans and r in gold_rels:
                    if t not in cur_ans:
                        cur_ans.append(t)
            last_ans = deepcopy(cur_ans)
    return last_ans


def get_abstract_answers_by_paths(weak_gold_paths, tris_per_hop):
    from copy import deepcopy

    result_ans = set()
    for path in weak_gold_paths:
        last_ans = [0]
        len_path = len(path)
        for hi, tris in tris_per_hop.items():
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

def extract_shortest_paths(short_paths, min_path_precision):
    paths = short_paths["Paths"]
    positive_rels = short_paths['PositiveRelations']

    precision_score_list = set([p[2] for p in paths])
    ordered_pre_score_list = sorted(precision_score_list, reverse=True)
    max_precision = ordered_pre_score_list[0]
    if len(ordered_pre_score_list) >= 2:
        sec_precision = ordered_pre_score_list[1]
    else:
        sec_precision = ordered_pre_score_list[0]

    if sec_precision >= min_path_precision:
        real_min_pre = sec_precision
    else:
        real_min_pre = max_precision
    selected_paths = [p[1] for p in paths if p[2] >= real_min_pre]
    selected_rels_per_hop = defaultdict(set)
    for path in paths:
        for idx, rel in enumerate(path[1]):
            selected_rels_per_hop[idx].add(rel)

    return (selected_rels_per_hop, selected_paths, positive_rels)


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


def get_subgraph(task_name, qid, question, tpe, cpes, gold_const_rels, split, gold_rel_structure, short_paths, filter_method, topk, filter_score,
                 max_hop, min_path_precision, max_num_triples, tokenizer, model, kg, rel_emb_cache):
    if short_paths is None:
        weak_gold_rels_per_hop, weak_gold_paths, positive_rels = defaultdict(set), [], []
        print("Qid: %s doesn't have any shortest path between qa." % (qid))
    else:
        weak_super_labels = extract_shortest_paths(short_paths, min_path_precision)
        weak_gold_rels_per_hop, weak_gold_paths, positive_rels = weak_super_labels

    tris_per_hop = defaultdict(set)
    last_paths = {()}
    tpe_id = 0
    cpes_id = set()
    placeholder_idx = 1
    all_triples = defaultdict(set)
    all_rel_and_score = defaultdict()
    reserved_rel_and_score = defaultdict()
    ans_placeholder_idx_during_search = set()
    all_retri_const_rels = set()
    cpes = set(cpes)
    for hop_id in range(max_hop):
        cur_paths = set()
        cur_relations = set()
        retrieved_const_rels = set()

        for path in last_paths:
            path_rels = []
            if len(path) > 0:
                for rel in path[1::2]:
                    path_rels.append(rel)
                tail = path[-1]
            else:
                tail = tpe_id

            rels, const_rels = kg.deduce_relation_leaves_and_nodes_by_path(kg="sparse", src=tpe, cpes=cpes,
                                                                           path_rels=path_rels, task_name=task_name)

            if len(rels) == 0:
                continue
            if hop_id == 0: # a trick for KBQA datasets.
                rels = list(set(rels)-set(const_rels))
                const_rels = []
            cur_relations.update(rels)
            for rel in rels:
                if rel in const_rels:
                    retrieved_const_rels.add(rel)
                    tris_per_hop[hop_id].add((tail, rel, placeholder_idx))
                    all_triples[tail].add((tail, rel, placeholder_idx))
                    cpes_id.add(placeholder_idx)
                else:
                    tris_per_hop[hop_id].add((tail, rel, placeholder_idx))
                    all_triples[tail].add((tail, rel, placeholder_idx))
                    if len(path) == 0:
                        cur_paths.add((tpe_id, rel, placeholder_idx))
                    elif len(path) > 0:
                        cur_paths.add(path+(rel, placeholder_idx))
                placeholder_idx = placeholder_idx + 1

        if len(cur_relations) == 0:
            break

        rels_scored_list, filtered_rel_socred_list = score_relations(question, cur_relations, tokenizer, model, rel_emb_cache={})
        ordered_rels_scored = sorted(filtered_rel_socred_list, key=lambda x: x[1], reverse=True)

        for rel_score in rels_scored_list:
            rel, score = rel_score
            all_rel_and_score[rel] = score

        reserved_rels = []
        if filter_method == "topk":
            reserved_rels = ordered_rels_scored[:topk]
        elif filter_method == "score":
            reserved_rels = [rs for rs in ordered_rels_scored if rs[1] > filter_score]
        elif filter_method == "mixture":
            reserved_rels = [rs for rs in ordered_rels_scored if rs[1] > filter_score][:topk]

        for rel_score in reserved_rels:
            rel, score = rel_score
            reserved_rel_and_score[rel] = score

        reserved_rels = [rs[0] for rs in reserved_rels]
        for rel in retrieved_const_rels:
            if rel not in reserved_rels:
                reserved_rels.append(rel)
        all_retri_const_rels.update(retrieved_const_rels)

        if split != 'test':  # This is a trick only used for train/dev set.
            weak_label_rels = weak_gold_rels_per_hop[hop_id] & cur_relations
            for rel in weak_label_rels:
                if rel not in reserved_rels:
                    reserved_rels.append(rel)
        cur_paths, tris_per_hop = filter_rels(reserved_rels, retrieved_const_rels, cur_paths, tris_per_hop, hop_id)
        last_paths = deepcopy(cur_paths)

    # sort the last paths from larger to lower
    scored_last_paths = []
    for path in last_paths:
        score = sum([all_rel_and_score[rel] for rel in path[1::2]])
        scored_last_paths.append((path, score))
    scored_last_paths = sorted(scored_last_paths, key=lambda ps: ps[1], reverse=True)
    last_paths = [ps[0] for ps in scored_last_paths]

    # sort all of reserved relations from lower to higher
    reserved_rels_sorted = sorted(reserved_rel_and_score.items(), key=lambda kv: (kv[1], kv[0]))

    # pad or truncate to maximal number of abs_sg
    num_selected_tris = sum([len(v) for k, v in tris_per_hop.items()])
    num_extra_tris = max_num_triples - num_selected_tris
    if num_extra_tris > 0:
        for path in last_paths:  # get top score path
            if num_extra_tris < 0:
                break
            for hop_id, ent_id in enumerate(path[2::2]):
                if num_extra_tris < 0:
                    break
                hop_id = hop_id + 1
                cands_tris = all_triples[ent_id]
                # filter very lower relation score triples
                if task_name == 'kgc':
                    cands_tris = [tri for tri in cands_tris if all_rel_and_score[tri[1]] > 0.0]
                else:
                    cands_tris = [tri for tri in cands_tris if all_rel_and_score[tri[1]] > 0.3]

                if len(cands_tris) >= num_extra_tris:
                    tris_per_hop[hop_id].update(random.sample(cands_tris, num_extra_tris))
                    num_extra_tris = 0
                else:
                    tris_per_hop[hop_id].update(cands_tris)
                    num_extra_tris = num_extra_tris - len(cands_tris)
    elif num_extra_tris < 0:
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

    num_selected_tris = sum([len(v) for k, v in tris_per_hop.items()])
    if max_num_triples < num_selected_tris:
        print("Qid abs subgraph more than %d nodes"%(qid))

    entities_set = set()
    triples_set = set()
    for hop_id, tris in tris_per_hop.items():
        triples_set.update(tris)
        for tri in tris:
            h, r, t = tri
            entities_set.add(h)
            entities_set.add(t)

    if weak_gold_paths:
        abs_answers = get_abstract_answers_by_paths(weak_gold_paths, tris_per_hop)
    else:
        abs_answers = []

    answers = []
    for a in abs_answers:
        answers.append({'kb_id': a, 'text': a})

    if len(gold_rel_structure) != 0:
        rel_struct_rec = compute_gold_rel_str_recall(tris_per_hop, gold_rel_structure)
    elif len(gold_const_rels) != 0:
        rel_struct_rec = compute_gold_const_rel_recall(all_retri_const_rels, gold_const_rels)
    else:
        rel_struct_rec = -1.0

    if split != "test" and len(answers) == 0:
        return None

    data_new = {'id': qid, 'question': question, 'entities': [0], 'const_entities': list(cpes_id), 'answers': answers,
                'rel_struct_rec': round(rel_struct_rec, 4),
                'subgraph': {"tuples": list(triples_set), "entities": list(entities_set)}}

    return data_new


def extract_abstract_sg_from_kb(idx, device, args, datasets, shortest_paths_dict, alrea_qid, output_path):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    from transformers import AutoModel, AutoTokenizer

    print("Start PID %d for processing %d-%d" % (os.getpid(), idx * split_index, (idx + 1) * split_index))

    retrieval_model_ckpt = args.model_path
    kg = KnowledgeGraph(args.dense_kg_source, (args.sparse_kg_source_path, args.sparse_ent_type_path),
                        args.sparse_ent2id_path, args.sparse_rel2id_path)
    tokenizer = AutoTokenizer.from_pretrained(retrieval_model_ckpt)
    model = AutoModel.from_pretrained(retrieval_model_ckpt)
    model.cuda()
    rel2embeddings = {}

    cur_op = output_path + "_" + str(idx)
    with open(cur_op, "w") as f:
        for data in tqdm(datasets, total=len(datasets), desc="PID: %d" % (os.getpid())):
            qid = data["ID"]
            split = data["Split"]
            if split != "test" and qid not in shortest_paths_dict:
                continue
            elif split == "test" and qid not in shortest_paths_dict:
                short_paths = None
            else:
                short_paths = shortest_paths_dict[qid]

            if qid in alrea_qid:
                continue

            parse = data["Parse"]
            if "GoldSqlStructure" in data:
                gold_rel_structure = data["GoldSqlStructure"]
            else:
                gold_rel_structure = defaultdict(list)

            # get the topic entities and answer entities
            tpe = parse["TopicEntityMid"]  # only one topic entity mid in default
            if tpe is None:
                # print("Qid: %s does not have any topic entity. We skip this parse of example." % qid)
                continue
            cpes = set(parse["GoldEntityMid"]) - {tpe}
            # answers = [ans["AnswerArgument"] for ans in parse["Answers"]]

            if args.use_masked_question and "SimplifiedQuestion" in parse:
                question = parse["SimplifiedQuestion"]
            else:
                question = data["ProcessedQuestion"]
            gold_const_rels = parse['GoldConstRels'] if 'GoldConstRels' in parse else []

            data_new = get_subgraph(task_name=args.task_name, qid=qid, question=question, tpe=tpe, cpes=cpes,
                                    gold_const_rels=gold_const_rels, split=split,
                                    gold_rel_structure=gold_rel_structure,
                                    short_paths=short_paths, filter_method=args.filter_method, topk=args.topk,
                                    filter_score=args.filter_score, max_hop=args.max_hop,
                                    min_path_precision=args.min_path_precision, max_num_triples=args.max_num_triples,
                                    tokenizer=tokenizer, model=model, kg=kg, rel_emb_cache=rel2embeddings)
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
    parser.add_argument('--min_path_precision', default=0.1, type=float)
    parser.add_argument('--device', default=[0, 1, 2, 3, 4, 5, 6, 7], nargs="+", help='the gpu device')
    parser.add_argument('--num_pro_each_device', default=3, type=int)
    parser.add_argument('--max_num_processes', default=1, type=int)
    parser.add_argument('--topk', default=10, type=int, help='retrieve the topk score paths')
    parser.add_argument('--filter_method', default="topk", type=str, help='methods for filtering relations')
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
    # if os.path.exists(out_path):
    #     with open(out_path, "r") as f:
    #         already_lines = f.readlines()
    #         already_lines = [json.loads(l) for l in already_lines]
    #         alrea_qid = [l['id'] for l in already_lines]
    # else:
    #     alrea_qid = []
    alrea_count = len(alrea_qid)
    print("There are %d samples have been already processed." % (alrea_count))

    device_list = args.device
    device_list = sum([device_list] * args.num_pro_each_device, [])
    num_process = min(len(device_list), num_process)
    print("Using %d process and %s gpu" % (num_process, device_list))
    split_index = len(ori_dataset) // num_process + 1

    # p = mp.Pool(num_process)
    # for cuda, idx in zip(device_list, range(num_process)):
    #     select_data = ori_dataset[idx * split_index: (idx + 1) * split_index]
    #     p.apply_async(extract_abstract_sg_from_kb,
    #                   args=(idx, cuda, args, select_data, inp_dataset_dict, alrea_qid, out_path))
    # p.close()
    # p.join()
    # print("All of the child processes over!")
    # print("Starting aggregate all samples to one file!")

    # used for deug
    lp = LineProfiler()
    lp.add_function(get_subgraph)
    lp.add_function(KnowledgeGraph.deduce_relation_leaves_and_nodes_by_path)
    lp.add_function(KnowledgeGraphSparse.deduce_relation_leaves_and_nodes_by_path)
    lp.add_function(KnowledgeGraphSparse.deduce_next_triples_by_path_wo_reverse)
    lp_wrapper = lp(extract_abstract_sg_from_kb)
    results = lp_wrapper(0, 0, args, ori_dataset[0:5], inp_dataset_dict, alrea_qid, out_path)
    lp.print_stats()
    exit(0)

    # print("Start time analysis.")
    # extract_abstract_sg_from_kb(0, 0, args, ori_dataset, inp_dataset_dict, alrea_qid, out_path)

    all_split_paths = []
    all_split_idx = []
    for idx in range(num_process):
        sp = out_path + "_" + str(idx)
        all_split_paths.append(sp)
        all_split_idx.append(idx)
    print("Loading %s split sub files based on %s" % (all_split_idx, out_path))

    all_abs_sg_data = []
    for path in all_split_paths:
        with open(path, "r") as f:
            split_abs_sg_data = f.readlines()
            # split_abs_sg_data = [json.loads(l) for l in split_abs_sg_data]
            all_abs_sg_data.extend(split_abs_sg_data)
    print("Totally load %d abstract subgraph data." % (len(all_abs_sg_data)))

    with open(out_path, "w") as f:
        for data in all_abs_sg_data:
            # f.write(json.dumps(data)+"\n")
            f.write(data.strip("\n")+"\n")