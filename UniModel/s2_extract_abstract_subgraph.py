import argparse
import logging
import math
import os
import sys
from copy import deepcopy

sys.path.append("..")
from KnowledgeBase.KG_api import KnowledgeGraph
from KnowledgeBase.sparql_executor import *
import multiprocessing as mp


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


def extract_abstract_sg_from_kb(idx, device, args, datasets, shortest_paths_dict, output_path):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    from transformers import AutoModel, AutoTokenizer

    print("Start PID %d for processing %d-%d" % (os.getpid(), idx * split_index, (idx + 1) * split_index))

    kg = KnowledgeGraph(args.sparse_kg_source_path, args.sparse_ent_type_path,
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

            question = data["ProcessedQuestion"]
            try:
                data_new = get_subgraph(task_name=args.task_name, qid=qid, question=question, tpe=tpe, cpes=cpes,
                                        split=split, topk=args.topk, short_paths=short_paths,
                                        filter_score=args.filter_score, not_filter_rels=args.not_filter_rels,
                                        max_hop=args.max_hop,
                                        max_num_triples=args.max_num_triples,
                                        tokenizer=tokenizer, model=model, kg=kg)
            except Exception as e:
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
    parser.add_argument('--not_filter_rels', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--device', default=[0, 1, 2, 3, 4, 5, 6, 7], nargs="+", help='the gpu device')
    parser.add_argument('--num_pro_each_device', default=3, type=int)
    parser.add_argument('--max_num_processes', default=1, type=int)
    parser.add_argument('--topk', default=10, type=int, help='retrieve the topk score paths')
    parser.add_argument('--filter_score', default=0.0, type=float, help='the minimal cosine similarity')
    parser.add_argument('--max_hop', default=2, type=int, help='retrieve the topk score paths')
    parser.add_argument('--max_num_triples', default=1500, type=int, help='retrieve the topk score paths')
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
    with open(ori_path, "r") as f:
        all_lines = f.readlines()
        if args.debug:
            ori_dataset = [json.loads(l) for l in all_lines[:100]]
        else:
            ori_dataset = [json.loads(l) for l in all_lines]
    ready_data = ori_dataset
    print('Load %d original samples from %s' % (len(ready_data), ori_path))

    inp_path = args.input_path
    with open(inp_path, "r") as f:
        all_lines = f.readlines()
        inp_dataset = [json.loads(l) for l in all_lines]
        inp_dataset_dict = {l["ID"]: l for l in inp_dataset}
    print('Load %d shortest paths from %s' % (len(inp_dataset), inp_path))

    if args.debug:
        # debug
        split_index = 0
        extract_abstract_sg_from_kb(0, 0, args, ori_dataset, inp_dataset_dict, out_path)
    else:
        device_list = sum([args.device] * args.num_pro_each_device, [])
        num_process = min(len(device_list), num_process)
        print("Using %d process and %s gpu" % (num_process, device_list))
        split_index = len(ready_data) // num_process + 1

        p = mp.Pool(num_process)
        for cuda, idx in zip(device_list, range(num_process)):
            select_data = ready_data[idx * split_index: (idx + 1) * split_index]
            p.apply_async(extract_abstract_sg_from_kb,
                          args=(idx, cuda, args, select_data, inp_dataset_dict, out_path))
        p.close()
        p.join()
        print("All of the child processes over!")
        print("Starting aggregate all samples to one file!")
