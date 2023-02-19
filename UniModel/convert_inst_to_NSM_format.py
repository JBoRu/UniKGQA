import json
import os
import argparse
import random
from tqdm import tqdm

def load_dict(filename):
    ele2id = {}
    with open(filename, "r") as f:
        all_lines = f.readlines()
        for l in all_lines:
            ele = l.strip("\n")
            if ele in ele2id:
                print("Already in dict:",l,ele)
                continue
            ele2id[ele] = len(ele2id)
    return ele2id

# def get_global_topic_entities(data, ent2id):
#     parses = data["Parses"]
#     topic_entities = set()
#     for p in parses:
#         ent = p["TopicEntityMid"]
#         try:
#             ent = str(ent)
#             topic_entities.add(ent2id[ent])
#         except:
#             print(ent, " not in ent2id mapping")
#     return list(topic_entities)

# def get_answers(data):
#     parses = data["Parses"]
#     answers_list = []
#     for p in parses:
#         ans = p["Answers"]
#         for a in ans:
#             ans_dict = {}
#             ans_dict["kb_id"] = a["AnswerArgument"]
#             ans_dict["text"] = a["EntityName"]
#             answers_list.append(ans_dict)
#     return answers_list

def get_global_topic_entity(data, ent2id):
    parse = data["Parse"]
    ent = parse["TopicEntityMid"]
    if ent is None:
        return []
    try:
        ent = ent2id[str(ent)]
    except:
        print(ent, " not in ent2id mapping")
    return [ent]

def get_global_const_entities(data, ent2id):
    parse = data["Parse"]
    topic_entities = set()
    # ent = parse["TopicEntityMid"]
    ent = parse["GoldEntityMid"]
    try:
        topic_entities.update([ent2id[str(e)] for e in ent])
    except:
        print(ent, " not in ent2id mapping")
    return list(topic_entities)

def get_answers(data):
    parse = data["Parse"]
    answers_list = []
    ans = parse["Answers"]
    for a in ans:
        ans_dict = {}
        ans_dict["kb_id"] = a["AnswerArgument"]
        ans_dict["text"] = a["EntityName"]
        answers_list.append(ans_dict)
    return answers_list

def get_subgraph(subgraph, ent2id, rel2id, max_nodes, new_sample, split, add_constraint_reverse):
    entities, triples = subgraph
    subgraph_new = {}
    entities_map = []
    triples_map = []
    cpes = new_sample["const_entities"]
    for hrt in triples:
        h, r, t = hrt
        assert h in entities
        assert t in entities
        if "type.object.name" in r:
            continue
        if add_constraint_reverse and t in cpes:
            triples_map.append([ent2id[t], rel2id[r], ent2id[h]])
        else:
            triples_map.append([ent2id[h], rel2id[r], ent2id[t]])
    for ent in entities:
        assert ent in ent2id
        entities_map.append(ent2id[ent])

    if len(entities_map) > max_nodes and split != "test":
        print("For %s set, more than max nodes: %d, so truncate it."%(split, max_nodes))
        keep_entities_id = set()
        keep_entities_id.update(new_sample["entities"])
        keep_entities_id.update(new_sample["const_entities"])
        answers = new_sample["answers"]
        for a in answers:
            keep_entities_id.add(a["kb_id"])

        common_entities = list(set(entities_map) & keep_entities_id)
        extra_entites = list(set(entities_map) - keep_entities_id)

        if len(extra_entites) >= max_nodes-len(common_entities):
            sampled_entities = random.sample(extra_entites, k=(max_nodes-len(common_entities)))
        else:
            sampled_entities = extra_entites

        entities_map = sampled_entities + common_entities
        triples_map_tmp = []
        for hrt in triples_map:
            h, r, t = hrt
            if h not in entities_map or t not in entities_map:
                continue
            else:
                triples_map_tmp.append(hrt)
        print("The triples truncate from %d to %d"%(len(triples_map), len(triples_map_tmp)))
        triples_map = triples_map_tmp

    subgraph_new["tuples"] = triples_map
    subgraph_new["entities"] = entities_map
    return subgraph_new

def convert_to_NSM_input_format(ori_path, in_path, out_path, split_list, kg_map_path, prefix, max_nodes):
    ent2id = load_dict(os.path.join(kg_map_path, prefix+'entities.txt'))
    rel2id = load_dict(os.path.join(kg_map_path, prefix+'relations.txt'))
    for split in split_list:
        ori_path_new = ori_path.replace("SPLIT", split)
        in_path_new = in_path.replace("SPLIT", split)
        out_path_new = out_path.replace("SPLIT", split)
        print("Input %s to Output %s: "%(in_path_new, out_path_new))
        with open(ori_path_new, "r") as f:
            ori_sample = f.readlines()
            ori_sample = [json.loads(s) for s in ori_sample]
            ori_sample_dict = {e["ID"]: e for e in ori_sample}
        with open(in_path_new, "r") as f:
            all_lines = f.readlines()
        with open(out_path_new, "w") as f:
            for line in tqdm(all_lines, total=len(all_lines)):
                new_sample = {}
                sample = json.loads(line)
                qid = sample["qid"]
                meta_info = ori_sample_dict[qid]
                new_sample["ID"] = qid
                query = sample["query"]
                new_sample["question"] = query.replace("[SEP]", "").strip()
                gold_entities = get_global_const_entities(meta_info, ent2id) # [45454]
                topic_entity = get_global_topic_entity(meta_info, ent2id) # [45454]
                new_sample["const_entities"] = list(set(gold_entities)-set(topic_entity))
                new_sample["entities"] = topic_entity
                if split == "train" and len(new_sample["entities"]) == 0:
                    print("Qid: %s does not have any valid topic entities, we skip this training example"%(qid))
                    continue
                new_sample["answers"] = get_answers(meta_info) #  [{"kb_id": "1971-10-13", "text": null}]
                ans_list = [a["kb_id"] for a in new_sample["answers"]]
                if split != "test" and len((set(sample["subgraph"][0]) & set(ans_list))) == 0:
                    print("Qid: %s does not retrieval valid answer entities, we skip this training example"%(qid))
                    continue
                ans_list = [{"kb_id": ent2id[a["kb_id"]], "text": a["text"]} for a in new_sample["answers"]]
                new_sample["answers"] = ans_list
                new_sample["subgraph"] = get_subgraph(sample["subgraph"], ent2id, rel2id, max_nodes, new_sample, split,
                                                      args.add_constraint_reverse) # {"tuples": [[h_glo,r_glo, t_glo], ], "entities": [glo]}
                if args.add_constraint_reverse:
                    new_sample["entities"] = list(gold_entities)
                f.write(json.dumps(new_sample)+"\n")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_path', required=True)
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--kg_map_path', required=True)
    parser.add_argument('--kg_map_prefix', default="")
    parser.add_argument('--max_nodes', default=1500, type=int)
    parser.add_argument('--add_constraint_reverse', action="store_true")
    parser.add_argument('--split_list', default=["train", "dev", "test"], nargs="+")
    args = parser.parse_args()

    print("Start convert to NSM input format.")
    return args

if __name__ == '__main__':
    # python convert_to_NSM_format.py --original_path "data/webqsp/data/WebQSP.split.SPLIT.jsonl" --input_path "data/webqsp/data/webqsp.split.SPLIT.retrieval.sg.nb.10.topk.10.jsonl" --output_path "data/webqsp/data/SPLIT_simple.json" --kg_map_path "data/webqsp/data"
    args = _parse_args()
    convert_to_NSM_input_format(args.original_path, args.input_path, args.output_path, args.split_list, args.kg_map_path, args.kg_map_prefix ,args.max_nodes)
