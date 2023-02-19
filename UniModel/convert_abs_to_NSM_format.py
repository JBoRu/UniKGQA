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

def get_global_topic_entities(data, ent2id):
    parses = data["Parses"]
    topic_entities = set()
    for p in parses:
        ent = p["TopicEntityMid"]
        try:
            ent = str(ent)
            topic_entities.add(ent2id[ent])
        except:
            print(ent, " not in ent2id mapping")
    return list(topic_entities)

def get_answers(data):
    parses = data["Parses"]
    answers_list = []
    for p in parses:
        ans = p["Answers"]
        for a in ans:
            ans_dict = {}
            ans_dict["kb_id"] = a["AnswerArgument"]
            ans_dict["text"] = a["EntityName"]
            answers_list.append(ans_dict)
    return answers_list

def get_subgraph(subgraph, ent2id, rel2id, max_nodes, new_sample, split):
    entities, triples = subgraph["entities"], subgraph["tuples"]
    subgraph_new = {}
    entities_map = []
    triples_map = []
    for hrt in triples:
        h, r, t = hrt
        h = str(h)
        t = str(t)
        assert h in ent2id, h
        assert t in ent2id, t
        assert r in rel2id, r
        triples_map.append([ent2id[h], rel2id[r], ent2id[t]])
    for ent in entities:
        ent = str(ent)
        assert ent in ent2id
        entities_map.append(ent2id[ent])

    if len(entities_map) > max_nodes and split != "test":
        print("For %s set, more than max nodes: %d, so truncate it."%(split, max_nodes))
        keep_entities_id = set()
        keep_entities_id.update(new_sample["entities"])
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

def convert_to_NSM_input_format(in_path, out_path, split_list, kg_map_path, prefix, max_nodes):
    ent2id = load_dict(os.path.join(kg_map_path, prefix+'entities.txt'))
    rel2id = load_dict(os.path.join(kg_map_path, prefix+'relations.txt'))
    for split in split_list:
        in_path_new = in_path.replace("SPLIT", split)
        out_path_new = out_path.replace("SPLIT", split)
        print("Input %s to Output %s: "%(in_path_new, out_path_new))
        with open(in_path_new, "r") as f:
            all_lines = f.readlines()
        with open(out_path_new, "w") as f:
            for line in tqdm(all_lines, total=len(all_lines)):
                new_sample = {}
                sample = json.loads(line)
                qid = sample["ID"]
                new_sample["ID"] = qid
                query = sample["question"]
                new_sample["question"] = query
                new_sample["entities"] = [ent2id[str(i)] for i in sample["entities"]]
                new_sample["const_entities"] = [ent2id[str(i)] for i in sample["const_entities"]]
                if split == "train" and len(new_sample["entities"]) == 0:
                    print("Qid: %s does not have any valid topic entities, we skip this training example"%(qid))
                    continue
                new_sample["answers"] = sample["answers"]
                ans_list = [a["kb_id"] for a in new_sample["answers"]]
                if split != "test" and len((set(sample["subgraph"]["entities"]) & set(ans_list))) == 0:
                    print("Qid: %s does not retrieval valid answer entities, we skip this training example"%(qid))
                    continue
                new_sample["answers"] = [{"kb_id": ent2id[str(a["kb_id"])], "text": ent2id[str(a["kb_id"])]} for a in
                                         new_sample["answers"]]
                new_sample["subgraph"] = get_subgraph(sample["subgraph"], ent2id, rel2id, max_nodes, new_sample, split) # {"tuples": [[h_glo,r_glo, t_glo], ], "entities": [glo]}
                f.write(json.dumps(new_sample)+"\n")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--all_output_path', required=True)
    parser.add_argument('--kg_map_path', required=True)
    parser.add_argument('--kg_map_prefix', default="")
    parser.add_argument('--max_nodes', default=2000, type=int)
    parser.add_argument('--split_list', default=["train", "dev", "test"], nargs="+")
    args = parser.parse_args()

    print("Start convert to NSM input format.")
    return args

if __name__ == '__main__':
    args = _parse_args()
    convert_to_NSM_input_format(args.input_path, args.output_path, args.split_list, args.kg_map_path, args.kg_map_prefix ,args.max_nodes)
    all_data = []
    for split in args.split_list:
        path = args.output_path.replace("SPLIT", split)
        with open(path, "r") as f:
            lines = f.readlines()
            all_data.extend(lines)
    with open(args.all_output_path, "w") as f:
        for line in all_data:
            f.write(line.strip("\n")+"\n")