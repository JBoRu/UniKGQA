import json
import os
import argparse

def output_dict(ele2id, filename):
    num_ele = len(ele2id)
    id2ele = {v: k for k, v in ele2id.items()}
    f = open(filename, "w")
    for i in range(num_ele):
        f.write(str(id2ele[i]) + "\n")
    f.close()

# def get_meta_info_gold_entities(data):
#     parses = data["Parses"]
#     gold_entities = set()
#     for p in parses:
#         ent = p["TopicEntityMid"]
#         gold_entities.add(ent)
#         ans = p["Answers"]
#         for a in ans:
#             gold_entities.add(a["AnswerArgument"])
#     return list(gold_entities)

def get_meta_info_gold_entities(data):
    parse = data["Parse"]
    gold_entities = set()
    # ent = parse["TopicEntityMid"]
    # gold_entities.add(ent)
    ent = parse["GoldEntityMid"] if 'GoldEntityMid' in parse else parse["TopicEntityMid"]
    gold_entities.update(ent)
    ans = parse["Answers"]
    for a in ans:
        gold_entities.add(a["AnswerArgument"])
    return list(gold_entities)

def map_dataset_specific_kg_to_global_id(in_path, out_path, split_list, ori_path, prefix):
    ent2id = {}
    rel2id = {}
    for split in split_list:
        in_path_new = in_path.replace("SPLIT", split)
        ori_path_new = ori_path.replace("SPLIT", split)
        print("Process %s and %s" % (in_path_new, ori_path_new))
        with open(in_path_new, "r") as f:
            all_lines = f.readlines()
        with open(ori_path_new, "r") as f:
            ori_sample = f.readlines()
            ori_sample = [json.loads(s) for s in ori_sample]
            ori_sample_dict = {e["ID"]: e for e in ori_sample}
        for idx, line in enumerate(all_lines):
            sample = json.loads(line)
            subgraph = sample["subgraph"]

            meta_info = ori_sample_dict[sample["qid"]]
            entities, triples = subgraph
            entities.extend(get_meta_info_gold_entities(meta_info))

            for ent in entities:
                if str(ent) not in ent2id:
                    ent2id[str(ent)] = len(ent2id)
            for tri in triples:
                h, r, t = tri
                if str(h) not in ent2id:
                    ent2id[str(h)] = len(ent2id)
                if str(t) not in ent2id:
                    ent2id[str(t)] = len(ent2id)
                if r not in rel2id:
                    rel2id[r] = len(rel2id)
    print("Number Entity : {}, Relation : {}".format(len(ent2id), len(rel2id)))
    output_dict(ent2id, os.path.join(out_path, prefix+"entities.txt"))
    output_dict(rel2id, os.path.join(out_path, prefix+"relations.txt"))

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', required=True)
    parser.add_argument('--ori_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--output_prefix', default="")
    parser.add_argument('--split_list', default=["train", "dev", "test"], nargs="+")
    args = parser.parse_args()

    print("Start mapping the KG to global id")
    return args

if __name__ == '__main__':
    args = _parse_args()
    map_dataset_specific_kg_to_global_id(args.input_path, args.output_path, args.split_list, args.ori_path, args.output_prefix)
