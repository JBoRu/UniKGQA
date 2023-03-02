import json
import os
from tqdm import tqdm
import re


def is_ent(tp_str):
    if len(tp_str) < 3:
        # print("%s is not entity" % (tp_str))
        return False
    if tp_str.startswith("m.") or tp_str.startswith("g."):
        # print(tp_str)
        return True
    # print("%s is not entity" % (tp_str))
    return False


def find_entity(sparql_str):
    str_lines = sparql_str.split("\n")
    ent_set = set()
    for line in str_lines[1:]:
        if "ns:" not in line:
            continue
        spline = line.strip().split(" ")
        for item in spline:
            item = item.strip().strip("\n")
            if not item.startswith("ns:"):
                # print(item)
                continue
            ent_str = item[3:].replace("(", "")
            ent_str = ent_str.replace(")", "")
            ent_str = ent_str.replace("\t.", "")
            ent_str = ent_str.strip("\n").strip("\t").strip()
            if is_ent(ent_str):
                ent_set.add(ent_str)
            elif "ns:m." in item or "ns:g." in item:
                print("The %s entity has not be identified"%(item))
    return ent_set

# data can be downloaded from https://github.com/lanyunshi/KBQA-GST
data_folder = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/webqsp"
data_file = ["train.jsonl", "dev.jsonl", "test.jsonl"]
output_file = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/webqsp/webqsp_step0.json"
# data_folder = "/home/jiangjinhao/work/QA/UniKBQA/UniModel/data/cwq"
# data_file = ["train.jsonl", "dev.jsonl", "test.jsonl"]
# output_file = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/cwq/cwq_step0.json"
f_out = open(output_file, "w")
for file in data_file:
    filename = os.path.join(data_folder, file)
    print("Process %s"%(filename))
    with open(filename, "r") as f_in:
        all_data = f_in.readlines()
        data = [json.loads(l) for l in all_data]
        for q_obj in tqdm(data, total=len(data)):
            ID = q_obj["QuestionId"]
            answer_list_new = []
            ent_list = []
            for parse in q_obj["Parses"]:
                for answer_obj in parse["Answers"]:
                    new_obj = {}
                    new_obj["kb_id"] = answer_obj["AnswerArgument"].strip()
                    new_obj["text"] = answer_obj["EntityName"]
                    answer_list_new.append(new_obj)

                tpes = parse["TopicEntityMid"]
                if tpes is None:
                    print("Qid:%s doesn't have any topic entities."%(ID))
                    continue
                elif isinstance(tpes, str):
                    ent_list.append({"kb_id": tpes, "text": tpes})
                elif isinstance(tpes, list):
                    # print("Qid: %s topice entities: %s" % (ID, tpes))
                    for ent in tpes:
                        ent_list.append({"kb_id": ent, "text": ent})
                else:
                    print("Qid: %s topice entities: %s" % (ID, tpes))
                    continue

                sparql_str = parse["Sparql"]
                ent_set = find_entity(sparql_str)
                for ent in ent_set:
                    if ent not in ent_list:
                        ent_list.append({"kb_id": ent, "text": ent})
            question = q_obj["RawQuestion"]
            new_obj = {
                "id": ID,
                "answers": answer_list_new,
                "question": question,
                "entities": ent_list,
            }
            f_out.write(json.dumps(new_obj) + "\n")
f_out.close()
