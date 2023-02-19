import json
import os
import numpy as np
from tqdm import tqdm
import pickle

def process_line(line):
    question, answers = line.strip().strip('\n').split('\t')
    tpe_s = question.find('[')
    tpe_e = question.find(']')
    if tpe_s == -1 or tpe_e == -1:
        tpe = None
    else:
        tpe = question[tpe_s+1:tpe_e].strip()
    question = question.replace('[','').replace(']','').strip()
    answers = answers.strip().split('|')
    answers = [a.strip() for a in answers]
    return question, tpe, answers


for hop_prefix in ['1', '2', '3']:
    print("Start processing %s hop"%hop_prefix)
    root_path = "./data/metaqa"
    processed_path = os.path.join(root_path, f'metaqa-{hop_prefix}hop')
    ori_path = os.path.join(root_path, f"metaqa_ori/{hop_prefix}-hop")

    global_ent_map_path = os.path.join(root_path, 'metaqa_ori/kb_entity_dict.txt')
    global_ent2id = {}
    with open(global_ent_map_path, "r") as f:
        for line in f.readlines():
            eid, etext = line.strip("\n").split("\t")
            eid = eid.strip()
            etext = etext.strip()
            global_ent2id[etext] = eid
    print("Load global entity text to mid dict")


    # 将cwq转换为webqsp数据格式，并抽取主题实体和限制实体
    new_path = ["dev.jsonl", "train.jsonl", "test.jsonl"]
    input_nsm_path = ["qa_dev.txt", "qa_train.txt", "qa_test.txt"]
    for op, np in zip(new_path, input_nsm_path):
        split = op.split(".")[0]
        op = os.path.join(processed_path, op)
        np = os.path.join(ori_path, np)
        print("Process %s"%(np))

        with open(np) as f:
            ori_data = f.readlines()
        print("Load %s: %d"%(input_nsm_path, len(ori_data)))

        new_samples = []
        no_tpe_count = 0
        for idx, d in enumerate(tqdm(ori_data, total=len(ori_data))):
            new_d = {}
            new_d["Split"] = split
            new_d["ID"] = f"{split}_{str(idx)}"
            question, topic_entity, answers = process_line(d)
            new_d["RawQuestion"] = question
            new_d["ProcessedQuestion"] = question
            topic_entity = topic_entity
            if not topic_entity:
                print("%s doesn't have any topic entity!" % (d["id"]))
                no_tpe_count += 1
            parse = {
                "Sparql": None,
                "TopicEntityMid": topic_entity,
                "GoldEntityMid": [topic_entity],
                "InferentialChain": [],
                "Constraints": [],
                "Time": None,
                "Order": None,
                "Answers": [
                    {
                        "AnswerType": "Value",
                        "AnswerArgument": a,
                        "EntityName": a
                    } for a in answers
                ],
                "GoldConstRels": []
            }
            new_d["Parse"] = parse
            new_samples.append(new_d)
        print("Start write to file!")
        with open(op, "w") as f:
            for sample in new_samples:
                f.write(json.dumps(sample)+"\n")
        print("%d samples no tpes!"%(no_tpe_count))

    import numpy as np
    # 合并文件抽取对应的qid
    input_path = ["dev.jsonl", "train.jsonl", "test.jsonl"]
    all_data = []
    for ip in input_path:
        print("Aggregate %s"%(ip))
        split = ip.split(".")[0]
        ip = os.path.join(processed_path, ip)
        op = os.path.join(processed_path, split+".qid")
        with open(ip,"r") as f:
            data = f.readlines()
            all_data.extend(data)
            data = [json.loads(d) for d in data]
        qid_set = []
        for d in data:
            assert d["ID"] not in qid_set, d["ID"]
            qid_set.append(d["ID"])
        qid_np = np.array(qid_set)
        np.save(op, qid_np)
    total_path = os.path.join(processed_path, "all_data.jsonl")
    with open(total_path,"w") as f:
        for line in all_data:
            f.write(line)