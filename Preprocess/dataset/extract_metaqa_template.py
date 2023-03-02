import json
import os
import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import pickle


for hop_prefix in ['1', '2', '3']:
    print("Start processing %s hop"%hop_prefix)
    root_path = "./data/metaqa"
    processed_path = os.path.join(root_path, f'metaqa-{hop_prefix}hop')
    output_path = os.path.join(root_path, f'metaqa-{hop_prefix}hop-os')
    ori_path = os.path.join(root_path, f"metaqa_ori/{hop_prefix}-hop")

    input_path = os.path.join(ori_path, "qa_train.txt")
    with open(input_path, "r") as f:
        ori_data = f.readlines()
        ori_data = [line.split("\t")[0] for line in ori_data]
    print("Load %d question samples."%len(ori_data))
    type_count = defaultdict(int)
    type_idx = defaultdict(list)
    for idx, que in enumerate(ori_data):
        tp_s = que.find("[")
        tp_e = que.find("]")
        tpe = que[tp_s:tp_e+1].strip()
        template = que.replace(tpe, '').replace("  ", " ")
        # template = " ".join(template.split(" ")
        if idx < 10:
            print(que, "--->", template)
        type_count[template] += 1
        type_idx[template].append(idx)
    count = list(type_count.values())
    print("There are total %d question types: [%d, %d, %.3f]"%(len(type_count), min(count), max(count), np.mean(count)))
    continue
    selected_idx = []
    for t, idx_list in type_idx.items():
        exclude_idx = [3612, 7806, 8212, 11527, 60831]
        flag = [si not in exclude_idx for si in idx_list]
        si = random.sample(idx_list, 1)[0]
        if any(flag):
            while si in exclude_idx:
                si = random.sample(idx_list, 1)[0]
                print(si)
        selected_idx.append(si)
    print("Total select %d training examples."%len(selected_idx))
    selected_idx_path = os.path.join(output_path, "sample_idx.txt")
    with open(selected_idx_path, 'w') as f:
        for idx in selected_idx:
            f.write(str(idx)+"\n")
    print("Save selected idx to %s"%selected_idx_path)
    template_path = os.path.join(output_path, "samples.txt")
    with open(template_path, "w") as f:
        for tem in type_count.keys():
            f.write(tem+"\n")
    print("Save templates to %s" % template_path)

    ip = os.path.join(processed_path, "train.jsonl")
    op = os.path.join(output_path, "train.jsonl")
    print("Process %s"%(ip))
    with open(ip, 'r') as f:
        processed_data = f.readlines()
    print("Load %s: %d"%(ip, len(processed_data)))

    oneshot_samples = []
    for idx in selected_idx:
        oneshot_samples.append(processed_data[idx])
    with open(op, 'w') as f:
        for line in oneshot_samples:
            f.write(line)

    # 合并文件抽取对应的qid
    input_path = ["dev.jsonl", "train.jsonl", "test.jsonl"]
    all_data = []
    for ip in input_path:
        print("Aggregate %s"%(ip))
        split = ip.split(".")[0]
        ip = os.path.join(output_path, ip)
        op = os.path.join(output_path, split+".qid")
        with open(ip, "r") as f:
            data = f.readlines()
            all_data.extend(data)
            data = [json.loads(d) for d in data]
        qid_set = []
        for d in data:
            assert d["ID"] not in qid_set, d["ID"]
            qid_set.append(d["ID"])
        qid_np = np.array(qid_set)
        np.save(op, qid_np)
    total_path = os.path.join(output_path, "all_data.jsonl")
    with open(total_path,"w") as f:
        for line in all_data:
            f.write(line)