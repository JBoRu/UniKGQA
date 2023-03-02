import json
import os

NSM_dp = "./data/webqsp/webqsp_NSM/split_simple.json"
data_path = "./data/webqsp/"

ori_dp = os.path.join(data_path, "webqsp_ori", "WebQSP.split.json")
split_dp = os.path.join(data_path, "split.jsonl")

with open(NSM_dp, "r") as f:
    all_lines = f.readlines()
with open(ori_dp, "r") as f:
    ori_data = json.load(f)
    ori_data = ori_data["Questions"]
    ori_data_dict = {o["QuestionId"]: o for o in ori_data}

all_split_lines = []
with open(split_dp, "w") as f:
    for i in all_lines:
        i = json.loads(i)
        qid = i["id"]
        sd = ori_data_dict[qid]
        sd = json.dumps(sd)
        all_split_lines.append(sd)
    print("Total %d" % (len(all_split_lines)))
    for line in all_split_lines:
        f.write(line + "\n")

path = "data/webqsp/data/WebQSP.test.json"
out = "data/webqsp/data/WebQSP.split.test.jsonl"
with open(path, "r") as f:
    data = json.load(f)
    data = data["Questions"]
    with open(out, "w") as fo:
        for l in data:
            fo.write(json.dumps(l) + "\n")
