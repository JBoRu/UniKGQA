import json
import os
import numpy as np
from tqdm import tqdm
import pickle

ent_path = "./data/cwq/cwq_NSM/entities.txt"
ent2id = {}
with open(ent_path) as f:
    for ent in f.readlines():
        ent = ent.strip("\n")
        ent2id[ent] = len(ent2id)
id2ent = {v:k for k,v in ent2id.items()}
print("Load NSM entities mapping!")

valid_ent_set_path = './data/cwq/graph_data/ent2id.pickle'
with open(valid_ent_set_path,"rb") as f:
    valid_ent_set = pickle.load(f)
print("Load ent2id file")

# valid_ent_set = set(valid_ent_set.keys())
# print("get valid enityt set")

def is_ent(tp_str):
    if len(tp_str) < 3:
        # print("%s is not entity" % (tp_str))
        return False
    if tp_str.startswith("m.") or tp_str.startswith("g."):
        # print(tp_str)
        return True
    # print("%s is not entity" % (tp_str))
    return False

def get_topic_entity(sql):
    lines = sql.split('\n')
    lines = [x.replace("  ", " ") for x in lines if x]
    # if lines[0] == '#MANUAL SPARQL':
    #     return None

    line_num = 0
    while line_num < len(lines):
        l = lines[line_num].strip().strip("\n").strip("\t")
        if l.startswith('WHERE {'):
            break
        line_num = line_num + 1
    assert line_num < len(lines), sql
    next_line = lines[line_num].strip().strip("\n").strip("\t")
    # assert next_line.startswith('SELECT DISTINCT ?x'), next_line
    # line_num = line_num + 1
    # next_line = lines[line_num].strip().strip("\n").strip("\t")
    assert next_line == 'WHERE {', sql
    assert '}' == lines[-1] or ' }' == lines[-1] or 'LIMIT' in lines[-1], sql

    lines = lines[line_num + 1: -1]
    lines = [l for l in lines if 'FILTER' not in l and "ORDER BY" not in l and "LIMIT" not in l]
    if lines[-1] == '}':
        lines = lines[:-1]
    lines = [l.replace("  ", " ").replace("EXISTS {", "") for l in lines]
    topic_entity = None
    for l in lines:
        if topic_entity is not None:
            break
        eles = l.strip().split(" ")
        for item in eles:
            item = item.strip().strip("\n")
            if not item.startswith("ns:"):
                continue
            ent_str = item[3:].replace("(", "")
            ent_str = ent_str.replace(")", "")
            ent_str = ent_str.replace("\t.", "")
            ent_str = ent_str.strip("\n").strip("\t").strip().replace("?x",'')
            if is_ent(ent_str):
                assert ent_str in valid_ent_set, ent_str
                topic_entity = ent_str
                break
    return topic_entity


def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass


def judge_ent_type(ent_str):
    a = ent_str
    if a.startswith("m.") or a.startswith("g."):
        return True
        # return False
    elif len(a.split("-")) == 3 and all([is_number(e) for e in a.split("-")]):  # 1921-09-13
        # print("date entity: %s" % (a))
        return False
    elif len(a.split("-")) == 2 and all([is_number(e) for e in a.split("-")]):  # 1921-09
        # print("date entity: %s" % (a))
        return False
    elif a.isdigit and len(a) == 4:  # 1990
        # print("date entity: %s" % (a))
        return False
    elif is_number(a):  # 3.99
        # print("digit entity: %s" % (a))
        return True
    elif len(a) < 3:
        # print("not string entity: %s" % (a))
        return False
    elif '?' in a or '#' in a:
        return False
    else:  # Elizabeth II
        # print("string entity: %s" % (a))
        return True


def extract_const_entities_from_sql(tpe, sql):
    const_entities = set()
    const_rels = set()
    lines = sql.split('\n')
    lines = [x.replace("  ", " ") for x in lines if x]
    if lines[0] == '#MANUAL SPARQL':
        return const_entities, const_rels

    line_num = 0
    while line_num < len(lines):
        l = lines[line_num].strip().strip("\n").strip("\t")
        if l.startswith('WHERE {'):
            break
        line_num = line_num + 1
    assert line_num < len(lines), sql
    next_line = lines[line_num].strip().strip("\n").strip("\t")
    # assert next_line.startswith('SELECT DISTINCT ?x'), next_line
    # line_num = line_num + 1
    # next_line = lines[line_num].strip().strip("\n").strip("\t")
    assert next_line == 'WHERE {', sql
    assert '}' == lines[-1] or ' }' == lines[-1] or 'LIMIT' in lines[-1], sql
    # if 'LIMIT' in lines[-1]:
    # print(lines[-1])

    lines = lines[line_num + 1: -1]

    # lines include 'filter'
    filter_lines = [l for l in lines if 'FILTER' in l and "?x != ?c" not in l
                    and '!isLiteral(?x)' not in l and '?x !=' not in l and "NOT EXISTS {?" not in l
                    and 'xsd:dateTime' not in l and '<' not in l and '>' not in l]
    str_entity_const_lines = [l for l in filter_lines if 'str(?sk' in l]
    for line in str_entity_const_lines:  # FILTER (str(?sk0) = "State")
        str_ent = line.strip().split('=')[1].strip()
        str_ent = str_ent.replace(')', '').replace('"', '').replace('?x', '')
        if str_ent in valid_ent_set:
            const_entities.add(str_ent)
        else:
            print(str_ent)
            continue
        placeholder = line.strip().split('=')[0].strip()[-5:-1]
        if placeholder != tpe:
            for line in lines:
                if placeholder in line and 'FILTER' not in line:
                    eles = line.strip().split()
                    if len(eles) == 4:
                        r = eles[1]
                        assert r.startswith("ns:"), eles
                        const_rels.add(r)

    # mid_entity_const_lines = [l for l in filter_lines if 'str(?sk' not in l]
    # for line in mid_entity_const_lines: # FILTER (?f1 = ?f)?x ns:book.author.works_written ns:m.067z76z
    #     eles = line.strip().split(" ")
    #     for item in eles:
    #         item = item.strip().strip("\n")
    #         if item.startswith("ns:") and not item.startswith("ns:m.") and not item.startswith("ns:g."):
    #             continue
    #         if item.startswith("ns:"):
    #             ent_str = item[3:].replace("(", "")
    #             ent_str = ent_str.replace(")", "")
    #             ent_str = ent_str.replace("\t.", "")
    #             ent_str = ent_str.strip("\n").strip("\t").strip()
    #             const_entities.add(ent_str)
    # lines include 'LIMIT'
    # limit_lines = [l for l in lines if "LIMIT" in l]

    lines = [l for l in lines if 'FILTER' not in l and "ORDER BY" not in l and "LIMIT" not in l]
    if lines[-1] == '}':
        lines = lines[:-1]
    # lines include 'dateTime'
    time_lines = [l for l in lines if "dateTime" in l and "=" in l and "<" not in l and ">" not in l]
    for l in time_lines:  # ?x ns:people.person.date_of_birth "1932-08-02"^^xsd:dateTime .
        eles = l.strip().split(" ")
        const_rel = None
        for ele in eles:
            if ele.startswith("ns:") and not ele.startswith("ns:m.") and not ele.startswith("ns:g."):
                # const_rels.add(ele)
                const_rel = ele
            if ele.startswith('"') and ele.endswith("^^xsd:dateTime"):
                time_ent = ele.replace('^^xsd:dateTime', '').replace('"', '').replace('?x','')
                if time_ent in valid_ent_set:
                    const_entities.add(time_ent)
                    if const_rel is not None:
                        const_rels.add(const_rel)
                else:
                    print(time_ent)

    lines = [l for l in lines if "dateTime" not in l]
    lines = [l.replace("  ", " ").replace("EXISTS {", "") for l in lines]
    # for line in lines:
    #     if 'ns:tv.tv_program.thetvdb_id' in line:
    #         print(sql)
    # return
    for l in lines:
        eles = l.strip().split(" ")
        h = eles[0]
        r = eles[1]
        t = ' '.join(eles[2:-1])
        for item in [h, t]:
            item = item.strip().strip("\n")
            if item.startswith("ns:") and not item.startswith("ns:m.") and not item.startswith("ns:g."):
                # print("Skip relation, ", item)
                continue
            if item.startswith("ns:"):
                ent_str = item[3:].replace("(", "")
            else:
                ent_str = item.replace("(", "")
            ent_str = ent_str.replace(")", "")
            ent_str = ent_str.replace("\t.", "")
            ent_str = ent_str.replace('"', '')
            ent_str = ent_str.replace('@en', '')
            ent_str = ent_str.strip("\n").strip("\t").strip().replace('?x','')
            if judge_ent_type(ent_str):
                # print(sql)
                if ent_str in valid_ent_set:
                    const_entities.add(ent_str)
                    if ent_str != tpe:
                        const_rels.add(r)
                else:
                    print(ent_str)
    const_rels = [rel.replace('ns:', '') for rel in const_rels]
    return const_entities, const_rels


# ori_path = ["ComplexWebQuestions_dev.json"]
# new_path = ["dev.jsonl"]
# nsm_path = ["dev_simple.json"]
# 将cwq转换为webqsp数据格式，并抽取主题实体和限制实体
ori_path = ["ComplexWebQuestions_dev.json","ComplexWebQuestions_train.json","ComplexWebQuestions_test_wans.json"]
new_path = ["dev.jsonl", "train.jsonl", "test.jsonl"]
nsm_path = ["dev_simple.json", "train_simple.json", "test_simple.json"]
for ip, op, np in zip(ori_path, new_path, nsm_path):
    split = op.split(".")[0]
    ip = "./data/cwq/cwq_ori/"+ip
    op = "./data/cwq/"+op
    np = "./data/cwq/cwq_NSM/"+np
    print("Process %s"%(ip))

    with open(ip) as f:
        data = json.load(f)
    print("Load original data over!")

    with open(np) as f:
        nsm_data = f.readlines()
        def func(line):
            line = json.loads(line)
            line.pop("subgraph")
            return line
        nsm_data = [func(d) for d in nsm_data]
        nsm_data_dict = {d["id"]:d for d in nsm_data}
    print("Load nsm data over!")

    new_samples = []
    no_tpe_count = 0
    for d in tqdm(data, total=len(data)):
        new_d = {}
        new_d["Split"] = "train"
        new_d["ID"] = d["ID"]
        new_d["RawQuestion"] = d["machine_question"]
        new_d["ProcessedQuestion"] = d["question"]
        nsm_d = nsm_data_dict[d["ID"]]
        topic_entity = get_topic_entity(d["sparql"])
        # print("Extract topic entity!")
        gold_entities_from_sql, const_rels = extract_const_entities_from_sql(topic_entity, d["sparql"])
        gold_entities_from_nsm = [id2ent[tpe].replace('?x', '') for tpe in nsm_d['entities']]
        # if len(gold_entities_from_sql) == 0:
        # gold_entities = gold_entities_from_nsm
        # else:
        # gold_entities = gold_entities_from_sql
        if len(set(gold_entities_from_nsm) - set(gold_entities_from_sql)) > 0:
            print(d["sparql"], gold_entities_from_sql, gold_entities_from_nsm)
        gold_entities = gold_entities_from_sql
        if topic_entity is None:
            other_entities = set(gold_entities_from_sql) | set(gold_entities_from_nsm)
            for ent in other_entities:
                if ent.startswith("m.") or ent.startswith("g."):
                    topic_entity = ent
                    break
            if topic_entity is None:
                print("%s doesn't have any topic entity!" % (d["ID"]))
                no_tpe_count += 1

        parse = {
            "Sparql": d["sparql"],
            "TopicEntityMid": topic_entity,
            "GoldEntityMid": list(gold_entities),
            "InferentialChain": [],
            "Constraints": [],
            "Time": None,
            "Order": None,
            "Answers": [
                {
                    "AnswerType": "Value",
                    "AnswerArgument": a["answer_id"][1:].replace('?x', '') if a["answer_id"].startswith(":") else a["answer_id"],
                    "EntityName": a["answer"]
                } for a in d["answers"]
            ],
            "GoldConstRels": list(const_rels)
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
    ip = "./data/cwq/"+ip
    op = "./data/cwq/"+split+".qid"
    with open(ip,"r") as f:
        data = f.readlines()
        all_data.extend(data)
        data = [json.loads(d) for d in data]
    qid_set = []
    for d in data:
        assert d["ID"] not in qid_set, d["QuestionId"]
        qid_set.append(d["ID"])
    qid_np = np.array(qid_set)
    np.save(op, qid_np)
total_path = "./data/cwq/all_data.jsonl"
with open(total_path,"w") as f:
    for line in all_data:
        f.write(line)