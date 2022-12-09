# %%
import json
import numpy as np

# %% [markdown]
# ### 统计检索回来的子图 answer 和 gold relation 的召回率

# %%
sg_path = "./data/webqsp/data/SPLIT.retrieval.sg.from_sg_freebase.nb.10.topk.5.jsonl"
ori_path = "./data/webqsp/data/WebQSP.split.SPLIT.jsonl"


# %%
def extract_gold_rels_and_ans_from_sparql(path):
    rels = set()
    sparql = p["Sparql"]
    lines = sparql.split("\n")
    for line in lines:
        if "ns:" in line:
            eles = line.split(" ")
            for ele in eles:
                if ele.startswith("ns:"):
                    if not ele.startswith("ns:m.") and not ele.startswith("ns:g.") and "." in ele:
                        rels.add(ele[3:])
                    else:
                        # print("%s is not relation"%(ele))
                        pass

    answers = set([a["AnswerArgument"] for a in p["Answers"]])

    return rels, answers


# %%
for split in ["dev", "test", "train"]:
    # for split in ["dev"]:
    sp = sg_path.replace("SPLIT", split)
    op = ori_path.replace("SPLIT", split)
    with open(sp, "r") as f:
        all_sg = f.readlines()
        all_sg = [json.loads(l) for l in all_sg]
    with open(op, "r") as f:
        all_ori = f.readlines()
        all_ori = [json.loads(l) for l in all_ori]
        all_ori_dict = {d["QuestionId"]: d for d in all_ori}

    all_right_count = 0
    all_ans_recall = []
    all_rel_recall = []

    for sample in all_sg:
        qid = sample["qid"]
        try:
            retrieval_rels = [r for ps in sample["paths"][0] for r in ps[0]]
        except:
            print("Error qid:", qid)
            continue
        retrieval_rels_ordered_list = []
        for r in retrieval_rels:
            if r not in retrieval_rels_ordered_list:
                retrieval_rels_ordered_list.append(r)
        retrieval_rels = set(retrieval_rels_ordered_list)

        retrieval_ents = set(sample["subgraph"][0])

        ori_sample = all_ori_dict[qid]
        parses = ori_sample["Parses"]

        for p in parses:
            pid = p["ParseId"]
            gold_rels, gold_ans = extract_gold_rels_and_ans_from_sparql(p)
            if len(gold_rels) == 0:
                print("%s not include valid relations" % (pid))
                continue
            if len(gold_ans) == 0:
                print("%s not include valid answers" % (pid))
            if len(gold_rels) == 0:
                rels_recall = 1.0
            else:
                rels_recall = len(retrieval_rels & gold_rels) / len(gold_rels)
            all_rel_recall.append(rels_recall)
            if len(gold_ans) == 0:
                ans_recall = 1.0
            else:
                ans_recall = len(retrieval_ents & gold_ans) / len(gold_ans)
            all_ans_recall.append(ans_recall)
            if rels_recall == ans_recall == 1:
                all_right_count += 1
            else:
                print("Qid:%s - Pid:%s" % (qid, pid))
                print("Retrieval relations:%s" % (retrieval_rels_ordered_list))
                print("Gold relations:%s" % (gold_rels))
                print("Retrieval entities:%s" % (retrieval_ents))
                print("Gold entities:%s" % (gold_ans))
    avg_rel_rec = np.mean(all_rel_recall)
    avg_ans_rec = np.mean(all_ans_recall)
    print("%s set, avg relation recall:%.2f answer recall:%.2f" % (split, avg_rel_rec, avg_ans_rec))

# %% [markdown]
# ### 统计抽取的二跳子图KG对所有实例的召回率

# %%
import numpy as np
import json
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib

sparql = SPARQLWrapper("http://localhost:3001/sparql")
sparql.setReturnFormat(JSON)

# %%
ori_path = "data/webqsp/data/WebQSP.split.SPLIT.jsonl"


# %%
def execute_query(query):
    answers = set()
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    for result in results['results']['bindings']:
        # print(result)
        answers.add(result['x']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return answers


# %%
for split in ["train", "dev", "test"]:
    path = ori_path.replace("SPLIT", split)
    with open(path, "r") as f:
        all_data = f.readlines()
        all_data = [json.loads(l) for l in all_data]
    total = 0
    valid = 0
    ans_recall = []
    for sample in all_data:
        if "WebQTrn-301" in sample['QuestionId']:
            continue
        flag = False
        for parse in sample["Parses"]:
            if flag:
                continue
            total += 1
            pid = parse["ParseId"]
            sql = parse["Sparql"]
            # if "#MANUAL" in sql:
            #     # print("Skip %s which is manual sql"%(pid))
            #     # continue
            #     sql = '''+ ''+ sql + '' + '''
            try:
                ans = execute_query(sql)
            except:
                print("Error in %s" % (pid))
                continue
            ans_gold = parse["Answers"]
            ans_gold = [a["AnswerArgument"] for a in ans_gold]
            if len(ans_gold) == 0:
                recall = 1.0
            else:
                recall = len(ans & set(ans_gold)) / len(set(ans_gold))
            if recall < 1:
                print("Not perfect parse:%s - %.2f" % (pid, recall))
                ans_recall.append(recall)
            else:
                ans_recall.append(recall)
                flag = True
            valid += 1
    print("%s %d/%d set average ans recall:%.2f" % (split, valid, total, np.mean(ans_recall)))

# %% [markdown]
# ### 将输入数据分为train/dev/test

# %%
import json
import os

# %%
data_path = "./data/webqsp"
NSM_dp = os.path.join(data_path, "webqsp_NSM", "train_simple.json")
ori_dp = os.path.join(data_path, "data", "WebQSP.train.json")
split_dp = os.path.join(data_path, "data", "WebQSP.split.train.jsonl")

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

# %% [markdown]
# ### 统计数据集的关系子图是否为逐跳往外

# %%
import json
import os


# %%
def check_infer_chain(data):
    no_ic_man_sql_cnt = 0
    no_ic_sql_cnt = 0
    no_ic_total = 0
    for d in data:
        qid = d["QuestionId"]
        parses = d["Parses"]
        for p in parses:
            no_ic_total += 1
            if p["InferentialChain"] is None:
                if "#MANUAL SPARQL" in p["Sparql"]:
                    print("PID %s doesn't have any inferchain which has manual sparql!" % (p["ParseId"]))
                    no_ic_man_sql_cnt += 1
                else:
                    print("PID %s doesn't have any inferchain!" % (p["ParseId"]))
                    no_ic_sql_cnt += 1
    print(
        "No infer chain: manual sql: %d/%d, sql: %d/%d" % (no_ic_man_sql_cnt, no_ic_total, no_ic_sql_cnt, no_ic_total))


def check_start_tpe(data):
    no_start_tpe_man_sql_cnt = 0
    no_start_tpe_sql_cnt = 0
    no_start_tpe_total = 0
    for d in data:
        qid = d["QuestionId"]
        parses = d["Parses"]
        for p in parses:
            no_start_tpe_total += 1
            sql = p['Sparql']
            lines = sql.split("\n")
            where_start = 0
            for line in lines:
                if "WHERE" in line:
                    break
                else:
                    where_start += 1
            tri_start = where_start
            for line in lines[where_start:]:
                if "FILTER" not in line and "ns:" in line:
                    break
                else:
                    tri_start += 1
            tpe = p["TopicEntityMid"]
            if tpe is None:
                print("PID %s doesn't have any tpe!" % (p["ParseId"]))
                continue
            elif not isinstance(tpe, str):
                print("PID %s has many tpes %s!" % (p["ParseId"], tpe))
                continue
            assertion = "ns:" + tpe
            # print(lines[tri_start], assertion)
            if not lines[tri_start].startswith(assertion):
                if "#MANUAL SPARQL" in p["Sparql"]:
                    print("PID %s doesn't start with tpe which has manual sparql!" % (p["ParseId"]))
                    no_start_tpe_man_sql_cnt += 1
                else:
                    print("PID %s doesn't start with tpe!" % (p["ParseId"]))
                    no_start_tpe_sql_cnt += 1
    print("No start with tpe: manual sql: %d/%d, sql: %d/%d" % (
    no_start_tpe_man_sql_cnt, no_start_tpe_total, no_start_tpe_sql_cnt, no_start_tpe_total))


def check_manual_sample(data):
    only_man_sql = 0
    total_sql = 0
    for d in data:
        total_sql += 1
        flag = False
        qid = d["QuestionId"]
        parses = d["Parses"]
        for p in parses:
            sql = p['Sparql']
            if "#MANUAL SPARQL" in sql:
                continue
            else:
                flag = True
                break
        if not flag:
            print("QID %s only has manual sql parse!" % (qid))
            only_man_sql += 1
    print("Only has manual sql sample: %d/%d" % (only_man_sql, total_sql))


def extract_sql_structure(data):
    count = 0.0
    total = 0.0
    for d in data:
        qid = d["QuestionId"]
        parses = d["Parses"]
        skip = False
        for p in parses:
            total += 1
            pif = p["ParseId"]
            query = p['Sparql']
            tpe = p['TopicEntityMid']
            if tpe is None:
                continue
            lines = query.split('\n')
            lines = [x for x in lines if x]

            # assert lines[0] != '#MANUAL SPARQL'
            if lines[0] == '#MANUAL SPARQL':
                continue

            prefix_stmts = []
            line_num = 0
            while True:
                l = lines[line_num]
                if l.startswith('PREFIX'):
                    prefix_stmts.append(l)
                else:
                    break
                line_num = line_num + 1

            next_line = lines[line_num]
            assert next_line.startswith('SELECT DISTINCT ?x')
            # if not next_line.startswith('SELECT DISTINCT ?x'):
            #     print("%s not match select distinct ?x!"%(qid))
            #     print(query)
            line_num = line_num + 1
            next_line = lines[line_num]
            assert next_line == 'WHERE {'
            # if not next_line == 'WHERE {':
            #     print("%s not match WHERE {"%(qid))
            #     print(query)
            assert lines[-1] in ['}', 'LIMIT 1', 'OFFSET 1', 'LIMIT 3']
            # if lines[-1] not in ['}', 'LIMIT 1', 'OFFSET 1', 'LIMIT 3',]:
            #     print("%s not match '}', 'LIMIT 1'"%(qid))
            #     print(d["ProcessedQuestion"])
            #     print(query)

            lines = lines[line_num + 1: -1]
            # assert all(['FILTER (str' not in x for x in lines])
            # if not all(['FILTER (str' not in x for x in lines]):
            #     print("%s not match all"%(qid))
            #     print(d["ProcessedQuestion"])
            #     print(query)
            lines = [l for l in lines if 'FILTER' not in l and "ORDER BY" not in l and "LIMIT" not in l]
            if lines[-1] == '}':
                lines = lines[:-1]
            lines = [l.replace("EXISTS {", "") for l in lines]
            tpe_str = "ns:" + tpe

            assert lines[0].startswith(tpe_str)
            flag = True
            if len(lines) == 1:
                # count += 1
                pass
            # tpe->y->x
            # tpe->x->e
            elif len(lines) == 2:
                if lines[0].split(" ")[2] == '?y' and lines[1].split(" ")[0] == '?y':
                    # count += 1
                    pass
                elif lines[0].split(" ")[2] == '?x' and lines[1].split(" ")[0] == '?x':
                    # count += 1
                    pass
                else:
                    print("%s not match all" % (qid))
                    print(lines)
            # tpe->?y->?x / ?y->t
            # tpe->?x->t / ?x->t
            elif len(lines) == 3:
                if lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?y":
                    # count += 1
                    pass
                elif lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?x":
                    # count += 1
                    pass
                elif lines[1].split(" ")[0] == "?x" and lines[2].split(" ")[0] == "?x":
                    # count += 1
                    pass
                else:
                    print("%s not match all" % (qid))
                    print(lines)
            elif len(lines) == 4:
                if lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?y" and lines[3].split(" ")[0] == "?y":
                    # count += 1
                    pass
                elif (lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?x" and lines[3].split(" ")[
                    0] == "?y") or \
                        (lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?y" and lines[3].split(" ")[
                            0] == "?x"):
                    # count += 1
                    pass
                elif lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?x" and lines[3].split(" ")[
                    0] == "?x":
                    # count += 1
                    pass
                elif lines[1].split(" ")[0] == "?x" and lines[2].split(" ")[0] == "?x" and lines[3].split(" ")[
                    0] == "?x":
                    # count += 1
                    pass
                else:
                    print("%s not match all" % (qid))
                    print(lines)
            elif len(lines) == 5:
                if lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?y" and lines[3].split(" ")[
                    0] == "?y" and lines[4].split(" ")[0] == "?y":
                    # count += 1
                    pass
                elif lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?x" and lines[3].split(" ")[
                    0] == "?y" and lines[4].split(" ")[0] == "?y":
                    # count += 1
                    pass
                else:
                    print("%s not match all" % (qid))
                    print(lines)
            elif len(lines) == 6:
                if lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?y" and lines[3].split(" ")[0] == "?y" \
                        and lines[4].split(" ")[0] == "?y" and lines[5].split(" ")[0] == "?y":
                    # count += 1
                    pass
                elif lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?y" and lines[3].split(" ")[
                    0] == "?x" \
                        and lines[4].split(" ")[0] == "?y" and lines[5].split(" ")[0] == "?y":
                    # count += 1
                    pass
                else:
                    print("%s not match all" % (qid))
                    print(lines)

    print("%d/%d=%.2f" % (count, total, count / total))


# %%
ori_path = './data/webqsp/data/WebQSP.split.SPLIT.jsonl'
for split in ["train", "dev", "test"]:
    # for split in ["dev"]:
    print(split)
    path = ori_path.replace("SPLIT", split)
    with open(path, "r") as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]
    # check_infer_chain(data)
    # check_start_tpe(data)
    # check_manual_sample(data)
    extract_sql_structure(data)

# %% [markdown]
# ### 根据SPARQL抽取对应的gold relation structure

# %%
import json
import os

# %%
from collections import defaultdict


def get_relation(triple_str):
    try:
        h, r, t = triple_str.split(" ")[0:3]
        assert r.startswith("ns:")
    except:
        print(triple_str.split(" "))
    return r[3:]


def extract_gold_relation_structure(data):
    qid = data["QuestionId"]
    parses = data["Parses"]
    new_parses = []
    for p in parses:
        pid = p["ParseId"]
        query = p['Sparql']
        tpe = p['TopicEntityMid']
        if tpe is None:
            continue
        lines = query.split('\n')
        lines = [x for x in lines if x]

        # assert lines[0] != '#MANUAL SPARQL'
        if lines[0] == '#MANUAL SPARQL':
            new_parses.append(p)
            continue

        prefix_stmts = []
        line_num = 0
        while True:
            l = lines[line_num]
            if l.startswith('PREFIX'):
                prefix_stmts.append(l)
            else:
                break
            line_num = line_num + 1

        next_line = lines[line_num]
        assert next_line.startswith('SELECT DISTINCT ?x')
        line_num = line_num + 1
        next_line = lines[line_num]
        assert next_line == 'WHERE {'
        assert lines[-1] in ['}', 'LIMIT 1', 'OFFSET 1', 'LIMIT 3']

        lines = lines[line_num + 1: -1]
        lines = [l for l in lines if 'FILTER' not in l and "ORDER BY" not in l and "LIMIT" not in l]
        if lines[-1] == '}':
            lines = lines[:-1]
        lines = [l.replace("EXISTS {", "") for l in lines]

        tpe_str = "ns:" + tpe
        assert lines[0].startswith(tpe_str)

        gold_sql_structure = defaultdict(list)
        if len(lines) == 1:
            # tpe->x
            gold_sql_structure[0].append(get_relation(lines[0]))
        elif len(lines) == 2:
            if lines[0].split(" ")[2] == '?y' and lines[1].split(" ")[0] == '?y':
                # tpe->y->x
                for idx, line in enumerate(lines):
                    gold_sql_structure[idx].append(get_relation(line))
            elif lines[0].split(" ")[2] == '?x' and lines[1].split(" ")[0] == '?x':
                # tpe->x->e
                for idx, line in enumerate(lines):
                    gold_sql_structure[idx].append(get_relation(line))
            else:
                print("%s not match all" % (qid))
                print(lines)
        elif len(lines) == 3:
            if lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?y":
                # tpe->y->x / y->t
                gold_sql_structure[0].append(get_relation(lines[0]))
                gold_sql_structure[1].append(get_relation(lines[1]))
                gold_sql_structure[1].append(get_relation(lines[2]))
            elif lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?x":
                # tpe->y->x->t
                gold_sql_structure[0].append(get_relation(lines[0]))
                gold_sql_structure[1].append(get_relation(lines[1]))
                gold_sql_structure[2].append(get_relation(lines[2]))
            elif lines[1].split(" ")[0] == "?x" and lines[2].split(" ")[0] == "?x":
                # tpe->x->t1 / x->t2
                gold_sql_structure[0].append(get_relation(lines[0]))
                gold_sql_structure[1].append(get_relation(lines[1]))
                gold_sql_structure[1].append(get_relation(lines[2]))
            else:
                print("%s not match all" % (qid))
                print(lines)
        elif len(lines) == 4:
            if lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?y" and lines[3].split(" ")[0] == "?y":
                # tpe->y->x / y->t1 / y->t2
                gold_sql_structure[0].append(get_relation(lines[0]))
                gold_sql_structure[1].append(get_relation(lines[1]))
                gold_sql_structure[1].append(get_relation(lines[2]))
                gold_sql_structure[1].append(get_relation(lines[3]))
            elif lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?x" and lines[3].split(" ")[0] == "?y":
                # tpe->y->x->t1 / y->t1
                gold_sql_structure[0].append(get_relation(lines[0]))
                gold_sql_structure[1].append(get_relation(lines[1]))
                gold_sql_structure[2].append(get_relation(lines[2]))
                gold_sql_structure[1].append(get_relation(lines[3]))
            elif lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?y" and lines[3].split(" ")[0] == "?x":
                # tpe->y->x / y->t1 / x->t2
                gold_sql_structure[0].append(get_relation(lines[0]))
                gold_sql_structure[1].append(get_relation(lines[1]))
                gold_sql_structure[1].append(get_relation(lines[2]))
                gold_sql_structure[2].append(get_relation(lines[3]))
            elif lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?x" and lines[3].split(" ")[0] == "?x":
                # tpe->y->x->t1 / x->t2
                gold_sql_structure[0].append(get_relation(lines[0]))
                gold_sql_structure[1].append(get_relation(lines[1]))
                gold_sql_structure[2].append(get_relation(lines[2]))
                gold_sql_structure[2].append(get_relation(lines[3]))
            elif lines[1].split(" ")[0] == "?x" and lines[2].split(" ")[0] == "?x" and lines[3].split(" ")[0] == "?x":
                # tpe->x->t1 / x->t2 / x->t3
                gold_sql_structure[0].append(get_relation(lines[0]))
                gold_sql_structure[1].append(get_relation(lines[1]))
                gold_sql_structure[1].append(get_relation(lines[2]))
                gold_sql_structure[1].append(get_relation(lines[3]))
            else:
                print("%s not match all" % (qid))
                print(lines)
        elif len(lines) == 5:
            if lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?y" and lines[3].split(" ")[0] == "?y" and \
                    lines[4].split(" ")[0] == "?y":
                # tpe->y->x / y->t1 / y->t2 / y->t3
                gold_sql_structure[0].append(get_relation(lines[0]))
                gold_sql_structure[1].append(get_relation(lines[1]))
                gold_sql_structure[1].append(get_relation(lines[2]))
                gold_sql_structure[1].append(get_relation(lines[3]))
                gold_sql_structure[1].append(get_relation(lines[4]))
            elif lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?x" and lines[3].split(" ")[
                0] == "?y" and lines[4].split(" ")[0] == "?y":
                # tpe->y->x->t1 / y->t2 / y->t3
                gold_sql_structure[0].append(get_relation(lines[0]))
                gold_sql_structure[1].append(get_relation(lines[1]))
                gold_sql_structure[2].append(get_relation(lines[2]))
                gold_sql_structure[1].append(get_relation(lines[3]))
                gold_sql_structure[1].append(get_relation(lines[4]))
            else:
                print("%s not match all" % (qid))
                print(lines)
        elif len(lines) == 6:
            if lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?y" and lines[3].split(" ")[0] == "?y" \
                    and lines[4].split(" ")[0] == "?y" and lines[5].split(" ")[0] == "?y":
                # tpe->y->x / y->t1 / y->t2 / y->t3 / y->t4
                gold_sql_structure[0].append(get_relation(lines[0]))
                gold_sql_structure[1].append(get_relation(lines[1]))
                gold_sql_structure[1].append(get_relation(lines[2]))
                gold_sql_structure[1].append(get_relation(lines[3]))
                gold_sql_structure[1].append(get_relation(lines[4]))
                gold_sql_structure[1].append(get_relation(lines[5]))
            elif lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?y" and lines[3].split(" ")[0] == "?x" \
                    and lines[4].split(" ")[0] == "?y" and lines[5].split(" ")[0] == "?y":
                # tpe->y->x / y->t1 / x->t2 / y->t3 / y->t4
                gold_sql_structure[0].append(get_relation(lines[0]))
                gold_sql_structure[1].append(get_relation(lines[1]))
                gold_sql_structure[1].append(get_relation(lines[2]))
                gold_sql_structure[2].append(get_relation(lines[3]))
                gold_sql_structure[1].append(get_relation(lines[4]))
                gold_sql_structure[1].append(get_relation(lines[5]))
            else:
                print("%s not match all" % (qid))
                print(lines)

        p["gold_sql_structure"] = gold_sql_structure
        new_parses.append(p)
    data["Parses"] = new_parses
    return data


# %%
ori_path = './data/webqsp/data/WebQSP.split.SPLIT.jsonl'
for split in ["train", "dev", "test"]:
    # for split in ["dev"]:
    print(split)
    path = ori_path.replace("SPLIT", split)
    with open(path, "r") as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]
    new_data = []
    for d in data:
        new_d = extract_gold_relation_structure(d)
        new_data.append(json.dumps(new_d) + "\n")
    assert len(new_data) == len(data)
    with open(path, "w") as f:
        for d in new_data:
            f.write(d)

# %% [markdown]
# ### 查看限制关系类型

# %%
import json
import os

# %%
from collections import defaultdict


def get_relation(triple_str):
    try:
        h, r, t = triple_str.split(" ")[0:3]
        assert r.startswith("ns:")
    except:
        print(triple_str.split(" "))
    return r[3:]


def extract_gold_relation_structure(data):
    qid = data["QuestionId"]
    parses = data["Parses"]
    new_parses = []
    all_constrain_relations = set()
    for p in parses:
        pid = p["ParseId"]
        query = p['Sparql']
        tpe = p['TopicEntityMid']
        if tpe is None:
            continue
        lines = query.split('\n')
        lines = [x for x in lines if x]

        # assert lines[0] != '#MANUAL SPARQL'
        if lines[0] == '#MANUAL SPARQL':
            new_parses.append(p)
            continue

        prefix_stmts = []
        line_num = 0
        while True:
            l = lines[line_num]
            if l.startswith('PREFIX'):
                prefix_stmts.append(l)
            else:
                break
            line_num = line_num + 1

        next_line = lines[line_num]
        assert next_line.startswith('SELECT DISTINCT ?x')
        line_num = line_num + 1
        next_line = lines[line_num]
        assert next_line == 'WHERE {'
        assert lines[-1] in ['}', 'LIMIT 1', 'OFFSET 1', 'LIMIT 3']

        lines = lines[line_num + 1: -1]
        lines = [l for l in lines if 'FILTER' not in l and "ORDER BY" not in l and "LIMIT" not in l]
        if lines[-1] == '}':
            lines = lines[:-1]
        lines = [l.replace("EXISTS {", "") for l in lines]

        tpe_str = "ns:" + tpe
        assert lines[0].startswith(tpe_str)

        constrain_relation = set()
        if len(lines) == 1:
            # tpe->x
            continue
        elif len(lines) == 2:
            if lines[0].split(" ")[2] == '?y' and lines[1].split(" ")[0] == '?y':
                # tpe->y->x
                continue
            elif lines[0].split(" ")[2] == '?x' and lines[1].split(" ")[0] == '?x':
                # tpe->x->e
                # constrain_relation.add(get_relation(line[1]))
                all_constrain_relations.add(get_relation(lines[1]))
                continue
            else:
                print("%s not match all" % (qid))
                print(lines)
        elif len(lines) == 3:
            if lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?y":
                # tpe->y->x / y->t
                all_constrain_relations.add(get_relation(lines[2]))
            elif lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?x":
                # tpe->y->x->t
                all_constrain_relations.add(get_relation(lines[2]))
            elif lines[1].split(" ")[0] == "?x" and lines[2].split(" ")[0] == "?x":
                # tpe->x->t1 / x->t2
                all_constrain_relations.add(get_relation(lines[1]))
                all_constrain_relations.add(get_relation(lines[2]))
            else:
                print("%s not match all" % (qid))
                print(lines)
        elif len(lines) == 4:
            if lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?y" and lines[3].split(" ")[0] == "?y":
                # tpe->y->x / y->t1 / y->t2
                all_constrain_relations.add(get_relation(lines[2]))
                all_constrain_relations.add(get_relation(lines[3]))
            elif lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?x" and lines[3].split(" ")[0] == "?y":
                # tpe->y->x->t1 / y->t1
                all_constrain_relations.add(get_relation(lines[2]))
                all_constrain_relations.add(get_relation(lines[3]))
            elif lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?y" and lines[3].split(" ")[0] == "?x":
                # tpe->y->x / y->t1 / x->t2
                all_constrain_relations.add(get_relation(lines[2]))
                all_constrain_relations.add(get_relation(lines[3]))
            elif lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?x" and lines[3].split(" ")[0] == "?x":
                # tpe->y->x->t1 / x->t2
                all_constrain_relations.add(get_relation(lines[2]))
                all_constrain_relations.add(get_relation(lines[3]))
            elif lines[1].split(" ")[0] == "?x" and lines[2].split(" ")[0] == "?x" and lines[3].split(" ")[0] == "?x":
                # tpe->x->t1 / x->t2 / x->t3
                all_constrain_relations.add(get_relation(lines[1]))
                all_constrain_relations.add(get_relation(lines[2]))
                all_constrain_relations.add(get_relation(lines[3]))
            else:
                print("%s not match all" % (qid))
                print(lines)
        elif len(lines) == 5:
            if lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?y" and lines[3].split(" ")[0] == "?y" and \
                    lines[4].split(" ")[0] == "?y":
                # tpe->y->x / y->t1 / y->t2 / y->t3
                all_constrain_relations.add(get_relation(lines[2]))
                all_constrain_relations.add(get_relation(lines[3]))
                all_constrain_relations.add(get_relation(lines[4]))
            elif lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?x" and lines[3].split(" ")[
                0] == "?y" and lines[4].split(" ")[0] == "?y":
                # tpe->y->x->t1 / y->t2 / y->t3
                all_constrain_relations.add(get_relation(lines[2]))
                all_constrain_relations.add(get_relation(lines[3]))
                all_constrain_relations.add(get_relation(lines[4]))
            else:
                print("%s not match all" % (qid))
                print(lines)
        elif len(lines) == 6:
            if lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?y" and lines[3].split(" ")[0] == "?y" \
                    and lines[4].split(" ")[0] == "?y" and lines[5].split(" ")[0] == "?y":
                # tpe->y->x / y->t1 / y->t2 / y->t3 / y->t4
                all_constrain_relations.add(get_relation(lines[2]))
                all_constrain_relations.add(get_relation(lines[3]))
                all_constrain_relations.add(get_relation(lines[4]))
                all_constrain_relations.add(get_relation(lines[5]))
            elif lines[1].split(" ")[0] == "?y" and lines[2].split(" ")[0] == "?y" and lines[3].split(" ")[0] == "?x" \
                    and lines[4].split(" ")[0] == "?y" and lines[5].split(" ")[0] == "?y":
                # tpe->y->x / y->t1 / x->t2 / y->t3 / y->t4
                all_constrain_relations.add(get_relation(lines[2]))
                all_constrain_relations.add(get_relation(lines[3]))
                all_constrain_relations.add(get_relation(lines[4]))
                all_constrain_relations.add(get_relation(lines[5]))
            else:
                print("%s not match all" % (qid))
                print(lines)

        # p["gold_sql_structure"] = gold_sql_structure
        print(all_constrain_relations)
        new_parses.append(p)
    data["Parses"] = new_parses
    return all_constrain_relations


# %%
ori_path = './data/webqsp/data/WebQSP.split.SPLIT.jsonl'
for split in ["train", "dev", "test"]:
    # for split in ["dev"]:
    print(split)
    path = ori_path.replace("SPLIT", split)
    with open(path, "r") as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]
    all_constrain_relations = set()
    for d in data:
        one = extract_gold_relation_structure(d)
        all_constrain_relations.update(one)

# %%
print(all_constrain_relations)

# %% [markdown]
# ### 统计标注信息是否匹配真实SPARQL模式

# %%
from collections import defaultdict

# %%
ori_path = './data/webqsp/data/WebQSP.split.SPLIT.jsonl'
for split in ["train", "dev", "test"]:
    print(split)
    path = ori_path.replace("SPLIT", split)
    with open(path, "r") as f:
        data = f.readlines()
        data = [json.loads(d) for d in data]
    pattern = defaultdict(lambda: 0)
    for d in data:
        qid = d["QuestionId"]
        parses = d["Parses"]
        for p in parses:
            if "#MANUAL SPARQL" in p["Sparql"]:
                continue
            gold_relation = defaultdict(set)
            pid = p["ParseId"]
            tpe = p["TopicEntityMid"]
            if tpe is None:
                print("Pid %s doesn't have any topic entity" % (pid))
                continue
            infer_chain = p["InferentialChain"]
            if infer_chain is None or len(infer_chain) == 0:
                print("Pid %s doesn't have any infer chain" % (pid))
                continue
            for idx, rel in enumerate(infer_chain):
                gold_relation[idx].add(rel)
            constraints = p["Constraints"]
            for const in constraints:
                rel = const["NodePredicate"]
                idx = const["SourceNodeIndex"] + 1
                gold_relation[idx].add(rel)
            pat = ""
            for hop_idx, rel_set in gold_relation.items():
                pat += str(hop_idx) + '_' + str(len(rel_set)) + "-"
            pat = pat[:-1]
            if pat not in pattern:
                print("Adding new pattern: %s" % (pat))
            pattern[pat] += 1
    for k, v in pattern.items():
        print("Pattern Key: %s Count: %d" % (k, v))

# %%


# %% [markdown]
# ### 验证可达路径

# %%
import numpy as np
import json
from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib

sparql = SPARQLWrapper("http://localhost:3001/sparql")
sparql.setReturnFormat(JSON)


# %%
def get_complete_ans(ans, type):
    if type == "uri_ans":
        ans_new = ':' + ans
    elif type == "date_ans":
        ans_new = '"' + ans + '"' + '^^<http://www.w3.org/2001/XMLSchema#date>'
    elif type == "gyear_ans":
        ans_new = '"' + ans + '"' + '^^<http://www.w3.org/2001/XMLSchema#gYear>'
    elif type == "gyear_month_ans":
        ans_new = '"' + ans + '"' + '^^<http://www.w3.org/2001/XMLSchema#gYearMonth>'
    elif type == "en_literal_ans":
        ans_new = '"' + ans + '"' + '@en'
    elif type == "digit_ans":
        ans_new = '"' + ans + '"'
    return ans_new


def search_paths(topic_ent, ans, hop, type):
    wrapper_ans = get_complete_ans(ans, type)
    if hop == 1:
        query = ("""
                    PREFIX : <http://rdf.freebase.com/ns/> 
                    SELECT distinct (?x0 as ?r0) WHERE {
                        """ ':' + topic_ent + ' ?x0 ' + wrapper_ans + ' .' """
                        FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                    }
                """)
    elif hop == 2:
        query = ("""
                    PREFIX : <http://rdf.freebase.com/ns/>
                    SELECT distinct ?x0 as ?r0 ?x1 as ?r1 WHERE {
                        """ ':' + topic_ent + ' ?x0 ' + ' ?t0 ' + '.\n' + '?t0 ?x1 ' + wrapper_ans + ' .' """
                        FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                        FILTER regex(?x1, "http://rdf.freebase.com/ns/")
                    }
                """)
    elif hop == 3:
        query = ("""
                    PREFIX : <http://rdf.freebase.com/ns/>
                    SELECT distinct ?x0 as ?r0 ?x1 as ?r1 ?x2 as ?r2 WHERE {
                        """ ':' + topic_ent + ' ?x0 ' + ' ?t0 ' + '.\n' + ' ?t0 ?x1 ?t1' + '.\n' + '?t1 ?x2 ' + wrapper_ans + ' .'"""
                        FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                        FILTER regex(?x1, "http://rdf.freebase.com/ns/")
                        FILTER regex(?x2, "http://rdf.freebase.com/ns/")
                    }
                """)
    paths = []
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    for result in results['results']['bindings']:
        # path = [topic_ent]
        path = []
        for value in result.values():
            rel = value["value"].replace("http://rdf.freebase.com/ns/", "")
            path.append(rel)
        paths.append(path)
    return paths


def instaniate_and_execute(path):
    te = path[0]
    if len(path) == 2:
        query = ("""
                    PREFIX : <http://rdf.freebase.com/ns/> 
                    SELECT (?x1 AS ?value) WHERE {
                        SELECT DISTINCT ?x1  WHERE {
                            """ ':' + te + ' :' + path[1] + ' ?x1 . ' """
                        }
                    }
                """)
    elif len(path) == 3:
        query = ("""
                    PREFIX : <http://rdf.freebase.com/ns/> 
                    SELECT (?x1 AS ?value) WHERE {
                        SELECT DISTINCT ?x1  WHERE {
                            """ ':' + te + ' :' + path[1] + ' ?t1 .\n' + '?t1' + ' :' + path[2] + ' ?x1 . ' """
                        }
                    }
                """)
    elif len(path) == 4:
        query = ("""
                    PREFIX : <http://rdf.freebase.com/ns/> 
                    SELECT (?t3 AS ?value) WHERE {
                        SELECT DISTINCT ?x1  WHERE {
                            """ ':' + te + ' :' + path[1] + ' ?t1 .\n' + '?t1' + ' :' + path[
            2] + ' ?t2 .\n' + '?t2' + ' :' + path[3] + ' ?t3 . ' """
                        }
                    }
                """)
    answers = set()
    if "rdf-schema#domain" in query:
        return answers
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    for result in results['results']['bindings']:
        answers.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return list(answers)


# %%
def compute_precision(candidate_ans, ans):
    return (len(set(candidate_ans) & set(ans)) + 0.0) / len(set(candidate_ans))


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


def get_shortest_paths(tpe, answers, answers_type):
    all_paths_one_tpe = []
    shortest_path_len = 0
    for at, ans in answers_type.items():
        for a in ans:
            print("Search paths from %s to %s(%s)" % (tpe, a, at))
            paths = search_paths(tpe, a, 1, at)
            print("Find paths:", paths)
            if len(paths) != 0:
                shortest_path_len = min(shortest_path_len, 1)
                for p in paths:
                    if p not in all_paths_one_tpe:
                        all_paths_one_tpe.append(p)
                continue
            paths = search_paths(tpe, a, 2, at)
            print("Find paths:", paths)
            if len(paths) != 0:
                shortest_path_len = min(shortest_path_len, 2)
                for p in paths:
                    if p not in all_paths_one_tpe:
                        all_paths_one_tpe.append(p)
                continue
            print("Not found arrived paths")
    if len(all_paths_one_tpe) == 0:
        return None
    score_paths_one_tpe = []
    max_pre = 0.0
    for path in all_paths_one_tpe:
        # path = ["film.actor.film", "film.performance.character"]
        path_tmp = [tpe] + path
        # instantiate to SPARQL
        can_ans = instaniate_and_execute(path_tmp)
        if len(can_ans) == 0:
            print("Path:%s doesn't get any answers" % (path_tmp))
            continue
        # compute the precision of this paths
        precision = compute_precision(can_ans, answers)
        max_pre = max(max_pre, precision)
        # filter the paths with too slower precision
        if precision > 0.1:
            score_paths_one_tpe.append((path, precision))
        else:
            print("Path:%s get lower precision recall:%.3f" % (path_tmp, precision))

    shortest_path_len = min([len(p[0]) for p in score_paths_one_tpe])
    filtered_paths_one_tpe = [p[0] for p in score_paths_one_tpe if len(p[0]) == shortest_path_len]
    print("filtered paths:", filtered_paths_one_tpe)

    gold_relation_per_hop = defaultdict(set)
    for path in filtered_paths_one_tpe:
        for idx in range(shortest_path_len):
            rel = path[idx]
            gold_relation_per_hop[idx].add(rel)
    return gold_relation_per_hop


# %%
data = {"QuestionId": "WebQTrn-196", "RawQuestion": "who did ben stiller play in megamind?",
        "ProcessedQuestion": "who did ben stiller play in megamind", "Parses": [
        {"ParseId": "WebQTrn-196.P0", "AnnotatorId": 1,
         "AnnotatorComment": {"ParseQuality": "Complete", "QuestionQuality": "Good", "Confidence": "Normal",
                              "FreeFormComment": "First-round parse verification"},
         "Sparql": "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\nFILTER (?x != ns:m.0mdqp)\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))\nns:m.0mdqp ns:film.actor.film ?y .\n?y ns:film.performance.character ?x .\n?y ns:film.performance.film ns:m.05229c_ .\n}\n",
         "PotentialTopicEntityMention": "ben stiller", "TopicEntityName": "Ben Stiller", "TopicEntityMid": "m.0mdqp",
         "InferentialChain": ["film.actor.film", "film.performance.character"], "Constraints": [
            {"Operator": "Equal", "ArgumentType": "Entity", "Argument": "m.05229c_", "EntityName": "MegaMind",
             "SourceNodeIndex": 0, "NodePredicate": "film.performance.film", "ValueType": "String"}], "Time": None,
         "Order": None, "Answers": [{"AnswerType": "Entity", "AnswerArgument": "m.0fndhvs", "EntityName": "Bernard"}],
         "gold_sql_structure": {"0": ["film.actor.film"],
                                "1": ["film.performance.character", "film.performance.film"]}},
        {"ParseId": "WebQTrn-196.P1", "AnnotatorId": 1,
         "AnnotatorComment": {"ParseQuality": "Complete", "QuestionQuality": "Good", "Confidence": "Normal",
                              "FreeFormComment": "First-round parse verification"},
         "Sparql": "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\nFILTER (?x != ns:m.05229c_)\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))\nns:m.05229c_ ns:film.film.starring ?y .\n?y ns:film.performance.character ?x .\n?y ns:film.performance.actor ns:m.0mdqp .\n}\n",
         "PotentialTopicEntityMention": "megamind", "TopicEntityName": "MegaMind", "TopicEntityMid": "m.05229c_",
         "InferentialChain": ["film.film.starring", "film.performance.character"], "Constraints": [
            {"Operator": "Equal", "ArgumentType": "Entity", "Argument": "m.0mdqp", "EntityName": "Ben Stiller",
             "SourceNodeIndex": 0, "NodePredicate": "film.performance.actor", "ValueType": "String"}], "Time": None,
         "Order": None, "Answers": [{"AnswerType": "Entity", "AnswerArgument": "m.0fndhvs", "EntityName": "Bernard"}],
         "gold_sql_structure": {"0": ["film.film.starring"],
                                "1": ["film.performance.character", "film.performance.actor"]}}]}

qid = data["QuestionId"]

question = data["RawQuestion"]
if question.endswith("?"):
    question = question[0:-1]

parses = data["Parses"]

tes_all = set()
ans_all = set()
ans_type_dict = {}
ans_type_dict["uri_ans"] = set()
ans_type_dict["date_ans"] = set()
ans_type_dict["gyear_ans"] = set()
ans_type_dict["gyear_month_ans"] = set()
ans_type_dict["en_literal_ans"] = set()
ans_type_dict["digit_ans"] = set()
for parse in parses:
    pid = parse["ParseId"]
    if "#MANUAL SPARQL" in parse['Sparql'] and split != "test":
        print("Pid: %s has manual sparql" % (pid))
        continue
    # get the topic entities and answer entities
    tes = [parse["TopicEntityMid"]]  # only one topic entity mid in default
    tes = [t for t in tes if t is not None]
    tes_all.update(tes)
    if len(tes) == 0:
        print("Pid: %s does not have topic entity, we skip it." % (pid))
        continue

    ans = parse["Answers"]
    ans = [a["AnswerArgument"] for a in ans]  # get the mid of each answer
    ans_all.update(ans)
    if len(ans) == 0 and split != "test":
        print("Pid: %s does not have answer entity" % (pid))
        continue

    for a in ans:
        if a.startswith("m.") or a.startswith("g."):
            ans_type_dict["uri_ans"].add(a)
        elif len(a.split("-")) == 3 and all([is_number(e) for e in a.split("-")]):  # 1921-09-13
            ans_type_dict["date_ans"].add(a)
        elif len(a.split("-")) == 2 and all([is_number(e) for e in a.split("-")]):  # 1921-09
            ans_type_dict["gyear_month_ans"].add(a)
        elif a.isdigit and len(a) == 4:  # 1990
            ans_type_dict["gyear_ans"].add(a)
        elif is_number(a):  # 3.99
            ans_type_dict["digit_ans"].add(a)
        else:  # Elizabeth II
            ans_type_dict["en_literal_ans"].add(a)
for ent in tes_all:
    data_new = get_shortest_paths(ent, ans_all, ans_type_dict)
    if data_new is not None:
        print(data_new)

# %%


# %% [markdown]
# ### 从训练 relation retriever 的数据中抽样100个负例（原始为150个）

# %%
import pandas as pd
import random

# %%
path = "data/webqsp/data/SPLIT.rels.retri.train.data.csv"
out_path = "data/webqsp/data/SPLIT.rels.retri.train.data.neg_100.csv"

# %%
for split in ["train", "dev"]:
    ip = path.replace('SPLIT', split)
    op = out_path.replace('SPLIT', split)
    df = pd.read_csv(ip, header=0)
    df_list = df.values.tolist()
    new_data = []
    for d in df_list:
        nd = d[0:2]
        samples = random.sample(d[2:], 100)
        nd.extend(samples)
        new_data.append(nd)
    new_data = pd.DataFrame(data=new_data)
    new_data.to_csv(op, index=False, header=True)

# %%
print(len(new_data[0]))

# %%
import torch
from transformers import RobertaModel

# %%
# path = "./retriever/results/rel_retriever_0/"
path = "outputs/nsm_retriever_v2/retriever_nsm_1-h1.ckpt"

# %%
state_dict = torch.load(path, map_location=torch.device("cpu"))

# %%
for kv in state_dict['state_dict']:
    # key = kv[0]
    print(kv)

# %%
model = RobertaModel.from_pretrained(path)

# %%
model.state_dict

# %%



