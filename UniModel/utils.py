import pickle
import json
import os
import shutil


def load_from_json(fname):
    with open(fname) as f:
        return json.load(f)


def dump_to_json(obj, fname, indent=None):
    with open(fname, 'w') as f:
        return json.dump(obj, f, indent=indent)


def compute_precision(candidate_ans, ans):
    return len(set(candidate_ans) & set(ans)) / len(set(candidate_ans))


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


def ordered_set_as_list(xs):
    ys = []
    for x in xs:
        if x not in ys:
            ys.append(x)
    return ys


def dump_to_bin(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_bin(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def load_json(fname):
    with open(fname) as f:
        return json.load(f)


def dump_json(obj, fname, indent=None):
    with open(fname, 'w') as f:
        return json.dump(obj, f, indent=indent)


def mkdir_f(prefix):
    if os.path.exists(prefix):
        shutil.rmtree(prefix)
    os.makedirs(prefix)


def mkdir_p(prefix):
    if not os.path.exists(prefix):
        os.makedirs(prefix)


def load_dict(filename):
    ele2id = {}
    with open(filename, "r") as f:
        all_lines = f.readlines()
        for l in all_lines:
            ele = l.strip("\n")
            if ele in ele2id:
                print("Already in dict:", l, ele)
                continue
            ele2id[ele] = len(ele2id)
    return ele2id
