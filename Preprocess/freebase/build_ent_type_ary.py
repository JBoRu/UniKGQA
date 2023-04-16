import pickle
import time
from deal_cvt import load_cvt, is_cvt
import numpy as np
from tqdm import tqdm
import sys

output_dir = sys.argv[1] # '/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/webqsp/subgraph_test/'
# output_dir = '/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/cwq/subgraph/'
start = time.time()
with open(output_dir+'ent2id.pickle', 'rb') as f:
    ent2id = pickle.load(f)
print("Load ent2id.pickle using %.2f s" % (time.time() - start))

cvt_map = load_cvt()


def is_ent(tp_str):
    if len(tp_str) < 3:
        return False
    if tp_str.startswith("m.") or tp_str.startswith("g."):
        # print(tp_str)
        return True
    return False


# 我们将实体切分为三种类型, "值", "普通实体", "CVT实体",
# 我们将上述实体分别定为 1, 2, 3
ent_type_ary = np.zeros(len(ent2id))


def judge_type(key):
    if is_ent(key):
        if is_cvt(key, cvt_map):
            return 3
        else:
            return 2
    else:
        return 1


for ent_key, ent_id in tqdm(ent2id.items(), total=len(ent2id)):
    ent_type = judge_type(ent_key)
    ent_type_ary[ent_id] = ent_type

print(sys.getsizeof(ent_type_ary))
ent_type_ary = ent_type_ary.astype(np.int8)
print(sys.getsizeof(ent_type_ary))
# [0: 0  1: 9353827 2: 15056141 3: 16725352]
print(np.bincount(ent_type_ary))

f = output_dir+"ent_type_ary.npy"
np.save(f, ent_type_ary)
