# 将全图上的字符串节点统一映射 Int 来
import pickle
from tqdm import tqdm
import numpy as np
import sys
import pickle
import time
from deal_cvt import load_cvt, is_cvt

#
# dataset_name = 'WN18RR'
# output_dir = f'/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/{dataset_name}/subgraph/'
# path = f"/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/{dataset_name}/train2id.txt"
# rel_path = f"/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/{dataset_name}/relation2id.txt"
# ent_path = f"/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/{dataset_name}/entity2id.txt"
#
# out_path = f"/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/{dataset_name}/subgraph/subgraph_hop2.txt"
# out_rel_path = f"/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/{dataset_name}/subgraph/relations.txt"
# out_ent_path = f"/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/{dataset_name}/subgraph/entities.txt"
#
#
# with open(ent_path,"r") as f:
#     all_lines = f.readlines()
#     total_entities = int(all_lines[0].strip("\n").strip())
#     all_lines = [line.strip("\n").strip() for line in all_lines[1:]]
# ent2id = {ei.split("\t")[0]: int(ei.split("\t")[1]) for ei in all_lines}
# id2ent = {i: e for e, i in ent2id.items()}
# assert total_entities == len(ent2id) == len(id2ent)
#
# with open(rel_path,"r") as f:
#     all_lines = f.readlines()
#     total_relations = int(all_lines[0].strip("\n").strip())
#     all_lines = [line.strip("\n").strip() for line in all_lines[1:]]
# rel2id = {ri.split("\t")[0]: int(ri.split("\t")[1]) for ri in all_lines}
# ori_num_rel = len(rel2id)
#
# if dataset_name == "FB15K-237":
#     reverse_rel2id = {ri[0] + "/reverse": ri[1] + ori_num_rel for ri in rel2id.items()}
# elif dataset_name == "WN18RR":
#     reverse_rel2id = {ri[0] + "_reverse": ri[1] + ori_num_rel for ri in rel2id.items()}
#
# rel2id.update(reverse_rel2id)
# id2rel = {i: r for r, i in rel2id.items()}
# assert 2*total_relations == len(rel2id) == len(id2rel)
#
# with open(out_rel_path, 'w') as f:
#     for relation in rel2id.keys():
#         f.write(relation+'\n')
# with open(out_ent_path, 'w') as f:
#     for entity in ent2id.keys():
#         f.write(entity+'\n')
#
# with open(output_dir+'ent2id.pickle', 'wb') as f:
#     pickle.dump(ent2id, f)
# with open(output_dir+'rel2id.pickle', 'wb') as f:
#     pickle.dump(rel2id, f)
#
# triple_list = []
# with open(path, "r") as f:
#     all_lines = f.readlines()
#     all_lines = all_lines[1:]
#     for line in tqdm(all_lines, total=len(all_lines)):
#         spline = line.strip().strip('\n').split()
#         h, r, t = int(spline[0]), int(spline[2]), int(spline[1])
#         triple_list.append([h, r, t])
#         triple_list.append([t, r+ori_num_rel, h])
#
# sp_g = np.array(triple_list)
# print(sp_g.shape)
# print(sp_g.dtype, sys.getsizeof(sp_g))
# sp_g = sp_g.astype(np.int32)
# print(sp_g.dtype, sys.getsizeof(sp_g))
#
# f = output_dir+'subgraph_2hop_triples.npy'
# np.save(f, sp_g)
#
# # build ent type arry
# start = time.time()
# with open(output_dir+'ent2id.pickle', 'rb') as f:
#     ent2id = pickle.load(f)
# print("Load ent2id.pickle using %.2f s" % (time.time() - start))
#
# # 我们将实体切分为三种类型, "值","普通实体","CVT实体",
# # 我们将上述实体分别定为 1, 2, 3
# # 对于KGC任务，使用实体都视为2
# ent_type_ary = np.ones(len(ent2id))*2
#
# print(sys.getsizeof(ent_type_ary))
# ent_type_ary = ent_type_ary.astype(np.int8)
# print(sys.getsizeof(ent_type_ary))
# # [0: 0  1: 9353827 2: 15056141 3: 16725352]
# print(np.bincount(ent_type_ary))
#
# f = output_dir+"ent_type_ary.npy"
# np.save(f, ent_type_ary)

# output_dir = '/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/fb15k-237/minerva/subgraph/'
# path = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/fb15k-237/minerva/graph.txt"
# rel_path = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/fb15k-237/minerva/subgraph/relations.txt"
# ent_path = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/fb15k-237/minerva/subgraph/entities.txt"
# output_dir = '/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/fb15k-237/minerva/di_subgraph/'
# path = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/fb15k-237/minerva/digraph.txt"
# rel_path = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/fb15k-237/minerva/di_subgraph/relations.txt"
# ent_path = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/fb15k-237/minerva/di_subgraph/entities.txt"
output_dir = '/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/WN18RR/minerva/subgraph/'
path = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/WN18RR/minerva/graph.txt"
rel_path = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/WN18RR/minerva/subgraph/relations.txt"
ent_path = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/WN18RR/minerva/subgraph/entities.txt"
entity_set = set()
relation_set = set()
with open(path, "r") as f:
    for line in tqdm(f):
        spline = line.strip().strip('\n').split('\t')
        entity_set.add(spline[0])
        entity_set.add(spline[2])
        relation_set.add(spline[1])
relation_list = sorted(relation_set)
entity_list = sorted(entity_set)
with open(rel_path, 'w') as f:
    for relation in relation_list:
        f.write(relation+'\n')
with open(ent_path, 'w') as f:
    for entity in entity_set:
        f.write(relation+'\n')
relation2id = {relation: idx for idx, relation in enumerate(relation_list)}
ent2id = {ent: idx for idx, ent in enumerate(entity_set)}

with open(output_dir+'ent2id.pickle', 'wb') as f:
    pickle.dump(ent2id, f)
with open(output_dir+'rel2id.pickle', 'wb') as f:
    pickle.dump(relation2id, f)

triple_list = []
with open(path, "r") as f:
    for line in tqdm(f):
        spline = line.strip().strip('\n').split('\t')
        h, r, t = spline[0], spline[1], spline[2]
        h, r, t = ent2id[h], relation2id[r], ent2id[t]
        triple_list.append([h, r, t])

sp_g = np.array(triple_list)
print(sp_g.shape)
print(sp_g.dtype, sys.getsizeof(sp_g))
sp_g = sp_g.astype(np.int32)
print(sp_g.dtype, sys.getsizeof(sp_g))

f = output_dir+'subgraph_2hop_triples.npy'
np.save(f, sp_g)

# build ent type arry
start = time.time()
with open(output_dir+'ent2id.pickle', 'rb') as f:
    ent2id = pickle.load(f)
print("Load ent2id.pickle using %.2f s" % (time.time() - start))

# 我们将实体切分为三种类型, "值","普通实体","CVT实体",
# 我们将上述实体分别定为 1, 2, 3
# 对于KGC任务，使用实体都视为2
ent_type_ary = np.ones(len(ent2id))*2

print(sys.getsizeof(ent_type_ary))
ent_type_ary = ent_type_ary.astype(np.int8)
print(sys.getsizeof(ent_type_ary))
# [0: 0  1: 9353827 2: 15056141 3: 16725352]
print(np.bincount(ent_type_ary))

f = output_dir+"ent_type_ary.npy"
np.save(f, ent_type_ary)