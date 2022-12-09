# 将全图上的字符串节点统一映射 Int 来
import pickle
from tqdm import tqdm
import numpy as np
import sys

output_dir = './webqsp/subgraph/'
path = "./webqsp/subgraph/subgraph_hop2.txt"
rel_path = "./webqsp/subgraph/relations.txt"
ent_path = "./webqsp/subgraph/entities.txt"
entity_set = set()
relation_set = set()
with open(path, "r") as f:
    for line in tqdm(f, total=150000000):
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
    for line in tqdm(f, total=150000000):
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
