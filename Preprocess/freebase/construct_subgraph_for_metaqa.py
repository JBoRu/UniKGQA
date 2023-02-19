# 将全图上的字符串节点统一映射 Int 来
import pickle
from tqdm import tqdm
import numpy as np
import sys
import pickle
import time

reverse_rels_mapping = {'directed_by': 'direct', 'has_genre': 'genre', 'has_imdb_rating': 'imdb_rating',
                        'has_imdb_votes': 'imdb_votes', 'has_tags': 'tags', 'in_language': 'language',
                        'release_year': 'released_year', 'starred_actors': 'star', 'written_by': 'write'}
# reverse_rels_mapping = {'directed_by': 'direct', 'starred_actors': 'star', 'written_by': 'write'}

output_dir = '/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/metaqa/subgraph/'
path = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/metaqa/kb.txt"
global_ent_map = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/metaqa/kb_entity_dict.txt"
rel_path = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/metaqa/subgraph/relations.txt"
ent_path = "/mnt/jiangjinhao/PLM4KBQA/data/Freebase/NSM_related/metaqa/subgraph/entities.txt"
entity_set = set()
relation_set = set()
with open(path, "r") as f:
    for line in tqdm(f):
        spline = line.strip().strip('\n').split('|')
        entity_set.add(spline[0])
        entity_set.add(spline[2])
        relation_set.add(spline[1])
        if spline[1] in reverse_rels_mapping:
            relation_set.add(reverse_rels_mapping[spline[1]])
relation_list = sorted(relation_set)
entity_list = sorted(entity_set)
with open(rel_path, 'w') as f:
    for relation in relation_list:
        f.write(relation+'\n')
with open(ent_path, 'w') as f:
    for entity in entity_set:
        f.write(relation+'\n')
relation2id = {relation: idx for idx, relation in enumerate(relation_list)}
# ent2id = {ent: idx for idx, ent in enumerate(entity_set)}
ent2id = {}
with open(global_ent_map, "r") as f:
    for line in f.readlines():
        eid, etext = line.strip("\n").split("\t")
        eid = eid.strip()
        etext = etext.strip()
        ent2id[etext] = int(eid)

with open(output_dir+'ent2id.pickle', 'wb') as f:
    pickle.dump(ent2id, f)
with open(output_dir+'rel2id.pickle', 'wb') as f:
    pickle.dump(relation2id, f)

triple_list = []
with open(path, "r") as f:
    for line in tqdm(f):
        spline = line.strip().strip('\n').split('|')
        h, r, t = spline[0], spline[1], spline[2]
        h, r, t = ent2id[h], relation2id[r], ent2id[t]
        triple_list.append([h, r, t])
        if spline[1] in reverse_rels_mapping:
            triple_list.append([t, relation2id[reverse_rels_mapping[spline[1]]], h])

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