import pandas as pd
from tqdm import tqdm
import random

inp_path = "./data/simplequestions/SimpleQuestions_v2/annotated_fb_data_SPLIT.txt"
out_path = "./data/simplequestions/SimpleQuestions_v2/qr_pair.csv"
train_path = "./data/simplequestions/train_qr_pair.csv"
dev_path = "./data/simplequestions/dev_qr_pair.csv"

all_lines = []
for split in ["train", 'valid', 'test']:
    path = inp_path.replace('SPLIT', split)
    with open(path, "r") as f:
        lines = f.readlines()
        all_lines.extend(lines)
print("Load total %d pairs" % (len(all_lines)))

all_pairs = []
for line in tqdm(all_lines, total=len(all_lines)):
    h, r, t, q = line.strip("\n").strip().split("\t")
    # assert r.startswith("www.freebase.com/") or r.startswith("<http://rdf.freebase.com/ns/"), r
    if r.startswith("www.freebase.com/"):
        r = r.replace('www.freebase.com/', '')
    elif r.startswith("<http://rdf.freebase.com/ns/"):
        r = r.replace('<http://rdf.freebase.com/ns/','').replace('>','')
    else:
        print("Not match: ",r)
    r = r.replace('/','.')
    assert '/' not in r, r
    if q.endswith("?"):
        q = q[:-1]
    all_pairs.append([q, r])
print("Filter from %d to %d"%(len(all_lines), len(all_pairs)))
print("Sample: ", all_pairs[-1])

random.shuffle(all_pairs)
count = len(all_pairs)
dev_count = 1000
dev_pairs = all_pairs[:dev_count]
train_pairs = all_pairs[dev_count:]

train_set = pd.DataFrame(data=train_pairs)
train_set.to_csv(train_path, index=False, header=True)
print("Extract %d train PN pairs and dump to %s." % (len(train_set), train_path))
dev_set = pd.DataFrame(data=dev_pairs)
dev_set.to_csv(dev_path, index=False, header=True)
print("Extract %d dev PN pairs and dump to %s." % (len(dev_set), dev_path))

