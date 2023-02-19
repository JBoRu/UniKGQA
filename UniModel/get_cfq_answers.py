import numpy as np
import json
from collections import defaultdict
import time
import os
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib
sparql = SPARQLWrapper("http://localhost:3001/sparql")
sparql.setReturnFormat(JSON)
from tqdm import tqdm
import multiprocessing

def run_sparql(query):
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
        return results
    except Exception as e:
        print(e)
        print(query)
        return None

simplify_all_data_path = "./data/cfq/cfq_origin/simplify_dataset.jsonl"
out_path = "./data/cfq/cfq_origin/simplify_dataset.jsonl"

with open(simplify_all_data_path, "r") as f:
    all_data = f.readlines()
    all_samples = [json.loads(l) for l in all_data]
print("Load %d samples from %s" % (len(all_samples), simplify_all_data_path))

def process_data(idx, data, out_path):
    print("Start PID %d for processing %d-%d" % (os.getpid(), idx * split_index, (idx + 1) * split_index))
    path = out_path+"_"+str(idx)
    with open(path, "w") as f:
        for sample in tqdm(data, total=len(data), desc="PID: %d"%(os.getpid())):
            qid = sample['ID']
            query = "PREFIX ns: <http://rdf.freebase.com/ns/>\n" + sample['sparql']
            query = query.replace("count(*)", "Distinct ?x0")
            answers = run_sparql(query)
            if answers is None:
                print("Qid:[%d] execute sql error!"%(qid))
            sample["answers"] = answers
            try:
                f.write(json.dumps(sample)+'\n')
            except Exception as e:
                print(e)
                continue

num_process = 40
split_index = len(all_samples) // num_process + 1

p = multiprocessing.Pool(num_process)
result_for_process = []
for idx in range(num_process):
    select_data = all_samples[idx*split_index: (idx+1)*split_index]
    p.apply_async(process_data, args=(idx, select_data, out_path))
p.close()
p.join()
print("All child process over!")