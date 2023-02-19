import argparse
import multiprocessing
import random
import sys

sys.path.append("..")
from KnowledgeBase.KG_api import KnowledgeGraph
from KnowledgeBase.sparql_executor import *


def get_neg_rels_neibouring(topic_entity, pos_rel_set, max_hop, num_neg):
    rels_within_spec_hop = set()
    hop_rels = {i: set() for i in range(max_hop)}
    ent = topic_entity
    try:
        neigbouring_rels_per_hop = kg.get_relations_within_specified_hop(ent, max_hop)
    except Exception as e:
        print("Exception during get relation with specified hop: \n", e)
        neigbouring_rels_per_hop = defaultdict(set)

    for hop_id, neigbouring_rels in neigbouring_rels_per_hop.items():
        rels_within_spec_hop.update(neigbouring_rels)
        hop_rels[hop_id].update(neigbouring_rels)

    hop_rels_filtered = {}
    rels_within_spec_hop_filtered = list(rels_within_spec_hop - pos_rel_set)
    if len(rels_within_spec_hop_filtered) == 0:
        return hop_rels_filtered

    for k, v in hop_rels.items():
        filtered_v = list(v - pos_rel_set)
        if len(filtered_v) < num_neg:
            if len(rels_within_spec_hop_filtered) < num_neg - len(filtered_v):
                extra_rels = random.choices(rels_within_spec_hop_filtered, k=(num_neg - len(filtered_v)))
            else:
                extra_rels = random.sample(rels_within_spec_hop_filtered, k=(num_neg - len(filtered_v)))
            filtered_v.extend(extra_rels)
        hop_rels_filtered[k] = filtered_v

    return hop_rels_filtered


def construct_contrastive_pos_neg_paths(sample):
    qid = sample["ID"]
    tpe = sample["TopicEntityMid"]
    if tpe is None:
        print("Qid: %s does not have any topic entity!"%(qid))
        return None
    paths = sample["Paths"]
    if len(paths) == 0:
        print("Qid: %s does not have any path!"%(qid))
        return None

    const_relations = sample["ConstRelations"]
    const_relations = [r.replace("ns:", "") for r in const_relations]

    filtered_paths = paths

    pos_rels_set = set()
    max_hop_of_ret_path = 0
    for p in filtered_paths:
        p_tmp = p[1]
        max_hop_of_ret_path = max(max_hop_of_ret_path, len(p_tmp))
        pos_rels_set.update(p_tmp)
    pos_rels_set.update(const_relations)

    if task_name == 'webqsp':
        max_hop_of_ret_path += 1  # used for constrained relation
    elif task_name == 'cwq':
        max_hop_of_ret_path += 2

    if max_hop_of_ret_path > max_hop:
        # print("Qid:%s has one path more than %d hop!"%(qid, max_hop))
        rea_max_hop = max_hop
    else:
        rea_max_hop = max_hop_of_ret_path

    one_training_sample = []
    neg_rels_neibouring = get_neg_rels_neibouring(tpe, pos_rels_set, rea_max_hop, num_neg)
    if len(neg_rels_neibouring) == 0:
        return None

    all_neg_rels = set()
    for k, v in neg_rels_neibouring.items():
        all_neg_rels.update(v)

    all_neg_rels = list(all_neg_rels)
    for ps in filtered_paths:
        cur_ques = ps[0]
        path = ps[1]
        if len(path) == 0:
            continue
        for idx, rel in enumerate(path):
            one_input_list = []
            query = cur_ques
            pos = rel
            neg = random.sample(neg_rels_neibouring[idx], k=num_neg)
            one_input_list.extend([query, pos])
            one_input_list.extend(neg)
            assert len(one_input_list) == 2 + num_neg
            one_training_sample.append(one_input_list)

    for cs in const_relations:
        # add the last end_of_hop
        one_input_list = []
        query = filtered_paths[0][0]
        pos = cs
        if len(all_neg_rels) >= num_neg:
            neg = random.sample(all_neg_rels, k=num_neg)
        else:
            neg = random.choices(all_neg_rels, k=num_neg)
        one_input_list.extend([query, pos])
        one_input_list.extend(neg)
        assert len(one_input_list) == 2 + num_neg
        one_training_sample.append(one_input_list)

    sample = {"ID": qid, "PosNegPairs": one_training_sample}
    return sample


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', required=True, type=str)
    parser.add_argument('--input_path', required=True,
                        help='the input data path used for extracting the shortest paths')
    parser.add_argument('--output_path', required=True,
                        help='the output data path used for extracting the shortest paths')
    parser.add_argument('--split_list', nargs="+")
    parser.add_argument('--max_hop', default=2, type=int, help='the max search hop of the shortest paths')
    parser.add_argument('--max_num_processes', default=1, type=int)
    parser.add_argument('--num_neg', default=15, type=int, help='the number of negative relationns')
    parser.add_argument('--sparse_kg_source_path', default=None, help='the sparse triples file')
    parser.add_argument('--sparse_ent_type_path', default=None, help='the file of entities type of sparse triples')
    parser.add_argument('--sparse_ent2id_path', default=None, help='the sparse ent2id file')
    parser.add_argument('--sparse_rel2id_path', default=None, help='the sparse rel2id file')
    args = parser.parse_args()

    print("Start constructing contrastive training data.")
    return args


if __name__ == '__main__':
    args = _parse_args()
    task_name = args.task_name

    kg = KnowledgeGraph(args.sparse_kg_source_path, args.sparse_ent_type_path, args.sparse_ent2id_path, args.sparse_rel2id_path)

    num_process = args.max_num_processes
    chunk_size = 1
    max_hop = args.max_hop
    num_neg = args.num_neg

    inp_path = args.input_path
    out_path = args.output_path
    print('Input %s to Output %s' % (inp_path, out_path))

    with open(inp_path, "r") as f:
        all_data = f.readlines()
        all_data = [json.loads(l) for l in all_data]

    with multiprocessing.Pool(num_process) as p:
        new_samples = p.imap_unordered(construct_contrastive_pos_neg_paths, all_data, chunksize=chunk_size)
        count = 0
        with open(out_path, 'w') as fout:
            for sample in tqdm(new_samples, total=len(all_data)):
                if sample is not None:
                    count += len(sample["PosNegPairs"])
                    fout.write(json.dumps(sample)+"\n")
            print("Total %d qpn / %d samples" % (count, len(all_data)))