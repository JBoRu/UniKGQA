#!/bin/bash

#task_name="webqsp"
#max_hop=2
#max_num_processes=60
#min_precision=0.1
#filter_order=-1
#num_neg=50

task_name="cwq"
max_hop=4
max_num_processes=50
min_precision=0.05
filter_order=-1
num_neg=50

python3 s0_extract_weak_super_relations.py --extra_hop_flag --task_name ${task_name} \
--sparse_kg_source_path ./data/${task_name}/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/${task_name}/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/${task_name}/graph_data/ent2id.pickle --sparse_rel2id_path ./data/${task_name}/graph_data/rel2id.pickle \
--input_path ./data/${task_name}/all_data.jsonl --output_path ./data/${task_name}/all_paths-mh_${max_hop}hop.jsonl \
--ids_path ./data/${task_name}/SPLIT.qid.npy --max_hop ${max_hop} --max_num_processes ${max_num_processes}

python extract_valid_weak_paths.py --min_precision ${min_precision} --filter_order ${filter_order} \
--input_path ./data/${task_name}/all_paths-mh_${max_hop}hop.jsonl --output_path ./data/${task_name}/all_paths-mh_${max_hop}hop-mp_${min_precision}-fo_${filter_order}.jsonl

python s1_construct_relation_retrieval_training_data.py --task_name ${task_name} \
--sparse_kg_source_path ./data/${task_name}/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/${task_name}/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/${task_name}/graph_data/ent2id.pickle --sparse_rel2id_path ./data/${task_name}/graph_data/rel2id.pickle \
--input_path ./data/${task_name}/all_paths-mh_${max_hop}hop-mp_${min_precision}-fo_${filter_order}.jsonl \
--output_path ./data/${task_name}/all_rels-mh_${max_hop}hop-neg_${num_neg}.jsonl \
--max_hop $max_hop --num_neg ${num_neg} --max_num_processes ${max_num_processes}

python extract_relation_retriever_training_data.py --split_qid_path ./data/${task_name}/SPLIT.qid.npy --split_list train dev \
--pn_pairs_path ./data/${task_name}/all_rels-mh_${max_hop}hop-neg_${num_neg}.jsonl --output_path ./data/${task_name}/SPLIT_rels-mh_${max_hop}hop-neg_${num_neg}.csv