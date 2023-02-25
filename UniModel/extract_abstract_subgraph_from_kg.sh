#!/bin/bash

#task_name="webqsp"
#hop_of_path=2
#max_hop=3
#max_num_processes=40
#num_pro_each_device=4
#min_path_precision=0.1
#filter_score=0.0
#filter_order=-1
#topk=15
#max_num_triples=1000
#max_nodes=1000

task_name="cwq"
hop_of_path=4
max_hop=4
max_num_processes=40
num_pro_each_device=4
min_path_precision=0.05
filter_score=0.0
filter_order=-1
topk=10
max_num_triples=2000
max_nodes=2000

# Note: you should manually specify the "--device" number!!!
python3 s2_extract_abstract_subgraph.py --max_num_processes ${max_num_processes} --task_name ${task_name} \
--sparse_kg_source_path ./data/${task_name}/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/${task_name}/graph_data/ent2id.pickle --sparse_rel2id_path ./data/${task_name}/graph_data/rel2id.pickle \
--ori_path ./data/${task_name}/all_data.jsonl --input_path ./data/${task_name}/all_paths-mh_${hop_of_path}hop.jsonl \
--output_path ./data/${task_name}/all_abs_sg-tk_${topk}-fs_${filter_score}-mt_${max_num_triples}-mh_${max_hop}hop.jsonl \
--model_path ./simcse_retriever/results/${task_name}_rel_retri/ --device 7 8 9 --topk ${topk} --max_num_triples ${max_num_triples} \
--filter_score ${filter_score} --max_hop ${max_hop} --num_pro_each_device ${num_pro_each_device}

# Note: you should manually specify the "--num_process" number, which is "len(--device)*num_pro_each_device"!!!
python3 extract_abstract_subgraph.py --split_qid_path ./data/${task_name}/SPLIT.qid.npy --split_list train dev test \
--abs_sg_path ./data/${task_name}/all_abs_sg-tk_${topk}-fs_${filter_score}-mt_${max_num_triples}-mh_${max_hop}hop.jsonl \
--output_path ./data/${task_name}/SPLIT_abs_sg-tk_${topk}-fs_${filter_score}-mt_${max_num_triples}-mh_${max_hop}hop.jsonl \
--all_output_path ./data/${task_name}/all_abs_sg-tk_${topk}-fs_${filter_score}-mt_${max_num_triples}-mh_${max_hop}hop.jsonl \
--all_shortest_path ./data/${task_name}/all_paths-mh_${hop_of_path}hop.jsonl \
--ori_path ./data/${task_name}/all_data.jsonl \
--min_path_precision ${min_path_precision} --filter_order ${filter_order} --num_process 12

python3 map_abs_kg_2_global_id.py --input_path data/${task_name}/SPLIT_abs_sg-tk_${topk}-fs_${filter_score}-mt_${max_num_triples}-mh_${max_hop}hop.jsonl \
--output_path data/${task_name}/ --output_prefix abs_sub_ --split_list train dev test

python3 convert_abs_to_NSM_format.py \
--input_path data/${task_name}/SPLIT_abs_sg-tk_${topk}-fs_${filter_score}-mt_${max_num_triples}-mh_${max_hop}hop.jsonl \
--output_path data/${task_name}/SPLIT_abs_sg-tk_${topk}-fs_${filter_score}-mt_${max_num_triples}-mh_${max_hop}hop-nsm.jsonl \
--all_output_path data/${task_name}/all_abs_sg-tk_${topk}-fs_${filter_score}-mt_${max_num_triples}-mh_${max_hop}hop-nsm.jsonl \
--kg_map_path data/${task_name}/ --kg_map_prefix abs_sub_ --max_nodes ${max_nodes} --split_list train dev test
