#!/bin/bash

task_name="webqsp"
max_num_processes=40
num_pro_each_device=1
model_name=retriever_p_t
max_hop=3
abs_subgraph_path=./data/webqsp/all_abs_sg-tk_15-fs_0.0-mt_1000-mh_3hop-nsm.jsonl
input_data_cache_abs_path=/mnt/jiangjinhao/UniKBQA/data/webqsp/abs_sg_for_retri.cache
topk=15
max_num_triples=300
max_nodes=700

python3 s3_retrieve_subgraph_by_nsm.py --max_num_processes ${max_num_processes} --task_name ${task_name} \
--sparse_kg_source_path ./data/${task_name}/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/${task_name}/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/${task_name}/graph_data/ent2id.pickle --sparse_rel2id_path ./data/${task_name}/graph_data/rel2id.pickle \
--ori_path data/${task_name}/all_data.jsonl \
--input_path ${abs_subgraph_path} \
--input_data_cache_abs_path ${input_data_cache_abs_path} \
--output_path data/${task_name}/all_ins_sg-tk_${topk}-mt_${max_num_triples}.jsonl \
--model_path ./ckpt/${task_name}_nsm/${model_name}-h1.ckpt \
--arg_path ./ckpt/${task_name}_nsm/${model_name}-args.json \
--relation2id_path data/${task_name}/abs_sub_relations.txt \
--entity2id_path data/${task_name}/abs_sub_entities.txt \
--device 7 8 9 --final_topk ${topk} --max_deduced_triples ${max_num_triples} --max_hop ${max_hop} --num_pro_each_device ${num_pro_each_device}

# Note: you should manually compute the "--num_process"
python extract_instantiate_subgraph.py --split_qid_path ./data/${task_name}/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/${task_name}/all_ins_sg-tk_${topk}-mt_${max_num_triples}.jsonl \
--output_path ./data/${task_name}/SPLIT_ins_sg-tk_${topk}-mt_${max_num_triples}.jsonl --num_process 3

python3 map_kg_2_global_id.py --input_path ./data/${task_name}/SPLIT_ins_sg-tk_${topk}-mt_${max_num_triples}.jsonl \
--ori_path data/${task_name}/SPLIT.jsonl --output_path data/${task_name}/ --output_prefix inst_rea_ \
--split_list train dev test

python3 convert_inst_to_NSM_format.py --original_path data/${task_name}/SPLIT.jsonl --max_nodes ${max_nodes} \
--input_path ./data/${task_name}/SPLIT_ins_sg-tk_${topk}-mt_${max_num_triples}.jsonl \
--output_path ./data/${task_name}/SPLIT_ins_sg-tk_${topk}-mt_${max_nodes}-cr.jsonl \
--kg_map_path ./data/${task_name}/ --kg_map_prefix inst_rea_ --split_list train dev test --add_constraint_reverse