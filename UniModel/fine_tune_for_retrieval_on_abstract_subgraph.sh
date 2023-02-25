#!/bin/bash

task_name="webqsp"
batch_size=40
gradient_accumulation_steps=1
test_batch_size=400
max_hop=3
num_step=3
num_epoch=100
patience=15
filter_score=0.0
entity_dim=768
word_dim=300
kg_dim=384
kge_dim=100
topk=15
max_num_triples=1000
linear_dropout=0.1
device=6

CUDA_VISIBLE_DEVICES=${device} python3 ./nsm/main_nsm.py --linear_dropout ${linear_dropout} --log_steps 50 \
--model_path ./simcse_retriever/results/${task_name}_rel_retri --agent_type PLM --instruct_model PLM \
--model_name gnn --data_folder ./data/${task_name}/ \
--data_name _abs_sg-tk_${topk}-fs_${filter_score}-mt_${max_num_triples}-mh_${max_hop}hop-nsm.jsonl \
--checkpoint_dir ./ckpt/${task_name}_nsm/ --experiment_name retriever_p_t \
--batch_size ${batch_size} --gradient_accumulation_steps ${gradient_accumulation_steps} --test_batch_size ${test_batch_size} \
--num_step ${num_step} --entity_dim ${entity_dim} --word_dim ${word_dim} --kg_dim ${kg_dim} --kge_dim ${kge_dim} \
--eval_every 1 --encode_type --eps 0.95 --num_epoch ${num_epoch} --patience ${patience} --use_self_loop \
--plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/${task_name}/abs_sg_for_retri.cache --relation2id abs_sub_relations.txt --entity2id abs_sub_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm --overwrite_cache
