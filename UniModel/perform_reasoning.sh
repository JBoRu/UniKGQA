#!/bin/bash

task_name="webqsp"
batch_size=40
gradient_accumulation_steps=1
test_batch_size=400
num_step=3
num_epoch=100
patience=15
entity_dim=768
word_dim=300
kg_dim=384
kge_dim=100
device=6
plm_ckpt_path=./simcse_retriever/results/webqsp_rel_retri/
nsm_ckpt_path=./ckpt/webqsp_nsm/retriever_p_t-f1.ckpt
data_suffix=_ins_sg-tk_15-mt_700-cr.jsonl
experiment_name=reason_p_t_f1 # Note: Whether update pre-trained params or transfer the retriever model params.

CUDA_VISIBLE_DEVICES=${device} python3 ./nsm/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ${nsm_ckpt_path} --model_path ${plm_ckpt_path} --relation_model_path ${plm_ckpt_path} \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/${task_name}/ --data_name ${data_suffix} \
--checkpoint_dir ./ckpt/${task_name}/ --experiment_name ${experiment_name} \
--batch_size ${batch_size} --gradient_accumulation_steps ${gradient_accumulation_steps} --test_batch_size ${test_batch_size} \
--num_step ${num_step} --entity_dim ${entity_dim} --word_dim ${word_dim} --kg_dim ${kg_dim} --kge_dim ${kge_dim} \
--eval_every 1 --encode_type --eps 0.95 --num_epoch ${num_epoch} --patience ${patience} --use_self_loop \
--plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id inst_rea_relations.txt --entity2id inst_rea_entities.txt --simplify_model \
--fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm