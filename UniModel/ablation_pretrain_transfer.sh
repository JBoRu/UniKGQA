## pretrain+transfer best 71.40/76.48  71.76/75.44
#CUDA_VISIBLE_DEVICES=1 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_0-f1.ckpt \
#--model_path ./retriever/results/webqsp_rel_retri_1/ --agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 100 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name webqsp_nsm_reason_4_reproduce --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt
#
## retrieval: pretrain, transfer, fix plm 不用pooler层
## webqsp: 71.89/83.40 72.66/84.26
#CUDA_VISIBLE_DEVICES=2 python3 main_nsm.py --linear_dropout 0.1 --log_steps 50 \
#--model_path ../retriever/results/new_webqsp_rel_retri_1/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
#--data_name .abs.sg.topk.15.min_sim.0.max_tris.1000.new_rel_retri_1.nsm.json \
#--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_p_t_wo_pooler \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 400 --num_step 3 \
#--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
# --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
#--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/new_abs_r1_sg_for_retri.cache \
#--relation2id new_abs_r1_relations.txt --entity2id new_abs_r1_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## reasoning: pretrain, transfer, fix plm, wo pooler f1 70.69/75.87 71.93/76.05
#CUDA_VISIBLE_DEVICES=2 python3 ./reader/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_wo_pooler-f1.ckpt \
#--model_path ./retriever/results/webqsp_rel_retri_1/ \
#--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_wo_pooler \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## reasoning: pretrain, transfer, fix plm, wo pooler h1 72.09/75.08 71.57/76.11
#CUDA_VISIBLE_DEVICES=2 python3 ./reader/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_wo_pooler-h1.ckpt \
#--model_path ./retriever/results/webqsp_rel_retri_1/ \
#--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_wo_pooler_h1 \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## retrieval: pretrain, transfer, fix plm
## cwq: 54.77/67.11 55.86/68.03
#CUDA_VISIBLE_DEVICES=3 python3 main_nsm.py --linear_dropout 0.1 --log_steps 50 \
#--model_path ../retriever/results/cwq_rel_retri_0/ --agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ../data/cwq/ --data_name .abstract.sg.max_tris.1500.nsm.json \
#--checkpoint_dir ../retriever_ckpt/cwq_nsm/ --experiment_name retriever_p_t_wo_pooler \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 500 --num_step 3 \
#--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
# --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
#--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/cwq/abs_r0_sg_for_retri.cache \
#--relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## reasoning: pretrain, transfer, fix plm, wo pooler f1 47.43/49.79 48.89/51.30
#CUDA_VISIBLE_DEVICES=2 python3 ./reader/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_p_t_wo_pooler-f1.ckpt \
#--model_path ./retriever/results/cwq_rel_retri_0/ \
#--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_wo_pooler \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## reasoning: pretrain, transfer, fix plm, wo pooler h1 46.47/50.71 48.61/49.97
#CUDA_VISIBLE_DEVICES=2 python3 ./reader/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_p_t_wo_pooler-h1.ckpt \
#--model_path ./retriever/results/cwq_rel_retri_0/ \
#--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_wo_pooler_h1 \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
#
#
## retrieval: pretrain, transfer, question updating 不用pooler 69.18/83.10 71.91/83.83
## webqsp
#CUDA_VISIBLE_DEVICES=2 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--model_path ../retriever/results/webqsp_rel_retri_1/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
#--relation_model_path ../retriever/results/webqsp_rel_retri_1/ \
#--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
#--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_p_t_q_wo_pooler \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 400 --num_step 3 \
#--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
# --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
#--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
#--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate
## reasoning: pretrain, transfer, question updating, wo pooler f1 69.03/74.40 71.07/74.71
#CUDA_VISIBLE_DEVICES=2 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_wo_pooler-f1.ckpt \
#--model_path ./retriever/results/webqsp_rel_retri_1/ \
#--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_wo_pooler \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate
## reasoning: pretrain, transfer, question updating, wo pooler h1 71.98/76.36 71.93/76.79
#CUDA_VISIBLE_DEVICES=1 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_wo_pooler-h1.ckpt \
#--model_path ./retriever/results/webqsp_rel_retri_1/ \
#--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_wo_pooler_h1 \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate
## retrieval: pretrain, transfer, question updating 不用pooler
## cwq 56.04/67.02 55.80/67.28
#CUDA_VISIBLE_DEVICES=5 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--model_path ../retriever/results/cwq_rel_retri_0/ --agent_type PLM --instruct_model PLM --model_name gnn \
#--relation_model_path ../retriever/results/cwq_rel_retri_0/ \
#--data_folder ../data/cwq/ --data_name .abstract.sg.max_tris.1500.nsm.json \
#--checkpoint_dir ../retriever_ckpt/cwq_nsm/ --experiment_name retriever_p_t_q_wo_pooler \
#--batch_size 5 --gradient_accumulation_steps 8 --test_batch_size 500 --num_step 3 \
#--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
# --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
#--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/cwq/abs_r0_sg_for_retri.cache \
#--relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate
## reasoning: pretrain, transfer, question updating, wo pooler f1 47.79/50.06 49.47/50.27
#CUDA_VISIBLE_DEVICES=3 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_p_t_q_wo_pooler-f1.ckpt \
#--model_path ./retriever/results/cwq_rel_retri_0/ \
#--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_q_wo_pooler \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate
## reasoning: pretrain, transfer, question updating, wo pooler h1 47.79/50.92 49.75/50.71
#CUDA_VISIBLE_DEVICES=3 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_p_t_q_wo_pooler-h1.ckpt \
#--model_path ./retriever/results/cwq_rel_retri_0/ \
#--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_q_wo_pooler_h1 \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate
#
## retrieval: pretrain, transfer, encoding relation but not updating 不用pooler层 71.68/82.73 71.47/82.73
## webqsp
#CUDA_VISIBLE_DEVICES=1 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--model_path ../retriever/results/webqsp_rel_retri_1/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
#--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
#--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler \
#--batch_size 10 --gradient_accumulation_steps 4 --test_batch_size 400 --num_step 3 \
#--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
# --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
#--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
#--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding
## reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 70.40/76.97 71.25/75.81
#CUDA_VISIBLE_DEVICES=4 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_er_wo_pooler-f1.ckpt \
#--model_path ./retriever/results/webqsp_rel_retri_1/ \
#--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_er_wo_pooler \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding
## reasoning: pretrain, transfer, encoding relation but not updating, wo pooler h1 71.05/77.03 71.37/77.15
#CUDA_VISIBLE_DEVICES=4 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_er_wo_pooler-h1.ckpt \
#--model_path ./retriever/results/webqsp_rel_retri_1/ \
#--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_er_wo_pooler_h1 \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding
## 使用roberta-large 7-1
#CUDA_VISIBLE_DEVICES=7 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--model_path ../retriever/results/webqsp_rel_retri_1_large/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
#--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
#--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler_w_large \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 100 --num_step 3 \
#--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
# --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
#--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
#--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding
## reasoning: pretrain, transfer, encoding relation but not updating, wo pooler, with attn aggre 67.22/75.32  71.97/75.26
#CUDA_VISIBLE_DEVICES=4 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_er_wo_pooler_w_attn-f1.ckpt \
#--model_path ./retriever/results/webqsp_rel_retri_1/ \
#--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_er_wo_pooler_w_attn \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding --aggregate_token_representation
## reasoning: pretrain, transfer, encoding relation but not updating, wo pooler with attn aggre h1 67.82/76.42 70.96/75.93
#CUDA_VISIBLE_DEVICES=4 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_er_wo_pooler_w_attn-h1.ckpt \
#--model_path ./retriever/results/webqsp_rel_retri_1/ \
#--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_er_wo_pooler_w_attn_h1 \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding --aggregate_token_representation
#
## retrieval: pretrain, transfer, encoding relation but not updating 不用pooler层
## cwq 53.88/66.62 55.55/67.60
#CUDA_VISIBLE_DEVICES=0 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--model_path ../retriever/results/cwq_rel_retri_0/ --agent_type PLM --instruct_model PLM --model_name gnn \
#--relation_model_path ../retriever/results/cwq_rel_retri_0/ \
#--data_folder ../data/cwq/ --data_name .abstract.sg.max_tris.1500.nsm.json \
#--checkpoint_dir ../retriever_ckpt/cwq_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler \
#--batch_size 20 --gradient_accumulation_steps 2 --test_batch_size 40 --num_step 3 \
#--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
# --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
#--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/cwq/abs_r0_sg_for_retri.cache \
#--relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding
## reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 46.83/50.12 49.77/51.09
#CUDA_VISIBLE_DEVICES=4 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_p_t_q_er_wo_pooler-f1.ckpt \
#--model_path ./retriever/results/cwq_rel_retri_0/ \
#--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_q_er_wo_pooler \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding
## reasoning: pretrain, transfer, encoding relation but not updating, wo pooler h1 48.44/50.68 50.26/51.63
#CUDA_VISIBLE_DEVICES=4 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_p_t_q_er_wo_pooler-f1.ckpt \
#--model_path ./retriever/results/cwq_rel_retri_0/ \
#--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_q_er_wo_pooler_h1 \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding
#
## retrieval: pretrain, transfer, update plm 不用pooler层 71.95/84.69 71.82/83.71
## webqsp
#CUDA_VISIBLE_DEVICES=0 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--model_path ../retriever/results/webqsp_rel_retri_1/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
#--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
#--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_p_t_q_r_wo_pooler \
#--batch_size 10 --gradient_accumulation_steps 4 --test_batch_size 400 --num_step 3 \
#--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
# --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
#--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
#--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt \
#--simplify_model
## reasoning: pretrain, transfer, update plm, wo pooler f1 71.97/76.66 72.75/76.42
#CUDA_VISIBLE_DEVICES=0 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_r_wo_pooler-f1.ckpt \
#--model_path ./retriever/results/webqsp_rel_retri_1/ \
#--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_r_wo_pooler \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
#--simplify_model
## reasoning: pretrain, transfer, update plm, wo pooler h1 72.13/77.15  72.44/77.15
#CUDA_VISIBLE_DEVICES=0 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_r_wo_pooler-h1.ckpt \
#--model_path ./retriever/results/webqsp_rel_retri_1/ \
#--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_r_wo_pooler_h1 \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
#--simplify_model
## cwq 55.00/67.28 55.39/66.76
#CUDA_VISIBLE_DEVICES=7 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--model_path ../retriever/results/cwq_rel_retri_0/ --agent_type PLM --instruct_model PLM --model_name gnn \
#--relation_model_path ../retriever/results/cwq_rel_retri_0/ \
#--data_folder ../data/cwq/ --data_name .abstract.sg.max_tris.1500.nsm.json \
#--checkpoint_dir ../retriever_ckpt/cwq_nsm/ --experiment_name retriever_p_t_q_r_wo_pooler \
#--batch_size 5 --gradient_accumulation_steps 8 --test_batch_size 40 --num_step 3 \
#--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
# --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
#--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/cwq/abs_r0_sg_for_retri.cache \
#--relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt \
#--simplify_model
## reasoning: pretrain, transfer, update plm, wo pooler f1 46.78/49.88 49.08/49.88
#CUDA_VISIBLE_DEVICES=6 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_p_t_q_r_wo_pooler-f1.ckpt \
#--model_path ./retriever/results/cwq_rel_retri_0/ \
#--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_q_r_wo_pooler \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
#--simplify_model
## reasoning: pretrain, transfer, update plm, wo pooler h1 48.79/50.47 49.01/50.65
#CUDA_VISIBLE_DEVICES=6 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_p_t_q_r_wo_pooler-h1.ckpt \
#--model_path ./retriever/results/cwq_rel_retri_0/ \
#--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_q_r_wo_pooler_h1 \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
#--simplify_model
#
#
## webqsp
## retrieval: no pretrain, transfer, encoding relation but not updating wo pooler 63.66/79.19 70.31/81.70
#CUDA_VISIBLE_DEVICES=1 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--model_path /mnt/jiangjinhao/hg_face/roberta-base --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
#--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
#--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_t_q_er_wo_pooler \
#--batch_size 10 --gradient_accumulation_steps 4 --test_batch_size 400 --num_step 3 \
#--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
# --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
#--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
#--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding
## reasoning: no pretrain, transfer, encoding relation but not updating, wo pooler f1 70.85/76.60 70.67/75.38
#CUDA_VISIBLE_DEVICES=4 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_t_q_er_wo_pooler-f1.ckpt \
#--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
#--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_t_q_er_wo_pooler \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding
## reasoning: no pretrain, transfer, encoding relation but not updating, wo pooler h1 69.24/75.44 70、38/74.53
#CUDA_VISIBLE_DEVICES=2 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_t_q_er_wo_pooler-h1.ckpt \
#--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
#--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_t_q_er_wo_pooler_h1 \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding
## cwq
## retrieval: no pretrain, transfer, encoding relation but not updating 不用pooler层 52.86/65.76
#CUDA_VISIBLE_DEVICES=5 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--model_path /mnt/jiangjinhao/hg_face/roberta-base/ --agent_type PLM --instruct_model PLM --model_name gnn \
#--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
#--data_folder ../data/cwq/ --data_name .abstract.sg.max_tris.1500.nsm.json \
#--checkpoint_dir ../retriever_ckpt/cwq_nsm/ --experiment_name retriever_t_q_er_wo_pooler \
#--batch_size 20 --gradient_accumulation_steps 2 --test_batch_size 40 --num_step 3 \
#--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
# --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
#--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/cwq/abs_r0_sg_for_retri.cache \
#--relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding
## reasoning: no pretrain, transfer, encoding relation but not updating, wo pooler f1 47.79/50.77 49.47/50.80
#CUDA_VISIBLE_DEVICES=6 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_t_q_er_wo_pooler-f1.ckpt \
#--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
#--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_t_q_er_wo_pooler \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding
## reasoning: no pretrain, transfer, encoding relation but not updating, wo pooler h1 46.86/49.82 49.46/50.50
#CUDA_VISIBLE_DEVICES=7 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_t_q_er_wo_pooler-h1.ckpt \
#--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
#--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_t_q_er_wo_pooler_h1 \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding

################################################################

# webqsp
## retrieval: no pretrain, transfer, update plm 不用pooler层 62.26/80.59 67.54/78.63
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ./data/webqsp/ \
--data_name .abs.sg.topk.15.min_sim.0.max_tris.1000.new_rel_retri_1.nsm.json \
--checkpoint_dir ./retriever_ckpt/webqsp_nsm/ --experiment_name retriever_q_r_wo_pooler \
--batch_size 10 --gradient_accumulation_steps 4 --test_batch_size 80 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ./data/webqsp/new_abs_r1_sg_for_retri.cache \
--relation2id new_abs_r1_relations.txt --entity2id new_abs_r1_entities.txt \
--simplify_model
## reasoning: no pretrain, transfer, update plm 不用pooler层
### transfer best f1 70.37/75.44 71.44/75.63
CUDA_VISIBLE_DEVICES=7 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_q_r_wo_pooler-f1.ckpt \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_t_q_r_wo_pooler_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt \
--simplify_model
### transfer best h1 70.21/75.57 70.43/75.02
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_q_r_wo_pooler-h1.ckpt \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_t_q_r_wo_pooler_h1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt \
--simplify_model
## retrieval: no pretrain, transfer, fix plm 不用pooler层 33.09/70.02 34.69/68.93
CUDA_VISIBLE_DEVICES=7 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ./data/webqsp/ \
--data_name .abs.sg.topk.15.min_sim.0.max_tris.1000.new_rel_retri_1.nsm.json \
--checkpoint_dir ./retriever_ckpt/webqsp_nsm/ --experiment_name retriever_wo_pooler \
--batch_size 20 --gradient_accumulation_steps 2 --test_batch_size 100 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ./data/webqsp/new_abs_r1_sg_for_retri.cache \
--relation2id new_abs_r1_relations.txt --entity2id new_abs_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## reasoning: no pretrain, transfer, fix plm 不用pooler层
### transfer best f1 59.86/68.97 58.75/68.30
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_wo_pooler-f1.ckpt \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_t_wo_pooler_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
### transfer best h1 58.02/68.54 57.64/68.17
CUDA_VISIBLE_DEVICES=7 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_wo_pooler-h1.ckpt \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_t_wo_pooler_h1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## retrieval: no pretrain, transfer, update plm for question encoding 不用pooler层 56.59/78.75 64.54/79.91
CUDA_VISIBLE_DEVICES=7 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn --data_folder ./data/webqsp/ \
--data_name .abs.sg.topk.15.min_sim.0.max_tris.1000.new_rel_retri_1.nsm.json \
--checkpoint_dir ./retriever_ckpt/webqsp_nsm/ --experiment_name retriever_q_wo_pooler \
--batch_size 20 --gradient_accumulation_steps 2 --test_batch_size 100 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ./data/webqsp/new_abs_r1_sg_for_retri.cache \
--relation2id new_abs_r1_relations.txt --entity2id new_abs_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate
## reasoning: no pretrain, transfer, update plm for question encoding 不用pooler层
### transfer best f1 66.75/74.16 67.82/74.10
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_q_wo_pooler-f1.ckpt \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_t_q_wo_pooler_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate
### transfer best h1 65.35/73.85 68.10/74.22
CUDA_VISIBLE_DEVICES=6 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_q_wo_pooler-h1.ckpt \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_t_q_wo_pooler_h1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate

## reasoning: pretrain, no transfer, update plm 不用pooler层 71.28/75.87 72.56/76.48
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ./retriever/results/new_ webqsp_rel_retri_1/ \
--relation_model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_q_r_wo_pooler \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt \
--simplify_mode
## reasoning: pretrain, no transfer, fix plm 不用pooler层 69.63/73.24 68.22/72.82
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--relation_model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_wo_pooler \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## reasoning: pretrain, no transfer, update plm for question encoding 不用pooler层 52.63/69.21 69.22/72.88
CUDA_VISIBLE_DEVICES=1 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--relation_model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_q_wo_pooler \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate

## reasoning: no pretrain, no transfer, update plm 不用pooler层 66.97/71.90 67.07/73.06
CUDA_VISIBLE_DEVICES=6 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_q_r_wo_pooler \
--batch_size 40 --gradient_accumulation_steps 2 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt \
--simplify_mode
## reasoning: no pretrain, no transfer, fix plm 不用pooler层 53.38/65.36 55.61/64.08
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_wo_pooler \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## reasoning: no pretrain, no transfer, update plm for question encoding 不用pooler层 64.27/71.17 65.93/70.89
CUDA_VISIBLE_DEVICES=6 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_q_wo_pooler \
--batch_size 40 --gradient_accumulation_steps 2 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate


# cwq
## retrieval: no pretrain, transfer, update plm 不用pooler层 50.03/74.09
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ --agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .abs.sg.max_tris.2000.nsm.json \
--checkpoint_dir ./retriever_ckpt/cwq_nsm_retri/ --batch_size 20 --gradient_accumulation_steps 4 \
--test_batch_size 70 --num_step 4 --entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--experiment_name retriever_q_r --eps 0.95 --num_epoch 100 --patience 10 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ./data/cwq/new_abs_r0_sg_for_retri.cache \
--relation2id new_abs_r0_relations.txt --entity2id new_abs_r0_entities.txt \
--simplify_model
## reasoning: no pretrain, transfer, update plm, wo pooler f1
### transfer best f1 50.67/51.69 50.87/51.48
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm_retri/retriever_q_r-f1.ckpt \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_t_q_r_wo_pooler_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 4 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/cwq/new_ins_r0_sg_for_reason.cache --overwrite_cache \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt \
--simplify_model
### transfer best h1 47.36/51.80 49.80/51.36
CUDA_VISIBLE_DEVICES=4 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm_retri/retriever_q_r-h1.ckpt \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_t_q_r_wo_pooler_h1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 4 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/cwq/new_ins_r0_sg_for_reason.cache \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt \
--simplify_model
## retrieval: no pretrain, transfer, fix plm 不用pooler层 15.16/61.12
CUDA_VISIBLE_DEVICES=2 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ --agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .abs.sg.max_tris.2000.nsm.json \
--checkpoint_dir ./retriever_ckpt/cwq_nsm_retri/ --batch_size 20 --gradient_accumulation_steps 4 \
--test_batch_size 50 --num_step 4 --entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--experiment_name retriever --eps 0.95 --num_epoch 100 --patience 10 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ./data/cwq/new_abs_r0_sg_for_retri.cache \
--relation2id new_abs_r0_relations.txt --entity2id new_abs_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## reasoning: no pretrain, transfer, fix plm, wo pooler f1
### transfer best f1 39.19/48.79 40.78/49.44
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm_retri/retriever-f1.ckpt \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_wo_pooler_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 4 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/cwq/new_ins_r0_sg_for_reason.cache \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
### transfer best h1 39.80/48.76 41.01/48.70
CUDA_VISIBLE_DEVICES=4 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm_retri/retriever-h1.ckpt \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_wo_pooler_h1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 4 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/cwq/new_ins_r0_sg_for_reason.cache \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## retrieval: no pretrain, transfer, update plm for question encoding 不用pooler层 52.23/76.35
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/  \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .abs.sg.max_tris.2000.nsm.json \
--checkpoint_dir ./retriever_ckpt/cwq_nsm_retri/ --batch_size 40 --gradient_accumulation_steps 2 \
--test_batch_size 70 --num_step 4 --entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--experiment_name retriever_q --eps 0.95 --num_epoch 100 --patience 10 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ./data/cwq/new_abs_r0_sg_for_retri.cache \
--relation2id new_abs_r0_relations.txt --entity2id new_abs_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate
## reasoning: no pretrain, transfer, update plm for question encoding, wo pooler f1
### transfer best f1 5-1
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm_retri/retriever_q-f1.ckpt \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_t_q_wo_pooler_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 4 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/cwq/new_ins_r0_sg_for_reason.cache --overwrite_cache \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate
### transfer best h1 5-2
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm_retri/retriever_q-h1.ckpt \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_t_q_wo_pooler_h1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 4 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/cwq/new_ins_r0_sg_for_reason.cache \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate

## reasoning: pretrain, no transfer, update plm 不用pooler层 50.30/52.28 51.51/52.66
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ./retriever/results/cwq_rel_retri_0/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_q_r_wo_pooler \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 4 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/cwq/new_ins_r0_sg_for_reason.cache \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt \
--simplify_model
## reasoning: pretrain, no transfer, fix plm 不用pooler层 48.49/51.39 49.70/51.45
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--model_path ./retriever/results/cwq_rel_retri_0/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_wo_pooler \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 4 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/cwq/new_ins_r0_sg_for_reason.cache \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## reasoning: pretrain, no transfer, update plm for question encoding 不用pooler层 49.27/51.39 50.78/51.69
CUDA_VISIBLE_DEVICES=6 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ./retriever/results/cwq_rel_retri_0/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_q_wo_pooler \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 4 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/cwq/new_ins_r0_sg_for_reason.cache \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate

## reasoning: no pretrain, no transfer, update plm 不用pooler层 45.43/50.74 50.84/50.95
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_q_r_wo_pooler \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 4 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/cwq/new_ins_r0_sg_for_reason.cache \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt \
--simplify_model
## reasoning: no pretrain, no transfer, fix plm 不用pooler层 38.89/48.64 39.58/48.94
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_wo_pooler \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 4 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/cwq/new_ins_r0_sg_for_reason.cache \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## reasoning: no pretrain, no transfer, update plm for question encoding 不用pooler层 45.13/47.90 47.23/47.87
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_q_wo_pooler \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 4 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/cwq/new_ins_r0_sg_for_reason.cache \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate

##############################################################

## webqsp
## reasoning: no pretrain, no transfer, encoding relation but not updating wo pooler h1 66.93/72.51 68.99/73.79
#CUDA_VISIBLE_DEVICES=2 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
#--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_q_er_wo_pooler_h1 \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding
## reasoning: pretrain, no transfer, encoding relation but not updating wo pooler h1 70.95/75.81 70.69/74.71
#CUDA_VISIBLE_DEVICES=4 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--model_path ./retriever/results/webqsp_rel_retri_1/ \
#--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_q_er_wo_pooler_h1 \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding
## cwq
## reasoning: no pretrain, no transfer, encoding relation but not updating, wo pooler 46.03/49.29 48.89/50.27
#CUDA_VISIBLE_DEVICES=6 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
#--relation_model_path /mnt/jiangjinhao/hg_face/roberta-base/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_q_er_wo_pooler \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding
## reasoning: pretrain, no transfer, encoding relation but not updating, wo pooler h1 48.69/51.24 49.53/50.56
#CUDA_VISIBLE_DEVICES=6 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
#--model_path ./retriever/results/cwq_rel_retri_0/ \
#--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
#--agent_type PLM --instruct_model PLM --model_name gnn \
#--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
#--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_q_er_wo_pooler \
#--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
#--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
#--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
#--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
#--simplify_model --fixed_plm_for_relation_encoding