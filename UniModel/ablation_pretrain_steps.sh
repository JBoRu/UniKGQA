# webqsp: pretrain, transfer, encoding relation but not updating 不用pooler层
# retrieval 400 70.34/82.12 70.65/82.25
CUDA_VISIBLE_DEVICES=0 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_bs_30/checkpoint-400/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler_400 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 400 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 70.07/75.50 71.16/77.09
CUDA_VISIBLE_DEVICES=0 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_er_wo_pooler_400-f1.ckpt \
--model_path ./retriever/results/webqsp_rel_retri_bs_30/checkpoint-400/ \
--relation_model_path ./retriever/results/webqsp_rel_retri_bs_30/checkpoint-400/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_er_wo_pooler_400_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, fix plm, wo pooler f1 4-1
CUDA_VISIBLE_DEVICES=0 python3 ./reader/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_er_wo_pooler_400-f1.ckpt \
--model_path ./retriever/results/webqsp_rel_retri_bs_30/checkpoint-400/ \
--relation_model_path ./retriever/results/webqsp_rel_retri_bs_30/checkpoint-400/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_er_wo_pooler_400_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm

# retrieval 1600
CUDA_VISIBLE_DEVICES=0 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_bs_30/checkpoint-1600/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler_1600 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 50 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 70.48/76.42 71.79/76.30
CUDA_VISIBLE_DEVICES=0 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_er_wo_pooler_1600-f1.ckpt \
--model_path ./retriever/results/webqsp_rel_retri_bs_30/checkpoint-1600/ \
--relation_model_path ./retriever/results/webqsp_rel_retri_bs_30/checkpoint-1600/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_er_wo_pooler_1600_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 4-2
CUDA_VISIBLE_DEVICES=0 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_er_wo_pooler_1600-f1.ckpt \
--model_path ./retriever/results/webqsp_rel_retri_bs_30/checkpoint-1600/ \
--relation_model_path ./retriever/results/webqsp_rel_retri_bs_30/checkpoint-1600/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_er_wo_pooler_1600_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm

# retrieval 2800 83.90
CUDA_VISIBLE_DEVICES=2 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_bs_30/checkpoint-2800/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler_2800 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 50 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 71.36/77.15 73.07/76.54
CUDA_VISIBLE_DEVICES=2 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_er_wo_pooler_2800-f1.ckpt \
--model_path ./retriever/results/webqsp_rel_retri_bs_30/checkpoint-2800/ \
--relation_model_path ./retriever/results/webqsp_rel_retri_bs_30/checkpoint-2800/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_er_wo_pooler_2800_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# retrieval 4000
CUDA_VISIBLE_DEVICES=2 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_bs_30/checkpoint-4000/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler_4000 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 50 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 69.82/72.18 70.60/75.08
CUDA_VISIBLE_DEVICES=2 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_er_wo_pooler_4000-f1.ckpt \
--model_path ./retriever/results/webqsp_rel_retri_bs_30/checkpoint-4000/ \
--relation_model_path ./retriever/results/webqsp_rel_retri_bs_30/checkpoint-4000/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_er_wo_pooler_4000_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# retrieval 5200
CUDA_VISIBLE_DEVICES=3 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_bs_30/checkpoint-5200/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler_5200 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 50 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 70.76/75.20 71.64/76.42
CUDA_VISIBLE_DEVICES=3 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_er_wo_pooler_5200-f1.ckpt \
--model_path ./retriever/results/webqsp_rel_retri_bs_30/checkpoint-5200/ \
--relation_model_path ./retriever/results/webqsp_rel_retri_bs_30/checkpoint-5200/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_er_wo_pooler_5200_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# retrieval 6400
CUDA_VISIBLE_DEVICES=3 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_bs_30/checkpoint-6400/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler_6400 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 71.91/76.85 72.14/76.05
CUDA_VISIBLE_DEVICES=3 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_er_wo_pooler_6400-f1.ckpt \
--model_path ./retriever/results/webqsp_rel_retri_bs_30/checkpoint-6400/ \
--relation_model_path ./retriever/results/webqsp_rel_retri_bs_30/checkpoint-6400/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_er_wo_pooler_6400_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# retrieval 7600
CUDA_VISIBLE_DEVICES=4 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_bs_30/checkpoint-7600/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler_7600 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 50 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 71.24/77.15 71.52/76.97
CUDA_VISIBLE_DEVICES=4 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_er_wo_pooler_7600-f1.ckpt \
--model_path ./retriever/results/webqsp_rel_retri_bs_30/checkpoint-7600/ \
--relation_model_path ./retriever/results/webqsp_rel_retri_bs_30/checkpoint-7600/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_er_wo_pooler_7600_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding


# cwq: no pretrain, transfer, encoding relation but not updating 不用pooler层
# retrieval:  3000 56.34/67.22 56.57/67.54
CUDA_VISIBLE_DEVICES=0 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/cwq_rel_retri_0/checkpoint-3000/ --agent_type PLM --instruct_model PLM --model_name gnn \
--relation_model_path ../retriever/results/cwq_rel_retri_0/checkpoint-3000/ \
--data_folder ../data/cwq/ --data_name .abstract.sg.max_tris.1500.nsm.json \
--checkpoint_dir ../retriever_ckpt/cwq_nsm/ --experiment_name retriever_t_q_er_wo_pooler_3000 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 50 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 10 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/cwq/abs_r0_sg_for_retri.cache \
--relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: 3000 f1 49.57/51.27 49.63/51.18
CUDA_VISIBLE_DEVICES=4 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_t_q_er_wo_pooler_3000-f1.ckpt \
--model_path ./retriever/results/cwq_rel_retri_0/checkpoint-3000/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/checkpoint-3000/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_t_q_er_wo_pooler_3000_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# retrieval:  15000 54.22/66.30 56.59/67.28
CUDA_VISIBLE_DEVICES=1 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/cwq_rel_retri_0/checkpoint-15000/ --agent_type PLM --instruct_model PLM --model_name gnn \
--relation_model_path ../retriever/results/cwq_rel_retri_0/checkpoint-15000/ \
--data_folder ../data/cwq/ --data_name .abstract.sg.max_tris.1500.nsm.json \
--checkpoint_dir ../retriever_ckpt/cwq_nsm/ --experiment_name retriever_t_q_er_wo_pooler_15000 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 50 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 10 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/cwq/abs_r0_sg_for_retri.cache \
--relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: f1 49.54/50.86 49.09/50.92
CUDA_VISIBLE_DEVICES=4 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_t_q_er_wo_pooler_15000-f1.ckpt \
--model_path ./retriever/results/cwq_rel_retri_0/checkpoint-15000/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/checkpoint-15000/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_t_q_er_wo_pooler_15000_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# retrieval:  27000 56.18/68.15 56.12/67.60
CUDA_VISIBLE_DEVICES=2 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/cwq_rel_retri_0/checkpoint-27000/ --agent_type PLM --instruct_model PLM --model_name gnn \
--relation_model_path ../retriever/results/cwq_rel_retri_0/checkpoint-27000/ \
--data_folder ../data/cwq/ --data_name .abstract.sg.max_tris.1500.nsm.json \
--checkpoint_dir ../retriever_ckpt/cwq_nsm/ --experiment_name retriever_t_q_er_wo_pooler_27000 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 50 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 10 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/cwq/abs_r0_sg_for_retri.cache \
--relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: f1 48.22/50.00 49.31/50.53
CUDA_VISIBLE_DEVICES=3 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_t_q_er_wo_pooler_27000-f1.ckpt \
--model_path ./retriever/results/cwq_rel_retri_0/checkpoint-27000/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/checkpoint-27000/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_t_q_er_wo_pooler_27000_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# retrieval:  39000 55.05/67.28 56.22/68.40
CUDA_VISIBLE_DEVICES=3 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/cwq_rel_retri_0/checkpoint-39000/ --agent_type PLM --instruct_model PLM --model_name gnn \
--relation_model_path ../retriever/results/cwq_rel_retri_0/checkpoint-39000/ \
--data_folder ../data/cwq/ --data_name .abstract.sg.max_tris.1500.nsm.json \
--checkpoint_dir ../retriever_ckpt/cwq_nsm/ --experiment_name retriever_t_q_er_wo_pooler_39000 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 50 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 10 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/cwq/abs_r0_sg_for_retri.cache \
--relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: f1 47.52/50.15 49.93/50.71
CUDA_VISIBLE_DEVICES=3 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_t_q_er_wo_pooler_39000-f1.ckpt \
--model_path ./retriever/results/cwq_rel_retri_0/checkpoint-39000/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/checkpoint-39000/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_t_q_er_wo_pooler_39000_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# retrieval:  51000 55.83/68.17 56.59/67.77
CUDA_VISIBLE_DEVICES=4 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/cwq_rel_retri_0/checkpoint-51000/ --agent_type PLM --instruct_model PLM --model_name gnn \
--relation_model_path ../retriever/results/cwq_rel_retri_0/checkpoint-51000/ \
--data_folder ../data/cwq/ --data_name .abstract.sg.max_tris.1500.nsm.json \
--checkpoint_dir ../retriever_ckpt/cwq_nsm/ --experiment_name retriever_t_q_er_wo_pooler_51000 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 50 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 10 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/cwq/abs_r0_sg_for_retri.cache \
--relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: f1 49.86/51.57 49.99/51.03
CUDA_VISIBLE_DEVICES=4 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_t_q_er_wo_pooler_51000-f1.ckpt \
--model_path ./retriever/results/cwq_rel_retri_0/checkpoint-51000/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/checkpoint-51000/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_t_q_er_wo_pooler_51000_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# retrieval:  63000 55.67/67.40 56.53/68.23
CUDA_VISIBLE_DEVICES=5 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/cwq_rel_retri_0/checkpoint-63000/ --agent_type PLM --instruct_model PLM --model_name gnn \
--relation_model_path ../retriever/results/cwq_rel_retri_0/checkpoint-63000/ \
--data_folder ../data/cwq/ --data_name .abstract.sg.max_tris.1500.nsm.json \
--checkpoint_dir ../retriever_ckpt/cwq_nsm/ --experiment_name retriever_t_q_er_wo_pooler_63000 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 50 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 10 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/cwq/abs_r0_sg_for_retri.cache \
--relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: f1 49.21/51.18 50.02/51.39
CUDA_VISIBLE_DEVICES=5 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_t_q_er_wo_pooler_63000-f1.ckpt \
--model_path ./retriever/results/cwq_rel_retri_0/checkpoint-63000/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/checkpoint-63000/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_t_q_er_wo_pooler_63000_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# retrieval:  69000 56.23/67.45 56.29/67.34
CUDA_VISIBLE_DEVICES=6 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/cwq_rel_retri_0/checkpoint-69000/ --agent_type PLM --instruct_model PLM --model_name gnn \
--relation_model_path ../retriever/results/cwq_rel_retri_0/checkpoint-69000/ \
--data_folder ../data/cwq/ --data_name .abstract.sg.max_tris.1500.nsm.json \
--checkpoint_dir ../retriever_ckpt/cwq_nsm/ --experiment_name retriever_t_q_er_wo_pooler_69000 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 50 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 10 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/cwq/abs_r0_sg_for_retri.cache \
--relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: f1 47.63/50.18 49.64/51.09
CUDA_VISIBLE_DEVICES=6 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_t_q_er_wo_pooler_69000-f1.ckpt \
--model_path ./retriever/results/cwq_rel_retri_0/checkpoint-69000/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/checkpoint-69000/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_t_q_er_wo_pooler_69000_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
