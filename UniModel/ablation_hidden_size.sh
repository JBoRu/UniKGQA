# webqsp: pretrain, transfer, encoding relation but not updating wo pooler
# retrieval: 64 67.78/83.34 71.64/83.47
CUDA_VISIBLE_DEVICES=4 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_1/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler_hd_64 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 400 --num_step 3 \
--entity_dim 64 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 71.63/76.24
CUDA_VISIBLE_DEVICES=4 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_er_wo_pooler_hd_64-f1.ckpt \
--model_path ./retriever/results/webqsp_rel_retri_1/ \
--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_er_wo_pooler_hd_64_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 64 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# retrieval: 128 71.89/84.14 71.50/83.16
CUDA_VISIBLE_DEVICES=5 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_1/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler_hd_128 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 400 --num_step 3 \
--entity_dim 128 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 71.45/77.21 71.78/77.15
CUDA_VISIBLE_DEVICES=5 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_er_wo_pooler_hd_128-f1.ckpt \
--model_path ./retriever/results/webqsp_rel_retri_1/ \
--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_er_wo_pooler_hd_128_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 128 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# retrieval: 256 71.68/83.77
CUDA_VISIBLE_DEVICES=5 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_1/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler_hd_256 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 50 --num_step 3 \
--entity_dim 256 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 72.37/76.05 72.50/76.73
CUDA_VISIBLE_DEVICES=5 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_er_wo_pooler_hd_256-f1.ckpt \
--model_path ./retriever/results/webqsp_rel_retri_1/ \
--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_er_wo_pooler_hd_256_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 256 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# retrieval: 512 71.91/83.71 71.54/82.92
CUDA_VISIBLE_DEVICES=6 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_1/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler_hd_512 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 400 --num_step 3 \
--entity_dim 512 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 69.33/76.79 72.10/75.38
CUDA_VISIBLE_DEVICES=6 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_er_wo_pooler_hd_512-f1.ckpt \
--model_path ./retriever/results/webqsp_rel_retri_1/ \
--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_er_wo_pooler_hd_512_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 512 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# retrieval: 1024 70.74/83.40
CUDA_VISIBLE_DEVICES=6 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_1/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler_hd_1024 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 400 --num_step 3 \
--entity_dim 1024 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 71.52/76.60  71.29/75.75
CUDA_VISIBLE_DEVICES=6 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_q_er_wo_pooler_hd_1024-f1.ckpt \
--model_path ./retriever/results/webqsp_rel_retri_1/ \
--relation_model_path ./retriever/results/webqsp_rel_retri_1/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_er_wo_pooler_hd_1024_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 1024 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r1_relations.txt --entity2id reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# cwq: pretrain, transfer, encoding relation but not updating 不用pooler层
# retrieval: 64 52.45/65.78 54.93/66.65
CUDA_VISIBLE_DEVICES=1 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/cwq_rel_retri_0/ --agent_type PLM --instruct_model PLM --model_name gnn \
--relation_model_path ../retriever/results/cwq_rel_retri_0/ \
--data_folder ../data/cwq/ --data_name .abstract.sg.max_tris.1500.nsm.json \
--checkpoint_dir ../retriever_ckpt/cwq_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler_hd_64 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 40 --num_step 3 \
--entity_dim 64 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/cwq/abs_r0_sg_for_retri.cache \
--relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 46.66/49.26 48.91/49.70
CUDA_VISIBLE_DEVICES=1 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_p_t_q_er_wo_pooler_hd_64-f1.ckpt \
--model_path ./retriever/results/cwq_rel_retri_0/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_q_er_wo_pooler_hd_64_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 64 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# retrieval: 128 54.77/67.65 55.05/67.28
CUDA_VISIBLE_DEVICES=1 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/cwq_rel_retri_0/ --agent_type PLM --instruct_model PLM --model_name gnn \
--relation_model_path ../retriever/results/cwq_rel_retri_0/ \
--data_folder ../data/cwq/ --data_name .abstract.sg.max_tris.1500.nsm.json \
--checkpoint_dir ../retriever_ckpt/cwq_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler_hd_128 \
--batch_size 20 --gradient_accumulation_steps 2 --test_batch_size 40 --num_step 3 \
--entity_dim 128 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/cwq/abs_r0_sg_for_retri.cache \
--relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 46.15/49.97  49.07/50.18
CUDA_VISIBLE_DEVICES=1 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_p_t_q_er_wo_pooler_hd_128-f1.ckpt \
--model_path ./retriever/results/cwq_rel_retri_0/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_q_er_wo_pooler_hd_128_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 128 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# retrieval: 256
CUDA_VISIBLE_DEVICES=2 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/cwq_rel_retri_0/ --agent_type PLM --instruct_model PLM --model_name gnn \
--relation_model_path ../retriever/results/cwq_rel_retri_0/ \
--data_folder ../data/cwq/ --data_name .abstract.sg.max_tris.1500.nsm.json \
--checkpoint_dir ../retriever_ckpt/cwq_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler_hd_256 \
--batch_size 20 --gradient_accumulation_steps 2 --test_batch_size 40 --num_step 3 \
--entity_dim 256 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/cwq/abs_r0_sg_for_retri.cache \
--relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 47.99/49.88 49.32/50.83
CUDA_VISIBLE_DEVICES=2 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_p_t_q_er_wo_pooler_hd_256-f1.ckpt \
--model_path ./retriever/results/cwq_rel_retri_0/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_q_er_wo_pooler_hd_256_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 256 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# retrieval: 512 54.89/66.85 55.76/66.85
CUDA_VISIBLE_DEVICES=3 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/cwq_rel_retri_0/ --agent_type PLM --instruct_model PLM --model_name gnn \
--relation_model_path ../retriever/results/cwq_rel_retri_0/ \
--data_folder ../data/cwq/ --data_name .abstract.sg.max_tris.1500.nsm.json \
--checkpoint_dir ../retriever_ckpt/cwq_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler_hd_512 \
--batch_size 20 --gradient_accumulation_steps 2 --test_batch_size 40 --num_step 3 \
--entity_dim 512 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/cwq/abs_r0_sg_for_retri.cache \
--relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 46.32/49.47 49.41/50.77
CUDA_VISIBLE_DEVICES=2 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_p_t_q_er_wo_pooler_hd_512-f1.ckpt \
--model_path ./retriever/results/cwq_rel_retri_0/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_q_er_wo_pooler_hd_512_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 512 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding

# retrieval: 1024 53.37/66.39 55.87/66.84
CUDA_VISIBLE_DEVICES=4 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/cwq_rel_retri_0/ --agent_type PLM --instruct_model PLM --model_name gnn \
--relation_model_path ../retriever/results/cwq_rel_retri_0/ \
--data_folder ../data/cwq/ --data_name .abstract.sg.max_tris.1500.nsm.json \
--checkpoint_dir ../retriever_ckpt/cwq_nsm/ --experiment_name retriever_p_t_q_er_wo_pooler_hd_1024 \
--batch_size 20 --gradient_accumulation_steps 2 --test_batch_size 40 --num_step 3 \
--entity_dim 1024 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/cwq/abs_r0_sg_for_retri.cache \
--relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
# reasoning: pretrain, transfer, encoding relation but not updating, wo pooler f1 48.47/49.29 49.30/50.98
CUDA_VISIBLE_DEVICES=1 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm/retriever_p_t_q_er_wo_pooler_hd_1024-f1.ckpt \
--model_path ./retriever/results/cwq_rel_retri_0/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_q_er_wo_pooler_hd_1024_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 1024 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding
