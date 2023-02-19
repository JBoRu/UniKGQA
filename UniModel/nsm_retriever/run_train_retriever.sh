CUDA_VISIBLE_DEVICES=6 /home/jiangjinhao/anaconda3/envs/pt1.8-transformer4.18/bin/python main_nsm.py --diff_lr --linear_dropout 0.0 --log_steps 300 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/data/ \
--data_name .super.subgraph.nsm.input.jsonl --checkpoint_dir ../outputs/retriever_training_0/ --batch_size 1 --gradient_accumulation_steps 20 \
--test_batch_size 20 --num_step 4 --entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 2 --encode_type \
--experiment_name webqsp_nsm_retriever_roberta_rel_query_adamw_linear_diff_lr_adamw_bs_20_wo_drop_ed_100_curriculum_not_self_loop --eps 0.95 --num_epoch 100 \
--plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ../data/webqsp/data/super_sg_for_retriever.cache
CUDA_VISIBLE_DEVICES=7 /home/jiangjinhao/anaconda3/envs/pt1.8-transformer4.18/bin/python main_nsm.py --diff_lr --linear_dropout 0.0 --log_steps 300 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/data/ \
--data_name .super.subgraph.nsm.input.jsonl --checkpoint_dir ../outputs/retriever_training_1/ --batch_size 1 --gradient_accumulation_steps 20 \
--test_batch_size 20 --num_step 4 --entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 2 --encode_type \
--experiment_name webqsp_nsm_retriever_roberta_bs_20_wo_drop_ed_100_curriculum_both_max_steps --eps 0.95 --num_epoch 100 \
--use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ../data/webqsp/data/super_sg_for_retriever.cache --overwrite_cache
# 特定步数之后的推理都计算loss 80.92
CUDA_VISIBLE_DEVICES=6 /home/jiangjinhao/anaconda3/envs/pt1.8-transformer4.18/bin/python main_nsm.py --diff_lr --linear_dropout 0.0 --log_steps 300 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/data/ \
--data_name .super.subgraph.nsm.input.jsonl --checkpoint_dir ../outputs/retriever_training_0/ --batch_size 1 --gradient_accumulation_steps 20 \
--test_batch_size 20 --num_step 4 --entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 2 --encode_type \
--experiment_name webqsp_nsm_retriever_roberta_bs_20_wo_drop_ed_100_curriculum_each_steps_loss --eps 0.95 --num_epoch 100 \
--use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ../data/webqsp/data/super_sg_for_retriever.cache --overwrite_cache
# 调整输入数据集 81.52
CUDA_VISIBLE_DEVICES=3 /home/jiangjinhao/anaconda3/envs/pt1.8-transformer4.18/bin/python main_nsm.py --diff_lr --linear_dropout 0.0 --log_steps 300 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/data/ \
--data_name .super.subgraph.v1.nsm.input.jsonl --checkpoint_dir ../outputs/retriever_training_4/ --batch_size 1 --gradient_accumulation_steps 20 \
--test_batch_size 20 --num_step 4 --entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 2 --encode_type \
--experiment_name webqsp_nsm_retriever_roberta_bs_20_wo_drop_ed_100_curriculum_each_steps_loss_v1 --eps 0.95 --num_epoch 100 \
--use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ../data/webqsp/data/super_sg_for_retriever.cache --overwrite_cache --relation2id super_sg_v1_relations.txt
# 调整数据集v2
CUDA_VISIBLE_DEVICES=7 /home/jiangjinhao/anaconda3/envs/pt1.8-transformer4.18/bin/python main_nsm.py --diff_lr --linear_dropout 0.0 --log_steps 300 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/data/ \
--data_name .super.subgraph.v2.nsm.input.jsonl --checkpoint_dir ../outputs/retriever_training_6/ --batch_size 1 --gradient_accumulation_steps 20 \
--test_batch_size 20 --num_step 4 --entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 2 --encode_type \
--experiment_name webqsp_nsm_retriever_roberta_bs_20_wo_drop_ed_100_curriculum_each_steps_loss_v2 --eps 0.95 --num_epoch 100 \
--use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ../data/webqsp/data/super_sg_for_retriever.cache --overwrite_cache --relation2id super_sg_v2_relations.txt
# 调整数据集v3
CUDA_VISIBLE_DEVICES=6 /home/jiangjinhao/anaconda3/envs/pt1.8-transformer4.18/bin/python main_nsm.py --diff_lr --linear_dropout 0.0 --log_steps 300 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/data/ \
--data_name .super.subgraph.v3.nsm.input.jsonl --checkpoint_dir ../outputs/retriever_training_7/ --batch_size 1 --gradient_accumulation_steps 20 \
--test_batch_size 20 --num_step 4 --entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 2 --encode_type \
--experiment_name webqsp_nsm_retriever_roberta_bs_20_wo_drop_ed_100_curriculum_each_steps_loss_v3 --eps 0.95 --num_epoch 100 \
--use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ../data/webqsp/data/super_sg_for_retriever.cache --overwrite_cache --relation2id super_sg_v3_relations.txt
# 数据集v3下调整参数
CUDA_VISIBLE_DEVICES=5 /home/jiangjinhao/anaconda3/envs/pt1.8-transformer4.18/bin/python main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 300 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/data/ \
--data_name .super.subgraph.v3.nsm.input.jsonl --checkpoint_dir ../outputs/retriever_training_8/ --batch_size 1 --gradient_accumulation_steps 20 \
--test_batch_size 20 --num_step 4 --entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 2 --encode_type \
--experiment_name webqsp_nsm_retriever_roberta_bs_20_linear_dp_0.1_ed_100_no_self_loop_curriculum_each_steps_loss_v3 --eps 0.95 --num_epoch 100 \
--plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl \
--data_cache ../data/webqsp/data/super_sg_for_retriever.cache --overwrite_cache --relation2id super_sg_v3_relations.txt

# 调整数据集，使用非结点的关系推理 74.81
CUDA_VISIBLE_DEVICES=6 /home/jiangjinhao/anaconda3/envs/pt1.8-transformer4.18/bin/python main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 300 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/data/ \
--data_name .super.subgraph.v1.nsm.input.jsonl --checkpoint_dir ../outputs/retriever_training_5/ --batch_size 1 --gradient_accumulation_steps 20 \
--test_batch_size 20 --num_step 4 --entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 2 --encode_type \
--experiment_name webqsp_nsm_retriever_roberta_not_use_ent_emb_bs_20_linear_dp_0.1_ed_100_curriculum_each_steps_loss_v1 --eps 0.95 --num_epoch 100 \
--use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ../data/webqsp/data/super_sg_for_retriever.cache --overwrite_cache --relation2id super_sg_v1_relations.txt --not_use_ent_emb
# 调参 batch size 增大
CUDA_VISIBLE_DEVICES=3 /home/jiangjinhao/anaconda3/envs/pt1.8-transformer4.18/bin/python main_nsm.py --diff_lr --linear_dropout 0.0 --log_steps 300 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/data/ \
--data_name .super.subgraph.v1.nsm.input.jsonl --checkpoint_dir ../outputs/retriever_training_5/ --batch_size 1 --gradient_accumulation_steps 40 \
--test_batch_size 20 --num_step 4 --entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 2 --encode_type \
--experiment_name webqsp_nsm_retriever_roberta_not_use_ent_emb_bs_40_wo_dp_curriculum_each_steps_loss_v1 --eps 0.95 --num_epoch 100 \
--use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl \
--data_cache ../data/webqsp/data/super_sg_for_retriever.cache --overwrite_cache --relation2id super_sg_v1_relations.txt --not_use_ent_emb


# 调参 81.22
CUDA_VISIBLE_DEVICES=7 /home/jiangjinhao/anaconda3/envs/pt1.8-transformer4.18/bin/python main_nsm.py --diff_lr --linear_dropout 0.0 --log_steps 300 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/data/ \
--data_name .super.subgraph.nsm.input.jsonl --checkpoint_dir ../outputs/retriever_training_1/ --batch_size 1 --gradient_accumulation_steps 20 \
--test_batch_size 20 --num_step 4 --entity_dim 200 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 2 --encode_type \
--experiment_name webqsp_nsm_retriever_roberta_bs_20_wo_drop_ed_200_curriculum_each_steps_loss --eps 0.95 --num_epoch 100 \
--use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ../data/webqsp/data/super_sg_for_retriever.cache --overwrite_cache
# 打乱数据集 81.22
CUDA_VISIBLE_DEVICES=5 /home/jiangjinhao/anaconda3/envs/pt1.8-transformer4.18/bin/python main_nsm.py --diff_lr --linear_dropout 0.0 --log_steps 300 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/data/ \
--data_name .super.subgraph.nsm.input.jsonl --checkpoint_dir ../outputs/retriever_training_2/ --batch_size 1 --gradient_accumulation_steps 20 \
--test_batch_size 20 --num_step 4 --entity_dim 200 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 2 --encode_type \
--experiment_name webqsp_nsm_retriever_roberta_bs_20_wo_drop_ed_200_each_steps_loss --eps 0.95 --num_epoch 100 \
--use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ../data/webqsp/data/super_sg_for_retriever.cache --overwrite_cache
# 使用plm初始化 80.92
CUDA_VISIBLE_DEVICES=4 /home/jiangjinhao/anaconda3/envs/pt1.8-transformer4.18/bin/python main_nsm.py --diff_lr --linear_dropout 0.0 --log_steps 300 \
--model_path /mnt/jiangjinhao/hg_face/roberta-base --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/data/ \
--data_name .super.subgraph.nsm.input.jsonl --checkpoint_dir ../outputs/retriever_training_3/ --batch_size 1 --gradient_accumulation_steps 20 \
--test_batch_size 20 --num_step 4 --entity_dim 200 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 2 --encode_type \
--experiment_name webqsp_nsm_retriever_roberta_bs_20_wo_drop_ed_200_plm_init_each_steps_loss --eps 0.95 --num_epoch 100 \
--use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ../data/webqsp/data/super_sg_for_retriever.cache --overwrite_cache --init_same_as_plm