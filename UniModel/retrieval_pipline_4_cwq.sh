# 流水线的NSM检索+NSM推理 更换数据集为CWQ
# 1.抽取最短可达路径，并通过一定的额外主题实体获得限制关系，最终得到所有可达路径及其召回的精度。
#train: Samples with paths-27441/27639-0.99 Const rels recall-0.995
#dev: Samples with paths-3498/3519-0.99 Const rels recall-0.997
#test: Samples with paths-3499/3531-0.99 Const rels recall-0.994
python3 s0_extract_weak_super_relations.py --dense_kg_source virtuoso --extra_hop_flag --task_name cwq \
--sparse_kg_source_path ./data/cwq/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/cwq/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/cwq/graph_data/ent2id.pickle --sparse_rel2id_path ./data/cwq/graph_data/rel2id.pickle \
--input_path ./data/cwq/all_data.jsonl --output_path ./data/cwq/all.shortest.paths.4hop.jsonl --ids_path ./data/cwq/SPLIT.qid.npy \
--max_hop 4 --max_num_processes 60 --overwrite

# 2.构建训练relation retriever的训练数据
# valid sample: 34438/34438-1.00 path count: [1, 1415, 12.10]
python extract_valid_weak_paths.py --min_precision 0.05 --filter_order 0 \
--input_path ./data/cwq/all.shortest.paths.4hop.jsonl --output_path ./data/cwq/all.shortest.paths.4hop.min_precision.0.05.jsonl

# train: 601705 dev: 92973
python s1_construct_relation_retrieval_training_data.py --dense_kg_source virtuoso --task_name cwq \
--sparse_kg_source_path ./data/cwq/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/cwq/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/cwq/graph_data/ent2id.pickle --sparse_rel2id_path ./data/cwq/graph_data/rel2id.pickle \
--input_path ./data/cwq/all.shortest.paths.4hop.min_precision.0.05.jsonl --output_path ./data/cwq/all.rels.retri.data.neg_50.0.05.jsonl \
--max_hop 4 --num_neg 50 --max_num_processes 50
python extract_relation_retriever_training_data.py --split_qid_path ./data/cwq/SPLIT.qid.npy --split_list train dev \
--pn_pairs_path ./data/cwq/all.rels.retri.data.neg_50.0.05.jsonl --output_path ./data/cwq/SPLIT.rels.retri.data.neg_50.0.05.csv

# 3.训练relation retriever
# dev hits1: 81.97
CUDA_VISIBLE_DEVICES=1,2,3,4,5 python3 -W ignore -m torch.distributed.launch \
--nproc_per_node 5 --master_port 1238 ./retriever/train.py --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
--train_file ./data/cwq/train.rels.retri.data.neg_50.0.05.csv --eval_file ./data/cwq/dev.rels.retri.data.neg_50.0.05.csv \
--output_dir ./retriever/results/cwq_rel_retri_0 --per_device_train_batch_size 8 --gradient_accumulation_steps 2 \
--learning_rate 5e-5 --do_train --num_train_epochs 15 --do_eval --evaluation_strategy steps --eval_steps 1500 \
--save_steps 1500 --save_total_limit 1 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05

# 将simcse对应的模型转换为huggingface通用的模型结构
python ./retriever/simcse_to_huggingface.py --path ./retriever/results/cwq_rel_retri_0/

# 4.构建主题实体周围三跳内的抽象子图，使用relation retriever过滤非常不相关关系 (记得注释掉稀疏检索中最大src_idx)
# train: (2,2001,1381)(ans:0.99-27320/27639 rel://)
# dev:   (6,2000,1437)(ans:0.98-3454/3519 rel://)
# test:  (0,2000,1432)(ans:0.98-3446/3531 rel://)
python3 s2_extract_abstract_subgraph.py --dense_kg_source virtuoso --max_num_processes 40 --task_name cwq \
--sparse_kg_source_path ./data/cwq/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/cwq/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/cwq/graph_data/ent2id.pickle --sparse_rel2id_path ./data/cwq/graph_data/rel2id.pickle \
--ori_path ./data/cwq/all_data.jsonl --input_path ./data/cwq/all.shortest.paths.4hop.jsonl \
--output_path ./data/cwq/all.abs.sg.topk.10.min_score.0.max_tris.2000.rel_retri_0.jsonl \
--model_path ./retriever/results/cwq_rel_retri_0/ --device 0 1 3 6 \
--topk 10 --max_num_triples 2000 --filter_score 0.0 --max_hop 4 --num_pro_each_device 4
python extract_abstract_subgraph.py --split_qid_path ./data/cwq/SPLIT.qid.npy --split_list train dev test \
--abs_sg_path ./data/cwq/all.abs.sg.topk.10.min_score.0.max_tris.2000.rel_retri_0.jsonl \
--output_path ./data/cwq/SPLIT.abs.sg.topk.10.min_score.0.max_tris.2000.rel_retri_0.jsonl \
--all_output_path ./data/cwq/all.abs.sg.topk.10.min_score.0.max_tris.2000.rel_retri_0.jsonl \
--all_shortest_path ./data/cwq/all.shortest.paths.4hop.jsonl \
--ori_path ./data/cwq/all_data.jsonl \
--min_path_precision 0.05 --filter_order 3 --num_process 16
# 如果要使用性能评测：kernprof -l -v 并注意添加注释
python extract_abstract_subgraph.py --split_qid_path ./data/cwq/SPLIT.qid.npy --split_list train dev test \
--abs_sg_path ./data/cwq/all.abstract.sg.topk.10.min_score.0.max_tris.1300.rel_retri_0.jsonl \
--output_path ./data/cwq/SPLIT.abstract.sg.topk.10.min_score.0.max_tris.1300.rel_retri_0.jsonl \
--all_output_path ./data/cwq/all.abstract.sg.topk.10.min_score.0.max_tris.1300.rel_retri_0.jsonl --num_process 24


# 5.将抽象子图的关系和节点进行映射
python3 map_abs_kg_2_global_id.py --input_path data/cwq/SPLIT.abs.sg.topk.10.min_score.0.max_tris.2000.rel_retri_0.jsonl \
--output_path data/cwq/ --output_prefix new_abs_r0_ --split_list train dev test


# 6.将抽象子图转换为NSM的输入形式
python3 convert_abs_to_NSM_format.py \
--input_path data/cwq/SPLIT.abs.sg.topk.10.min_score.0.max_tris.2000.rel_retri_0.jsonl \
--output_path data/cwq/SPLIT.abs.sg.max_tris.2000.nsm.json \
--all_output_path data/cwq/all.abs.sg.max_tris.2000.nsm.json \
--kg_map_path data/cwq/ --kg_map_prefix new_abs_r0_ --max_nodes 2000 --split_list train dev test


# 7.训练NSM retriever
# 45.00/75.93 56.71/75.02
CUDA_VISIBLE_DEVICES=4 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--model_path ./retriever/results/cwq_rel_retri_0/ --agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .abs.sg.max_tris.2000.nsm.json \
--checkpoint_dir ./retriever_ckpt/cwq_nsm_retri/ --batch_size 80 --gradient_accumulation_steps 1 \
--test_batch_size 100 --num_step 4 --entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--experiment_name new_retriever_nsm_0 --eps 0.95 --num_epoch 100 --patience 10 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ./data/cwq/new_abs_r0_sg_for_retri.cache --overwrite_cache \
--relation2id new_abs_r0_relations.txt --entity2id new_abs_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm


# 8.基于第4步抽取出的子图 使用NSM retriever检索 topk 15
# train: (0,1817,251) sg: 27320/27639-0.99 ans: 0.93
# dev:   (0,1755,266) sg: 3454/3519-0.98 ans: 0.90
# test:  (0,1375,251) sg: 3508/3531-0.99 ans: 0.88
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name cwq \
--sparse_kg_source_path ./data/cwq/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/cwq/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/cwq/graph_data/ent2id.pickle --sparse_rel2id_path ./data/cwq/graph_data/rel2id.pickle \
--ori_path data/cwq/all_data.jsonl \
--input_path data/cwq/all.abs.sg.max_tris.2000.nsm.json \
--output_path data/cwq/all.ins.sg.15-500.jsonl \
--model_path retriever_ckpt/cwq_nsm_retri/new_retriever_nsm_0-f1.ckpt \
--arg_path retriever_ckpt/cwq_nsm_retri/new_retriever_nsm_0-args.json \
--relation2id_path data/cwq/new_abs_r0_relations.txt \
--entity2id_path data/cwq/new_abs_r0_entities.txt \
--device 0 1 5 6 --final_topk 15 --max_deduced_triples 500 --max_hop 4 --num_pro_each_device 3
python extract_instantiate_subgraph.py --split_qid_path ./data/cwq/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/cwq/all.ins.sg.15-500.jsonl \
--output_path ./data/cwq/SPLIT.ins.sg.15-500.jsonl --num_process 8

# 9.转换为NSM输入形式
python3 map_kg_2_global_id.py --input_path ./data/cwq/SPLIT.ins.sg.15-500.jsonl \
--ori_path data/cwq/SPLIT.jsonl --output_path data/cwq/ --output_prefix new_reason_r0_ \
--split_list train dev test

python3 convert_inst_to_NSM_format.py --original_path data/cwq/SPLIT.jsonl --max_nodes 1000 \
--input_path ./data/cwq/SPLIT.ins.sg.15-500.jsonl \
--output_path ./data/cwq/SPLIT.reason.sg.15-500.max_tris.1000.const_reverse.json \
--kg_map_path ./data/cwq/ --kg_map_prefix new_reason_r0_ --split_list train dev test --add_constraint_reverse


# 10.跑NSM Reasoner
# 原始LSTM效果
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--instruct_model LSTM \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 20 --test_batch_size 500 --num_step 4 --entity_dim 50 --word_dim 300 \
--kg_dim 100 --kge_dim 100 --eval_every 2 --encode_type --experiment_name cwq_nsm --eps 0.95 --num_epoch 100 \
--use_self_loop --lr 5e-4 --word2id vocab_new.txt --word_emb_file word_emb_300d.npy --loss_type kl --patience 40 --reason_kb \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt

# reasoning: pretrain, transfer, fix plm, wo pooler f1
## transfer best f1 49.97/52.63 50.23/51.83
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm_retri/new_retriever_nsm_0-f1.ckpt \
--model_path ./retriever/results/cwq_rel_retri_0/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_wo_pooler_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 4 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/cwq/new_ins_r0_sg_for_reason.cache --overwrite_cache \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## transfer best h1 49.13/52.07 49.67/51.51
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm_retri/new_retriever_nsm_0-h1.ckpt \
--model_path ./retriever/results/cwq_rel_retri_0/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_wo_pooler_h1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 4 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/cwq/new_ins_r0_sg_for_reason.cache \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm

# reasoning: pretrain, transfer, question updating, wo pooler f1
## transfer best f1 50.05/52.42 51.32/53.31
CUDA_VISIBLE_DEVICES=2 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm_retri/new_retriever_nsm_0-f1.ckpt \
--model_path ./retriever/results/cwq_rel_retri_0/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_q_wo_pooler_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 4 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/cwq/new_ins_r0_sg_for_reason.cache \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate
## transfer best h1 46.72/50.80 51.49/52.66
CUDA_VISIBLE_DEVICES=2 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm_retri/new_retriever_nsm_0-h1.ckpt \
--model_path ./retriever/results/cwq_rel_retri_0/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_q_wo_pooler_h1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 4 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/cwq/new_ins_r0_sg_for_reason.cache \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate

# reasoning: pretrain, transfer, update plm, wo pooler f1
## transfer best f1 49.90/52.63 50.94/52.72
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm_retri/new_retriever_nsm_0-f1.ckpt \
--model_path ./retriever/results/cwq_rel_retri_0/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_q_r_wo_pooler_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 4 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/cwq/new_ins_r0_sg_for_reason.cache \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt \
--simplify_model
## transfer best h1 50.12/51.83 51.41/52.10
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/cwq_nsm_retri/new_retriever_nsm_0-h1.ckpt \
--model_path ./retriever/results/cwq_rel_retri_0/ \
--relation_model_path ./retriever/results/cwq_rel_retri_0/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/cwq/ --data_name .reason.sg.15-500.max_tris.1000.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/cwq_rea/ --experiment_name reason_p_t_q_r_wo_pooler_h1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 4 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--data_cache ./data/cwq/new_ins_r0_sg_for_reason.cache \
--relation2id new_reason_r0_relations.txt --entity2id new_reason_r0_entities.txt \
--simplify_model