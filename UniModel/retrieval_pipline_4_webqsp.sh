#!/bin/bash
# 流水线的NSM检索+NSM推理
# 1.抽取最短可达路径，并通过一定的额外主题实体获得限制关系，最终得到所有可达路径及其召回的精度。
# train:  Samples with paths-3145/3178-0.99 Const rels recall-1.000
# dev:    Samples with paths-267/275-0.97   Const rels recall-1.000
# test:   Samples with paths-1618/1639-0.99 Const rels recall-0.999
python3 s0_extract_weak_super_relations.py --dense_kg_source virtuoso --extra_hop_flag --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--input_path ./data/webqsp/all_data.jsonl --output_path ./data/webqsp/new.all.shortest.paths.2hop.jsonl \
--ids_path ./data/webqsp/SPLIT.qid.npy --max_hop 2 --max_num_processes 60

# 2.构建训练relation retriever的训练数据
# valid sample: 5030/5030-0.94 path count: [0, 50, 3.03]
python extract_valid_weak_paths.py --min_precision 0.1 --filter_order 0 \
--input_path ./data/webqsp/new.all.shortest.paths.2hop.jsonl --output_path ./data/webqsp/new.all.shortest.paths.2hop.min_pre.0.1.jsonl

# neg_50_0.1 train：19290 dev：1653
python s1_construct_relation_retrieval_training_data.py --dense_kg_source virtuoso --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--input_path ./data/webqsp/new.all.shortest.paths.2hop.min_pre.0.1.jsonl --output_path ./data/webqsp/new.all.rels.retri.data.neg_50.0.1.jsonl \
--max_hop 3 --num_neg 50 --max_num_processes 60
python extract_relation_retriever_training_data.py --split_qid_path ./data/webqsp/SPLIT.qid.npy --split_list train dev \
--pn_pairs_path ./data/webqsp/new.all.rels.retri.data.neg_50.0.1.jsonl --output_path ./data/webqsp/new.SPLIT.rels.retri.data.neg_50.0.1.csv

# 3.训练relation retriever
# dev hits@1 75.19
CUDA_VISIBLE_DEVICES=1,2,3 python3 -W ignore -m torch.distributed.launch \
--nproc_per_node 3 --master_port 1237 ./retriever/train.py --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
--train_file ./data/webqsp/new.train.rels.retri.data.neg_50.0.1.csv \
--eval_file ./data/webqsp/new.dev.rels.retri.data.neg_50.0.1.csv \
--output_dir ./retriever/results/new_webqsp_rel_retri_1 --per_device_train_batch_size 10 --gradient_accumulation_steps 2 \
--learning_rate 5e-5 --do_train --num_train_epochs 15 --do_eval --evaluation_strategy steps --eval_steps 300 \
--save_steps 300 --save_total_limit 1 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05
# 占卡程序
CUDA_VISIBLE_DEVICES=0 python3 ./retriever_for_gpu/train.py --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
--train_file ./data/webqsp/train.rels.retri.data.neg_50.0.1.csv \
--eval_file ./data/webqsp/dev.rels.retri.data.neg_50.0.1.csv \
--output_dir ./retriever/results/debug --per_device_train_batch_size 10 --gradient_accumulation_steps 1 \
--learning_rate 5e-5 --do_train --num_train_epochs 150000 --do_eval --evaluation_strategy steps --eval_steps 20000 \
--save_steps 20000 --save_total_limit 1 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05

# 将simcse对应的模型转换为huggingface通用的模型结构
python ./retriever/simcse_to_huggingface.py --path ./retriever/results/new_webqsp_rel_retri_1/

# 4.构建主题实体周围三跳内的抽象子图，使用relation retriever过滤非常不相关关系
# train:  (8,1001,694)  (ans:0.98-3145/3178 rel_acc:0.97 rel_recall:0.99 rel_valid:0.96-3063/3178)
# dev:    (22,1000,708) (ans:0.97-267/275   rel_acc:0.97 rel_recall:0.98 rel_valid:0.95-262/275)
# test:   (0,1001,694)   (ans:0.98-1602/1639 rel_acc:0.95 rel_recall:0.97 rel_valid:0.97-1596/1639)
python3 s2_extract_abstract_subgraph.py --dense_kg_source virtuoso --max_num_processes 40 --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--ori_path ./data/webqsp/all_data.jsonl --input_path ./data/webqsp/new.all.shortest.paths.2hop.jsonl \
--output_path ./data/webqsp/all.abs.sg.topk.15.min_sim.0.max_tris.1000.new_rel_retri_1.jsonl \
--model_path ./retriever/results/new_webqsp_rel_retri_1/ --device 0 1 2 3 --topk 15 --max_num_triples 1000 \
--filter_score 0.0 --max_hop 3 --num_pro_each_device 4
python extract_abstract_subgraph.py --split_qid_path ./data/webqsp/SPLIT.qid.npy --split_list train dev test \
--abs_sg_path ./data/webqsp/all.abs.sg.topk.15.min_sim.0.max_tris.1000.new_rel_retri_1.jsonl \
--output_path ./data/webqsp/SPLIT.abs.sg.topk.15.min_sim.0.max_tris.1000.new_rel_retri_1.jsonl \
--all_output_path ./data/webqsp/all.abs.sg.topk.15.min_sim.0.max_tris.1000.new_rel_retri_1.jsonl \
--all_shortest_path ./data/webqsp/new.all.shortest.paths.2hop.jsonl \
--ori_path ./data/webqsp/all_data.jsonl \
--min_path_precision 0.1 --filter_order 0 --num_process 16

# 5.将抽象子图的关系和节点进行映射
python3 map_abs_kg_2_global_id.py --input_path data/webqsp/SPLIT.abs.sg.topk.15.min_sim.0.max_tris.1000.new_rel_retri_1.jsonl \
--output_path data/webqsp/ --output_prefix new_abs_r1_ --split_list train dev test

# 6.将抽象子图转换为NSM的输入形式
python3 convert_abs_to_NSM_format.py \
--input_path data/webqsp/SPLIT.abs.sg.topk.15.min_sim.0.max_tris.1000.new_rel_retri_1.jsonl \
--output_path data/webqsp/SPLIT.abs.sg.topk.15.min_sim.0.max_tris.1000.new_rel_retri_1.nsm.json \
--all_output_path data/webqsp/all.abs.sg.topk.15.min_sim.0.max_tris.1000.new_rel_retri_1.nsm.json \
--kg_map_path data/webqsp/ --kg_map_prefix new_abs_r1_ --max_nodes 1000 --split_list train dev test

# 7.训练NSM retriever
# retrieval: pretrain, transfer, fix plm 不用pooler层
# 66.43/83.27 67.90/82.48
CUDA_VISIBLE_DEVICES=2 python3 main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/new_webqsp_rel_retri_1/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abs.sg.topk.15.min_sim.0.max_tris.1000.new_rel_retri_1.nsm.json \
--checkpoint_dir ../retriever_ckpt/webqsp_nsm/ --experiment_name retriever_p_t_wo_pooler \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 400 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
 --eps 0.95 --num_epoch 100 --patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/new_abs_r1_sg_for_retri.cache \
--relation2id new_abs_r1_relations.txt --entity2id new_abs_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm

# 8.基于第4步抽取出的子图 使用NSM retriever检索
# train:  (0,1387,187)   (valid subgraph: 3145/3178-0.99 ans:0.98 rel:0.95)
# dev:    (3,920,178)   (valid subgraph: 267/275-0.97 ans:0.97 rel:0.95)
# test:   (0,854,158)   (valid subgraph: 1637/1639-0.999 ans:0.96 rel:0.92)
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--ori_path data/webqsp/all_data.jsonl \
--input_path data/webqsp/all.abs.sg.topk.15.min_sim.0.max_tris.1000.new_rel_retri_1.nsm.json \
--output_path data/webqsp/all.inst.sg.new_rel_retri_1.15-300.jsonl \
--model_path retriever_ckpt/webqsp_nsm/retriever_p_t_wo_pooler-h1.ckpt \
--arg_path retriever_ckpt/webqsp_nsm/retriever_p_t_wo_pooler-args.json \
--relation2id_path data/webqsp/new_abs_r1_relations.txt \
--entity2id_path data/webqsp/new_abs_r1_entities.txt \
--device 5 6 7 --final_topk 15 --max_deduced_triples 300 --max_hop 3 --num_pro_each_device 5
python extract_instantiate_subgraph.py --split_qid_path ./data/webqsp/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/webqsp/all.inst.sg.new_rel_retri_1.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.inst.sg.new_rel_retri_1.15-300.jsonl --num_process 1

# 9.将实例化子图的关系和节点进行映射
python3 map_kg_2_global_id.py --input_path ./data/webqsp/SPLIT.inst.sg.new_rel_retri_1.15-300.jsonl \
--ori_path data/webqsp/SPLIT.jsonl --output_path data/webqsp/ --output_prefix new_reason_r1_ \
--split_list train dev test
# 使用T5直接进行检索
python3 map_kg_2_global_id.py --input_path ./data/webqsp/SPLIT.sg.topk.8.min_sim.0.max_tris.300.t5_base.jsonl \
--ori_path data/webqsp/SPLIT.jsonl --output_path data/webqsp/ --output_prefix reason_t5_base_0 \
--split_list train dev test

# 10.转换为NSM输入形式
## 添加限制实体的逆关系
python3 convert_inst_to_NSM_format.py --original_path data/webqsp/SPLIT.jsonl --max_nodes 700 \
--input_path ./data/webqsp/SPLIT.inst.sg.new_rel_retri_1.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.reason.sg.max_tris.700.const_reverse.json \
--kg_map_path ./data/webqsp/ --kg_map_prefix new_reason_r1_ --split_list train dev test --add_constraint_reverse
# 使用T5直接进行检索
python3 convert_inst_to_NSM_format.py --original_path data/webqsp/SPLIT.jsonl --max_nodes 300 \
--input_path ./data/webqsp/SPLIT.sg.topk.8.min_sim.0.max_tris.300.t5_base.jsonl \
--output_path ./data/webqsp/SPLIT.reason.sg.t5_base.max_tris.300.const_reverse.json \
--kg_map_path ./data/webqsp/ --kg_map_prefix reason_t5_base_0 --split_list train dev test --add_constraint_reverse

# 11.跑NSM Reasoner (注意加载数据时，answer映射问题)
# original NSN
CUDA_VISIBLE_DEVICES=1 python3 ./nsm_retriever/main_nsm.py --model_name gnn \
--data_folder ./dataset/webqsp/webqsp_NSM/ --data_name _simple.json \
--instruct_model LSTM \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ \
--batch_size 20 --test_batch_size 40 --num_step 3 --entity_dim 50 --word_dim 300 \
--kg_dim 100 --kge_dim 100 --eval_every 2 --encode_type --experiment_name ori_nsm --eps 0.95 --num_epoch 200 \
--use_self_loop --lr 5e-4 --word2id vocab_new.txt --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id relations.txt --entity2id entities.txt
# original NSM on our retrieved data 67.7
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--instruct_model LSTM \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ \
--batch_size 40 --test_batch_size 500 --num_step 3 --entity_dim 50 --word_dim 300 --eval_every 2 \
--kg_dim 100 --kge_dim 100 --eval_every 1 --encode_type --experiment_name nsm --eps 0.95 --num_epoch 200 \
--use_self_loop --lr 5e-4 --word2id vocab_new.txt --word_emb_file word_emb_300d.npy --loss_type kl --patience 40 \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt
# original NSM 51.19/69.66 50.93/69.53 使用T5直接进行检索
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.t5_base.max_tris.300.const_reverse.json \
--instruct_model LSTM \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ \
--batch_size 40 --test_batch_size 500 --num_step 4 --entity_dim 50 --word_dim 300 \
--kg_dim 100 --kge_dim 100 --eval_every 1 --encode_type --experiment_name nsm_1 --eps 0.95 --num_epoch 100 \
--use_self_loop --lr 5e-4 --word2id vocab_new.txt --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_t5_base_0relations.txt --entity2id reason_t5_base_0entities.txt

# pretrain, transfer, fix plm
## transfer best f1 71.45/76.36 72.77/77.64
CUDA_VISIBLE_DEVICES=1 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_wo_pooler-f1.ckpt \
--model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--relation_model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_wo_pooler_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## transfer best f1 71.05/75.43 71.53/75.25 使用T5直接进行检索
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_wo_pooler-f1.ckpt \
--model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--relation_model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.t5_base.max_tris.300.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_wo_pooler_f1_1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_t5_base_0relations.txt --entity2id reason_t5_base_0entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## transfer best h1 72.23/77.03 72.12/77.83
CUDA_VISIBLE_DEVICES=1 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_wo_pooler-h1.ckpt \
--model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--relation_model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_wo_pooler_h1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## transfer best h1 72.09/74.69 70.70/73.77 使用T5直接进行检索
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_wo_pooler-h1.ckpt \
--model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--relation_model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.t5_base.max_tris.300.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_wo_pooler_h1_1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_t5_base_0relations.txt --entity2id reason_t5_base_0entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm

# pretrain, transfer, question updating
## transfer best f1 72.77/77.34 72.00/77.95
CUDA_VISIBLE_DEVICES=2 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_wo_pooler-f1.ckpt \
--model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--relation_model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_wo_pooler_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate
## transfer best h1 72.71/77.46 72.50/76.97
CUDA_VISIBLE_DEVICES=2 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_wo_pooler-h1.ckpt \
--model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--relation_model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_wo_pooler_h1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate

# pretrain, transfer, update plm
## transfer best f1 72.50/76.18 72.72/76.66
CUDA_VISIBLE_DEVICES=1 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_wo_pooler-f1.ckpt \
--model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--relation_model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_r_wo_pooler_f1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt \
--simplify_model
## transfer best h1 72.39/76.36 72.83/77.34
CUDA_VISIBLE_DEVICES=1 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm/retriever_p_t_wo_pooler-h1.ckpt \
--model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--relation_model_path ./retriever/results/new_webqsp_rel_retri_1/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.max_tris.700.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/webqsp_rea/ --experiment_name reason_p_t_q_r_wo_pooler_h1 \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt \
--simplify_model
