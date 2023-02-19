# FB15K-237
# 1.抽取最短可达路径，并通过一定的额外主题实体获得限制关系，最终得到所有可达路径及其召回的精度。
python3 s0_extract_weak_super_relations.py --dense_kg_source virtuoso --task_name kgc \
--sparse_kg_source_path ./data/FB15K-237/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/FB15K-237/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/FB15K-237/graph_data/ent2id.pickle --sparse_rel2id_path ./data/FB15K-237/graph_data/rel2id.pickle \
--input_path ./data/FB15K-237/all_data.jsonl --output_path ./data/FB15K-237/all.shortest.paths.3hop.jsonl \
--ids_path ./data/FB15K-237/SPLIT.qid.npy --max_hop 3 --max_num_processes 40 --overwrite

#train: Samples with paths-92743/93372-0.99 Const rels recall-1.000
#dev: Samples with paths-12055/12079-1.00 Const rels recall-1.000
#test: Samples with paths-13679/13737-1.00 Const rels recall-1.000
python3 s0_extract_weak_super_relations.py --dense_kg_source virtuoso --task_name kgc \
--sparse_kg_source_path ./data/FB15K-237/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/FB15K-237/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/FB15K-237/graph_data/ent2id.pickle --sparse_rel2id_path ./data/FB15K-237/graph_data/rel2id.pickle \
--input_path ./data/FB15K-237/all_data_aggregate.jsonl --output_path ./data/FB15K-237/all.shortest.paths.3hop.aggre.jsonl \
--ids_path ./data/FB15K-237/SPLIT.aggregate.qid.npy --max_hop 3 --max_num_processes 60 --overwrite

python extract_valid_weak_paths.py --min_precision 0.05 \
--input_path ./data/FB15K-237/all.shortest.paths.3hop.aggre.jsonl \
--output_path ./data/FB15K-237/all.shortest.paths.3hop.aggre.min_precision.0.05.jsonl \
--filter_order 1

# valid sample: 101985/118477-0.86 path count: [0, 185, 2.98]
python extract_valid_weak_paths.py --min_precision 0.1 \
--input_path ./data/FB15K-237/all.shortest.paths.3hop.aggre.jsonl \
--output_path ./data/FB15K-237/all.shortest.paths.3hop.aggre.min_precision.0.1.jsonl \
--filter_order 1


# 2.构建训练relation retriever的训练数据
# train: 793548 dev: 140484
python s1_construct_relation_retrieval_training_data.py --dense_kg_source virtuoso --task_name kgc \
--sparse_kg_source_path ./data/FB15K-237/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/FB15K-237/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/FB15K-237/graph_data/ent2id.pickle --sparse_rel2id_path ./data/FB15K-237/graph_data/rel2id.pickle \
--input_path ./data/FB15K-237/all.shortest.paths.3hop.aggre.min_precision.0.1.jsonl --output_path ./data/FB15K-237/all.rels.retri.data.neg_5.0.1.aggre.jsonl \
--max_hop 3 --num_neg 5 --max_num_processes 80
python extract_relation_retriever_training_data.py --split_qid_path ./data/FB15K-237/SPLIT.aggregate.qid.npy --split_list train dev \
--pn_pairs_path ./data/FB15K-237/all.rels.retri.data.neg_5.0.1.aggre.jsonl --output_path ./data/FB15K-237/SPLIT.rels.retri.data.neg_5.0.1.aggre.csv

# 3.训练relation retriever
# dev hits1: 77.98
CUDA_VISIBLE_DEVICES=1,4,5,6 python3 -W ignore -m torch.distributed.launch \
--nproc_per_node 4 --master_port 1237 ./retriever/train.py --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
--train_file ./data/FB15K-237/train.rels.retri.data.neg_5.0.1.aggre.csv --eval_file ./data/FB15K-237/dev.rels.retri.data.neg_5.0.1.aggre.csv \
--output_dir ./retriever/results/FB15K_rel_retri_0 --per_device_train_batch_size 50 --gradient_accumulation_steps 10 \
--learning_rate 5e-5 --do_train --num_train_epochs 5 --do_eval --evaluation_strategy steps --eval_steps 350 \
--save_steps 350 --save_total_limit 1 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05

# 4.构建主题实体周围三跳内的抽象子图，使用relation retriever过滤非常不相关关系
# train:  (2,234,40)  (ans:1-86835/86835)
# dev:    (4,223,42) (ans:0.67-2019/3034)
# test:   (0,219,27)   (ans:0.46-1455/3134)
python3 s2_extract_abstract_subgraph.py --dense_kg_source virtuoso --max_num_processes 40 --task_name kgc_tmp \
--sparse_kg_source_path ./data/FB15K-237/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/FB15K-237/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/FB15K-237/graph_data/ent2id.pickle --sparse_rel2id_path ./data/FB15K-237/graph_data/rel2id.pickle \
--ori_path ./data/FB15K-237/all_data_aggregate.jsonl --input_path ./data/FB15K-237/all.shortest.paths.3hop.aggre.jsonl \
--output_path ./data/FB15K-237/all.abstract.sg.topk.10.min_score.0.max_tris.1000.rel_retri_0.aggre.jsonl \
--model_path ./retriever/results/FB15K_rel_retri_0/ --device 1 4 5 6 --topk 10 --max_num_triples 1000 \
--filter_score 0.0 --filter_method mixture --max_hop 3 --min_path_precision 0.5 --num_pro_each_device 5 --filter_order 2
python3 s2_extract_abstract_subgraph.py --dense_kg_source virtuoso --max_num_processes 50 --task_name kgc_tmp \
--sparse_kg_source_path ./data/FB15K-237/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/FB15K-237/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/FB15K-237/graph_data/ent2id.pickle --sparse_rel2id_path ./data/FB15K-237/graph_data/rel2id.pickle \
--ori_path ./data/FB15K-237/all_data_aggregate.jsonl --input_path ./data/FB15K-237/all.shortest.paths.3hop.aggre.jsonl \
--output_path ./data/FB15K-237/all.abstract.sg.aggre.jsonl \
--model_path ./retriever/results/FB15K_rel_retri_0/ --device 1 \
--max_hop 3 --min_path_precision 0.1 --num_pro_each_device 50 --filter_order 2 --not_filter_rels

python extract_abstract_subgraph.py --split_qid_path ./data/FB15K-237/SPLIT.aggregate.qid.npy --split_list train dev test \
--abs_sg_path ./data/FB15K-237/all.abstract.sg.aggre.jsonl \
--output_path ./data/FB15K-237/SPLIT.abstract.sg.aggre.jsonl \
--all_output_path ./data/FB15K-237/all.abstract.sg.aggre.jsonl \
--num_process 50

# 5.将抽象子图的关系和节点进行映射
python3 map_abs_kg_2_global_id.py --input_path data/FB15K-237/SPLIT.abstract.sg.topk.10.min_score.0.max_tris.1000.rel_retri_0.jsonl \
--output_path data/FB15K-237/ --output_prefix abs_r0_ --split_list train dev test

# 6.将抽象子图转换为NSM的输入形式
python3 convert_abs_to_NSM_format.py \
--input_path data/FB15K-237/SPLIT.abstract.sg.topk.10.min_score.0.max_tris.1000.rel_retri_0.jsonl \
--output_path data/FB15K-237/SPLIT.abstract.sg.topk.10.min_score.0.max_tris.1000.rel_retri_0.nsm.json \
--all_output_path data/FB15K-237/all.abstract.sg.topk.10.min_score.0.max_tris.1000.rel_retri_0.nsm.json \
--kg_map_path data/FB15K-237/ --kg_map_prefix abs_r0_ --max_nodes 1000 --split_list train dev test

# 7.训练NSM retriever
# best h1: F1:45.59 H1:22.14 -> 76.16 best f1: F1:46.22 H1:21.64 -> 65.63
CUDA_VISIBLE_DEVICES=3 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 100 \
--model_path ../retriever/results/WN18RR_rel_retri_0/ --agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ../data/FB15K-237/ --data_name .abstract.sg.topk.10.min_score.0.max_tris.1000.rel_retri_0.nsm.json \
--checkpoint_dir ../retriever_ckpt/WN18RR_nsm_retri_v0/ \
--batch_size 800 --gradient_accumulation_steps 1 --test_batch_size 4000 --num_step 3 \
--entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--experiment_name retriever_nsm_0 --eps 0.95 --num_epoch 100 --patience 20 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--loss_type kl --reason_kb --data_cache ../data/FB15K-237/abs_r0_sg_for_retri.cache \
--overwrite_cache --relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt



                                                    #######    WN18RR   #######
# 1.抽取最短可达路径
#train: Samples with paths-62547/62547-1.00 Const rels recall-1.000
#dev: Samples with paths-1930/2916-0.66 Const rels recall-1.000
#test: Samples with paths-1974/3022-0.65 Const rels recall-1.000
python3 s0_extract_weak_super_relations.py --dense_kg_source virtuoso --task_name kgc \
--sparse_kg_source_path ./data/WN18RR/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/WN18RR/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/WN18RR/graph_data/ent2id.pickle --sparse_rel2id_path ./data/WN18RR/graph_data/rel2id.pickle \
--input_path ./data/WN18RR/all_data_aggregate.jsonl --output_path ./data/WN18RR/all.shortest.paths.3hop.aggre.jsonl \
--ids_path ./data/WN18RR/SPLIT.aggregate.qid.npy --max_hop 3 --max_num_processes 40 --overwrite

python3 s0_extract_weak_super_relations.py --dense_kg_source virtuoso --task_name kgc \
--sparse_kg_source_path ./data/WN18RR/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/WN18RR/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/WN18RR/graph_data/ent2id.pickle --sparse_rel2id_path ./data/WN18RR/graph_data/rel2id.pickle \
--input_path ./data/WN18RR/all_data.jsonl --output_path ./data/WN18RR/all.shortest.paths.3hop.jsonl \
--ids_path ./data/WN18RR/SPLIT.qid.npy --max_hop 3 --max_num_processes 40 --overwrite

python extract_valid_weak_paths.py --min_precision 0.01 \
--input_path ./data/WN18RR/all.shortest.paths.3hop.jsonl \
--output_path ./data/WN18RR/all.shortest.paths.3hop.min_precision.0.01.jsonl

python extract_valid_weak_paths.py --min_precision 0.01 \
--input_path ./data/WN18RR/all.shortest.paths.3hop.aggre.jsonl \
--output_path ./data/WN18RR/all.shortest.paths.3hop.aggre.min_precision.0.01.jsonl


# 2.构建训练relation retriever的训练数据
# train: 107861 dev: 5073
python s1_construct_relation_retrieval_training_data.py --dense_kg_source virtuoso --task_name kgc \
--sparse_kg_source_path ./data/WN18RR/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/WN18RR/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/WN18RR/graph_data/ent2id.pickle --sparse_rel2id_path ./data/WN18RR/graph_data/rel2id.pickle \
--input_path ./data/WN18RR/all.shortest.paths.3hop.aggre.min_precision.0.01.jsonl --output_path ./data/WN18RR/all.rels.retri.data.neg_10.0.01.aggre.jsonl \
--max_hop 3 --num_neg 10 --max_num_processes 80
python extract_relation_retriever_training_data.py --split_qid_path ./data/WN18RR/SPLIT.aggregate.qid.npy --split_list train dev \
--pn_pairs_path ./data/WN18RR/all.rels.retri.data.neg_10.0.01.aggre.jsonl --output_path ./data/WN18RR/SPLIT.rels.retri.data.neg_10.0.01.aggre.csv

# 3.训练relation retriever
# dev hits1: 77.82
CUDA_VISIBLE_DEVICES=6,7 python3 -W ignore -m torch.distributed.launch \
--nproc_per_node 2 --master_port 1236 ./retriever/train.py --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
--train_file ./data/WN18RR/train.rels.retri.data.neg_10.0.01.aggre.csv --eval_file ./data/WN18RR/dev.rels.retri.data.neg_10.0.01.aggre.csv \
--output_dir ./retriever/results/WN18RR_rel_retri_0 --per_device_train_batch_size 200 --gradient_accumulation_steps 1 \
--learning_rate 1e-5 --do_train --num_train_epochs 10 --do_eval --evaluation_strategy steps --eval_steps 100 \
--save_steps 100 --save_total_limit 1 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05
python ./retriever/simcse_to_huggingface.py --path ./retriever/results/WN18RR_rel_retri_0/

# 4.构建主题实体周围三跳内的抽象子图，使用relation retriever过滤非常不相关关系
# train:  (4,288,67)  (ans:1-86835/86835)
# dev:    (4,267,66) (ans:0.67-2019/3034)
# test:   (0,288,54)   (ans:0.65-2043/3134)
python3 s2_extract_abstract_subgraph.py --dense_kg_source virtuoso --max_num_processes 40 --task_name kgc \
--sparse_kg_source_path ./data/WN18RR/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/WN18RR/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/WN18RR/graph_data/ent2id.pickle --sparse_rel2id_path ./data/WN18RR/graph_data/rel2id.pickle \
--ori_path ./data/WN18RR/all_data_aggregate.jsonl --input_path ./data/WN18RR/all.shortest.paths.3hop.aggre.jsonl \
--output_path ./data/WN18RR/all.abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_0.jsonl \
--model_path ./retriever/results/WN18RR_rel_retri_0/ --device 6 7  --topk 15 --max_num_triples 1000 \
--filter_score 0.0 --filter_method mixture --max_hop 3 --min_path_precision 0.01 --num_pro_each_device 5

python extract_abstract_subgraph.py --split_qid_path ./data/WN18RR/SPLIT.aggregate.qid.npy --split_list train test dev \
--abs_sg_path ./data/WN18RR/all.abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_0.jsonl \
--output_path ./data/WN18RR/SPLIT.abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_0.jsonl \
--num_process 10

# 5.将抽象子图的关系和节点进行映射
python3 map_abs_kg_2_global_id.py --input_path data/WN18RR/SPLIT.abstract.sg.topk.10.min_score.0.max_tris.1000.rel_retri_0.jsonl \
--output_path data/WN18RR/ --output_prefix abs_r0_ --split_list train dev test

# 6.将抽象子图转换为NSM的输入形式
python3 convert_abs_to_NSM_format.py \
--input_path data/WN18RR/SPLIT.abstract.sg.topk.10.min_score.0.max_tris.1000.rel_retri_0.jsonl \
--output_path data/WN18RR/SPLIT.abstract.sg.max_tris.300.rel_retri_0.nsm.json \
--all_output_path data/WN18RR/all.abstract.sg.max_tris.300.rel_retri_0.nsm.json \
--kg_map_path data/WN18RR/ --kg_map_prefix abs_r0_ --max_nodes 300 --split_list train dev test

# 7.训练NSM retriever
# best h1: F1:40.67 H1:44.28 -> 63.85 best f1: F1:40.86 H1:16.99 -> 46.55
CUDA_VISIBLE_DEVICES=3 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 100 \
--model_path ../retriever/results/WN18RR_rel_retri_0/ --agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ../data/WN18RR/ --data_name .abstract.sg.max_tris.300.rel_retri_0.nsm.json \
--checkpoint_dir ../retriever_ckpt/WN18RR_nsm_retri_v0/ \
--batch_size 800 --gradient_accumulation_steps 1 --test_batch_size 4000 --num_step 3 \
--entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--experiment_name retriever_nsm_0 --eps 0.95 --num_epoch 100 --patience 20 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--loss_type kl --reason_kb --data_cache ../data/WN18RR/abs_r0_sg_for_retri.cache \
--overwrite_cache --relation2id abs_r0_relations.txt --entity2id abs_r0_entities.txt

# 8.基于第4步抽取出的子图 使用NSM retriever检索
# train:  (2,213,27)   (valid subgraph: 86835/86835-1.0 ans:0.99)
# dev:    (2,200,32)   (valid subgraph: 2019/3034-0.67 ans:0.70)
# test:   (2,213,20)   (valid subgraph: 2599/3134-0.83 ans:0.5)
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 \
--sparse_kg_source_path ./data/WN18RR/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/WN18RR/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/WN18RR/graph_data/ent2id.pickle --sparse_rel2id_path ./data/WN18RR/graph_data/rel2id.pickle \
--ori_path data/WN18RR/all_data.jsonl \
--input_path data/WN18RR/all.abstract.sg.max_tris.300.rel_retri_0.nsm.json \
--output_path data/WN18RR/all.instantiate.sg.rel_retri_0.10-100.jsonl \
--model_path retriever_ckpt/WN18RR_nsm_retri_v0/retriever_nsm_0-h1.ckpt \
--arg_path retriever_ckpt/WN18RR_nsm_retri_v0/retriever_nsm_0-args.json \
--relation2id_path data/WN18RR/abs_r0_relations.txt \
--entity2id_path data/WN18RR/abs_r0_entities.txt \
--device 0 1 2 3 --final_topk 10 --max_deduced_triples 100 --max_hop 3 --num_pro_each_device 3
python extract_instantiate_subgraph.py --split_qid_path ./data/WN18RR/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/WN18RR/all.instantiate.sg.rel_retri_0.10-100.jsonl \
--output_path ./data/WN18RR/SPLIT.instantiate.sg.rel_retri_0.10-100.jsonl \
--num_process 12

# 9.将实例化子图的关系和节点进行映射
python3 map_kg_2_global_id.py --input_path ./data/WN18RR/SPLIT.instantiate.sg.topk.10.max_tris.1000.rel_retri_0.10-100.jsonl \
--ori_path data/WN18RR/SPLIT.jsonl --output_path data/WN18RR/ --output_prefix reason_r0_ \
--split_list train dev test

# 10.转换为NSM输入形式
python3 convert_to_NSM_format.py --original_path data/WN18RR/SPLIT.jsonl --max_nodes 250 \
--input_path ./data/WN18RR/SPLIT.instantiate.sg.topk.10.max_tris.1000.rel_retri_0.10-100.jsonl \
--output_path ./data/WN18RR/SPLIT.reason.sg.max_tris.250.json \
--kg_map_path ./data/WN18RR/ --kg_map_prefix reason_r0_ --split_list train dev test

# 11.跑NSM Reasoner (注意加载数据时，answer映射问题)
# best h1:  -> 0.027 best f1:  -> 0.02 LSTM效果
CUDA_VISIBLE_DEVICES=0 python3 ./reader/main_nsm.py --model_name gnn \
--data_folder ./data/WN18RR/ --data_name .reason.sg.max_tris.250.json \
--checkpoint_dir ./retriever_ckpt/WN18RR/nsm_lstm_reasoner/ --batch_size 800 --test_batch_size 4000 --num_step 3 \
--entity_dim 50 --word_dim 300 --kg_dim 100 --kge_dim 100 --eval_every 2 --encode_type \
--experiment_name wn18rr_nsm_reason_0 --eps 0.95 --num_epoch 100 --use_self_loop \
--lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt

# best h1:  -> 13.81 best f1:  -> 14.16 迁移relation retriever
CUDA_VISIBLE_DEVICES=1 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ./retriever/results/WN18RR_rel_retri_0/ --agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/WN18RR/ --data_name .reason.sg.max_tris.250.json \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 400 --gradient_accumulation_steps 2 --test_batch_size 2000 --num_step 3 --entity_dim 100 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name wn18rr_nsm_reason_0 --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt

# best h1:  -> 25.01 best f1:  -> 24.16 迁移retrieval NSM最好的h1
CUDA_VISIBLE_DEVICES=2 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/WN18RR_nsm_retri_v0/retriever_nsm_0-h1.ckpt \
--model_path ./retriever/results/WN18RR_rel_retri_0/ --agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/WN18RR/ --data_name .reason.sg.max_tris.250.json \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 400 --gradient_accumulation_steps 2 --test_batch_size 2000 --num_step 3 --entity_dim 100 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name wn18rr_nsm_reason_1 --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt

# best h1: -> 13.43 best f1: -> 10.23 迁移retrieval NSM最好的f1
CUDA_VISIBLE_DEVICES=2 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/WN18RR_nsm_retri_v0/retriever_nsm_0-f1.ckpt \
--model_path ./retriever/results/WN18RR_rel_retri_0/ --agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/WN18RR/ --data_name .reason.sg.max_tris.250.json \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 400 --gradient_accumulation_steps 2 --test_batch_size 2000 --num_step 3 --entity_dim 100 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name wn18rr_nsm_reason_2 --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r0_relations.txt --entity2id reason_r0_entities.txt