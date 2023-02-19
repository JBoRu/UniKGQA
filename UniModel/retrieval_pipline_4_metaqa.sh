#!/bin/bash
# 流水线的NSM检索+NSM推理
# 1.抽取最短可达路径，并通过一定的额外主题实体获得限制关系，最终得到所有可达路径及其召回的精度。
#train: Samples with paths-96101/96106-1.00 Const rels recall-1.000
#dev: Samples with paths-9992/9992-1.00 Const rels recall-1.000
#test: Samples with paths-9947/9947-1.00 Const rels recall-1.000
python3 s0_extract_weak_super_relations.py --dense_kg_source virtuoso --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--input_path ./data/metaqa/metaqa-1hop/all_data.jsonl --output_path ./data/metaqa/metaqa-1hop/new.all.shortest.paths.1hop.jsonl \
--ids_path ./data/metaqa/metaqa-1hop/SPLIT.qid.npy --log_path ./data/metaqa/metaqa-1hop/output.log --max_hop 1 --max_num_processes 60
# 1hop-os
python3 s0_extract_weak_super_relations.py --dense_kg_source virtuoso --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--input_path ./data/metaqa/metaqa-1hop-os/all_data.jsonl --output_path ./data/metaqa/metaqa-1hop-os/all.shortest.paths.1hop.jsonl \
--ids_path ./data/metaqa/metaqa-1hop-os/SPLIT.qid.npy --log_path ./data/metaqa/metaqa-1hop-os/output.log --max_hop 1 --max_num_processes 60
#train: Samples with paths-118980/118980-1.00 Const rels recall-1.000
#dev: Samples with paths-14872/14872-1.00 Const rels recall-1.000
#test: Samples with paths-14872/14872-1.00 Const rels recall-1.000
python3 s0_extract_weak_super_relations.py --dense_kg_source virtuoso --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--input_path ./data/metaqa/metaqa-2hop/all_data.jsonl --output_path ./data/metaqa/metaqa-2hop/all.shortest.paths.2hop.jsonl \
--ids_path ./data/metaqa/metaqa-2hop/SPLIT.qid.npy --max_hop 2 --max_num_processes 60
# 2hop-os
python3 s0_extract_weak_super_relations.py --dense_kg_source virtuoso --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--input_path ./data/metaqa/metaqa-2hop-os/all_data.jsonl --output_path ./data/metaqa/metaqa-2hop-os/all.shortest.paths.2hop.jsonl \
--ids_path ./data/metaqa/metaqa-2hop-os/SPLIT.qid.npy --max_hop 2 --max_num_processes 60

python3 s0_extract_weak_super_relations.py --dense_kg_source virtuoso --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--input_path ./data/metaqa/metaqa-3hop/all_data.jsonl --output_path ./data/metaqa/metaqa-3hop/all.shortest.paths.3hop.jsonl \
--ids_path ./data/metaqa/metaqa-3hop/SPLIT.qid.npy --max_hop 3 --max_num_processes 60
# 3hop-os
python3 s0_extract_weak_super_relations.py --dense_kg_source virtuoso --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--input_path ./data/metaqa/metaqa-3hop-os/all_data.jsonl --output_path ./data/metaqa/metaqa-3hop-os/all.shortest.paths.3hop.jsonl \
--ids_path ./data/metaqa/metaqa-3hop-os/SPLIT.qid.npy --max_hop 3 --max_num_processes 60

# 2.构建训练relation retriever的训练数据
#valid sample: 116040/116040-1.00 path count: [1, 3, 1.19]
python extract_valid_weak_paths.py --min_precision 1 --filter_order 0 \
--input_path ./data/metaqa/metaqa-1hop/all.shortest.paths.1hop.jsonl --output_path ./data/metaqa/metaqa-1hop/all.shortest.paths.1hop.min_pre.1.jsonl
#valid sample: 109917/116040-0.95 path count: [0, 3, 1.13]
python extract_valid_weak_paths.py --min_precision 1 --filter_order -1 \
--input_path ./data/metaqa/metaqa-1hop/new.all.shortest.paths.1hop.jsonl --output_path ./data/metaqa/metaqa-1hop/new.all.shortest.paths.1hop.min_pre.1.jsonl
#valid sample: 20099/20099-1.00 path count: [1, 3, 1.19]
python extract_valid_weak_paths.py --min_precision 1 --filter_order 0 \
--input_path ./data/metaqa/metaqa-1hop-os/all.shortest.paths.1hop.jsonl --output_path ./data/metaqa/metaqa-1hop-os/all.shortest.paths.1hop.min_pre.1.jsonl
#valid sample: 148724/148724-1.00 path count: [1, 9, 1.38]
python extract_valid_weak_paths.py --min_precision 1 --filter_order 0 \
--input_path ./data/metaqa/metaqa-2hop/all.shortest.paths.2hop.jsonl --output_path ./data/metaqa/metaqa-2hop/all.shortest.paths.2hop.min_pre.1.jsonl
#valid sample: 29954/29954-1.00 path count: [1, 9, 1.38]
python extract_valid_weak_paths.py --min_precision 1 --filter_order 0 \
--input_path ./data/metaqa/metaqa-2hop-os/all.shortest.paths.2hop.jsonl --output_path ./data/metaqa/metaqa-2hop-os/all.shortest.paths.2hop.min_pre.1.jsonl
#valid sample: 142744/142744-1.00 path count: [1, 30, 2.47]
python extract_valid_weak_paths.py --min_precision 1 --filter_order 0 \
--input_path ./data/metaqa/metaqa-3hop/all.shortest.paths.3hop.jsonl --output_path ./data/metaqa/metaqa-3hop/all.shortest.paths.3hop.min_pre.1.jsonl
#valid sample: 28698/28698-1.00 path count: [1, 30, 2.49]
python extract_valid_weak_paths.py --min_precision 1 --filter_order 0 \
--input_path ./data/metaqa/metaqa-3hop-os/all.shortest.paths.3hop.jsonl --output_path ./data/metaqa/metaqa-3hop-os/all.shortest.paths.3hop.min_pre.1.jsonl


# 1hop neg_5_1 train：84318 dev：8801
python s1_construct_relation_retrieval_training_data.py --dense_kg_source virtuoso --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--input_path ./data/metaqa/metaqa-1hop/all.shortest.paths.1hop.min_pre.1.jsonl --output_path ./data/metaqa/metaqa-1hop/all.rels.retri.data.neg_5.1.jsonl \
--max_hop 1 --num_neg 5 --max_num_processes 60
python extract_relation_retriever_training_data.py --split_qid_path ./data/metaqa/metaqa-1hop/SPLIT.qid.npy --split_list train dev \
--pn_pairs_path ./data/metaqa/metaqa-1hop/all.rels.retri.data.neg_5.1.jsonl --output_path ./data/metaqa/metaqa-1hop/SPLIT.rels.retri.data.neg_5.1.csv
# 1hop os neg_5_1 train：132 dev：8801
python s1_construct_relation_retrieval_training_data.py --dense_kg_source virtuoso --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--input_path ./data/metaqa/metaqa-1hop-os/all.shortest.paths.1hop.min_pre.1.jsonl --output_path ./data/metaqa/metaqa-1hop-os/all.rels.retri.data.neg_5.1.jsonl \
--max_hop 1 --num_neg 5 --max_num_processes 60
python extract_relation_retriever_training_data.py --split_qid_path ./data/metaqa/metaqa-1hop-os/SPLIT.qid.npy --split_list train dev \
--pn_pairs_path ./data/metaqa/metaqa-1hop-os/all.rels.retri.data.neg_5.1.jsonl --output_path ./data/metaqa/metaqa-1hop-os/SPLIT.rels.retri.data.neg_5.1.csv
# neg_5_1 train：327228 dev：40716
python s1_construct_relation_retrieval_training_data.py --dense_kg_source virtuoso --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--input_path ./data/metaqa/metaqa-2hop/all.shortest.paths.2hop.min_pre.1.jsonl --output_path ./data/metaqa/metaqa-2hop/all.rels.retri.data.neg_5.1.jsonl \
--max_hop 2 --num_neg 5 --max_num_processes 60
python extract_relation_retriever_training_data.py --split_qid_path ./data/metaqa/metaqa-2hop/SPLIT.qid.npy --split_list train dev \
--pn_pairs_path ./data/metaqa/metaqa-2hop/all.rels.retri.data.neg_5.1.jsonl --output_path ./data/metaqa/metaqa-2hop/SPLIT.rels.retri.data.neg_5.1.csv
# 2hoop os neg_5_1 train：554 dev：40716
python s1_construct_relation_retrieval_training_data.py --dense_kg_source virtuoso --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--input_path ./data/metaqa/metaqa-2hop-os/all.shortest.paths.2hop.min_pre.1.jsonl --output_path ./data/metaqa/metaqa-2hop-os/all.rels.retri.data.neg_5.1.jsonl \
--max_hop 2 --num_neg 5 --max_num_processes 60
python extract_relation_retriever_training_data.py --split_qid_path ./data/metaqa/metaqa-2hop-os/SPLIT.qid.npy --split_list train dev \
--pn_pairs_path ./data/metaqa/metaqa-2hop-os/all.rels.retri.data.neg_5.1.jsonl --output_path ./data/metaqa/metaqa-2hop-os/SPLIT.rels.retri.data.neg_5.1.csv
# neg_5_1 train：845115 dev：105774
python s1_construct_relation_retrieval_training_data.py --dense_kg_source virtuoso --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--input_path ./data/metaqa/metaqa-3hop/all.shortest.paths.3hop.min_pre.1.jsonl --output_path ./data/metaqa/metaqa-3hop/all.rels.retri.data.neg_5.1.jsonl \
--max_hop 3 --num_neg 5 --max_num_processes 60
python extract_relation_retriever_training_data.py --split_qid_path ./data/metaqa/metaqa-3hop/SPLIT.qid.npy --split_list train dev \
--pn_pairs_path ./data/metaqa/metaqa-3hop/all.rels.retri.data.neg_5.1.jsonl --output_path ./data/metaqa/metaqa-3hop/SPLIT.rels.retri.data.neg_5.1.csv
# 3hop os neg_5_1 train：1290 dev：105774
python s1_construct_relation_retrieval_training_data.py --dense_kg_source virtuoso --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--input_path ./data/metaqa/metaqa-3hop-os/all.shortest.paths.3hop.min_pre.1.jsonl --output_path ./data/metaqa/metaqa-3hop-os/all.rels.retri.data.neg_5.1.jsonl \
--max_hop 3 --num_neg 5 --max_num_processes 60
python extract_relation_retriever_training_data.py --split_qid_path ./data/metaqa/metaqa-3hop-os/SPLIT.qid.npy --split_list train dev \
--pn_pairs_path ./data/metaqa/metaqa-3hop-os/all.rels.retri.data.neg_5.1.jsonl --output_path ./data/metaqa/metaqa-3hop-os/SPLIT.rels.retri.data.neg_5.1.csv

# 3.训练relation retriever
# 1hop dev hits@1 99.34
CUDA_VISIBLE_DEVICES=0 python3 ./retriever/train.py --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
--train_file ./data/metaqa/metaqa-1hop/train.rels.retri.data.neg_5.1.csv \
--eval_file ./data/metaqa/metaqa-1hop/dev.rels.retri.data.neg_5.1.csv \
--output_dir ./retriever/results/metaqa_rel_retri_for_1hop --per_device_train_batch_size 80 --gradient_accumulation_steps 1 \
--learning_rate 5e-5 --do_train --num_train_epochs 5 --do_eval --evaluation_strategy steps --eval_steps 500 \
--save_steps 500 --save_total_limit 1 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05
# 1hop os dev hits@1 98.97
CUDA_VISIBLE_DEVICES=3 python3 ./retriever/train.py --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
--train_file ./data/metaqa/metaqa-1hop-os/train.rels.retri.data.neg_5.1.csv \
--eval_file ./data/metaqa/metaqa-1hop-os/dev.rels.retri.data.neg_5.1.csv \
--output_dir ./retriever/results/metaqa_rel_retri_for_1hop_os --per_device_train_batch_size 8 --gradient_accumulation_steps 1 \
--learning_rate 1e-5 --do_train --num_train_epochs 5 --do_eval --evaluation_strategy steps --eval_steps 15 \
--save_steps 15 --save_total_limit 1 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05
# 2hop dev hits@1 97.05
CUDA_VISIBLE_DEVICES=3 python3 ./retriever/train.py --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
--train_file ./data/metaqa/metaqa-2hop/train.rels.retri.data.neg_5.1.csv \
--eval_file ./data/metaqa/metaqa-2hop/dev.rels.retri.data.neg_5.1.csv \
--output_dir ./retriever/results/metaqa_rel_retri_for_2hop --per_device_train_batch_size 80 --gradient_accumulation_steps 1 \
--learning_rate 5e-5 --do_train --num_train_epochs 5 --do_eval --evaluation_strategy steps --eval_steps 2000 \
--save_steps 2000 --save_total_limit 1 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05
# 2hop os dev hits@1 93.26
CUDA_VISIBLE_DEVICES=3 python3 ./retriever/train.py --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
--train_file ./data/metaqa/metaqa-2hop-os/train.rels.retri.data.neg_5.1.csv \
--eval_file ./data/metaqa/metaqa-2hop-os/dev.rels.retri.data.neg_5.1.csv \
--output_dir ./retriever/results/metaqa_rel_retri_for_2hop-os --per_device_train_batch_size 55 --gradient_accumulation_steps 1 \
--learning_rate 5e-5 --do_train --num_train_epochs 10 --do_eval --evaluation_strategy steps --eval_steps 10 \
--save_steps 20 --save_total_limit 1 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05
# 3hop dev hits@1 94.39
CUDA_VISIBLE_DEVICES=5 python3 ./retriever/train.py --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
--train_file ./data/metaqa/metaqa-3hop/train.rels.retri.data.neg_5.1.csv \
--eval_file ./data/metaqa/metaqa-3hop/dev.rels.retri.data.neg_5.1.csv \
--output_dir ./retriever/results/metaqa_rel_retri_for_3hop --per_device_train_batch_size 80 --gradient_accumulation_steps 1 \
--learning_rate 5e-5 --do_train --num_train_epochs 5 --do_eval --evaluation_strategy steps --eval_steps 5000 \
--save_steps 5000 --save_total_limit 1 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05
# 3hop os dev hits@1 92.00
CUDA_VISIBLE_DEVICES=3 python3 ./retriever/train.py --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
--train_file ./data/metaqa/metaqa-3hop-os/train.rels.retri.data.neg_5.1.csv \
--eval_file ./data/metaqa/metaqa-3hop-os/dev.rels.retri.data.neg_5.1.csv \
--output_dir ./retriever/results/metaqa_rel_retri_for_3hop-os --per_device_train_batch_size 64 --gradient_accumulation_steps 1 \
--learning_rate 5e-5 --do_train --num_train_epochs 10 --do_eval --evaluation_strategy steps --eval_steps 20 \
--save_steps 20 --save_total_limit 1 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05

# 将simcse对应的模型转换为huggingface通用的模型结构
python ./retriever/simcse_to_huggingface.py --path ./retriever/results/metaqa_rel_retri_for_1hop/
python ./retriever/simcse_to_huggingface.py --path ./retriever/results/metaqa_rel_retri_for_2hop/
python ./retriever/simcse_to_huggingface.py --path ./retriever/results/metaqa_rel_retri_for_3hop/

# 4.构建主题实体周围三跳内的抽象子图，使用relation retriever过滤非常不相关关系
# 1hop train: (2,5,3) (ans:0.98-96101/96106) dev: (2,5,3) (ans:1-9992/9992) test: (0,4,3) (ans:0.99-9946/9947)
python3 s2_extract_abstract_subgraph.py --dense_kg_source virtuoso --max_num_processes 40 --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--ori_path ./data/metaqa/metaqa-1hop/all_data.jsonl --input_path ./data/metaqa/metaqa-1hop/all.shortest.paths.1hop.jsonl \
--output_path ./data/metaqa/metaqa-1hop/all.abs.sg.topk.3.min_sim.0.max_tris.100.metaqa_rel_retri_for_1hop.jsonl \
--model_path ./retriever/results/metaqa_rel_retri_for_1hop/ --device 0 --topk 3 --max_num_triples 100 \
--filter_score 0.0 --max_hop 1 --num_pro_each_device 10
python extract_abstract_subgraph.py --split_qid_path ./data/metaqa/metaqa-1hop/SPLIT.qid.npy --split_list train dev test \
--abs_sg_path ./data/metaqa/metaqa-1hop/all.abs.sg.topk.3.min_sim.0.max_tris.100.metaqa_rel_retri_for_1hop.jsonl \
--output_path ./data/metaqa/metaqa-1hop/SPLIT.abs.sg.topk.3.min_sim.0.max_tris.100.metaqa_rel_retri_for_1hop.jsonl \
--all_output_path ./data/metaqa/metaqa-1hop/all.abs.sg.topk.3.min_sim.0.max_tris.100.metaqa_rel_retri_for_1hop.jsonl \
--all_shortest_path ./data/metaqa/metaqa-1hop/all.shortest.paths.1hop.jsonl \
--ori_path ./data/metaqa/metaqa-1hop/all_data.jsonl \
--filter_order 1 --num_process 10
# 1hop os train: (2,4,3) (ans:0.99-160/161) dev: (2,5,3) (ans:1-9992/9992) test: (0,4,3) (ans:0.99-9946/9947)
python3 s2_extract_abstract_subgraph.py --dense_kg_source virtuoso --max_num_processes 40 --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--ori_path ./data/metaqa/metaqa-1hop-os/all_data.jsonl --input_path ./data/metaqa/metaqa-1hop-os/all.shortest.paths.1hop.jsonl \
--output_path ./data/metaqa/metaqa-1hop-os/all.abs.sg.topk.3.min_sim.0.max_tris.100.metaqa_rel_retri_for_1hop.jsonl \
--model_path ./retriever/results/metaqa_rel_retri_for_1hop_os/ --device 5 --topk 3 --max_num_triples 100 \
--filter_score 0.0 --max_hop 1 --num_pro_each_device 10
python extract_abstract_subgraph.py --split_qid_path ./data/metaqa/metaqa-1hop/SPLIT.qid.npy --split_list train dev test \
--abs_sg_path ./data/metaqa/metaqa-1hop-os/all.abs.sg.topk.3.min_sim.0.max_tris.100.metaqa_rel_retri_for_1hop.jsonl \
--output_path ./data/metaqa/metaqa-1hop-os/SPLIT.abs.sg.topk.3.min_sim.0.max_tris.100.metaqa_rel_retri_for_1hop.jsonl \
--all_output_path ./data/metaqa/metaqa-1hop-os/all.abs.sg.topk.3.min_sim.0.max_tris.100.metaqa_rel_retri_for_1hop.jsonl \
--all_shortest_path ./data/metaqa/metaqa-1hop-os/all.shortest.paths.1hop.jsonl \
--ori_path ./data/metaqa/metaqa-1hop-os/all_data.jsonl \
--filter_order 1 --num_process 10
# 2 hop train:  (3,28,6) (ans:1-118980/118980) dev: (3,22,6) (ans:1-14872/14872) test: (3,13,5) (ans:1-14872/14872)
python3 s2_extract_abstract_subgraph.py --dense_kg_source virtuoso --max_num_processes 40 --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--ori_path ./data/metaqa/metaqa-2hop/all_data.jsonl --input_path ./data/metaqa/metaqa-2hop/all.shortest.paths.2hop.jsonl \
--output_path ./data/metaqa/metaqa-2hop/all.abs.sg.topk.3.min_sim.0.max_tris.200.metaqa_rel_retri_for_2hop.jsonl \
--model_path ./retriever/results/metaqa_rel_retri_for_2hop/ --device 3 --topk 3 --max_num_triples 200 \
--filter_score 0.0 --max_hop 2 --num_pro_each_device 10
python extract_abstract_subgraph.py --split_qid_path ./data/metaqa/metaqa-2hop/SPLIT.qid.npy --split_list train dev test \
--abs_sg_path ./data/metaqa/metaqa-2hop/all.abs.sg.topk.3.min_sim.0.max_tris.200.metaqa_rel_retri_for_2hop.jsonl \
--output_path ./data/metaqa/metaqa-2hop/SPLIT.abs.sg.topk.3.min_sim.0.max_tris.200.metaqa_rel_retri_for_2hop.jsonl \
--all_output_path ./data/metaqa/metaqa-2hop/all.abs.sg.topk.3.min_sim.0.max_tris.200.metaqa_rel_retri_for_2hop.jsonl \
--all_shortest_path ./data/metaqa/metaqa-2hop/all.shortest.paths.2hop.jsonl \
--ori_path ./data/metaqa/metaqa-2hop/all_data.jsonl \
--filter_order 1 --num_process 10
# 2hop os train: (4,19,7) (ans:1-210/210) dev: (3,23,7) (ans:1-14872/14872) test: (3,13,7) (ans:0.9986-14872/14872)
python3 s2_extract_abstract_subgraph.py --dense_kg_source virtuoso --max_num_processes 40 --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--ori_path ./data/metaqa/metaqa-2hop-os/all_data.jsonl --input_path ./data/metaqa/metaqa-2hop-os/all.shortest.paths.2hop.jsonl \
--output_path ./data/metaqa/metaqa-2hop-os/all.abs.sg.topk.3.min_sim.0.max_tris.200.metaqa_rel_retri_for_2hop.jsonl \
--model_path ./retriever/results/metaqa_rel_retri_for_2hop-os/ --device 3 --topk 3 --max_num_triples 100 \
--filter_score 0.0 --max_hop 2 --num_pro_each_device 10
python extract_abstract_subgraph.py --split_qid_path ./data/metaqa/metaqa-2hop-os/SPLIT.qid.npy --split_list train dev test \
--abs_sg_path ./data/metaqa/metaqa-2hop-os/all.abs.sg.topk.3.min_sim.0.max_tris.200.metaqa_rel_retri_for_2hop.jsonl \
--output_path ./data/metaqa/metaqa-2hop-os/SPLIT.abs.sg.topk.3.min_sim.0.max_tris.200.metaqa_rel_retri_for_2hop.jsonl \
--all_output_path ./data/metaqa/metaqa-2hop-os/all.abs.sg.topk.3.min_sim.0.max_tris.200.metaqa_rel_retri_for_2hop.jsonl \
--all_shortest_path ./data/metaqa/metaqa-2hop-os/all.shortest.paths.2hop.jsonl \
--ori_path ./data/metaqa/metaqa-2hop-os/all_data.jsonl \
--filter_order 1 --num_process 10
# 3 hop train: (6,110,34) (ans:1-114196/114196) dev: (6,100,34) (ans:1-14274/14274) test: (6,40,21) (ans:0.9957-14212/14274)
python3 s2_extract_abstract_subgraph.py --dense_kg_source virtuoso --max_num_processes 40 --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--ori_path ./data/metaqa/metaqa-3hop/all_data.jsonl --input_path ./data/metaqa/metaqa-3hop/all.shortest.paths.3hop.jsonl \
--output_path ./data/metaqa/metaqa-3hop/all.abs.sg.topk.3.min_sim.0.max_tris.400.metaqa_rel_retri_for_3hop.jsonl \
--model_path ./retriever/results/metaqa_rel_retri_for_3hop/ --device 5 --topk 3 --max_num_triples 400 \
--filter_score 0.0 --max_hop 3 --num_pro_each_device 10
python extract_abstract_subgraph.py --split_qid_path ./data/metaqa/metaqa-3hop/SPLIT.qid.npy --split_list train dev test \
--abs_sg_path ./data/metaqa/metaqa-3hop/all.abs.sg.topk.3.min_sim.0.max_tris.400.metaqa_rel_retri_for_3hop.jsonl \
--output_path ./data/metaqa/metaqa-3hop/SPLIT.abs.sg.topk.3.min_sim.0.max_tris.400.metaqa_rel_retri_for_3hop.jsonl \
--all_output_path ./data/metaqa/metaqa-3hop/all.abs.sg.topk.3.min_sim.0.max_tris.400.metaqa_rel_retri_for_3hop.jsonl \
--all_shortest_path ./data/metaqa/metaqa-3hop/all.shortest.paths.3hop.jsonl \
--ori_path ./data/metaqa/metaqa-3hop/all_data.jsonl \
--filter_order 1 --num_process 10
# 3 hop os train: (11,73,35)  (ans:1-150/114196) dev: (6,100,35) (ans:1-14274/14274) test: (6,40,22)   (ans:0.9941-14190/14274)
python3 s2_extract_abstract_subgraph.py --dense_kg_source virtuoso --max_num_processes 40 --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--ori_path ./data/metaqa/metaqa-3hop-os/all_data.jsonl --input_path ./data/metaqa/metaqa-3hop-os/all.shortest.paths.3hop.jsonl \
--output_path ./data/metaqa/metaqa-3hop-os/all.abs.sg.topk.3.min_sim.0.max_tris.400.metaqa_rel_retri_for_3hop.jsonl \
--model_path ./retriever/results/metaqa_rel_retri_for_3hop-os/ --device 5 --topk 3 --max_num_triples 100 \
--filter_score 0.0 --max_hop 3 --num_pro_each_device 10
python extract_abstract_subgraph.py --split_qid_path ./data/metaqa/metaqa-3hop-os/SPLIT.qid.npy --split_list train dev test \
--abs_sg_path ./data/metaqa/metaqa-3hop-os/all.abs.sg.topk.3.min_sim.0.max_tris.400.metaqa_rel_retri_for_3hop.jsonl \
--output_path ./data/metaqa/metaqa-3hop-os/SPLIT.abs.sg.topk.3.min_sim.0.max_tris.400.metaqa_rel_retri_for_3hop.jsonl \
--all_output_path ./data/metaqa/metaqa-3hop-os/all.abs.sg.topk.3.min_sim.0.max_tris.400.metaqa_rel_retri_for_3hop.jsonl \
--all_shortest_path ./data/metaqa/metaqa-3hop-os/all.shortest.paths.3hop.jsonl \
--ori_path ./data/metaqa/metaqa-3hop-os/all_data.jsonl \
--filter_order 1 --num_process 10


# 5.将抽象子图的关系和节点进行映射
python3 map_abs_kg_2_global_id.py --input_path data/metaqa/metaqa-1hop/SPLIT.abs.sg.topk.3.min_sim.0.max_tris.100.metaqa_rel_retri_for_1hop.jsonl \
--output_path data/metaqa/metaqa-1hop/ --output_prefix abs_ --split_list train dev test
python3 map_abs_kg_2_global_id.py --input_path data/metaqa/metaqa-2hop/SPLIT.abs.sg.topk.3.min_sim.0.max_tris.200.metaqa_rel_retri_for_2hop.jsonl \
--output_path data/metaqa/metaqa-2hop/ --output_prefix abs_ --split_list train dev test
python3 map_abs_kg_2_global_id.py --input_path data/metaqa/metaqa-3hop/SPLIT.abs.sg.topk.3.min_sim.0.max_tris.400.metaqa_rel_retri_for_3hop.jsonl \
--output_path data/metaqa/metaqa-3hop/ --output_prefix abs_ --split_list train dev test

# 6.将抽象子图转换为NSM的输入形式
python3 convert_abs_to_NSM_format.py \
--input_path data/metaqa/metaqa-1hop/SPLIT.abs.sg.topk.3.min_sim.0.max_tris.100.metaqa_rel_retri_for_1hop.jsonl \
--output_path data/metaqa/metaqa-1hop/SPLIT.abs.sg.topk.3.min_sim.0.max_tris.100.metaqa_rel_retri_for_1hop.nsm.json \
--all_output_path data/metaqa/metaqa-1hop/all.abs.sg.topk.3.min_sim.0.max_tris.100.metaqa_rel_retri_for_1hop.nsm.json \
--kg_map_path data/metaqa/metaqa-1hop/ --kg_map_prefix abs_ --max_nodes 10 --split_list train dev test
python3 convert_abs_to_NSM_format.py \
--input_path data/metaqa/metaqa-2hop/SPLIT.abs.sg.topk.3.min_sim.0.max_tris.200.metaqa_rel_retri_for_2hop.jsonl \
--output_path data/metaqa/metaqa-2hop/SPLIT.abs.sg.topk.3.min_sim.0.max_tris.200.metaqa_rel_retri_for_2hop.nsm.json \
--all_output_path data/metaqa/metaqa-2hop/all.abs.sg.topk.3.min_sim.0.max_tris.200.metaqa_rel_retri_for_2hop.nsm.json \
--kg_map_path data/metaqa/metaqa-2hop/ --kg_map_prefix abs_ --max_nodes 30 --split_list train dev test
python3 convert_abs_to_NSM_format.py \
--input_path data/metaqa/metaqa-3hop/SPLIT.abs.sg.topk.3.min_sim.0.max_tris.400.metaqa_rel_retri_for_3hop.jsonl \
--output_path data/metaqa/metaqa-3hop/SPLIT.abs.sg.topk.3.min_sim.0.max_tris.400.metaqa_rel_retri_for_3hop.nsm.json \
--all_output_path data/metaqa/metaqa-3hop/all.abs.sg.topk.3.min_sim.0.max_tris.400.metaqa_rel_retri_for_3hop.nsm.json \
--kg_map_path data/metaqa/metaqa-3hop/ --kg_map_prefix abs_ --max_nodes 120 --split_list train dev test

# 7.训练NSM retriever
# retrieval: pretrain, transfer, fix plm 不用pooler层
# 1hop 90.84/99.72
CUDA_VISIBLE_DEVICES=0 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--model_path ./retriever/results/metaqa_rel_retri_for_1hop/ --agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-1hop/ \
--data_name .abs.sg.topk.3.min_sim.0.max_tris.100.metaqa_rel_retri_for_1hop.nsm.json \
--checkpoint_dir ./retriever_ckpt/metaqa_1hop_nsm/ --experiment_name retriever_p_t_wo_pooler \
--batch_size 500 --gradient_accumulation_steps 1 --test_batch_size 4000 --num_step 1 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--eps 0.95 --num_epoch 50 --patience 10 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--loss_type kl --reason_kb --data_cache ./data/metaqa/abs_1hop_sg_for_retri.cache --overwrite_cache \
--relation2id abs_relations.txt --entity2id abs_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
# 1hop os 94.87/99.59
CUDA_VISIBLE_DEVICES=4 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--model_path ./retriever/results/metaqa_rel_retri_for_1hop_os/ --agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-1hop-os/ \
--data_name .abs.sg.topk.3.min_sim.0.max_tris.100.metaqa_rel_retri_for_1hop.nsm.json \
--checkpoint_dir ./retriever_ckpt/metaqa_1hop_nsm_os/ --experiment_name retriever_p_t_wo_pooler \
--batch_size 8 --gradient_accumulation_steps 1 --test_batch_size 4000 --num_step 1 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--eps 0.95 --num_epoch 50 --patience 10 --use_self_loop --plm_lr 1e-5 --lr 1e-4 \
--loss_type kl --reason_kb --data_cache ./data/metaqa/abs_1hop_os_sg_for_retri.cache --overwrite_cache \
--relation2id abs_relations.txt --entity2id abs_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
# 2hop 87.53/95.45
CUDA_VISIBLE_DEVICES=7 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--model_path ./retriever/results/metaqa_rel_retri_for_2hop/ --agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-2hop/ \
--data_name .abs.sg.topk.3.min_sim.0.max_tris.200.metaqa_rel_retri_for_2hop.nsm.json \
--checkpoint_dir ./retriever_ckpt/metaqa_2hop_nsm/ --experiment_name retriever_p_t_wo_pooler \
--batch_size 500 --gradient_accumulation_steps 1 --test_batch_size 4000 --num_step 2 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--eps 0.95 --num_epoch 50 --patience 10 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--loss_type kl --reason_kb --data_cache ./data/metaqa/abs_2hop_sg_for_retri.cache --overwrite_cache \
--relation2id abs_relations.txt --entity2id abs_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
# 2hop os 86.36/89.21
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--model_path ./retriever/results/metaqa_rel_retri_for_2hop-os/ --agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-2hop-os/ \
--data_name .abs.sg.topk.3.min_sim.0.max_tris.200.metaqa_rel_retri_for_2hop.nsm.json \
--checkpoint_dir ./retriever_ckpt/metaqa_2hop_nsm_os/ --experiment_name retriever_p_t_wo_pooler \
--batch_size 8 --gradient_accumulation_steps 1 --test_batch_size 4000 --num_step 2 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--eps 0.95 --num_epoch 50 --patience 10 --use_self_loop --plm_lr 1e-5 --lr 1e-4 \
--loss_type kl --reason_kb --data_cache ./data/metaqa/abs_2hop_os_sg_for_retri.cache --overwrite_cache \
--relation2id abs_relations.txt --entity2id abs_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
# 3 hop 72.36/89.50
CUDA_VISIBLE_DEVICES=2 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--model_path ./retriever/results/metaqa_rel_retri_for_3hop/ --agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-3hop/ \
--data_name .abs.sg.topk.3.min_sim.0.max_tris.400.metaqa_rel_retri_for_3hop.nsm.json \
--checkpoint_dir ./retriever_ckpt/metaqa_3hop_nsm/ --experiment_name retriever_p_t_wo_pooler \
--batch_size 500 --gradient_accumulation_steps 1 --test_batch_size 2000 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--eps 0.95 --num_epoch 60 --patience 6 --use_self_loop --plm_lr 1e-5 --lr 1e-4 \
--loss_type kl --reason_kb --data_cache ./data/metaqa/abs_3hop_sg_for_retri.cache --overwrite_cache \
--relation2id abs_relations.txt --entity2id abs_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
# 3 hop os 72.83/84.39
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--model_path ./retriever/results/metaqa_rel_retri_for_3hop-os/ --agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-3hop-os/ \
--data_name .abs.sg.topk.3.min_sim.0.max_tris.400.metaqa_rel_retri_for_3hop.nsm.json \
--checkpoint_dir ./retriever_ckpt/metaqa_3hop_nsm_os/ --experiment_name retriever_p_t_wo_pooler \
--batch_size 8 --gradient_accumulation_steps 1 --test_batch_size 2000 --num_step 3 \
--entity_dim 768 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--eps 0.95 --num_epoch 60 --patience 6 --use_self_loop --plm_lr 1e-5 --lr 1e-4 \
--loss_type kl --reason_kb --data_cache ./data/metaqa/abs_3hop_os_sg_for_retri.cache --overwrite_cache \
--relation2id abs_relations.txt --entity2id abs_entities.txt \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm

# 8.基于第4步抽取出的子图 使用NSM retriever检索
# 1hop train: (2,97,4) (valid subgraph: 96101/96106-0.99 ans:1) dev: (2,67,4) (valid subgraph: 9992/9992-1.0 ans:1) test: (1,96,4) (valid subgraph: 9946/9947-0.999 ans:1)
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--ori_path data/metaqa/metaqa-1hop/all_data.jsonl \
--input_path data/metaqa/metaqa-1hop/all.abs.sg.topk.3.min_sim.0.max_tris.100.metaqa_rel_retri_for_1hop.nsm.json \
--output_path data/metaqa/metaqa-1hop/all.inst.sg.metaqa_rel_retri_for_1hop.3-100.jsonl \
--model_path retriever_ckpt/metaqa_1hop_nsm/retriever_p_t_wo_pooler-h1.ckpt \
--arg_path retriever_ckpt/metaqa_1hop_nsm/retriever_p_t_wo_pooler-args.json \
--relation2id_path data/metaqa/metaqa-1hop/abs_relations.txt \
--entity2id_path data/metaqa/metaqa-1hop/abs_entities.txt \
--device 2 --final_topk 3 --max_deduced_triples 100 --max_hop 1 --num_pro_each_device 10
python extract_instantiate_subgraph.py --split_qid_path ./data/metaqa/metaqa-1hop/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/metaqa/metaqa-1hop/all.inst.sg.metaqa_rel_retri_for_1hop.3-100.jsonl \
--output_path ./data/metaqa/metaqa-1hop/SPLIT.inst.sg.metaqa_rel_retri_for_1hop.3-100.jsonl --num_process 10
# 1hop os train: (2,32,5)   (valid sg: 96101/96106-0.99 ans:1) dev: (0,48,5)   (valid sg: 9992/9992-1.0 ans:0.9998) test: (2,46,5)   (valid sg: 9947/9947-1.0 ans:0.9999)
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--ori_path data/metaqa/metaqa-1hop-os/all_data.jsonl \
--input_path data/metaqa/metaqa-1hop-os/all.abs.sg.topk.3.min_sim.0.max_tris.100.metaqa_rel_retri_for_1hop.nsm.json \
--output_path data/metaqa/metaqa-1hop-os/all.inst.sg.metaqa_rel_retri_for_1hop.3-100.jsonl \
--model_path retriever_ckpt/metaqa_1hop_nsm_os/retriever_p_t_wo_pooler-h1.ckpt \
--arg_path retriever_ckpt/metaqa_1hop_nsm_os/retriever_p_t_wo_pooler-args.json \
--relation2id_path data/metaqa/metaqa-1hop-os/abs_relations.txt \
--entity2id_path data/metaqa/metaqa-1hop-os/abs_entities.txt \
--device 4 --final_topk 3 --max_deduced_triples 100 --max_hop 1 --num_pro_each_device 10
python extract_instantiate_subgraph.py --split_qid_path ./data/metaqa/metaqa-1hop-os/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/metaqa/metaqa-1hop-os/all.inst.sg.metaqa_rel_retri_for_1hop.3-100.jsonl \
--output_path ./data/metaqa/metaqa-1hop-os/SPLIT.inst.sg.metaqa_rel_retri_for_1hop.3-100.jsonl --num_process 10
# 2 hop train: (2,362,15) (valid sg: 118980/118980-1.0 ans:1) dev: (3,287,15) (valid sg: 14872/14872-1.0 ans:1) test: (3,225,14)   (valid sg: 14872/14872-1.0 ans:1)
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--ori_path data/metaqa/metaqa-2hop/all_data.jsonl \
--input_path data/metaqa/metaqa-2hop/all.abs.sg.topk.3.min_sim.0.max_tris.200.metaqa_rel_retri_for_2hop.nsm.json \
--output_path data/metaqa/metaqa-2hop/all.inst.sg.metaqa_rel_retri_for_2hop.3-200.jsonl \
--model_path retriever_ckpt/metaqa_2hop_nsm/retriever_p_t_wo_pooler-h1.ckpt \
--arg_path retriever_ckpt/metaqa_2hop_nsm/retriever_p_t_wo_pooler-args.json \
--relation2id_path data/metaqa/metaqa-2hop/abs_relations.txt \
--entity2id_path data/metaqa/metaqa-2hop/abs_entities.txt \
--device 2 --final_topk 3 --max_deduced_triples 200 --max_hop 2 --num_pro_each_device 10
python extract_instantiate_subgraph.py --split_qid_path ./data/metaqa/metaqa-2hop/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/metaqa/metaqa-2hop/all.inst.sg.metaqa_rel_retri_for_2hop.3-200.jsonl \
--output_path ./data/metaqa/metaqa-2hop/SPLIT.inst.sg.metaqa_rel_retri_for_2hop.3-200.jsonl --num_process 10
# 2 hop os train: (2,123,16) (valid sg: 210/210-1.0 ans:1) dev: (2,251,16) (valid sg: 14872/14872-1.0 ans:0.9989) test: (2,286,12)   (valid sg: 14872/14872-1.0 ans:0.9993)
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--ori_path data/metaqa/metaqa-2hop-os/all_data.jsonl \
--input_path data/metaqa/metaqa-2hop-os/all.abs.sg.topk.3.min_sim.0.max_tris.200.metaqa_rel_retri_for_2hop.nsm.json \
--output_path data/metaqa/metaqa-2hop-os/all.inst.sg.metaqa_rel_retri_for_2hop.3-200.jsonl \
--model_path retriever_ckpt/metaqa_2hop_nsm_os/retriever_p_t_wo_pooler-h1.ckpt \
--arg_path retriever_ckpt/metaqa_2hop_nsm_os/retriever_p_t_wo_pooler-args.json \
--relation2id_path data/metaqa/metaqa-2hop-os/abs_relations.txt \
--entity2id_path data/metaqa/metaqa-2hop-os/abs_entities.txt \
--device 3 --final_topk 3 --max_deduced_triples 200 --max_hop 2 --num_pro_each_device 10
python extract_instantiate_subgraph.py --split_qid_path ./data/metaqa/metaqa-2hop-os/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/metaqa/metaqa-2hop-os/all.inst.sg.metaqa_rel_retri_for_2hop.3-200.jsonl \
--output_path ./data/metaqa/metaqa-2hop-os/SPLIT.inst.sg.metaqa_rel_retri_for_2hop.3-200.jsonl --num_process 10
# 3 hop train: (0,344,46) (valid sg: 114196/114196-1.0 ans:0.99) dev: (0,312,46) (valid sg: 14274/14274-1.0 ans:0.99) test: (0,299,46) (valid sg: 14274/14274-1.0 ans:0.99)
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--ori_path data/metaqa/metaqa-3hop/all_data.jsonl \
--input_path data/metaqa/metaqa-3hop/all.abs.sg.topk.3.min_sim.0.max_tris.400.metaqa_rel_retri_for_3hop.nsm.json \
--output_path data/metaqa/metaqa-3hop/all.inst.sg.metaqa_rel_retri_for_3hop.3-200.jsonl \
--model_path retriever_ckpt/metaqa_3hop_nsm/retriever_p_t_wo_pooler-h1.ckpt \
--arg_path retriever_ckpt/metaqa_3hop_nsm/retriever_p_t_wo_pooler-args.json \
--relation2id_path data/metaqa/metaqa-3hop/abs_relations.txt \
--entity2id_path data/metaqa/metaqa-3hop/abs_entities.txt \
--device 2 --final_topk 3 --max_deduced_triples 200 --max_hop 3 --num_pro_each_device 10
python extract_instantiate_subgraph.py --split_qid_path ./data/metaqa/metaqa-3hop/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/metaqa/metaqa-3hop/all.inst.sg.metaqa_rel_retri_for_3hop.3-200.jsonl \
--output_path ./data/metaqa/metaqa-3hop/SPLIT.inst.sg.metaqa_rel_retri_for_3hop.3-200.jsonl --num_process 10
# 3 hop os train: (3,224,41) (valid sg: 150/150-1.0 ans:0.96) dev: (0,361,47) (valid sg: 14274/14274-1.0 ans:0.97) test: (0,382,46) (valid sg: 14274/14274-1.0 ans:0.97)
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name metaqa \
--sparse_kg_source_path ./data/metaqa/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/metaqa/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/metaqa/graph_data/ent2id.pickle --sparse_rel2id_path ./data/metaqa/graph_data/rel2id.pickle \
--ori_path data/metaqa/metaqa-3hop-os/all_data.jsonl \
--input_path data/metaqa/metaqa-3hop-os/all.abs.sg.topk.3.min_sim.0.max_tris.400.metaqa_rel_retri_for_3hop.nsm.json \
--output_path data/metaqa/metaqa-3hop-os/all.inst.sg.metaqa_rel_retri_for_3hop.3-200.jsonl \
--model_path retriever_ckpt/metaqa_3hop_nsm_os/retriever_p_t_wo_pooler-h1.ckpt \
--arg_path retriever_ckpt/metaqa_3hop_nsm_os/retriever_p_t_wo_pooler-args.json \
--relation2id_path data/metaqa/metaqa-3hop-os/abs_relations.txt \
--entity2id_path data/metaqa/metaqa-3hop-os/abs_entities.txt \
--device 5 --final_topk 3 --max_deduced_triples 200 --max_hop 3 --num_pro_each_device 10
python extract_instantiate_subgraph.py --split_qid_path ./data/metaqa/metaqa-3hop-os/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/metaqa/metaqa-3hop-os/all.inst.sg.metaqa_rel_retri_for_3hop.3-200.jsonl \
--output_path ./data/metaqa/metaqa-3hop-os/SPLIT.inst.sg.metaqa_rel_retri_for_3hop.3-200.jsonl --num_process 10


# 9.将实例化子图的关系和节点进行映射
python3 map_kg_2_global_id.py --input_path ./data/metaqa/metaqa-1hop/SPLIT.inst.sg.metaqa_rel_retri_for_1hop.3-100.jsonl \
--ori_path data/metaqa/metaqa-1hop/SPLIT.jsonl --output_path data/metaqa/metaqa-1hop/ --output_prefix reason_ \
--split_list train dev test
python3 map_kg_2_global_id.py --input_path ./data/metaqa/metaqa-2hop/SPLIT.inst.sg.metaqa_rel_retri_for_2hop.3-200.jsonl \
--ori_path data/metaqa/metaqa-2hop/SPLIT.jsonl --output_path data/metaqa/metaqa-2hop/ --output_prefix reason_ \
--split_list train dev test
python3 map_kg_2_global_id.py --input_path ./data/metaqa/metaqa-3hop/SPLIT.inst.sg.metaqa_rel_retri_for_3hop.3-200.jsonl \
--ori_path data/metaqa/metaqa-3hop/SPLIT.jsonl --output_path data/metaqa/metaqa-3hop/ --output_prefix reason_ \
--split_list train dev test

# 10.转换为NSM输入形式
## 添加限制实体的逆关系
python3 convert_inst_to_NSM_format.py --original_path data/metaqa/metaqa-1hop/SPLIT.jsonl --max_nodes 100 \
--input_path ./data/metaqa/metaqa-1hop/SPLIT.inst.sg.metaqa_rel_retri_for_1hop.3-100.jsonl \
--output_path ./data/metaqa/metaqa-1hop/SPLIT.reason.sg.max_tris.100.json \
--kg_map_path ./data/metaqa/metaqa-1hop/ --kg_map_prefix reason_ --split_list train dev test
python3 convert_inst_to_NSM_format.py --original_path data/metaqa/metaqa-2hop/SPLIT.jsonl --max_nodes 400 \
--input_path ./data/metaqa/metaqa-2hop/SPLIT.inst.sg.metaqa_rel_retri_for_2hop.3-200.jsonl \
--output_path ./data/metaqa/metaqa-2hop/SPLIT.reason.sg.max_tris.400.json \
--kg_map_path ./data/metaqa/metaqa-2hop/ --kg_map_prefix reason_ --split_list train dev test
python3 convert_inst_to_NSM_format.py --original_path data/metaqa/metaqa-3hop/SPLIT.jsonl --max_nodes 400 \
--input_path ./data/metaqa/metaqa-3hop/SPLIT.inst.sg.metaqa_rel_retri_for_3hop.3-200.jsonl \
--output_path ./data/metaqa/metaqa-3hop/SPLIT.reason.sg.max_tris.400.json \
--kg_map_path ./data/metaqa/metaqa-3hop/ --kg_map_prefix reason_ --split_list train dev test

# 11.跑NSM Reasoner (注意加载数据时，answer映射问题)
# original NSM 67.7
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --model_name gnn \
--data_folder ./data/metaqa/metaqa-1hop-os/ --data_name .reason.sg.max_tris.100.json \
--instruct_model LSTM \
--checkpoint_dir ./retriever_ckpt/metaqa_1hop_rea_os/ \
--batch_size 8 --test_batch_size 500 --num_step 1 --entity_dim 80 --word_dim 300 \
--kg_dim 80 --kge_dim 100 --eval_every 1 --encode_type --experiment_name nsm --eps 0.95 --num_epoch 100 \
--use_self_loop --use_inverse_relation --lr 5e-4 --word2id vocab_new.txt --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id new_reason_r1_relations.txt --entity2id new_reason_r1_entities.txt
# reproduce original data
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --model_name gnn \
--data_folder ./data/metaqa/metaqa_NSM/metaqa-1hop/ --data_name _simple.json \
--instruct_model LSTM \
--checkpoint_dir ./retriever_ckpt/metaqa_1hop_rea/ \
--batch_size 80 --test_batch_size 500 --num_step 1 --entity_dim 80 --word_dim 300 \
--kg_dim 80 --eval_every 1 --encode_type --experiment_name nsm --eps 0.95 --num_epoch 100 \
--use_self_loop --use_inverse_relation --lr 5e-4 --q_type seq --word_emb_file word_emb_300d.npy \
--word2id vocab_new.txt --encode_type --reason_kb --loss_type kl \
--relation2id relations.txt --entity2id entities.txt --overwrite_cache --data_cache ./data/metaqa/nsm_1hop_sg_for_reason.cache \
# 1 hop one shot original data
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --model_name gnn \
--data_folder ./data/metaqa/metaqa_NSM/metaqa-1hop/ --data_name _simple.json \
--instruct_model LSTM \
--checkpoint_dir ./retriever_ckpt/metaqa_1hop_rea/ \
--batch_size 8 --test_batch_size 100 --num_step 1 --entity_dim 80 --word_dim 300 \
--kg_dim 80 --eval_every 1 --encode_type --experiment_name nsm --eps 0.95 --num_epoch 100 \
--use_self_loop --use_inverse_relation --lr 5e-4 --q_type seq --word_emb_file word_emb_300d.npy \
--word2id vocab_new.txt --encode_type --reason_kb --loss_type kl \
--one_shot --sample_idx_path /home/jiangjinhao/work/QA/UniKBQA/UniModel/data/metaqa/metaqa-1hop-os/sample_idx.txt \
--relation2id relations.txt --entity2id entities.txt --overwrite_cache --data_cache ./data/metaqa/nsm_1hop_os_sg_for_reason.cache \
# 2 hop one shot original data
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --model_name gnn \
--data_folder ./data/metaqa/metaqa_NSM/metaqa-2hop/ --data_name _simple.json \
--instruct_model LSTM --lstm_dropout 0.0 --linear_dropout 0.0 \
--checkpoint_dir ./retriever_ckpt/metaqa_2hop_rea/ \
--batch_size 8 --test_batch_size 300 --num_step 2 --entity_dim 80 --word_dim 300 \
--kg_dim 80 --eval_every 2 --encode_type --experiment_name nsm --eps 0.95 --num_epoch 100 \
--use_self_loop --lr 5e-4 --q_type seq --word_emb_file word_emb_300d.npy \
--word2id vocab_new.txt --encode_type --reason_kb --loss_type kl \
--one_shot --sample_idx_path /home/jiangjinhao/work/QA/UniKBQA/UniModel/data/metaqa/metaqa-2hop-os/sample_idx.txt \
--relation2id relations.txt --entity2id entities.txt --overwrite_cache --data_cache ./data/metaqa/nsm_2hop_os_sg_for_reason.cache \
# 3 hop one shot original data
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --model_name gnn \
--data_folder ./data/metaqa/metaqa_NSM/metaqa-3hop/ --data_name _simple.json \
--instruct_model LSTM --lstm_dropout 0.0 --linear_dropout 0.0 \
--checkpoint_dir ./retriever_ckpt/metaqa_3hop_rea/ \
--batch_size 8 --test_batch_size 300 --num_step 3 --entity_dim 80 --word_dim 300 \
--kg_dim 80 --eval_every 1 --encode_type --experiment_name nsm --eps 0.95 --num_epoch 100 \
--use_self_loop --lr 5e-4 --q_type seq --word_emb_file word_emb_300d.npy \
--word2id vocab_new.txt --encode_type --reason_kb --loss_type kl \
--one_shot --sample_idx_path /home/jiangjinhao/work/QA/UniKBQA/UniModel/data/metaqa/metaqa-3hop-os/sample_idx.txt \
--relation2id relations.txt --entity2id entities.txt --overwrite_cache --data_cache ./data/metaqa/nsm_3hop_os_sg_for_reason.cache \



# 1hop
## pretrain,transfer,fix plm
### 1hop transfer best f1 98.03/97.33 98.04/97.37
CUDA_VISIBLE_DEVICES=7 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/metaqa_1hop_nsm/retriever_p_t_wo_pooler-f1.ckpt \
--model_path ./retriever/results/metaqa_rel_retri_for_1hop/ \
--relation_model_path ./retriever/results/metaqa_rel_retri_for_1hop/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-1hop/ --data_name .reason.sg.max_tris.100.json \
--checkpoint_dir ./retriever_ckpt/metaqa_1hop_rea/ --experiment_name reason_p_t_wo_pooler_f1 \
--batch_size 500 --gradient_accumulation_steps 1 --test_batch_size 4000 --num_step 1 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 60 \
--patience 6 --use_self_loop --plm_lr 1e-5 --lr 1e-4 --loss_type kl --reason_kb \
--relation2id reason_relations.txt --entity2id reason_entities.txt --overwrite_cache --data_cache ./data/metaqa/inst_1hop_sg_for_reason.cache \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
### 1hop os transfer best f1 96.65/97.04 96.74/97.03
CUDA_VISIBLE_DEVICES=4 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/metaqa_1hop_nsm_os/retriever_p_t_wo_pooler-f1.ckpt \
--model_path ./retriever/results/metaqa_rel_retri_for_1hop_os/ \
--relation_model_path ./retriever/results/metaqa_rel_retri_for_1hop_os/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-1hop-os/ --data_name .reason.sg.max_tris.100.json \
--checkpoint_dir ./retriever_ckpt/metaqa_1hop_rea_os/ --experiment_name reason_p_t_wo_pooler_f1 \
--batch_size 8 --gradient_accumulation_steps 1 --test_batch_size 4000 --num_step 1 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 60 \
--patience 6 --use_self_loop --plm_lr 1e-5 --lr 1e-4 --loss_type kl --reason_kb \
--relation2id reason_relations.txt --entity2id reason_entities.txt --overwrite_cache --data_cache ./data/metaqa/inst_1hop_os_sg_for_reason.cache \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
# only evaluate
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 --is_eval \
--load_experiment ./retriever_ckpt/metaqa_1hop_rea/reason_p_t_wo_pooler_f1-h1.ckpt \
--retriever_ckpt_path ./retriever_ckpt/metaqa_1hop_nsm/retriever_p_t_wo_pooler-f1.ckpt \
--model_path ./retriever/results/metaqa_rel_retri_for_1hop/ \
--relation_model_path ./retriever/results/metaqa_rel_retri_for_1hop/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-1hop/ --data_name .reason.sg.max_tris.100.json \
--checkpoint_dir ./retriever_ckpt/metaqa_1hop_rea/ --experiment_name reason_p_t_wo_pooler_f1 \
--batch_size 500 --gradient_accumulation_steps 1 --test_batch_size 4000 --num_step 1 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 60 \
--patience 6 --use_self_loop --plm_lr 1e-5 --lr 1e-4 --loss_type kl --reason_kb \
--relation2id reason_relations.txt --entity2id reason_entities.txt --overwrite_cache --data_cache ./data/metaqa/inst_1hop_sg_for_reason.cache \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
### transfer best h1 98.04/97.35 98.04/97.34
CUDA_VISIBLE_DEVICES=6 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/metaqa_1hop_nsm/retriever_p_t_wo_pooler-h1.ckpt \
--model_path ./retriever/results/metaqa_rel_retri_for_1hop/ \
--relation_model_path ./retriever/results/metaqa_rel_retri_for_1hop/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-1hop/ --data_name .reason.sg.max_tris.100.json \
--checkpoint_dir ./retriever_ckpt/metaqa_1hop_rea/ --experiment_name reason_p_t_wo_pooler_h1 \
--batch_size 500 --gradient_accumulation_steps 1 --test_batch_size 4000 --num_step 1 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 60 \
--patience 6 --use_self_loop --plm_lr 1e-5 --lr 1e-4 --loss_type kl --reason_kb \
--relation2id reason_relations.txt --entity2id reason_entities.txt --data_cache ./data/metaqa/inst_1hop_sg_for_reason.cache \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## pretrain, transfer, question updating
### transfer best f1 98.04/97.37 98.04/97.34
CUDA_VISIBLE_DEVICES=1 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/metaqa_1hop_nsm/retriever_p_t_wo_pooler-f1.ckpt \
--model_path ./retriever/results/metaqa_rel_retri_for_1hop/ \
--relation_model_path ./retriever/results/metaqa_rel_retri_for_1hop/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-1hop/ --data_name .reason.sg.max_tris.100.json \
--checkpoint_dir ./retriever_ckpt/metaqa_1hop_rea/ --experiment_name reason_p_t_q_wo_pooler_f1 \
--batch_size 250 --gradient_accumulation_steps 2 --test_batch_size 1000 --num_step 1 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 60 \
--patience 6 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --loss_type kl --reason_kb \
--relation2id reason_relations.txt --entity2id reason_entities.txt --data_cache ./data/metaqa/inst_1hop_sg_for_reason.cache \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate
### transfer best h1 98.04/97.33 98.04/97.27
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/metaqa_1hop_nsm/retriever_p_t_wo_pooler-h1.ckpt \
--model_path ./retriever/results/metaqa_rel_retri_for_1hop/ \
--relation_model_path ./retriever/results/metaqa_rel_retri_for_1hop/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-1hop/ --data_name .reason.sg.max_tris.100.json \
--checkpoint_dir ./retriever_ckpt/metaqa_1hop_rea/ --experiment_name reason_p_t_q_wo_pooler_h1 \
--batch_size 125 --gradient_accumulation_steps 4 --test_batch_size 500 --num_step 1 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 60 \
--patience 6 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --loss_type kl --reason_kb \
--relation2id reason_relations.txt --entity2id reason_entities.txt --data_cache ./data/metaqa/inst_1hop_sg_for_reason.cache \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate


# 2hop
## pretrain, transfer, fix plm
### transfer best f1 99.81/99.99 99.81/99.99
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/metaqa_2hop_nsm/retriever_p_t_wo_pooler-f1.ckpt \
--model_path ./retriever/results/metaqa_rel_retri_for_2hop/ \
--relation_model_path ./retriever/results/metaqa_rel_retri_for_2hop/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-2hop/ --data_name .reason.sg.max_tris.400.json \
--checkpoint_dir ./retriever_ckpt/metaqa_2hop_rea/ --experiment_name reason_p_t_wo_pooler_f1 \
--batch_size 500 --gradient_accumulation_steps 1 --test_batch_size 1000 --num_step 2 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 60 \
--patience 6 --use_self_loop --plm_lr 1e-5 --lr 1e-4 --loss_type kl --reason_kb \
--relation2id reason_relations.txt --entity2id reason_entities.txt --overwrite_cache --data_cache ./data/metaqa/inst_2hop_sg_for_reason.cache \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
### one shot transfer best f1 92.86/97.81 93.57/97.59
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/metaqa_2hop_nsm_os/retriever_p_t_wo_pooler-f1.ckpt \
--model_path ./retriever/results/metaqa_rel_retri_for_2hop-os/ \
--relation_model_path ./retriever/results/metaqa_rel_retri_for_2hop-os/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-2hop-os/ --data_name .reason.sg.max_tris.400.json \
--checkpoint_dir ./retriever_ckpt/metaqa_2hop_rea_os/ --experiment_name reason_p_t_wo_pooler_f1 \
--batch_size 8 --gradient_accumulation_steps 1 --test_batch_size 1000 --num_step 2 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 60 \
--patience 6 --use_self_loop --plm_lr 1e-5 --lr 1e-4 --loss_type kl --reason_kb \
--relation2id reason_relations.txt --entity2id reason_entities.txt --overwrite_cache --data_cache ./data/metaqa/inst_2hop_os_sg_for_reason.cache \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
### transfer best h1 99.79/99.99 99.82/99.99
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/metaqa_2hop_nsm/retriever_p_t_wo_pooler-h1.ckpt \
--model_path ./retriever/results/metaqa_rel_retri_for_2hop/ \
--relation_model_path ./retriever/results/metaqa_rel_retri_for_2hop/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-2hop/ --data_name .reason.sg.max_tris.400.json \
--checkpoint_dir ./retriever_ckpt/metaqa_2hop_rea/ --experiment_name reason_p_t_wo_pooler_h1 \
--batch_size 500 --gradient_accumulation_steps 1 --test_batch_size 1000 --num_step 2 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 60 \
--patience 6 --use_self_loop --plm_lr 1e-5 --lr 1e-4 --loss_type kl --reason_kb \
--relation2id reason_relations.txt --entity2id reason_entities.txt --data_cache ./data/metaqa/inst_2hop_sg_for_reason.cache \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
### one shot transfer best h1 93.33/98.10 93.63/97.98
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/metaqa_2hop_nsm_os/retriever_p_t_wo_pooler-h1.ckpt \
--model_path ./retriever/results/metaqa_rel_retri_for_2hop-os/ \
--relation_model_path ./retriever/results/metaqa_rel_retri_for_2hop-os/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-2hop-os/ --data_name .reason.sg.max_tris.400.json \
--checkpoint_dir ./retriever_ckpt/metaqa_2hop_rea_os/ --experiment_name reason_p_t_wo_pooler_h1 \
--batch_size 8 --gradient_accumulation_steps 1 --test_batch_size 1000 --num_step 2 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 60 \
--patience 6 --use_self_loop --plm_lr 1e-5 --lr 1e-4 --loss_type kl --reason_kb \
--relation2id reason_relations.txt --entity2id reason_entities.txt --data_cache ./data/metaqa/inst_2hop_os_sg_for_reason.cache \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm

# 3hop
## pretrain, transfer, fix plm
### transfer best f1 98.41/99.66 98.40/99.64
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/metaqa_3hop_nsm/retriever_p_t_wo_pooler-f1.ckpt \
--model_path ./retriever/results/metaqa_rel_retri_for_3hop/ \
--relation_model_path ./retriever/results/metaqa_rel_retri_for_3hop/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-3hop/ --data_name .reason.sg.max_tris.400.json \
--checkpoint_dir ./retriever_ckpt/metaqa_3hop_rea/ --experiment_name reason_p_t_wo_pooler_f1 \
--batch_size 250 --gradient_accumulation_steps 2 --test_batch_size 1000 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 60 \
--patience 6 --use_self_loop --plm_lr 1e-5 --lr 1e-4 --loss_type kl --reason_kb \
--relation2id reason_relations.txt --entity2id reason_entities.txt --overwrite_cache --data_cache ./data/metaqa/inst_3hop_sg_for_reason.cache \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
### one shot transfer best f1 85.94/92.66 87.54/92.64
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/metaqa_3hop_nsm_os/retriever_p_t_wo_pooler-f1.ckpt \
--model_path ./retriever/results/metaqa_rel_retri_for_3hop-os/ \
--relation_model_path ./retriever/results/metaqa_rel_retri_for_3hop-os/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-3hop-os/ --data_name .reason.sg.max_tris.400.json \
--checkpoint_dir ./retriever_ckpt/metaqa_3hop_rea_os/ --experiment_name reason_p_t_wo_pooler_f1 \
--batch_size 8 --gradient_accumulation_steps 1 --test_batch_size 2000 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 60 \
--patience 6 --use_self_loop --plm_lr 1e-5 --lr 1e-4 --loss_type kl --reason_kb \
--relation2id reason_relations.txt --entity2id reason_entities.txt --overwrite_cache --data_cache ./data/metaqa/inst_3hop_os_sg_for_reason.cache \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
### transfer best h1 98.43/99.65 98.43/99.61
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/metaqa_3hop_nsm/retriever_p_t_wo_pooler-h1.ckpt \
--model_path ./retriever/results/metaqa_rel_retri_for_3hop/ \
--relation_model_path ./retriever/results/metaqa_rel_retri_for_3hop/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-3hop/ --data_name .reason.sg.max_tris.400.json \
--checkpoint_dir ./retriever_ckpt/metaqa_3hop_rea/ --experiment_name reason_p_t_wo_pooler_h1 \
--batch_size 250 --gradient_accumulation_steps 2 --test_batch_size 1000 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 60 \
--patience 6 --use_self_loop --plm_lr 1e-5 --lr 1e-4 --loss_type kl --reason_kb \
--relation2id reason_relations.txt --entity2id reason_entities.txt --data_cache ./data/metaqa/inst_3hop_sg_for_reason.cache \
--simplify_model --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
## pretrain, transfer, question updating
### transfer best f1 98.40/99.66 98.41/99.66
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/metaqa_3hop_nsm/retriever_p_t_wo_pooler-f1.ckpt \
--model_path ./retriever/results/metaqa_rel_retri_for_3hop/ \
--relation_model_path ./retriever/results/metaqa_rel_retri_for_3hop/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-3hop/ --data_name .reason.sg.max_tris.400.json \
--checkpoint_dir ./retriever_ckpt/metaqa_3hop_rea/ --experiment_name reason_p_t_q_wo_pooler_f1 \
--batch_size 250 --gradient_accumulation_steps 2 --test_batch_size 1000 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 60 \
--patience 6 --use_self_loop --plm_lr 1e-5 --lr 1e-4 --loss_type kl --reason_kb \
--relation2id reason_relations.txt --entity2id reason_entities.txt --data_cache ./data/metaqa/inst_3hop_sg_for_reason.cache \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate
### transfer best h1 98.39/99.67 98.43/99.67
CUDA_VISIBLE_DEVICES=3 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/metaqa_3hop_nsm/retriever_p_t_wo_pooler-h1.ckpt \
--model_path ./retriever/results/metaqa_rel_retri_for_3hop/ \
--relation_model_path ./retriever/results/metaqa_rel_retri_for_3hop/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-3hop/ --data_name .reason.sg.max_tris.400.json \
--checkpoint_dir ./retriever_ckpt/metaqa_3hop_rea/ --experiment_name reason_p_t_q_wo_pooler_h1 \
--batch_size 250 --gradient_accumulation_steps 2 --test_batch_size 1000 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 60 \
--patience 6 --use_self_loop --plm_lr 1e-5 --lr 1e-4 --loss_type kl --reason_kb \
--relation2id reason_relations.txt --entity2id reason_entities.txt --data_cache ./data/metaqa/inst_3hop_sg_for_reason.cache \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate
## pretrain, transfer, update plm
### transfer best f1 98.40/99.66 98.41/99.66 1-2
CUDA_VISIBLE_DEVICES=5 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/metaqa_3hop_nsm/retriever_p_t_wo_pooler-f1.ckpt \
--model_path ./retriever/results/metaqa_rel_retri_for_3hop/ \
--relation_model_path ./retriever/results/metaqa_rel_retri_for_3hop/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-3hop/ --data_name .reason.sg.max_tris.400.json \
--checkpoint_dir ./retriever_ckpt/metaqa_3hop_rea/ --experiment_name reason_p_t_q_r_wo_pooler_f1 \
--batch_size 250 --gradient_accumulation_steps 2 --test_batch_size 1000 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 60 \
--patience 6 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --loss_type kl --reason_kb \
--relation2id reason_relations.txt --entity2id reason_entities.txt --data_cache ./data/metaqa/inst_3hop_sg_for_reason.cache \
--simplify_model
### transfer best h1 98.39/99.67 98.43/99.67 1-3
CUDA_VISIBLE_DEVICES=7 python3 ./nsm_retriever/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/metaqa_3hop_nsm/retriever_p_t_wo_pooler-h1.ckpt \
--model_path ./retriever/results/metaqa_rel_retri_for_3hop/ \
--relation_model_path ./retriever/results/metaqa_rel_retri_for_3hop/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/metaqa/metaqa-3hop/ --data_name .reason.sg.max_tris.400.json \
--checkpoint_dir ./retriever_ckpt/metaqa_3hop_rea/ --experiment_name reason_p_t_q_r_wo_pooler_h1 \
--batch_size 250 --gradient_accumulation_steps 2 --test_batch_size 1000 --num_step 3 --entity_dim 768 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --eps 0.95 --num_epoch 60 \
--patience 6 --use_self_loop --plm_lr 1e-5 --lr 1e-4 --loss_type kl --reason_kb \
--relation2id reason_relations.txt --entity2id reason_entities.txt --data_cache ./data/metaqa/inst_3hop_sg_for_reason.cache \
--simplify_model --fixed_plm_for_relation_encoding --encode_relation_separate