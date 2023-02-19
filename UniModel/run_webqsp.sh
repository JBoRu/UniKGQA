############################## PLM均不更新 ##############################
### 1. 使用预训练PLM 79.68/79.68
CUDA_VISIBLE_DEVICES=0 python3 main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever_ckpt/rel_tri_pretrain_0/checkpoint-60000/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json --checkpoint_dir ../retriever_ckpt/webqsp_nsm_retri_v0/ \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 400 --num_step 3 \
--entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--experiment_name retriever_nsm_3 --eps 0.95 --num_epoch 100 --patience 20 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
# 0.98 0.98 0.95
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--ori_path data/webqsp/all_data.jsonl \
--input_path data/webqsp/all.abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--output_path data/webqsp/all.instantiate.sg.retriever_nsm_3.15-300.jsonl \
--model_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_3-h1.ckpt \
--arg_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_3-args.json \
--relation2id_path data/webqsp/abs_r1_relations.txt \
--entity2id_path data/webqsp/abs_r1_entities.txt \
--device 0 1 --final_topk 15 --max_deduced_triples 300 --max_hop 3 --num_pro_each_device 6 --overwrite
python extract_instantiate_subgraph.py --split_qid_path ./data/webqsp/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/webqsp/all.instantiate.sg.retriever_nsm_3.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_3.15-300.jsonl --num_process 12
python3 map_kg_2_global_id.py --input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_3.15-300.jsonl \
--ori_path data/webqsp/SPLIT.jsonl --output_path data/webqsp/ --output_prefix reason_r3_ \
--split_list train dev test
python3 convert_to_NSM_format.py --original_path data/webqsp/SPLIT.jsonl --max_nodes 800 \
--input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_3.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.reason.sg.retriever_nsm_3.max_tris.800.const_reverse.json \
--kg_map_path ./data/webqsp/ --kg_map_prefix reason_r3_ --split_list train dev test --add_constraint_reverse
# reason 72.82/72.88
CUDA_VISIBLE_DEVICES=1 python3 ./reader/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_3-f1.ckpt \
--model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ --relation_model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.retriever_nsm_3.max_tris.800.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 100 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name webqsp_nsm_reason_3 --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r3_relations.txt --entity2id reason_r3_entities.txt --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
# wrmup reason 75.81/75.69
CUDA_VISIBLE_DEVICES=4 python3 ./reader/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_7-f1.ckpt \
--model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ --relation_model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.retriever_nsm_3.max_tris.800.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 100 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name webqsp_nsm_reason_3_1 --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r3_relations.txt --entity2id reason_r3_entities.txt --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm

### 2.使用warmup PLM 83.71/82.73
CUDA_VISIBLE_DEVICES=4 python3 main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_1/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json --checkpoint_dir ../retriever_ckpt/webqsp_nsm_retri_v0/ \
--batch_size 10 --gradient_accumulation_steps 4 --test_batch_size 400 --num_step 3 \
--entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--experiment_name retriever_nsm_7 --eps 0.95 --num_epoch 100 --patience 20 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
# 0.98 0.98 0.95
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--ori_path data/webqsp/all_data.jsonl \
--input_path data/webqsp/all.abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--output_path data/webqsp/all.instantiate.sg.retriever_nsm_7.15-300.jsonl \
--model_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_7-h1.ckpt \
--arg_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_7-args.json \
--relation2id_path data/webqsp/abs_r1_relations.txt \
--entity2id_path data/webqsp/abs_r1_entities.txt \
--device 5 6 7 --final_topk 15 --max_deduced_triples 300 --max_hop 3 --num_pro_each_device 6 --overwrite
python extract_instantiate_subgraph.py --split_qid_path ./data/webqsp/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/webqsp/all.instantiate.sg.retriever_nsm_7.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_7.15-300.jsonl --num_process 18
python3 map_kg_2_global_id.py --input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_7.15-300.jsonl \
--ori_path data/webqsp/SPLIT.jsonl --output_path data/webqsp/ --output_prefix reason_r7_ \
--split_list train dev test
python3 convert_to_NSM_format.py --original_path data/webqsp/SPLIT.jsonl --max_nodes 800 \
--input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_7.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.reason.sg.retriever_nsm_7.max_tris.800.const_reverse.json \
--kg_map_path ./data/webqsp/ --kg_map_prefix reason_r7_ --split_list train dev test --add_constraint_reverse
# reason 75.08/74.89
CUDA_VISIBLE_DEVICES=4 python3 ./reader/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_7-f1.ckpt \
--model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ --relation_model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.retriever_nsm_7.max_tris.800.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 100 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name webqsp_nsm_reason_7 --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r7_relations.txt --entity2id reason_r7_entities.txt --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm

### 3.使用 Continue Pretrain PLM 82.37/83.16
CUDA_VISIBLE_DEVICES=0 python3 main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_1_contiune/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json --checkpoint_dir ../retriever_ckpt/webqsp_nsm_retri_v0/ \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 400 --num_step 3 \
--entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--experiment_name retriever_nsm_10 --eps 0.95 --num_epoch 100 --patience 20 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm
# 0.98 0.98 0.96
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--ori_path data/webqsp/all_data.jsonl \
--input_path data/webqsp/all.abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--output_path data/webqsp/all.instantiate.sg.retriever_nsm_10.15-300.jsonl \
--model_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_10-h1.ckpt \
--arg_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_10-args.json \
--relation2id_path data/webqsp/abs_r1_relations.txt \
--entity2id_path data/webqsp/abs_r1_entities.txt \
--device 1 --final_topk 15 --max_deduced_triples 300 --max_hop 3 --num_pro_each_device 10 --overwrite
python extract_instantiate_subgraph.py --split_qid_path ./data/webqsp/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/webqsp/all.instantiate.sg.retriever_nsm_10.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_10.15-300.jsonl --num_process 10
python3 map_kg_2_global_id.py --input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_10.15-300.jsonl \
--ori_path data/webqsp/SPLIT.jsonl --output_path data/webqsp/ --output_prefix reason_r10_ \
--split_list train dev test
python3 convert_to_NSM_format.py --original_path data/webqsp/SPLIT.jsonl --max_nodes 800 \
--input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_10.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.reason.sg.retriever_nsm_10.max_tris.800.const_reverse.json \
--kg_map_path ./data/webqsp/ --kg_map_prefix reason_r10_ --split_list train dev test --add_constraint_reverse
# reason 75.50/75.08
CUDA_VISIBLE_DEVICES=0 python3 ./reader/main_nsm.py --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_10-f1.ckpt \
--model_path ./retriever/results/webqsp_rel_retri_1_contiune/ --relation_model_path ./retriever/results/webqsp_rel_retri_1_contiune/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.retriever_nsm_10.max_tris.800.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 100 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name webqsp_nsm_reason_10 --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r10_relations.txt --entity2id reason_r10_entities.txt --fixed_plm_for_relation_encoding --fixed_plm_for_query_encoding --fix_all_plm


############################## PLM仅更新Q且仅编码一次relation ##############################
### 1.使用预训练PLM 79.01/78.95
CUDA_VISIBLE_DEVICES=0 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever_ckpt/rel_tri_pretrain_0/checkpoint-60000/ --relation_model_path ../retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ \
--agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json --checkpoint_dir ../retriever_ckpt/webqsp_nsm_retri_v0/ \
--batch_size 10 --gradient_accumulation_steps 4 --test_batch_size 400 --num_step 3 \
--entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--experiment_name retriever_nsm_5 --eps 0.95 --num_epoch 100 --patience 20 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt --fixed_plm_for_relation_encoding --encode_relation_separate
# 0.98 0.97 0.95
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--ori_path data/webqsp/all_data.jsonl \
--input_path data/webqsp/all.abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--output_path data/webqsp/all.instantiate.sg.retriever_nsm_5.15-300.jsonl \
--model_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_5-h1.ckpt \
--arg_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_5-args.json \
--relation2id_path data/webqsp/abs_r1_relations.txt \
--entity2id_path data/webqsp/abs_r1_entities.txt \
--device 0 1 --final_topk 15 --max_deduced_triples 300 --max_hop 3 --num_pro_each_device 6 --overwrite
python extract_instantiate_subgraph.py --split_qid_path ./data/webqsp/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/webqsp/all.instantiate.sg.retriever_nsm_5.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_5.15-300.jsonl --num_process 12
python3 map_kg_2_global_id.py --input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_5.15-300.jsonl \
--ori_path data/webqsp/SPLIT.jsonl --output_path data/webqsp/ --output_prefix reason_r5_ \
--split_list train dev test
python3 convert_to_NSM_format.py --original_path data/webqsp/SPLIT.jsonl --max_nodes 800 \
--input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_5.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.reason.sg.retriever_nsm_5.max_tris.800.const_reverse.json \
--kg_map_path ./data/webqsp/ --kg_map_prefix reason_r5_ --split_list train dev test --add_constraint_reverse
# reason 72.02/71.96
CUDA_VISIBLE_DEVICES=1 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_5-f1.ckpt \
--model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ --relation_model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.retriever_nsm_5.max_tris.800.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 100 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name webqsp_nsm_reason_5 --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r5_relations.txt --entity2id reason_r5_entities.txt --fixed_plm_for_relation_encoding --encode_relation_separate
# warmup reason 73.79/74.83
CUDA_VISIBLE_DEVICES=3 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_9-f1.ckpt \
--model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ --relation_model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.retriever_nsm_5.max_tris.800.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 100 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name webqsp_nsm_reason_5_1 --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r5_relations.txt --entity2id reason_r5_entities.txt --fixed_plm_for_relation_encoding --encode_relation_separate

### 2.使用warmup PLM 83.22/82.73
CUDA_VISIBLE_DEVICES=5 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_1/ --relation_model_path ../retriever/results/webqsp_rel_retri_1/ \
--agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json --checkpoint_dir ../retriever_ckpt/webqsp_nsm_retri_v0/ \
--batch_size 10 --gradient_accumulation_steps 4 --test_batch_size 400 --num_step 3 \
--entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--experiment_name retriever_nsm_9 --eps 0.95 --num_epoch 100 --patience 20 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt --fixed_plm_for_relation_encoding --encode_relation_separate
# 0.98 0.98 0.95
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--ori_path data/webqsp/all_data.jsonl \
--input_path data/webqsp/all.abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--output_path data/webqsp/all.instantiate.sg.retriever_nsm_9.15-300.jsonl \
--model_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_9-h1.ckpt \
--arg_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_9-args.json \
--relation2id_path data/webqsp/abs_r1_relations.txt \
--entity2id_path data/webqsp/abs_r1_entities.txt \
--device 5 6 7 --final_topk 15 --max_deduced_triples 300 --max_hop 3 --num_pro_each_device 6 --overwrite
python extract_instantiate_subgraph.py --split_qid_path ./data/webqsp/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/webqsp/all.instantiate.sg.retriever_nsm_9.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_9.15-300.jsonl --num_process 18
python3 map_kg_2_global_id.py --input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_9.15-300.jsonl \
--ori_path data/webqsp/SPLIT.jsonl --output_path data/webqsp/ --output_prefix reason_r9_ \
--split_list train dev test
python3 convert_to_NSM_format.py --original_path data/webqsp/SPLIT.jsonl --max_nodes 800 \
--input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_9.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.reason.sg.retriever_nsm_9.max_tris.800.const_reverse.json \
--kg_map_path ./data/webqsp/ --kg_map_prefix reason_r9_ --split_list train dev test --add_constraint_reverse
# reason 76.11/75.44
CUDA_VISIBLE_DEVICES=4 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_9-f1.ckpt \
--model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ --relation_model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.retriever_nsm_9.max_tris.800.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 100 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name webqsp_nsm_reason_9 --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r9_relations.txt --entity2id reason_r9_entities.txt --fixed_plm_for_relation_encoding --encode_relation_separate

### 2.使用 Continue Pretrain PLM 79.56/81.70
CUDA_VISIBLE_DEVICES=0 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_1_contiune/ --relation_model_path ../retriever/results/webqsp_rel_retri_1/ \
--agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json --checkpoint_dir ../retriever_ckpt/webqsp_nsm_retri_v0/ \
--batch_size 10 --gradient_accumulation_steps 4 --test_batch_size 400 --num_step 3 \
--entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--experiment_name retriever_nsm_11 --eps 0.95 --num_epoch 100 --patience 20 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt --fixed_plm_for_relation_encoding --encode_relation_separate
# 0.98 0.98 0.95
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--ori_path data/webqsp/all_data.jsonl \
--input_path data/webqsp/all.abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--output_path data/webqsp/all.instantiate.sg.retriever_nsm_11.15-300.jsonl \
--model_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_11-h1.ckpt \
--arg_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_11-args.json \
--relation2id_path data/webqsp/abs_r1_relations.txt \
--entity2id_path data/webqsp/abs_r1_entities.txt \
--device 1 --final_topk 15 --max_deduced_triples 300 --max_hop 3 --num_pro_each_device 6 --overwrite
python extract_instantiate_subgraph.py --split_qid_path ./data/webqsp/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/webqsp/all.instantiate.sg.retriever_nsm_11.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_11.15-300.jsonl --num_process 6
python3 map_kg_2_global_id.py --input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_11.15-300.jsonl \
--ori_path data/webqsp/SPLIT.jsonl --output_path data/webqsp/ --output_prefix reason_r11_ \
--split_list train dev test
python3 convert_to_NSM_format.py --original_path data/webqsp/SPLIT.jsonl --max_nodes 800 \
--input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_11.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.reason.sg.retriever_nsm_11.max_tris.800.const_reverse.json \
--kg_map_path ./data/webqsp/ --kg_map_prefix reason_r11_ --split_list train dev test --add_constraint_reverse
# reason 4-2
CUDA_VISIBLE_DEVICES=0 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_11-f1.ckpt \
--model_path ./retriever/results/webqsp_rel_retri_1_contiune/ --relation_model_path ./retriever/results/webqsp_rel_retri_1_contiune/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.retriever_nsm_11.max_tris.800.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 100 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name webqsp_nsm_reason_11 --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r11_relations.txt --entity2id reason_r11_entities.txt --fixed_plm_for_relation_encoding --encode_relation_separate


############################## PLM仅更新Q同时编码relation ##############################
### 1.使用预训练PLM 81.33/80.29
CUDA_VISIBLE_DEVICES=0 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever_ckpt/rel_tri_pretrain_0/checkpoint-60000/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json --checkpoint_dir ../retriever_ckpt/webqsp_nsm_retri_v0/ \
--batch_size 10 --gradient_accumulation_steps 4 --test_batch_size 400 --num_step 3 \
--entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--experiment_name retriever_nsm_4 --eps 0.95 --num_epoch 100 --patience 20 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt --fixed_plm_for_relation_encoding
# 0.98 0.98 0.95
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--ori_path data/webqsp/all_data.jsonl \
--input_path data/webqsp/all.abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--output_path data/webqsp/all.instantiate.sg.retriever_nsm_4.15-300.jsonl \
--model_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_4-h1.ckpt \
--arg_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_4-args.json \
--relation2id_path data/webqsp/abs_r1_relations.txt \
--entity2id_path data/webqsp/abs_r1_entities.txt \
--device 5 6 7 --final_topk 15 --max_deduced_triples 300 --max_hop 3 --num_pro_each_device 6 --overwrite
python extract_instantiate_subgraph.py --split_qid_path ./data/webqsp/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/webqsp/all.instantiate.sg.retriever_nsm_4.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_4.15-300.jsonl --num_process 18
python3 map_kg_2_global_id.py --input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_4.15-300.jsonl \
--ori_path data/webqsp/SPLIT.jsonl --output_path data/webqsp/ --output_prefix reason_r4_ \
--split_list train dev test
python3 convert_to_NSM_format.py --original_path data/webqsp/SPLIT.jsonl --max_nodes 800 \
--input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_4.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.reason.sg.retriever_nsm_4.max_tris.800.const_reverse.json \
--kg_map_path ./data/webqsp/ --kg_map_prefix reason_r4_ --split_list train dev test --add_constraint_reverse
# reason
CUDA_VISIBLE_DEVICES=5 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_4-f1.ckpt \
--model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ --relation_model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.retriever_nsm_4.max_tris.800.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 100 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name webqsp_nsm_reason_4 --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r4_relations.txt --entity2id reason_r4_entities.txt --fixed_plm_for_relation_encoding
# warmup reason 75.75/74.65
CUDA_VISIBLE_DEVICES=3 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_8-f1.ckpt \
--model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ --relation_model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.retriever_nsm_4.max_tris.800.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 100 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name webqsp_nsm_reason_4_1 --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r4_relations.txt --entity2id reason_r4_entities.txt --fixed_plm_for_relation_encoding

### 2.使用warmup PLM 83.47/82.73
CUDA_VISIBLE_DEVICES=5 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_1/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json --checkpoint_dir ../retriever_ckpt/webqsp_nsm_retri_v0/ \
--batch_size 10 --gradient_accumulation_steps 4 --test_batch_size 400 --num_step 3 \
--entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--experiment_name retriever_nsm_8 --eps 0.95 --num_epoch 100 --patience 20 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt --fixed_plm_for_relation_encoding
# 0.98 0.98 0.95
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--ori_path data/webqsp/all_data.jsonl \
--input_path data/webqsp/all.abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--output_path data/webqsp/all.instantiate.sg.retriever_nsm_8.15-300.jsonl \
--model_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_8-h1.ckpt \
--arg_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_8-args.json \
--relation2id_path data/webqsp/abs_r1_relations.txt \
--entity2id_path data/webqsp/abs_r1_entities.txt \
--device 6 7 --final_topk 15 --max_deduced_triples 300 --max_hop 3 --num_pro_each_device 6 --overwrite
python extract_instantiate_subgraph.py --split_qid_path ./data/webqsp/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/webqsp/all.instantiate.sg.retriever_nsm_8.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_8.15-300.jsonl --num_process 12
python3 map_kg_2_global_id.py --input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_8.15-300.jsonl \
--ori_path data/webqsp/SPLIT.jsonl --output_path data/webqsp/ --output_prefix reason_r8_ \
--split_list train dev test
python3 convert_to_NSM_format.py --original_path data/webqsp/SPLIT.jsonl --max_nodes 800 \
--input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_8.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.reason.sg.retriever_nsm_8.max_tris.800.const_reverse.json \
--kg_map_path ./data/webqsp/ --kg_map_prefix reason_r8_ --split_list train dev test --add_constraint_reverse
# reason 76.60/75.75
CUDA_VISIBLE_DEVICES=5 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_8-f1.ckpt \
--model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ --relation_model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.retriever_nsm_8.max_tris.800.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 100 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name webqsp_nsm_reason_8 --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r8_relations.txt --entity2id reason_r8_entities.txt --fixed_plm_for_relation_encoding

### 3.使用 Continue PLM 82.25/82.67
CUDA_VISIBLE_DEVICES=0 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_1_contiune/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json --checkpoint_dir ../retriever_ckpt/webqsp_nsm_retri_v0/ \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 400 --num_step 3 \
--entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--experiment_name retriever_nsm_12 --eps 0.95 --num_epoch 100 --patience 20 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt --fixed_plm_for_relation_encoding
# 0.98 0.98 0.95
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--ori_path data/webqsp/all_data.jsonl \
--input_path data/webqsp/all.abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--output_path data/webqsp/all.instantiate.sg.retriever_nsm_12.15-300.jsonl \
--model_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_12-h1.ckpt \
--arg_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_12-args.json \
--relation2id_path data/webqsp/abs_r1_relations.txt \
--entity2id_path data/webqsp/abs_r1_entities.txt \
--device 1 --final_topk 15 --max_deduced_triples 300 --max_hop 3 --num_pro_each_device 6 --overwrite
python extract_instantiate_subgraph.py --split_qid_path ./data/webqsp/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/webqsp/all.instantiate.sg.retriever_nsm_12.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_12.15-300.jsonl --num_process 6
python3 map_kg_2_global_id.py --input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_12.15-300.jsonl \
--ori_path data/webqsp/SPLIT.jsonl --output_path data/webqsp/ --output_prefix reason_r12_ \
--split_list train dev test
python3 convert_to_NSM_format.py --original_path data/webqsp/SPLIT.jsonl --max_nodes 800 \
--input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_12.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.reason.sg.retriever_nsm_12.max_tris.800.const_reverse.json \
--kg_map_path ./data/webqsp/ --kg_map_prefix reason_r12_ --split_list train dev test --add_constraint_reverse
# reason
CUDA_VISIBLE_DEVICES=5 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_8-f1.ckpt \
--model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ --relation_model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ \
--agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.retriever_nsm_8.max_tris.800.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 100 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name webqsp_nsm_reason_8 --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r8_relations.txt --entity2id reason_r8_entities.txt --fixed_plm_for_relation_encoding


############################## PLM均更新 ##############################
########使用预训练PLM retrieve 81.76/81.51
CUDA_VISIBLE_DEVICES=1 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever_ckpt/rel_tri_pretrain_0/checkpoint-60000/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json --checkpoint_dir ../retriever_ckpt/webqsp_nsm_retri_v0/ \
--batch_size 10 --gradient_accumulation_steps 4 --test_batch_size 400 --num_step 3 \
--entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--experiment_name retriever_nsm_2 --eps 0.95 --num_epoch 100 --patience 20 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt
# 0.98 0.98 0.95
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--ori_path data/webqsp/all_data.jsonl \
--input_path data/webqsp/all.abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--output_path data/webqsp/all.instantiate.sg.retriever_nsm_2.15-300.jsonl \
--model_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_2-h1.ckpt \
--arg_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_2-args.json \
--relation2id_path data/webqsp/abs_r1_relations.txt \
--entity2id_path data/webqsp/abs_r1_entities.txt \
--device 3 4 --final_topk 15 --max_deduced_triples 300 --max_hop 3 --num_pro_each_device 6 --overwrite
python extract_instantiate_subgraph.py --split_qid_path ./data/webqsp/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/webqsp/all.instantiate.sg.retriever_nsm_2.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_2.15-300.jsonl --num_process 12
python3 map_kg_2_global_id.py --input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_2.15-300.jsonl \
--ori_path data/webqsp/SPLIT.jsonl --output_path data/webqsp/ --output_prefix reason_r2_ \
--split_list train dev test
python3 convert_to_NSM_format.py --original_path data/webqsp/SPLIT.jsonl --max_nodes 800 \
--input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_2.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.reason.sg.retriever_nsm_2.max_tris.800.const_reverse.json \
--kg_map_path ./data/webqsp/ --kg_map_prefix reason_r2_ --split_list train dev test --add_constraint_reverse
# reason 7-3
CUDA_VISIBLE_DEVICES=7 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_2-f1.ckpt \
--model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ --agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.retriever_nsm_2.max_tris.800.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 100 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name webqsp_nsm_reason_2 --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r2_relations.txt --entity2id reason_r2_entities.txt
# warmup reason 75.75/74.22
CUDA_VISIBLE_DEVICES=4 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_6-f1.ckpt \
--model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ --agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.retriever_nsm_2.max_tris.800.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 100 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name webqsp_nsm_reason_2_1 --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r2_relations.txt --entity2id reason_r2_entities.txt


# 使用warmup PLM 83.53/82.98
CUDA_VISIBLE_DEVICES=3 python3 main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--model_path ../retriever/results/webqsp_rel_retri_1/ --agent_type PLM --instruct_model PLM --model_name gnn --data_folder ../data/webqsp/ \
--data_name .abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json --checkpoint_dir ../retriever_ckpt/webqsp_nsm_retri_v0/ \
--batch_size 10 --gradient_accumulation_steps 4 --test_batch_size 400 --num_step 3 \
--entity_dim 100 --word_dim 300 --kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type \
--experiment_name retriever_nsm_6 --eps 0.95 --num_epoch 100 --patience 20 --use_self_loop --plm_lr 1e-5 --lr 5e-4 \
--word_emb_file word_emb_300d.npy --loss_type kl --reason_kb --data_cache ../data/webqsp/abs_r1_sg_for_retri.cache \
--relation2id abs_r1_relations.txt --entity2id abs_r1_entities.txt
# 0.98 0.99 0.95
python3 s3_retrieve_subgraph_by_nsm.py --dense_kg_source virtuoso --max_num_processes 40 --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--ori_path data/webqsp/all_data.jsonl \
--input_path data/webqsp/all.abstract.sg.topk.15.min_score.0.max_tris.1000.rel_retri_1.nsm.json \
--output_path data/webqsp/all.instantiate.sg.retriever_nsm_6.15-300.jsonl \
--model_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_6-h1.ckpt \
--arg_path retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_6-args.json \
--relation2id_path data/webqsp/abs_r1_relations.txt \
--entity2id_path data/webqsp/abs_r1_entities.txt \
--device 6 7 --final_topk 15 --max_deduced_triples 300 --max_hop 3 --num_pro_each_device 6 --overwrite
python extract_instantiate_subgraph.py --split_qid_path ./data/webqsp/SPLIT.qid.npy --split_list train dev test \
--ins_sg_path ./data/webqsp/all.instantiate.sg.retriever_nsm_6.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_6.15-300.jsonl --num_process 12
python3 map_kg_2_global_id.py --input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_6.15-300.jsonl \
--ori_path data/webqsp/SPLIT.jsonl --output_path data/webqsp/ --output_prefix reason_r6_ \
--split_list train dev test
python3 convert_to_NSM_format.py --original_path data/webqsp/SPLIT.jsonl --max_nodes 800 \
--input_path ./data/webqsp/SPLIT.instantiate.sg.retriever_nsm_6.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.reason.sg.retriever_nsm_6.max_tris.800.const_reverse.json \
--kg_map_path ./data/webqsp/ --kg_map_prefix reason_r6_ --split_list train dev test --add_constraint_reverse
# reason 75.14/75.32
CUDA_VISIBLE_DEVICES=5 python3 ./reader/main_nsm.py --diff_lr --linear_dropout 0.1 --log_steps 50 \
--retriever_ckpt_path ./retriever_ckpt/webqsp_nsm_retri_v0/retriever_nsm_6-f1.ckpt \
--model_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-140000/ --agent_type PLM --instruct_model PLM --model_name gnn \
--data_folder ./data/webqsp/ --data_name .reason.sg.retriever_nsm_6.max_tris.800.const_reverse.json \
--checkpoint_dir ./retriever_ckpt/nsm_reasoner/ \
--batch_size 40 --gradient_accumulation_steps 1 --test_batch_size 80 --num_step 3 --entity_dim 100 --word_dim 300 \
--kg_dim 384 --kge_dim 100 --eval_every 1 --encode_type --experiment_name webqsp_nsm_reason_6 --eps 0.95 --num_epoch 120 \
--patience 15 --use_self_loop --plm_lr 1e-5 --lr 5e-4 --word_emb_file word_emb_300d.npy --loss_type kl --reason_kb \
--relation2id reason_r6_relations.txt --entity2id reason_r6_entities.txt



