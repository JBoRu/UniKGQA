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

python3 map_kg_2_global_id.py --input_path ./data/webqsp/SPLIT.inst.sg.new_rel_retri_1.15-300.jsonl \
--ori_path data/webqsp/SPLIT.jsonl --output_path data/webqsp/ --output_prefix new_reason_r1_ \
--split_list train dev test

python3 convert_inst_to_NSM_format.py --original_path data/webqsp/SPLIT.jsonl --max_nodes 700 \
--input_path ./data/webqsp/SPLIT.inst.sg.new_rel_retri_1.15-300.jsonl \
--output_path ./data/webqsp/SPLIT.reason.sg.max_tris.700.const_reverse.json \
--kg_map_path ./data/webqsp/ --kg_map_prefix new_reason_r1_ --split_list train dev test --add_constraint_reverse