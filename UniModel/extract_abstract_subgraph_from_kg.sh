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

python3 map_abs_kg_2_global_id.py --input_path data/webqsp/SPLIT.abs.sg.topk.15.min_sim.0.max_tris.1000.new_rel_retri_1.jsonl \
--output_path data/webqsp/ --output_prefix new_abs_r1_ --split_list train dev test

python3 convert_abs_to_NSM_format.py \
--input_path data/webqsp/SPLIT.abs.sg.topk.15.min_sim.0.max_tris.1000.new_rel_retri_1.jsonl \
--output_path data/webqsp/SPLIT.abs.sg.topk.15.min_sim.0.max_tris.1000.new_rel_retri_1.nsm.json \
--all_output_path data/webqsp/all.abs.sg.topk.15.min_sim.0.max_tris.1000.new_rel_retri_1.nsm.json \
--kg_map_path data/webqsp/ --kg_map_prefix new_abs_r1_ --max_nodes 1000 --split_list train dev test
