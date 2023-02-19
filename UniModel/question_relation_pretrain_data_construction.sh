

python3 s0_extract_weak_super_relations.py --extra_hop_flag --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--input_path ./data/webqsp/all_data.jsonl --output_path ./data/webqsp/all_paths-mh_2hop.jsonl \
--ids_path ./data/webqsp/SPLIT.qid.npy --max_hop 2 --max_num_processes 60 --debug

python extract_valid_weak_paths.py --min_precision 0.1 --filter_order 0 \
--input_path ./data/webqsp/all_paths-mh_2hop.jsonl --output_path ./data/webqsp/all_paths-mh_2hop-mp_0.1-fo_0.jsonl

python s1_construct_relation_retrieval_training_data.py --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--input_path ./data/webqsp/all_paths-mh_2hop-mp_0.1-fo_0.jsonl --output_path ./data/webqsp/all_rels-mh_3hop-neg_50.jsonl \
--max_hop 3 --num_neg 50 --max_num_processes 60

python extract_relation_retriever_training_data.py --split_qid_path ./data/webqsp/SPLIT.qid.npy --split_list train dev \
--pn_pairs_path ./data/webqsp/all_rels-mh_3hop-neg_50.jsonl --output_path ./data/webqsp/SPLIT_rels-mh_3hop-neg_50.csv