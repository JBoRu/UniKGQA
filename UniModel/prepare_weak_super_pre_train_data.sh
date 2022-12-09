python3 s0_extract_weak_super_relations.py --dense_kg_source virtuoso --extra_hop_flag --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--input_path ./data/webqsp/all_data.jsonl --output_path ./data/webqsp/new.all.shortest.paths.2hop.jsonl \
--ids_path ./data/webqsp/SPLIT.qid.npy --max_hop 2 --max_num_processes 60

python extract_valid_weak_paths.py --min_precision 0.1 --filter_order 0 \
--input_path ./data/webqsp/new.all.shortest.paths.2hop.jsonl --output_path ./data/webqsp/new.all.shortest.paths.2hop.min_pre.0.1.jsonl

python s1_construct_relation_retrieval_training_data.py --dense_kg_source virtuoso --task_name webqsp \
--sparse_kg_source_path ./data/webqsp/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/webqsp/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/webqsp/graph_data/ent2id.pickle --sparse_rel2id_path ./data/webqsp/graph_data/rel2id.pickle \
--input_path ./data/webqsp/new.all.shortest.paths.2hop.min_pre.0.1.jsonl --output_path ./data/webqsp/new.all.rels.retri.data.neg_50.0.1.jsonl \
--max_hop 3 --num_neg 50 --max_num_processes 60

python extract_relation_retriever_training_data.py --split_qid_path ./data/webqsp/SPLIT.qid.npy --split_list train dev \
--pn_pairs_path ./data/webqsp/new.all.rels.retri.data.neg_50.0.1.jsonl --output_path ./data/webqsp/new.SPLIT.rels.retri.data.neg_50.0.1.csv
