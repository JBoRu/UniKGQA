#!/bin/bash
#SBATCH --nodes 1

#SBATCH --tasks-per-node=1

#SBATCH --cpus-per-task=40 # change this parameter to 2,4,6,... and increase "--num_workers" accordingly to see the effect on performance

#SBATCH --mem=200G

#SBATCH --time=07-00:00

#SBATCH --output=s0_cwq.out

#SBATCH --account=def-jynie # rrg-jynie

module load cuda cudnn
source ~/pt1.8/bin/activate

echo "starting training..."
python3 s0_extract_weak_super_relations.py --dense_kg_source virtuoso --extra_hop_flag --task_name cwq \
--sparse_kg_source_path ./data/cwq/graph_data/subgraph_2hop_triples.npy --sparse_ent_type_path ./data/cwq/graph_data/ent_type_ary.npy \
--sparse_ent2id_path ./data/cwq/graph_data/ent2id.pickle --sparse_rel2id_path ./data/cwq/graph_data/rel2id.pickle \
--input_path ./data/cwq/all_data.jsonl --output_path ./data/cwq/all.shortest.paths.4hop.jsonl --ids_path ./data/cwq/SPLIT.qid.npy \
--max_hop 4 --max_num_processes 60 --overwrite