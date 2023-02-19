# 预训练步数对retrieval和reasoning的影响 26337
#  bs: 30 save steps: 400
CUDA_VISIBLE_DEVICES=0,1,2 python3 -W ignore -m torch.distributed.launch \
--nproc_per_node 3 --master_port 1235 ./retriever/train.py --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
--train_file ./data/webqsp/train.rels.retri.data.neg_50.0.1.csv \
--eval_file ./data/webqsp/dev.rels.retri.data.neg_50.0.1.csv \
--output_dir ./retriever/results/webqsp_rel_retri_bs_30 --per_device_train_batch_size 10 --gradient_accumulation_steps 1 \
--learning_rate 5e-5 --do_train --num_train_epochs 10 --do_eval --evaluation_strategy steps --eval_steps 400 \
--save_steps 400 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05 > ./retriever/results/webqsp_rel_retri_bs_30.log
#  bs: 60 save steps: 200
CUDA_VISIBLE_DEVICES=0,1,2 python3 -W ignore -m torch.distributed.launch \
--nproc_per_node 3 --master_port 1235 ./retriever/train.py --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
--train_file ./data/webqsp/train.rels.retri.data.neg_50.0.1.csv \
--eval_file ./data/webqsp/dev.rels.retri.data.neg_50.0.1.csv \
--output_dir ./retriever/results/webqsp_rel_retri_bs_60 --per_device_train_batch_size 10 --gradient_accumulation_steps 2 \
--learning_rate 5e-5 --do_train --num_train_epochs 10 --do_eval --evaluation_strategy steps --eval_steps 200 \
--save_steps 200 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05 > ./retriever/results/webqsp_rel_retri_bs_60.log
#  bs: 120 save steps: 100
CUDA_VISIBLE_DEVICES=0,1,2 python3 -W ignore -m torch.distributed.launch \
--nproc_per_node 3 --master_port 1235 ./retriever/train.py --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
--train_file ./data/webqsp/train.rels.retri.data.neg_50.0.1.csv \
--eval_file ./data/webqsp/dev.rels.retri.data.neg_50.0.1.csv \
--output_dir ./retriever/results/webqsp_rel_retri_bs_120 --per_device_train_batch_size 10 --gradient_accumulation_steps 4 \
--learning_rate 5e-5 --do_train --num_train_epochs 10 --do_eval --evaluation_strategy steps --eval_steps 100 \
--save_steps 100 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05 > ./retriever/results/webqsp_rel_retri_bs_120.log
#  bs: 240 save steps: 50
CUDA_VISIBLE_DEVICES=0,1,2 python3 -W ignore -m torch.distributed.launch \
--nproc_per_node 3 --master_port 1235 ./retriever/train.py --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
--train_file ./data/webqsp/train.rels.retri.data.neg_50.0.1.csv \
--eval_file ./data/webqsp/dev.rels.retri.data.neg_50.0.1.csv \
--output_dir ./retriever/results/webqsp_rel_retri_bs_240 --per_device_train_batch_size 10 --gradient_accumulation_steps 8 \
--learning_rate 5e-5 --do_train --num_train_epochs 10 --do_eval --evaluation_strategy steps --eval_steps 50 \
--save_steps 50 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05 > ./retriever/results/webqsp_rel_retri_bs_240.log

