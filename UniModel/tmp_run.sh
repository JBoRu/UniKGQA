CUDA_VISIBLE_DEVICES=3,4,5,6 python3 -W ignore -m torch.distributed.launch \
--nproc_per_node 4 --master_port 1237 ./retriever/train.py --model_name_or_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-60000/ \
--train_file ./data/webqsp/train.rels.retri.data.neg_50.0.2.csv \
--eval_file ./data/webqsp/dev.rels.retri.data.neg_50.0.2.csv \
--output_dir ./retriever/results/webqsp_rel_retri_1_contiune --per_device_train_batch_size 10 --gradient_accumulation_steps 1 \
--learning_rate 5e-5 --do_train --num_train_epochs 15 --do_eval --evaluation_strategy steps --eval_steps 200 \
--save_steps 200 --save_total_limit 1 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05
CUDA_VISIBLE_DEVICES=3,4,5,6 python3 -W ignore -m torch.distributed.launch \
--nproc_per_node 4 --master_port 1238 ./retriever/train.py --model_name_or_path ./retriever_ckpt/rel_tri_pretrain_0/checkpoint-60000/ \
--train_file ./data/cwq/train.rels.retri.data.neg_50.0.05.csv --eval_file ./data/cwq/dev.rels.retri.data.neg_50.0.05.csv \
--output_dir ./retriever/results/cwq_rel_retri_0_contiune --per_device_train_batch_size 8 --gradient_accumulation_steps 2 \
--learning_rate 5e-5 --do_train --num_train_epochs 15 --do_eval --evaluation_strategy steps --eval_steps 1500 \
--save_steps 1500 --save_total_limit 1 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05

