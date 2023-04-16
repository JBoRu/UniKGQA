# webqsp command
CUDA_VISIBLE_DEVICES=9 python3 ./simcse_retriever/train.py --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
--train_file ./data/webqsp/train_rels-mh_2hop-neg_50.csv \
--eval_file ./data/webqsp/dev_rels-mh_2hop-neg_50.csv \
--output_dir ./simcse_retriever/results/debug --per_device_train_batch_size 10 --gradient_accumulation_steps 2 \
--learning_rate 5e-5 --do_train --num_train_epochs 10 --do_eval --evaluation_strategy steps --eval_steps 120 \
--save_steps 120 --save_total_limit 1 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05
python ./simcse_retriever/simcse_to_huggingface.py --path ./simcse_retriever/results/webqsp_rel_retri/
# cwq command
CUDA_VISIBLE_DEVICES=4,5,6,7,8,9 python3 -W ignore -m torch.distributed.launch \
--nproc_per_node 6 --master_port 1238 ./simcse_retriever/train.py --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
--train_file ./data/cwq/train_rels-mh_4hop-neg_50.csv --eval_file ./data/cwq/dev_rels-mh_4hop-neg_50.csv \
--output_dir ./simcse_retriever/results/cwq_rel_retri --per_device_train_batch_size 10 --gradient_accumulation_steps 4 \
--learning_rate 5e-5 --do_train --num_train_epochs 6 --do_eval --evaluation_strategy steps --eval_steps 2500 \
--save_steps 2500 --save_total_limit 1 --load_best_model_at_end --metric_for_best_model eval_hits1 --pooler_type cls \
--overwrite_output_dir --temp 0.05
python ./simcse_retriever/simcse_to_huggingface.py --path ./simcse_retriever/results/cwq_rel_retri/

