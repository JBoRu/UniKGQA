#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.
#    --train_file ../../WebQSP/outputs/webqsp.test.multi_hop.15.debug.csv \


CUDA_VISIBLE_DEVICES=2 python train.py \
    --model_name_or_path roberta-base \
    --train_file ../data/webqsp/data/webqsp.split.train.weak.multi_hop.neg_10.csv \
    --output_dir results/my-roberta-base \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --max_seq_length 64 \
    --save_steps 1000 \
    --save_total_limit 5 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --fp16 \
    "$@"
