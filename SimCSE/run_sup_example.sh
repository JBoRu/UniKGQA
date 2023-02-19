#!/bin/bash

# In this example, we show how to train SimCSE using multiple GPU cards and PyTorch's distributed data parallel on supervised NLI dataset.
# Set how many GPUs to use

NUM_GPU=3

# Randomly set a port number
# If you encounter "address already used" error, just run again or manually set an available port id.
# PORT_ID=$(expr $RANDOM + 1000)
PORT_ID=1238

# Allow multiple threads
export OMP_NUM_THREADS=60

# Use distributed data parallel
# If you only want to use one card, uncomment the following line and comment the line with "torch.distributed.launch"
# python train.py \
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port $PORT_ID train.py \
    --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
    --train_file ./data/simplequestions/train_qr_pair.csv \
    --eval_file ./data/simplequestions/dev_qr_pair.csv \
    --output_dir ./retriever_ckpt/rel_tri_pretrain_with_simpleqa \
    --num_train_epochs 10 \
    --per_device_train_batch_size 200 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --max_seq_length 40 \
    --metric_for_best_model eval_hits1 \
    --load_best_model_at_end \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 90 \
    --fp16 \
    "$@"
