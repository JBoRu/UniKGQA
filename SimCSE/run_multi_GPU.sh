#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:2          # Request 2 GPU "generic resources”.
#SBATCH --tasks-per-node=2   # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in parallel.
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=07-00:00
#SBATCH --output=pretrain.out
#SBATCH --account=rrg-jynie

source ~/pt1.8/bin/activate
module load cuda cudnn
module load java/14.0.2
module load gcc/9.3.0 arrow python scipy-stack

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times

srun python train.py --slurm_flag True --dist_backend nccl --init_method tcp://$MASTER_ADDR:3456 --world_size $SLURM_NTASKS \
    --model_name_or_path /mnt/jiangjinhao/hg_face/roberta-base \
    --train_file ./data/30MQA/train_qr_pair.csv \
    --eval_file ./data/30MQA/dev_qr_pair.csv \
    --output_dir ./retriever_ckpt/rel_tri_pretrain_0 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 400 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --max_seq_length 36 \
    --metric_for_best_model eval_hits1 \
    --load_best_model_at_end \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps 10000 \
    --fp16 \
    "$@"