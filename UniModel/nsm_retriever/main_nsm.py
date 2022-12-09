import argparse
import sys
from NSM.train.trainer_nsm import Trainer_KBQA
from NSM.util.utils import create_logger
import time
import torch
import numpy as np
import os
import datetime
import copy
import wandb
from NSM.util.utils import setup_seed
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser()

parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--find_unused_parameters", action="store_true")

# datasets
parser.add_argument('--name', default='webqsp', type=str)
parser.add_argument('--model_name', default='rw', type=str)
parser.add_argument('--data_folder', default='dataset/webqsp/', type=str)
parser.add_argument('--data_name', default='_simple.json', type=str)
parser.add_argument('--overwrite_cache', action='store_true')
parser.add_argument('--data_cache', default=None, type=str) # 'dataset/webqsp/dataset.cache'
parser.add_argument('--one_shot', action='store_true') # load only one shot idx samples for original NSM
parser.add_argument('--sample_idx_path', type=str) # load only one shot idx samples for original NSM

# embeddings
# parser.add_argument('--word2id', default='vocab_new.txt', type=str)
parser.add_argument('--word2id', type=str)
parser.add_argument('--relation2id', default='super_sg_relations.txt', type=str)
parser.add_argument('--entity2id', default='entities.txt', type=str)
parser.add_argument('--char2id', default='chars.txt', type=str)
parser.add_argument('--entity_emb_file', default=None, type=str)
parser.add_argument('--entity_kge_file', default=None, type=str)
parser.add_argument('--relation_emb_file', default=None, type=str)
parser.add_argument('--relation_kge_file', default=None, type=str)
parser.add_argument('--word_emb_file', default='word_emb_300d.npy', type=str)
parser.add_argument('--rel_word_ids', default='rel_word_idx.npy', type=str)
parser.add_argument('--not_use_ent_emb', action='store_true')

# GraftNet embeddings
parser.add_argument('--pretrained_entity_kge_file', default='entity_emb_100d.npy', type=str)

# dimensions, layers, dropout
parser.add_argument('--entity_dim', default=100, type=int)
parser.add_argument('--kge_dim', default=100, type=int)
parser.add_argument('--kg_dim', default=100, type=int)
parser.add_argument('--word_dim', default=300, type=int)
parser.add_argument('--lstm_dropout', default=0.3, type=float)
parser.add_argument('--linear_dropout', default=0.2, type=float)

# optimization
parser.add_argument('--num_epoch', default=100, type=int)
parser.add_argument('--fact_scale', default=3, type=int)
parser.add_argument('--eval_every', default=5, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--gradient_clip', default=1.0, type=float)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--plm_lr', default=0.001, type=float)
parser.add_argument('--decay_rate', default=0.0, type=float)
parser.add_argument('--seed', default=19960626, type=int)
parser.add_argument('--lr_schedule', action='store_true')
parser.add_argument('--label_smooth', default=0.1, type=float)
parser.add_argument('--fact_drop', default=0, type=float)
parser.add_argument('--diff_lr', default=False, action='store_true')
parser.add_argument('--weight_decay', default=0.01, type=float)
parser.add_argument('--fix_plm_layer', default=False, action='store_true')
parser.add_argument('--fix_all_plm', default=False, action='store_true')
parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
parser.add_argument('--init_same_as_plm', action='store_true')

# model options
parser.add_argument('--q_type', default='seq', type=str)
parser.add_argument('--share_encoder', action='store_true')
parser.add_argument('--use_inverse_relation', action='store_true')
parser.add_argument('--use_self_loop', action='store_true')
parser.add_argument('--train_KL', action='store_true')
parser.add_argument('--is_eval', action='store_true')
parser.add_argument('--checkpoint_dir', default='checkpoint/', type=str)
parser.add_argument('--log_level', type=str, default='info')
parser.add_argument('--experiment_name', default='debug', type=str)
parser.add_argument('--load_experiment', default=None, type=str)
parser.add_argument('--load_ckpt_file', default=None, type=str)
parser.add_argument('--eps', default=0.05, type=float) # threshold for f1
parser.add_argument('--model_path', type=str, help='PLM model path')
parser.add_argument('--relation_model_path', type=str, help='PLM model path')
parser.add_argument('--instruct_model', type=str, default='LSTM')
parser.add_argument('--agent_type', type=str, default=None)
parser.add_argument('--report_to', type=str, default="wandb")
parser.add_argument('--record_weight_gradient', action='store_true')
parser.add_argument('--log_steps', type=int, default=1)
parser.add_argument('--patience', type=int, default=15)
parser.add_argument('--retriever_ckpt_path', default=None, type=str)
parser.add_argument('--fixed_plm_for_query_encoding', action='store_true')
parser.add_argument('--fixed_plm_for_relation_encoding', action='store_true')
parser.add_argument('--encode_relation_separate', action='store_true')
parser.add_argument('--simplify_model', action='store_true')
parser.add_argument('--use_pooler_output', action='store_true')
parser.add_argument('--aggregate_token_representation', action='store_true')


# RL options
parser.add_argument('--filter_sub', action='store_true')
parser.add_argument('--encode_type', action='store_true')
parser.add_argument('--reason_kb', action='store_true')
parser.add_argument('--num_layer', default=1, type=int)
parser.add_argument('--test_batch_size', default=20, type=int)
parser.add_argument('--num_step', default=1, type=int)
parser.add_argument('--mode', default='teacher', type=str)
parser.add_argument('--entropy_weight', default=0.0, type=float)

parser.add_argument('--use_label', action='store_true')
parser.add_argument('--tree_soft', action='store_true')
parser.add_argument('--filter_label', action='store_true')
parser.add_argument('--share_embedding', action='store_true')
parser.add_argument('--share_instruction', action='store_true')
parser.add_argument('--encoder_type', default='lstm', type=str)
parser.add_argument('--lambda_label', default=0.1, type=float)
parser.add_argument('--lambda_auto', default=0.01, type=float)
parser.add_argument('--label_f1', default=0.5, type=float)
parser.add_argument('--loss_type', default='kl', type=str)
parser.add_argument('--label_file', default=None, type=str)

args = parser.parse_args()
args.use_cuda = torch.cuda.is_available()

setup_seed(args.seed)

if args.experiment_name == None:
    timestamp = str(int(time.time()))
    args.experiment_name = "{}-{}-{}".format(
        args.dataset,
        args.model_name,
        timestamp,
    )


def main():
    print("Self local rank is: ", args.local_rank)
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        local_rank = dist.get_rank()
        print("Using DDP in %d - %d" % (local_rank, args.local_rank))
    else:
        print("Using single GPU")

    if not os.path.exists(args.checkpoint_dir) and args.local_rank <= 0:
        os.mkdir(args.checkpoint_dir)

    if args.local_rank <= 0:
        logger = create_logger(args)

        if args.report_to == "wandb":
            logger.info("Set wandb")
            wandb.init(
                project="UniKGQA",
                name=args.experiment_name,
                entity='UniKGQA',
            )
            wandb.config.update(args, allow_val_change=True)

    trainer = Trainer_KBQA(args=vars(args))

    if not args.is_eval:
        trainer.train(0, args.num_epoch - 1)
    else:
        if args.local_rank <= 0:
            assert args.load_experiment is not None
            if args.load_experiment is not None:
                ckpt_path = os.path.join(args.checkpoint_dir, args.load_experiment)
                print("Loading pre trained model from {}".format(ckpt_path))
            else:
                ckpt_path = None
            trainer.evaluate_single(ckpt_path)


if __name__ == '__main__':
    main()
