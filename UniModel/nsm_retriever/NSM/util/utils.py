import logging
import os
import numpy as np
import wandb
import torch
import random
from torch.backends import cudnn

def setup_seed(seed):
    # CUDNN
    # cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    # cudnn.deterministic = True
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # python & numpy
    np.random.seed(seed)
    random.seed(seed)


def create_logger(args):
    log_file = os.path.join(args.checkpoint_dir, args.experiment_name + ".log")
    logger = logging.getLogger("NSM")
    if args.log_level == "debug":
        log_level = logging.DEBUG
    elif args.log_level == "info":
        log_level = logging.INFO
    else:
        log_level = logging.ERROR
    logger.setLevel(level=log_level)
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # FileHandler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # StreamHandler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("PARAMETER" + "-" * 10)
    for attr, value in sorted(args.__dict__.items()):
        logger.info("{}={}".format(attr.upper(), value))
    logger.info("---------" + "-" * 10)

    return logger


def get_dict(data_folder, filename):
    filename_true = os.path.join(data_folder, filename)
    word2id = dict()
    with open(filename_true, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id

def reverse_dict(dict):
    re_dict = {}
    for k, v in dict.items():
        re_dict[v] = k
    return re_dict

class State(object):
    def __init__(self, log_steps, log_flag):
        self.global_steps = 0
        self.loss = []
        self.hits1 = []
        self.f1 = []
        self.log_steps = log_steps
        self.log_flag = log_flag

    def reset_state(self):
        self.loss = []
        self.hits1 = []
        self.f1 = []

    def log(self, loss, h1, f1, mode):
        log_dict = {}
        if loss is not None:
            log_dict[mode+"/loss"] = loss
        if h1 is not None:
            log_dict[mode + "/h1"] = h1
        if f1 is not None:
            log_dict[mode + "/f1"] = f1
        if self.log_flag == "wandb":
            wandb.log(log_dict, step=self.global_steps)

    def record_each_step(self, loss, hits1, f1):
        self.loss.append(loss)
        self.hits1.append(hits1)
        self.f1.append(f1)
        self.global_steps += 1
        if self.global_steps % self.log_steps == 0:
            self.log(loss=np.mean(self.loss), h1=np.mean(self.hits1), f1=np.mean(self.f1), mode="train")
            self.reset_state()