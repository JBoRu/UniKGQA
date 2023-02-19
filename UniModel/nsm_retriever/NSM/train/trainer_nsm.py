import collections
import pickle
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np
import os, math
from NSM.train.init import init_nsm
from NSM.util.utils import State
from NSM.train.evaluate_nsm import Evaluator_nsm
from NSM.data.load_data_super import load_data
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import torch.optim as optim
tqdm.monitor_iterval = 0
import logging
from functools import reduce
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger("NSM")

class Trainer_KBQA(object):
    def __init__(self, args):
        self.args = args
        self.logger = logger
        self.best_dev_performance = 0.0
        self.best_h1 = 0.0
        self.best_f1 = 0.0
        self.eps = args['eps']
        self.learning_rate = self.args['lr']
        self.plm_learning_rate = self.args['plm_lr']
        self.test_batch_size = args['test_batch_size']
        self.train_kl = args['train_KL']
        self.num_step = args['num_step']
        self.use_label = args['use_label']
        self.reset_time = 0
        self.local_rank = args["local_rank"]
        self.world_size = 1 if self.local_rank < 0 else torch.distributed.get_world_size()
        self.device = torch.device("cuda", self.local_rank) if self.local_rank >= 0 else \
            torch.device('cuda' if args['use_cuda'] else 'cpu')
        self.load_data(args)
        if 'decay_rate' in args:
            self.decay_rate = args['decay_rate']
        else:
            self.decay_rate = 0.98
        self.mode = None
        self.model_name = self.args['model_name']
        self.student = init_nsm(self.args, self.entity2id, self.num_kb_relation, self.word2id, self.relation2id, self.device)
        if self.args["retriever_ckpt_path"] is not None:
            self.load_ckpt_from_retrieval()
        elif args['load_experiment'] is not None:
            self.load_pretrain()
        self.student.to(self.device)
        if self.local_rank != -1:
            find_unused_parameters = self.args["find_unused_parameters"]
            self.student = DDP(self.student, device_ids=[self.local_rank], output_device=self.local_rank,
                               find_unused_parameters=find_unused_parameters)
        self.evaluator = Evaluator_nsm(args=args, student=self.student, entity2id=self.entity2id,
                                       relation2id=self.relation2id, num_relation=self.student.num_relation)
        self.optim_def()
        self.state = State(args['log_steps'], log_flag=args["report_to"])

    def get_parameter_names(self, model, forbidden_layer_types):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in self.get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(model._parameters.keys())
        return result

    def optim_def(self):
        if self.args["fix_plm_layer"]:
            if self.local_rank <= 0:
                logger.info("Fix the bottom layers of plm")
            for n, p in self.student.named_parameters():
                if "query_encoder" in n:
                    if "encoder.layer.9" in n or "encoder.layer.10" in n or "encoder.layer.11" in n:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
        elif self.args["fix_all_plm"]:
            logger.info("Fix the whole of plm")
            print("Fix the whole of plm")
            for n, p in self.student.named_parameters():
                if "query_encoder" in n:
                    p.requires_grad = False

        if self.args["encode_relation_separate"]:
            for n,p in self.student.named_parameters():
                if "relation_encoder" in n:
                    p.requires_grad = False

        if self.args["diff_lr"] and not self.args["fix_all_plm"]:
            if self.local_rank <= 0:
                logger.info("Using different lr for PLM and other params with weight decay")
            decay_parameters = self.get_parameter_names(self.student, [nn.LayerNorm])
            # print(decay_parameters)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            log_group_params = [
                {
                    "plm params": [n for n, p in self.student.named_parameters() if p.requires_grad and
                               "query_encoder" in n and n in decay_parameters],
                    "weight_decay": self.args['weight_decay'],
                    "lr": self.plm_learning_rate
                },
                {
                    "plm params": [n for n, p in self.student.named_parameters() if p.requires_grad and
                               "query_encoder" in n and n not in decay_parameters],
                    "weight_decay": 0.0,
                    "lr": self.plm_learning_rate
                },
                {
                    "other params": [n for n, p in self.student.named_parameters() if p.requires_grad and
                               "query_encoder" not in n and n in decay_parameters],
                    "weight_decay": self.args['weight_decay'],
                    "lr": self.learning_rate
                },
                {
                    "other params": [n for n, p in self.student.named_parameters() if p.requires_grad and
                               "query_encoder" not in n and n not in decay_parameters],
                    "weight_decay": 0.0,
                    "lr": self.learning_rate
                },
            ]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.student.named_parameters() if p.requires_grad and
                               "query_encoder" in n and n in decay_parameters],
                    "weight_decay": self.args['weight_decay'],
                    "lr": self.plm_learning_rate
                },
                {
                    "params": [p for n, p in self.student.named_parameters() if p.requires_grad and
                               "query_encoder" in n and n not in decay_parameters],
                    "weight_decay": 0.0,
                    "lr": self.plm_learning_rate
                },
                {
                    "params": [p for n, p in self.student.named_parameters() if p.requires_grad and
                               "query_encoder" not in n and n in decay_parameters],
                    "weight_decay": self.args['weight_decay'],
                    "lr": self.learning_rate
                },
                {
                    "params": [p for n, p in self.student.named_parameters() if p.requires_grad and
                               "query_encoder" not in n and n not in decay_parameters],
                    "weight_decay": 0.0,
                    "lr": self.learning_rate
                },
            ]

            self.optim_student = optim.AdamW(optimizer_grouped_parameters)
            if self.decay_rate > 0:
                if self.local_rank <= 0:
                    logger.info("Using ExponentialLR with decay rate of %.2f"%(self.decay_rate))
                self.scheduler = ExponentialLR(self.optim_student, self.decay_rate)
            if self.local_rank <= 0:
                logger.info("The trainable params group: %s"%(log_group_params))
        else:
            if self.local_rank <= 0:
                logger.info("Using Adam optimizer with lr %s"%(str(self.learning_rate)))
            trainable = filter(lambda p: p.requires_grad, self.student.parameters())
            # self.optim_student = optim.AdamW(trainable, lr=self.learning_rate)
            self.optim_student = optim.Adam(trainable, lr=self.learning_rate)
            if self.decay_rate > 0:
                if self.local_rank <= 0:
                    logger.info("Using ExponentialLR with decay rate of %.2f" % (self.decay_rate))
                self.scheduler = ExponentialLR(self.optim_student, self.decay_rate)

        total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0)
                            for w in self.student.parameters() if w.requires_grad])
        if self.local_rank <= 0:
            logger.info("Trainable model params: {}".format(total_params))

    def load_data(self, args):
        if args["overwrite_cache"]:
            if self.local_rank <= 0:
                logger.info("Preprocess data from scratch!")

            dataset = load_data(args)
            if self.local_rank <= 0 and args["data_cache"] is not None:
                with open(args["data_cache"], "wb") as f:
                    pickle.dump(dataset, f)
                    logger.info("Cache preprocessed data to %s" % (args["data_cache"]))
        else:
            if args["data_cache"] is not None and os.path.exists(args["data_cache"]):
                if self.local_rank <= 0:
                    logger.info("Load preprocessed data cache from %s" % (args["data_cache"]))
                with open(args["data_cache"], "rb") as f:
                    dataset = pickle.load(f)
            else:
                if self.local_rank <= 0:
                    logger.info("Preprocess data from scratch!")
                dataset = load_data(args)

        self.train_data = dataset["train"]
        self.valid_data = dataset["valid"]
        self.test_data = dataset["test"]
        self.entity2id = dataset["entity2id"]
        self.relation2id = dataset["relation2id"]
        self.word2id = dataset["word2id"]
        self.num_kb_relation = self.train_data.num_kb_relation
        self.num_entity = len(self.entity2id)

    def load_pretrain(self):
        args = self.args
        ckpt_path = os.path.join(args['checkpoint_dir'], args['load_experiment'])
        logger.info("Load ckpt from %s"%ckpt_path)
        self.load_ckpt(ckpt_path)

    def evaluate(self, data, test_batch_size=20, mode="teacher", write_info=False):
        if mode == "retrieve":
            return self.evaluator.evaluate_single_sample(data, test_batch_size, write_info)
        else:
            return self.evaluator.evaluate(data, test_batch_size, write_info)

    def reduce_loss(self, loss):
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
        loss /= torch.distributed.get_world_size()
        return loss

    def train(self, start_epoch, end_epoch):
        # self.load_pretrain()
        eval_every = self.args['eval_every']
        # eval_acc = inference(self.model, self.valid_data, self.entity2id, self.args)
        if self.local_rank <= 0:
            self.save_ckpt("debug")
            self.save_args("args")
            filename = os.path.join(self.args['checkpoint_dir'], "{}-debug.ckpt".format(self.args['experiment_name']))
            self.load_ckpt(filename)
            eval_f1, eval_h1 = self.evaluate(self.valid_data, self.test_batch_size, mode="teacher")
            self.logger.info("Before finetune: Dev F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
            test_f1, test_h1 = self.evaluate(self.test_data, self.test_batch_size, mode="teacher")
            self.logger.info("Before finetune: Test F1: {:.4f}, H1: {:.4f}".format(test_f1, test_h1))
            # self.evaluate(self.valid_data, self.test_batch_size, mode="teacher")
        print("Strat Training------------------")
        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            loss, extras, h1_list_all, f1_list_all = self.train_epoch()
            if self.local_rank >= 0:
                # print("LR: %d loss %.2f when epoch %d" % (self.local_rank, loss, epoch))
                loss = self.reduce_loss(torch.tensor(loss).to("cuda")).item()
                # print("LR: %d loss %.2f when epoch %d after reduce" % (self.local_rank, loss, epoch))
            if self.decay_rate > 0:
                self.scheduler.step()
            # if self.mode == "student":
            #     self.student.update_target()
            # actor_loss, ent_loss = extras
            if self.local_rank <= 0:
                self.logger.info("Epoch: {}, loss : {:.4f}, time: {}".format(epoch + 1, loss, time.time() - st))
                self.logger.info("Training h1 : {:.4f}, f1 : {:.4f}".format(np.mean(h1_list_all), np.mean(f1_list_all)))
            # print("actor : {:.4f}, ent : {:.4f}".format(actor_loss, ent_loss))
            if (epoch + 1) % eval_every == 0 and epoch + 1 > 0 and self.local_rank <= 0:
                eval_f1, eval_h1 = self.evaluate(self.valid_data, self.test_batch_size, mode="teacher")
                test_f1, test_h1 = self.evaluate(self.test_data, self.test_batch_size, mode="teacher")
                self.logger.info("EVAL F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
                self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(test_f1, test_h1))
                self.state.log(loss=None, h1=eval_h1, f1=eval_f1, mode="dev")
                if eval_h1 > self.best_h1:
                    self.best_h1 = eval_h1
                    self.save_ckpt("h1")
                    self.reset_time = 0
                elif eval_f1 > self.best_f1:
                    self.best_f1 = eval_f1
                    self.save_ckpt("f1")
                    self.reset_time = 0
                else:
                    self.reset_time += 1
                    self.logger.info('No improvement after %d evaluation iter.'%(self.reset_time))

                if self.reset_time >= self.args["patience"]:
                    self.logger.info('No improvement after %d evaluation. Early Stopping.'%(self.reset_time))
                    break
        self.save_ckpt("final")
        if self.local_rank <= 0:
            self.logger.info('Train Done! Evaluate on testset with saved model')
            print("End Training------------------")
            if self.model_name != "back":
                self.evaluate_best(self.mode)

    def evaluate_best(self, mode):
        filename = os.path.join(self.args['checkpoint_dir'], "{}-h1.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1 = self.evaluate(self.test_data, self.test_batch_size, mode="teacher", write_info=False)
        self.logger.info("Best h1 evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
        self.state.log(loss=None, h1=eval_h1, f1=eval_f1, mode='test/best_h1')

        filename = os.path.join(self.args['checkpoint_dir'], "{}-f1.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1 = self.evaluate(self.test_data, self.test_batch_size, mode="teacher", write_info=False)
        self.logger.info("Best f1 evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
        self.state.log(loss=None, h1=eval_h1, f1=eval_f1, mode='test/best_f1')

        filename = os.path.join(self.args['checkpoint_dir'], "{}-final.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1 = self.evaluate(self.test_data, self.test_batch_size, mode="teacher", write_info=False)
        self.logger.info("Final evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
        self.state.log(loss=None, h1=eval_h1, f1=eval_f1, mode='test/final')

    def evaluate_single(self, filename):
        if filename is not None:
            self.load_ckpt(filename)
        test_f1, test_hits = self.evaluate(self.test_data, self.test_batch_size, mode="teacher", write_info=False)
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(test_f1, test_hits))

    def retrieve_single(self, sample):
        self.evaluate(sample, 1, mode="retrieve", write_info=False)

    def train_epoch(self):
        self.student.train()
        self.train_data.reset_batches(is_sequential=True)
        losses = []
        actor_losses = []
        ent_losses = []
        num_steps = math.ceil(self.train_data.num_data / self.args['batch_size'])
        # print("LR: %d Total num step: %d / %d = %d"%(self.local_rank, self.train_data.num_data, self.args['batch_size'], num_steps))
        iter_step_ids = [i for i in range(num_steps)]
        if self.local_rank == -1:
            cur_iter_step_ids = iter_step_ids
        else:
            num_steps_per_node = int(num_steps / self.world_size)
            cur_iter_step_ids = iter_step_ids[self.local_rank * num_steps_per_node: (self.local_rank+1) * num_steps_per_node]
            # print("LR:%d Current iteration step ids: %s"%(self.local_rank, cur_iter_step_ids))
        # print("LR:%d, len of current iteration step:%d"%(self.local_rank, len(cur_iter_step_ids)))
        h1_list_all = []
        f1_list_all = []
        self.optim_student.zero_grad()
        for iteration in tqdm(cur_iter_step_ids):
            batch = self.train_data.get_batch(iteration, self.args['batch_size'], self.args['fact_drop'])
            # label_dist, label_valid = self.train_data.get_label()
            # loss = self.train_step_student(batch, label_dist, label_valid)
            loss, _, _, tp_list = self.student(batch, training=True)

            if self.args["gradient_accumulation_steps"] > 1:
                loss = loss / self.args["gradient_accumulation_steps"]
            loss.backward()
            # if tp_list is not None:
            h1_list, f1_list = tp_list
            h1 = np.mean(h1_list)
            f1 = np.mean(f1_list)
            loss = loss.item()

            losses.append(loss)
            h1_list_all.append(h1)
            f1_list_all.append(f1)

            if self.local_rank >= 0:
                h1 = self.reduce_loss(torch.tensor(h1).to("cuda")).item()
                f1 = self.reduce_loss(torch.tensor(f1).to("cuda")).item()
                # print("LR: %d loss %.2f when iter %d" % (self.local_rank, loss, iteration))
                loss = self.reduce_loss(torch.tensor(loss).to("cuda")).item()
                # print("LR: %d loss %.2f when iter %d after reduce" % (self.local_rank, loss, iteration))

            if self.local_rank <= 0:
                self.state.record_each_step(loss, h1, f1)

            if (iteration+1) % self.args["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_([param for name, param in self.student.named_parameters()],
                                               self.args['gradient_clip'])
                self.optim_student.step()
                self.optim_student.zero_grad()
            # print("LR:%d iteration %d of %s" % (self.local_rank, iteration, cur_iter_step_ids))
        # print("LR:%d over one epoch!"%(self.local_rank))
        extras = [0, 0]
        return np.mean(losses), extras, h1_list_all, f1_list_all

    def save_ckpt(self, reason="h1"):
        state = {
            'state_dict': self.student.state_dict(),
        }
        model_name = os.path.join(self.args['checkpoint_dir'], "{}-{}.ckpt".format(self.args['experiment_name'],
                                                                                   reason))
        if self.local_rank == 0:
            saved_dict = collections.OrderedDict()
            for key, val in state.items():
                if (key == 'state_dict'):
                    for state_dict_key, state_dict_val in val.items():
                        if (state_dict_key[0:7] == 'module.'):
                            changed_key = state_dict_key[7:]
                        else:
                            changed_key = state_dict_key
                        saved_dict[changed_key] = state_dict_val
                    state[key] = saved_dict
            torch.save(state, model_name)
            print("Best %s, save model as %s" %(reason, model_name))
        elif self.local_rank < 0:
            torch.save(state, model_name)
            print("Best %s, save model as %s" %(reason, model_name))

    def load_ckpt(self, filename):
        if self.local_rank <= 0:
            checkpoint = torch.load(filename, map_location=torch.device("cpu"))["state_dict"]
            selected_state_dict = {}
            # filtered_keys = ["model.relation_embedding", "model.word_embedding"]
            # filtered_keys = ["relation_embedding", "word_embedding"]
            filtered_keys = ["relation_embedding", "word_embedding", "rel_features"]
            for k, v in checkpoint.items():
                select_flag = True
                for fk in filtered_keys:
                    if fk in k:
                        select_flag = False
                        break
                if select_flag:
                    if k.startswith("model."):
                        k = k[6:]
                    selected_state_dict[k] = v

            ori_model_dict = self.student.model.state_dict()
            ori_model_dict.update(selected_state_dict)
            self.student.model.load_state_dict(ori_model_dict)
            # if self.logger is not None:
            #     self.logger.info("Load param of {} from {}.".format(", ".join(list(checkpoint['state_dict'].keys())), filename))
            # else:
            #     print("Load param of {} from {}.".format(", ".join(list(checkpoint['state_dict'].keys())), filename))

    def save_args(self, suffix):
        model_name = os.path.join(self.args['checkpoint_dir'], "{}-{}.json".format(self.args['experiment_name'],
                                                                                   suffix))
        with open(model_name, "wb") as f:
            pickle.dump(self.args, f)

    def load_ckpt_from_retrieval(self):
        retriever_ckpt_path = self.args["retriever_ckpt_path"]
        if retriever_ckpt_path is not None:
            checkpoint = torch.load(retriever_ckpt_path, map_location=torch.device("cpu"))["state_dict"]
            selected_state_dict = {}
            # filtered_keys = ["model.relation_embedding", "model.word_embedding"]
            filtered_keys = ["relation_embedding", "word_embedding", "rel_features"]
            for k, v in checkpoint.items():
                select_flag = True
                for fk in filtered_keys:
                    if fk in k:
                        select_flag = False
                        break
                if select_flag:
                    if k.startswith("model."):
                        k = k[6:]
                    selected_state_dict[k] = v

            # ori_model_dict = self.student.state_dict()
            # ori_model_dict.update(selected_state_dict)
            # self.student.load_state_dict(ori_model_dict)
            ori_model_dict = self.student.model.state_dict()
            ori_model_dict.update(selected_state_dict)
            self.student.model.load_state_dict(ori_model_dict)
            self.logger.info(
                "Load param of {} from {}.".format(", ".join(list(selected_state_dict.keys())), retriever_ckpt_path))
