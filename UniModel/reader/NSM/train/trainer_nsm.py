import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np
import os, math
try:
    from NSM.train.init import init_nsm
    from NSM.train.evaluate_nsm import Evaluator_nsm
    from NSM.data.load_data_super import load_data
except:
    from init import init_nsm
    from evaluate_nsm import Evaluator_nsm
    from UniModel.reader.NSM.data.load_data_super import load_data
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import torch.optim as optim
tqdm.monitor_iterval = 0
import logging
from functools import reduce

logger = logging.getLogger("NSM")
logger.setLevel(logging.ERROR)

class Trainer_KBQA(object):
    def __init__(self, args, logger=None):
        self.args = args
        self.logger = logger
        self.best_dev_performance = 0.0
        self.best_h1 = 0.0
        self.best_f1 = 0.0
        self.eps = args['eps']
        self.learning_rate = self.args['lr']
        self.plm_learning_rate = self.args['plm_lr']
        self.test_batch_size = args['test_batch_size']
        self.device = torch.device('cuda' if args['use_cuda'] else 'cpu')
        self.train_kl = args['train_KL']
        self.num_step = args['num_step']
        self.use_label = args['use_label']
        self.reset_time = 0
        self.load_data(args)
        if 'decay_rate' in args:
            self.decay_rate = args['decay_rate']
        else:
            self.decay_rate = 0.98
        # self.mode = args['mode']
        # self.use_middle = args['use_middle']
        self.mode = "teacher"
        self.model_name = self.args['model_name']
        self.student = init_nsm(self.args, self.logger, len(self.entity2id), self.num_kb_relation,
                                len(self.word2id), self.relation2id)
        # self.student.insert_relation_tokens_to_model(self.relation_tokens)
        self.load_ckpt_from_retrieval()
        self.student.to(self.device)
        self.evaluator = Evaluator_nsm(args=args, student=self.student, entity2id=self.entity2id,
                                       relation2id=self.relation2id, num_relation=self.student.num_relation)
        # self.load_pretrain()
        self.optim_def()

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

        if self.args["diff_lr"]:
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
                logger.info("Using ExponentialLR with decay rate of %.2f" % (self.decay_rate))
                self.scheduler = ExponentialLR(self.optim_student, self.decay_rate)
            logger.info("The trainable params group: %s" % (log_group_params))
        else:
            logger.info("Using Adam optimizer with lr %s" % (str(self.learning_rate)))
            trainable = filter(lambda p: p.requires_grad, self.student.parameters())
            self.optim_student = optim.AdamW(trainable, lr=self.learning_rate)
            if self.decay_rate > 0:
                logger.info("Using ExponentialLR with decay rate of %.2f" % (self.decay_rate))
                self.scheduler = ExponentialLR(self.optim_student, self.decay_rate)

        total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0)
                            for w in self.student.parameters() if w.requires_grad])
        logger.info("Trainable model params: {}".format(total_params))

    def load_data(self, args):
        dataset = load_data(args)
        self.train_data = dataset["train"]
        self.valid_data = dataset["valid"]
        self.test_data = dataset["test"]
        self.entity2id = dataset["entity2id"]
        self.relation2id = dataset["relation2id"]
        # self.relation_tokens = torch.from_numpy(dataset['relation_tokens']).to(self.device)
        self.word2id = dataset["word2id"]
        self.num_kb_relation = self.test_data.num_kb_relation
        self.num_entity = len(self.entity2id)

    def load_pretrain(self):
        args = self.args
        if args['load_experiment'] is not None:
            ckpt_path = os.path.join(args['checkpoint_dir'], args['load_experiment'])
            print("Load ckpt from", ckpt_path)
            self.load_ckpt(ckpt_path)

    def evaluate(self, data, test_batch_size=20, mode="teacher", write_info=False):
        return self.evaluator.evaluate(data, test_batch_size, write_info)

    def train(self, start_epoch, end_epoch):
        # self.load_pretrain()
        eval_every = self.args['eval_every']
        # eval_acc = inference(self.model, self.valid_data, self.entity2id, self.args)
        self.save_ckpt("debug")
        self.save_args("args")
        filename = os.path.join(self.args['checkpoint_dir'], "{}-debug.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1 = self.evaluate(self.valid_data, self.test_batch_size, mode="teacher")
        self.logger.info("Before finetune: Dev F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
        test_f1, test_h1 = self.evaluate(self.test_data, self.test_batch_size, mode="teacher")
        self.logger.info("Before finetune: Test F1: {:.4f}, H1: {:.4f}".format(test_f1, test_h1))
        print("Strat Training------------------")
        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            loss, extras, h1_list_all, f1_list_all = self.train_epoch()
            if self.decay_rate > 0:
                self.scheduler.step()
            # if self.mode == "student":
            #     self.student.update_target()
            # actor_loss, ent_loss = extras
            self.logger.info("Epoch: {}, loss : {:.4f}, time: {}".format(epoch + 1, loss, time.time() - st))
            self.logger.info("Training h1 : {:.4f}, f1 : {:.4f}".format(np.mean(h1_list_all), np.mean(f1_list_all)))
            # print("actor : {:.4f}, ent : {:.4f}".format(actor_loss, ent_loss))
            if (epoch + 1) % eval_every == 0 and epoch + 1 > 0:
                if self.model_name == "back":
                    eval_f1 = np.mean(f1_list_all)
                    eval_h1 = np.mean(h1_list_all)
                else:
                    eval_f1, eval_h1 = self.evaluate(self.valid_data, self.test_batch_size, mode="teacher")
                    test_f1, test_h1 = self.evaluate(self.test_data, self.test_batch_size, mode="teacher")
                self.logger.info("EVAL F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))
                self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(test_f1, test_h1))
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
                    self.logger.info('No improvement after %d evaluation iter.' % (self.reset_time))

                if self.reset_time >= self.args["patience"]:
                    self.logger.info('No improvement after %d evaluation. Early Stopping.' % (self.reset_time))
                    break
        self.save_ckpt("final")
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

        filename = os.path.join(self.args['checkpoint_dir'], "{}-f1.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1 = self.evaluate(self.test_data, self.test_batch_size, mode="teacher", write_info=False)
        self.logger.info("Best f1 evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))

        filename = os.path.join(self.args['checkpoint_dir'], "{}-final.ckpt".format(self.args['experiment_name']))
        self.load_ckpt(filename)
        eval_f1, eval_h1 = self.evaluate(self.test_data, self.test_batch_size, mode="teacher", write_info=False)
        self.logger.info("Final evaluation")
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(eval_f1, eval_h1))

    def evaluate_single(self, filename):
        if filename is not None:
            self.load_ckpt(filename)
        test_f1, test_hits = self.evaluate(self.test_data, self.test_batch_size, mode=self.mode, write_info=True)
        self.logger.info("TEST F1: {:.4f}, H1: {:.4f}".format(test_f1, test_hits))

    def train_epoch(self):
        self.student.train()
        self.train_data.reset_batches(is_sequential=False)
        losses = []
        actor_losses = []
        ent_losses = []
        num_epoch = math.ceil(self.train_data.num_data / self.args['batch_size'])
        h1_list_all = []
        f1_list_all = []
        for iteration in tqdm(range(num_epoch)):
            batch = self.train_data.get_batch(iteration, self.args['batch_size'], self.args['fact_drop'])
            # label_dist, label_valid = self.train_data.get_label()
            # loss = self.train_step_student(batch, label_dist, label_valid)
            self.optim_student.zero_grad()
            loss, _, _, tp_list = self.student(batch, training=True)
            # if tp_list is not None:
            h1_list, f1_list = tp_list
            h1_list_all.extend(h1_list)
            f1_list_all.extend(f1_list)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([param for name, param in self.student.named_parameters()],
                                           self.args['gradient_clip'])
            self.optim_student.step()
            losses.append(loss.item())
        extras = [0, 0]
        return np.mean(losses), extras, h1_list_all, f1_list_all

    def save_ckpt(self, reason="h1"):
        model = self.student.model
        checkpoint = {
            'state_dict': model.state_dict()
        }
        model_name = os.path.join(self.args['checkpoint_dir'], "{}-{}.ckpt".format(self.args['experiment_name'],
                                                                                   reason))
        torch.save(checkpoint, model_name)
        print("Best %s, save model as %s" %(reason, model_name))

    def load_ckpt(self, filename):
        checkpoint = torch.load(filename)
        model_state_dict = checkpoint["state_dict"]
        model = self.student.model
        # model = self.student
        # self.logger.info("Load param of {} from {}.".format(", ".join(list(model_state_dict.keys())), filename))
        model.load_state_dict(model_state_dict, strict=False)

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
            self.logger.info("Load param of {} from {}.".format(", ".join(list(selected_state_dict.keys())), retriever_ckpt_path))

    def save_args(self, suffix):
        model_name = os.path.join(self.args['checkpoint_dir'], "{}-{}.json".format(self.args['experiment_name'],
                                                                                   suffix))
        with open(model_name, "wb") as f:
            pickle.dump(self.args, f)