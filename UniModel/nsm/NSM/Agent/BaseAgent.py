import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import numpy as np
import logging
logger = logging.getLogger('NSM')
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class BaseAgent(nn.Module):
    def __init__(self, args, num_entity, num_relation):
        super(BaseAgent, self).__init__()
        self.parse_args(args, num_entity, num_relation)

    def parse_args(self, args, num_entity, num_relation):
        self.args = args
        self.local_rank = args['local_rank']
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.use_inverse_relation = args['use_inverse_relation']
        self.use_self_loop = args['use_self_loop']
        if self.local_rank <= 0:
            logger.info("Entity: {}, Relation: {}".format(num_entity, num_relation))
        self.learning_rate = self.args['lr']
        self.q_type = args['q_type']
        self.num_step = args['num_step']

        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file') or k.endswith('kge_file'):
                if v is None:
                    setattr(self, k, None)
                else:
                    setattr(self, k, args['data_folder'] + v)

        self.reset_time = 0

    @staticmethod
    def get_node_emb(query_hidden_emb, action):
        '''

        :param query_hidden_emb: (batch_size, max_hyper, emb)
        :param action: (batch_size)
        :return: (batch_size, emb)
        '''
        batch_size, max_hyper, _ = query_hidden_emb.size()
        row_idx = torch.arange(0, batch_size).type(torch.LongTensor)
        q_rep = query_hidden_emb[row_idx, action, :]
        return q_rep

    def forward(self, *args):
        pass

    @staticmethod
    def mask_max(values, mask, keepdim=True):
        return torch.max(values + (1 - mask) * VERY_NEG_NUMBER, dim=-1, keepdim=keepdim)[0]

    @staticmethod
    def mask_argmax(values, mask):
        return torch.argmax(values + (1 - mask) * VERY_NEG_NUMBER, dim=-1, keepdim=True)
