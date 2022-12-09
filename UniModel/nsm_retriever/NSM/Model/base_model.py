import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
from NSM.Modules.layer_nsm import TypeLayer

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


def f1_and_hits_new(answers, candidate2prob, eps=0.5):
    retrieved = []
    correct = 0
    cand_list = sorted(candidate2prob, key=lambda x:x[1], reverse=True)
    if len(cand_list) == 0:
        best_ans = -1
    else:
        best_ans = cand_list[0][0]
    # max_prob = cand_list[0][1]
    tp_prob = 0.0
    for c, prob in cand_list:
        retrieved.append((c, prob))
        tp_prob += prob
        if c in answers:
            correct += 1
        if tp_prob > eps:
            break
    if len(answers) == 0:
        if len(retrieved) == 0:
            return 1.0, 1.0, 1.0, 1.0  # precision, recall, f1, hits
        else:
            return 0.0, 1.0, 0.0, 1.0  # precision, recall, f1, hits
    else:
        hits = float(best_ans in answers)
        if len(retrieved) == 0:
            return 1.0, 0.0, 0.0, hits  # precision, recall, f1, hits
        else:
            p, r = correct / len(retrieved), correct / len(answers)
            f1 = 2.0 / (1.0 / p + 1.0 / r) if p != 0 and r != 0 else 0.0
            return p, r, f1, hits


class BaseModel(torch.nn.Module):
    def __init__(self, args, num_entity, num_relation):
        super(BaseModel, self).__init__()
        self.num_relation = num_relation
        self.num_entity = num_entity
        self.use_inverse_relation = args['use_inverse_relation']
        self.use_self_loop = args['use_self_loop']
        self.has_entity_kge = False
        self.has_relation_kge = False
        self.num_step = args['num_step']
        self.linear_dropout = args['linear_dropout']
        self.encode_type = args["encode_type"]
        self.reason_kb = args['reason_kb']
        self.eps = args['eps']
        self.loss_type = args['loss_type']
        self.entropy_weight = args['entropy_weight']
        self.init_same_as_plm = args["init_same_as_plm"]

        for k, v in args.items():
            if k.endswith('dim'):
                setattr(self, k, v)
            if k.endswith('emb_file') or k.endswith('kge_file'):
                if v is None:
                    setattr(self, k, None)
                else:
                    setattr(self, k, args['data_folder'] + v)

        self.reset_time = 0


    def get_ent_init(self, local_entity, kb_adj_mat, rel_features):
        # batch_size, max_local_entity = local_entity.size()
        # hidden_size = self.entity_dim
        if self.encode_type:
            local_entity_emb = self.type_layer(local_entity=local_entity,
                                               edge_list=kb_adj_mat,
                                               rel_features=rel_features)
        else:
            local_entity_emb = self.entity_embedding(local_entity)  # batch_size, max_local_entity, word_dim
            if self.has_entity_kge:
                local_entity_emb = torch.cat((local_entity_emb, self.entity_kge(local_entity)),
                                             dim=2)  # batch_size, max_local_entity, word_dim + kge_dim
            if self.word_dim != self.entity_dim:
                local_entity_emb = self.entity_linear(local_entity_emb)  # batch_size, max_local_entity, entity_dim
        return local_entity_emb

    def load_relation_file(self, filename):
        half_tensor = np.load(filename)
        num_pad = 0
        if self.use_self_loop:
            num_pad = 1
        if self.use_inverse_relation:
            load_tensor = np.concatenate([half_tensor, half_tensor])
        else:
            load_tensor = half_tensor
        return np.pad(load_tensor, ((0, num_pad), (0, 0)), 'constant')

    def get_rel_feature(self):
        rel_features = self.relation_embedding.weight
        if self.has_relation_kge:
            rel_features = torch.cat((rel_features, self.relation_kge.weight), dim=-1)
        rel_features = self.relation_linear(rel_features)
        return rel_features

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return self.instruction.init_hidden(num_layer, batch_size, hidden_size)

    def encode_question(self, q_input):
        return self.instruction.encode_question(q_input)

    def get_instruction(self, query_hidden_emb, query_mask, states):
        return self.instruction.get_instruction(query_hidden_emb, query_mask, states)

    def get_loss_bce(self, pred_dist_score, answer_dist):
        answer_dist = (answer_dist > 0).float() * 0.9   # label smooth
        # answer_dist = answer_dist * 0.9  # label smooth
        loss = self.bce_loss_logits(pred_dist_score, answer_dist)
        return loss

    def get_loss_kl(self, pred_dist, answer_dist):
        answer_len = torch.sum(answer_dist, dim=1, keepdim=True)
        answer_len[answer_len == 0] = 1.0
        answer_prob = answer_dist.div(answer_len)
        log_prob = torch.log(pred_dist + 1e-8)
        loss = self.kld_loss(log_prob, answer_prob)
        return loss

    def get_loss_new(self, pred_dist, answer_dist, reduction='mean'):
        if self.loss_type == "bce":
            tp_loss = self.get_loss_bce(pred_dist, answer_dist)
            if reduction == 'none':
                return tp_loss
            else:
                # mean
                return torch.mean(tp_loss)
        else:
            tp_loss = self.get_loss_kl(pred_dist, answer_dist)
            if reduction == 'none':
                return tp_loss
            else:
                # batchmean
                return torch.sum(tp_loss) / pred_dist.size(0)

    def calc_f1_new(self, curr_dist, dist_ans, h1_vec):
        batch_size = curr_dist.size(0)
        max_local_entity = curr_dist.size(1)
        seed_dist = self.dist_history[0]
        local_entity = self.local_entity
        ignore_prob = (1 - self.eps) / max_local_entity
        pad_ent_id = self.num_entity
        # hits_list = []
        f1_list = []
        for batch_id in range(batch_size):
            if h1_vec[batch_id].item() == 0.0:
                f1_list.append(0.0)
                # we consider cases which own hit@1 as prior to reduce computation time
                continue
            candidates = local_entity[batch_id, :].tolist()
            probs = curr_dist[batch_id, :].tolist()
            answer_prob = dist_ans[batch_id, :].tolist()
            seed_entities = seed_dist[batch_id, :].tolist()
            answer_list = []
            candidate2prob = []
            for c, p, p_a, s in zip(candidates, probs, answer_prob, seed_entities):
                if s > 0:
                    # ignore seed entities
                    continue
                if c == pad_ent_id:
                    continue
                if p_a > 0:
                    answer_list.append(c)
                if p < ignore_prob:
                    continue
                candidate2prob.append((c, p))
            precision, recall, f1, hits = f1_and_hits_new(answer_list, candidate2prob, self.eps)
            # hits_list.append(hits)
            f1_list.append(f1)
        # hits_vec = torch.FloatTensor(hits_list).to(self.device)
        f1_vec = torch.FloatTensor(f1_list).to(self.device)
        return f1_vec

    def calc_h1(self, curr_dist, dist_ans, eps=0.01):
        greedy_option = curr_dist.argmax(dim=-1, keepdim=True)
        dist_top1 = torch.zeros_like(curr_dist).scatter_(1, greedy_option, 1.0)
        dist_ans = (dist_ans > eps).float()
        h1 = torch.sum(dist_top1 * dist_ans, dim=-1)
        return (h1 > 0).float()
    
    def get_eval_metric(self, pred_dist, answer_dist):
        with torch.no_grad():
            h1 = self.calc_h1(curr_dist=pred_dist, dist_ans=answer_dist, eps=VERY_SMALL_NUMBER)
            f1 = self.calc_f1_new(pred_dist, answer_dist, h1)
        return h1, f1

    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        # elif isinstance(module, BertLayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()