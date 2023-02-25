import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
try:
    from NSM.Model.base_model import BaseModel
    from NSM.Modules.Instruction.seq_instruction import LSTMInstruction
    from NSM.Modules.Reasoning.gnn_reasoning import GNNReasoning
except:
    try:
        from nsm.NSM.Model.base_model import BaseModel
        from nsm.NSM.Modules.Instruction.seq_instruction import LSTMInstruction
        from nsm.NSM.Modules.Reasoning.gnn_reasoning import GNNReasoning
    except:
        from UniModel.nsm.NSM.Model.base_model import BaseModel
        from UniModel.nsm.NSM.Modules.Instruction.seq_instruction import LSTMInstruction
        from UniModel.nsm.NSM.Modules.Reasoning.gnn_reasoning import GNNReasoning

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


def build_model(args, num_entity, num_relation, num_word):
    model_cls = RWModel
    return model_cls(args, num_entity, num_relation, num_word)


class ForwardReasonModel(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word):
        """
        num_relation: number of relation including self-connection
        """
        super(ForwardReasonModel, self).__init__(args, num_entity, num_relation, num_word)
        self.embedding_def()
        self.share_module_def()
        self.instruction_def(args)
        self.reasoning_def(args, num_entity, num_relation)
        self.loss_type = args['loss_type']
        self.model_name = args['model_name'].lower()
        self.lambda_label = args['lambda_label']
        self.filter_label = args['filter_label']
        self.to(self.device)

    def instruction_def(self, args):
        self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)

    def reasoning_def(self, args, num_entity, num_relation):
        self.reasoning = GNNReasoning(args, num_entity, num_relation)

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input):
        # batch_size = local_entity.size(0)
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        query_node_emb = self.instruction.query_node_emb
        self.rel_features = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, self.rel_features)
        self.curr_dist = curr_dist
        self.dist_history = [curr_dist]
        self.action_probs = []
        self.reasoning.init_reason(local_entity=local_entity,
                                   kb_adj_mat=kb_adj_mat,
                                   local_entity_emb=self.local_entity_emb,
                                   rel_features=self.rel_features,
                                   query_node_emb=query_node_emb)

    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss_new(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    def train_batch(self, batch, middle_dist, label_valid=None):
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, \
        local_entity, query_entities, true_batch_id = batch
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input)
        for i in range(self.num_step):
            self.one_step(num_step=i)
        # loss, extras = self.calc_loss_basic(answer_dist)
        pred_dist = self.dist_history[-1]
        main_loss = self.get_loss_new(pred_dist, answer_dist)
        distill_loss = None
        for i in range(self.num_step - 1):
            curr_dist = self.dist_history[i + 1]
            # teacher_dist = middle_dist[i].detach()
            teacher_dist = middle_dist[i].squeeze(1).detach()
            # print(curr_dist.size(), teacher_dist.size())
            if self.filter_label:
                assert not (label_valid is None)
                tp_label_loss = self.calc_loss_label(curr_dist=curr_dist,
                                                    teacher_dist=teacher_dist,
                                                    label_valid=label_valid)
            else:
                tp_label_loss = self.get_loss_new(curr_dist, teacher_dist)
            if distill_loss is None:
                distill_loss = tp_label_loss
            else:
                distill_loss += tp_label_loss
        # pred = torch.max(pred_dist, dim=1)[1]
        extras = [main_loss.item(), distill_loss.item()]
        # tp_list = [h1.tolist(), f1.tolist()]
        loss = main_loss + distill_loss * self.lambda_label
        h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
        tp_list = [h1.tolist(), f1.tolist()]
        return loss, extras, pred_dist, tp_list

    def one_step(self, num_step):
        # relational_ins, attn_weight = self.instruction.get_instruction(self.relational_ins, query_mask, step=num_step)
        relational_ins = self.instruction_list[num_step]
        # attn_weight = self.attn_list[num_step]
        # self.relational_ins = relational_ins
        self.curr_dist = self.reasoning(self.curr_dist, relational_ins, step=num_step)
        self.dist_history.append(self.curr_dist)

    def forward(self, batch, training=False):
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, \
        local_entity, query_entities, true_batch_id = batch
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input)
        for i in range(self.num_step):
            self.one_step(num_step=i)
        # loss, extras = self.calc_loss_basic(answer_dist)
        pred_dist = self.dist_history[-1]
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        # loss = self.get_loss_new(pred_dist, answer_dist)
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        pred = torch.max(pred_dist, dim=1)[1]
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, pred, pred_dist, tp_list
