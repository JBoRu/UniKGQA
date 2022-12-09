import sys
import logging
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from NSM.Model.base_model import BaseModel
from NSM.Modules.Instruction.seq_instruction import LSTMInstruction, PLMInstruction
from NSM.Modules.Reasoning.gnn_reasoning import GNNReasoning, PLMGNNReasoning
from NSM.util.utils import reverse_dict
from NSM.Modules.layer_nsm import TypeLayer
from transformers import AutoTokenizer, AutoModel, AutoConfig

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000
logger = logging.getLogger("NSM")

class GNNModel(BaseModel):
    def __init__(self, args, num_entity, num_relation, num_word, device):
        """
        num_relation: number of relation including self-connection
        """
        super(GNNModel, self).__init__(args, num_entity, num_relation)
        self.device = device
        self.loss_type = "kl"
        self.num_layer = args['num_layer']
        self.lstm_dropout = args['lstm_dropout']
        self.num_word = num_word
        print("Entity: {}, Relation: {}, Word: {}".format(num_entity, num_relation, num_word))

        self.embedding_def()
        self.share_module_def()
        self.private_module_def(args, num_entity, num_relation)

    def embedding_def(self):
        word_dim = self.word_dim
        kge_dim = self.kge_dim
        kg_dim = self.kg_dim
        num_entity = self.num_entity
        num_relation = self.num_relation
        num_word = self.num_word

        if not self.encode_type:
            self.entity_embedding = nn.Embedding(num_embeddings=num_entity + 1, embedding_dim=kg_dim,
                                                 padding_idx=num_entity)
            if self.init_same_as_plm:
                self.init_weights(self.entity_embedding)
            if self.entity_emb_file is not None:
                self.entity_embedding.weight = nn.Parameter(
                    torch.from_numpy(np.pad(np.load(self.entity_emb_file), ((0, 1), (0, 0)), 'constant')).type(
                        'torch.FloatTensor'))
                self.entity_embedding.weight.requires_grad = False

            if self.entity_kge_file is not None:
                self.has_entity_kge = True
                self.entity_kge = nn.Embedding(num_embeddings=num_entity + 1, embedding_dim=kge_dim,
                                               padding_idx=num_entity)
                self.entity_kge.weight = nn.Parameter(
                    torch.from_numpy(np.pad(np.load(self.entity_kge_file), ((0, 1), (0, 0)), 'constant')).type(
                        'torch.FloatTensor'))
                self.entity_kge.weight.requires_grad = False
            else:
                self.entity_kge = None

        # initialize relation embedding
        self.relation_embedding = nn.Embedding(num_embeddings=num_relation, embedding_dim=2 * kg_dim)
        if self.init_same_as_plm:
            self.init_weights(self.relation_embedding)
        if self.relation_emb_file is not None:
            np_tensor = self.load_relation_file(self.relation_emb_file)
            self.relation_embedding.weight = nn.Parameter(torch.from_numpy(np_tensor).type('torch.FloatTensor'))
        if self.relation_kge_file is not None:
            self.has_relation_kge = True
            self.relation_kge = nn.Embedding(num_embeddings=num_relation, embedding_dim=kge_dim)
            np_tensor = self.load_relation_file(self.relation_kge_file)
            self.relation_kge.weight = nn.Parameter(torch.from_numpy(np_tensor).type('torch.FloatTensor'))
        else:
            self.relation_kge = None

        # initialize text embeddings
        self.word_embedding = nn.Embedding(num_embeddings=num_word + 1, embedding_dim=word_dim,
                                           padding_idx=num_word)
        if self.init_same_as_plm:
            self.init_weights(self.word_embedding)
        if self.word_emb_file is not None:
            self.word_embedding.weight = nn.Parameter(
                torch.from_numpy(
                    np.pad(np.load(self.word_emb_file), ((0, 1), (0, 0)), 'constant')).type(
                    'torch.FloatTensor'))
            self.word_embedding.weight.requires_grad = False

    def share_module_def(self):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        kge_dim = self.kge_dim
        entity_dim = self.entity_dim

        if self.has_entity_kge:
            self.entity_linear = nn.Linear(in_features=kg_dim + kge_dim, out_features=entity_dim)
        else:
            self.entity_linear = nn.Linear(in_features=kg_dim, out_features=entity_dim)
        if self.init_same_as_plm:
            self.init_weights(self.entity_linear)

        if self.has_relation_kge:
            self.relation_linear = nn.Linear(in_features=2 * kg_dim + kge_dim, out_features=entity_dim)
        else:
            self.relation_linear = nn.Linear(in_features=2 * kg_dim, out_features=entity_dim)
        if self.init_same_as_plm:
            self.init_weights(self.relation_linear)

        # dropout
        self.lstm_drop = nn.Dropout(p=self.lstm_dropout)
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

        if self.encode_type:
            self.type_layer = TypeLayer(in_features=entity_dim, out_features=entity_dim,
                                        linear_drop=self.linear_drop, device=self.device)
            if self.init_same_as_plm:
                self.init_weights(self.type_layer.kb_self_linear)
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

    def private_module_def(self, args, num_entity, num_relation):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        kge_dim = self.kge_dim
        entity_dim = self.entity_dim
        self.reasoning = GNNReasoning(args, num_entity, num_relation, 768)
        self.instruction = LSTMInstruction(args, self.word_embedding, self.num_word)

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input):
        # batch_size = local_entity.size(0)
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        self.rel_features = self.get_rel_feature()
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, self.rel_features)
        # self.local_entity_emb = None
        self.curr_dist = curr_dist
        self.dist_history = [curr_dist]
        self.action_probs = []
        self.reasoning.init_reason(local_entity=local_entity,
                                   kb_adj_mat=kb_adj_mat,
                                   local_entity_emb=self.local_entity_emb,
                                   rel_features=self.rel_features)

    def one_step(self, num_step):
        # relational_ins, attn_weight = self.instruction.get_instruction(self.relational_ins, query_mask, step=num_step)
        relational_ins = self.instruction_list[num_step]
        # attn_weight = self.attn_list[num_step]
        # self.relational_ins = relational_ins
        self.curr_dist = self.reasoning(self.curr_dist, relational_ins, step=num_step)
        self.dist_history.append(self.curr_dist)

    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss_new(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    def forward(self, batch, training=False):
        current_dist, q_input, query_mask, kb_adj_mat, answer_dist, \
        local_entity, query_entities, true_batch_id = batch
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input)
        for i in range(self.num_step):
            self.one_step(num_step=i)
        pred_dist = self.dist_history[-1]
        # pred_dist = torch.sum(torch.stack(self.dist_history[1:], dim=1), dim=1)
        # loss, extras = self.calc_loss_basic(answer_dist)
        # tp_loss = self.get_loss_kl(pred_dist, answer_dist)
        # (batch_size, max_local_entity)
        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        # filter no answer training case
        # loss = torch.sum(tp_loss * case_valid) / pred_dist.size(0)
        main_loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        distill_loss = 0
        for i in range(self.num_step - 1):
            curr_dist = self.dist_history[i + 1]
            # teacher_dist = middle_dist[i].detach()
            tp_label_loss = self.calc_loss_label(curr_dist=curr_dist,
                                                 teacher_dist=answer_dist,
                                                 label_valid=case_valid)
            distill_loss += tp_label_loss
        loss = main_loss + distill_loss
        pred_dist = torch.sum(torch.stack(self.dist_history[1:], dim=1), dim=1)
        pred = torch.max(pred_dist, dim=1)[1]
        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, pred, pred_dist, tp_list


class GNNPLMModel(BaseModel):
    def __init__(self, args, num_entity, num_relation, rel2id, device):
        """
        num_relation: number of relation including self-connection
        """
        super(GNNPLMModel, self).__init__(args, num_entity, num_relation)
        self.device = device
        self.args = args
        self.id2rel = reverse_dict(rel2id)
        if self.args["use_self_loop"]:
            self.id2rel[len(self.id2rel)] = "self.loop.edge"
            assert len(self.id2rel) == num_relation
        self.embedding_def()
        self.private_module_def(args, num_entity, num_relation)

    def embedding_def(self):
        kg_dim = self.kg_dim
        kge_dim = self.kge_dim
        entity_dim = self.entity_dim
        num_entity = self.num_entity
        num_relation = self.num_relation

        # dropout
        self.linear_drop = nn.Dropout(p=self.linear_dropout)

        # if not use 1-hop neighboring relation representations
        if not self.encode_type:
            self.entity_embedding = nn.Embedding(num_embeddings=num_entity + 1, embedding_dim=kg_dim,
                                                 padding_idx=num_entity)
            if self.init_same_as_plm:
                self.init_weights(self.entity_embedding)
            if self.entity_emb_file is not None:
                self.entity_embedding.weight = nn.Parameter(
                    torch.from_numpy(np.pad(np.load(self.entity_emb_file), ((0, 1), (0, 0)), 'constant')).type(
                        'torch.FloatTensor'))
                self.entity_embedding.weight.requires_grad = False

            if self.entity_kge_file is not None:
                self.has_entity_kge = True
                self.entity_kge = nn.Embedding(num_embeddings=num_entity + 1, embedding_dim=kge_dim,
                                               padding_idx=num_entity)
                self.entity_kge.weight = nn.Parameter(
                    torch.from_numpy(np.pad(np.load(self.entity_kge_file), ((0, 1), (0, 0)), 'constant')).type(
                        'torch.FloatTensor'))
                self.entity_kge.weight.requires_grad = False
            else:
                self.entity_kge = None
        else:
            self.type_layer = TypeLayer(in_features=2 * kg_dim, out_features=entity_dim,
                                        linear_drop=self.linear_drop, device=self.device)
            if self.init_same_as_plm:
                self.init_weights(self.type_layer.kb_self_linear)
        if self.has_entity_kge:
            self.entity_linear = nn.Linear(in_features=kg_dim + kge_dim, out_features=entity_dim)
        else:
            self.entity_linear = nn.Linear(in_features=kg_dim, out_features=entity_dim)
        if self.init_same_as_plm:
            self.init_weights(self.entity_linear)

        # initialize relation embedding
        self.relation_embedding = nn.Embedding(num_embeddings=num_relation, embedding_dim=2 * kg_dim)
        if self.init_same_as_plm:
            self.init_weights(self.relation_embedding)
        if self.relation_emb_file is not None:
            np_tensor = self.load_relation_file(self.relation_emb_file)
            self.relation_embedding.weight = nn.Parameter(torch.from_numpy(np_tensor).type('torch.FloatTensor'))
        if self.relation_kge_file is not None:
            self.has_relation_kge = True
            self.relation_kge = nn.Embedding(num_embeddings=num_relation, embedding_dim=kge_dim)
            np_tensor = self.load_relation_file(self.relation_kge_file)
            self.relation_kge.weight = nn.Parameter(torch.from_numpy(np_tensor).type('torch.FloatTensor'))
        else:
            self.relation_kge = None
        if self.has_relation_kge:
            self.relation_linear = nn.Linear(in_features=2 * kg_dim + kge_dim, out_features=entity_dim)
        else:
            self.relation_linear = nn.Linear(in_features=2 * kg_dim, out_features=entity_dim)
        if self.init_same_as_plm:
            self.init_weights(self.relation_linear)

        # loss
        self.kld_loss = nn.KLDivLoss(reduction='none')
        self.bce_loss_logits = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss()

    def private_module_def(self, args, num_entity, num_relation):
        self.plm_config = AutoConfig.from_pretrained(self.args["model_path"])
        self.query_tokenizer = AutoTokenizer.from_pretrained(self.args["model_path"])
        self.query_encoder = AutoModel.from_pretrained(self.args["model_path"])
        if self.args["encode_relation_separate"]:
            self.relation_encoder = AutoModel.from_pretrained(self.args["relation_model_path"])
        self.reasoning = PLMGNNReasoning(args, num_entity, num_relation, self.plm_config.hidden_size)
        self.instruction = PLMInstruction(args, self.plm_config, self.query_tokenizer, self.query_encoder)

    def get_rel_feature_from_plm(self, kb_adj_mat):
        rels_batch = kb_adj_mat[1]
        rels_batch = rels_batch.flatten().tolist()
        rels_batch = list(set(rels_batch))
        all_relation_text = [self.id2rel[rid] for rid in rels_batch]
        # print(all_relation_text)
        # print("total relation", len(all_relation_text))
        relation_embedding_plm = []
        bs = 32
        num_batch = int(len(all_relation_text) / bs) + 1
        start = 0
        for i in range(num_batch):
            if start < len(all_relation_text):
                sample_relation = all_relation_text[start : start+bs]
                rel_inputs = self.query_tokenizer(sample_relation, padding="longest", return_tensors="pt")
                rel_inputs = rel_inputs.to(self.device)
                if self.args["encode_relation_separate"]:
                    rel_outputs = self.relation_encoder(**rel_inputs)
                else:
                    rel_outputs = self.query_encoder(**rel_inputs)
                if self.args["simplify_model"]:
                    if self.args["use_pooler_output"]:
                        rel_cls_embs = rel_outputs['pooler_output']
                    else:
                        rel_cls_embs = rel_outputs["last_hidden_state"][:, 0, :]
                else:
                    rel_cls_embs = rel_outputs["last_hidden_state"][:, 0, :]
                if self.args["fixed_plm_for_relation_encoding"] or self.args["encode_relation_separate"]:
                    rel_cls_embs = rel_cls_embs.detach()
                relation_embedding_plm.append(rel_cls_embs)
                start += bs
        rels_batch = torch.tensor(rels_batch, dtype=torch.long).to(self.device)
        relation_embedding_plm = torch.cat(relation_embedding_plm, dim=0) # (num_rel, hid_dim)
        relation_embedding_data = self.relation_embedding.weight.data
        relation_embedding_data[rels_batch, :] = relation_embedding_plm
        self.relation_embedding.weight.data = relation_embedding_data
        rel_features = self.relation_embedding.weight
        # print("rel_features.requires_grad: ", rel_features.requires_grad)
        if not self.args["simplify_model"]:
            rel_features = self.relation_linear(rel_features)
        # rel_features = self.relation_linear(rel_features)
        return rel_features

    def init_reason(self, curr_dist, local_entity, kb_adj_mat, q_input):
        # batch_size = local_entity.size(0)
        self.local_entity = local_entity
        self.instruction_list, self.attn_list = self.instruction(q_input)
        # self.rel_features = self.get_rel_feature()
        self.rel_features = self.get_rel_feature_from_plm(kb_adj_mat)
        self.local_entity_emb = self.get_ent_init(local_entity, kb_adj_mat, self.rel_features)
        self.curr_dist = curr_dist
        self.dist_history = [curr_dist]
        self.action_probs = []
        self.reasoning.init_reason(local_entity=local_entity,
                                   kb_adj_mat=kb_adj_mat,
                                   local_entity_emb=self.local_entity_emb,
                                   rel_features=self.rel_features)

    def one_step(self, num_step):
        # relational_ins, attn_weight = self.instruction.get_instruction(self.relational_ins, query_mask, step=num_step)
        relational_ins = self.instruction_list[num_step]
        # attn_weight = self.attn_list[num_step]
        # self.relational_ins = relational_ins
        self.curr_dist = self.reasoning(self.curr_dist, relational_ins, step=num_step)
        self.dist_history.append(self.curr_dist)

    def calc_loss_label(self, curr_dist, teacher_dist, label_valid):
        tp_loss = self.get_loss_new(pred_dist=curr_dist, answer_dist=teacher_dist, reduction='none')
        tp_loss = tp_loss * label_valid
        cur_loss = torch.sum(tp_loss) / curr_dist.size(0)
        return cur_loss

    def forward(self, batch, new_id2rel=None, training=False):
        if new_id2rel is not None:
            self.id2rel = new_id2rel
        current_dist, q_input, kb_adj_mat, answer_dist, \
        local_entity, query_entities, true_batch_id = batch
        self.init_reason(curr_dist=current_dist, local_entity=local_entity,
                         kb_adj_mat=kb_adj_mat, q_input=q_input)
        # for i in range(self.num_step):
        for i in range(self.num_step):
            self.one_step(num_step=i)

        pred_dist = self.dist_history[-1]

        # pred_list_for_real_hop = []
        # answer_dist_for_real_hop = []
        # pred_dist = []
        # for sid, hid in enumerate(hop_num): # 1,2,3,4
        #     pred_for_score_per_sample = []
        #     for other_hid in range(hid, self.num_step+1): # 1,2,3,4 | 2,3,4 | 3,4 | 4
        #         pred_list_for_real_hop.append(self.dist_history[other_hid][sid])
        #         pred_for_score_per_sample.append(self.dist_history[other_hid][sid])
        #         answer_dist_for_real_hop.append(answer_dist[sid])
        #
        #     pred_for_score_per_sample = torch.stack(pred_for_score_per_sample, dim=0)
        #     pred_for_score_per_sample = torch.sum(pred_for_score_per_sample, dim=0)
        #     pred_dist.append(pred_for_score_per_sample)
        # pred_dist = torch.stack(pred_dist, dim=0)

        # pred_dist_for_loss = torch.stack(pred_list_for_real_hop, dim=0)
        # answer_dist_for_loss = torch.stack(answer_dist_for_real_hop, dim=0)

        # answer_number = torch.sum(answer_dist_for_loss, dim=1, keepdim=True)
        # case_valid = (answer_number > 0).float()
        # loss = self.calc_loss_label(curr_dist=pred_dist_for_loss, teacher_dist=answer_dist_for_loss, label_valid=case_valid)
        # pred = torch.max(pred_dist, dim=1)[1]

        answer_number = torch.sum(answer_dist, dim=1, keepdim=True)
        case_valid = (answer_number > 0).float()
        loss = self.calc_loss_label(curr_dist=pred_dist, teacher_dist=answer_dist, label_valid=case_valid)
        pred = torch.max(pred_dist, dim=1)[1]

        if training:
            h1, f1 = self.get_eval_metric(pred_dist, answer_dist)
            tp_list = [h1.tolist(), f1.tolist()]
        else:
            tp_list = None
        return loss, pred, pred_dist, tp_list