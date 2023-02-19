import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import time
import numpy as np
try:
    from NSM.Modules.Instruction.base_instruction import BaseInstruction
except:
    try:
        from nsm_retriever.NSM.Modules.Instruction.base_instruction import BaseInstruction
    except:
        from UniModel.nsm_retriever.NSM.Modules.Instruction.base_instruction import BaseInstruction
VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000


class LSTMInstruction(BaseInstruction):

    def __init__(self, args, word_embedding, num_word):
        super(LSTMInstruction, self).__init__(args)
        self.word_embedding = word_embedding
        self.num_word = num_word
        self.encoder_def()
        entity_dim = self.entity_dim
        self.cq_linear = nn.Linear(in_features=2 * entity_dim, out_features=entity_dim)
        self.ca_linear = nn.Linear(in_features=entity_dim, out_features=1)
        for i in range(self.num_step):
            self.add_module('question_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))

    def encoder_def(self):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        kge_dim = self.kge_dim
        entity_dim = self.entity_dim
        self.node_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim,
                                    batch_first=True, bidirectional=False)

    def encode_question(self, query_text):
        batch_size = query_text.size(0)
        query_word_emb = self.word_embedding(query_text)  # batch_size, max_query_word, word_dim
        query_hidden_emb, (h_n, c_n) = self.node_encoder(self.lstm_drop(query_word_emb),
                                                         self.init_hidden(1, batch_size,
                                                                          self.entity_dim))  # 1, batch_size, entity_dim
        self.instruction_hidden = h_n
        self.instruction_mem = c_n
        self.query_node_emb = h_n.squeeze(dim=0).unsqueeze(dim=1)  # batch_size, 1, entity_dim
        self.query_hidden_emb = query_hidden_emb
        self.query_mask = (query_text != self.num_word).float()
        return query_hidden_emb, self.query_node_emb

    def init_reason(self, query_text):
        batch_size = query_text.size(0)
        self.encode_question(query_text)
        self.relational_ins = torch.zeros(batch_size, self.entity_dim).to(self.device)
        self.instructions = []
        self.attn_list = []

    def get_instruction(self, relational_ins, step=0, query_node_emb=None):
        query_hidden_emb = self.query_hidden_emb
        query_mask = self.query_mask
        if query_node_emb is None:
            query_node_emb = self.query_node_emb
        relational_ins = relational_ins.unsqueeze(1)
        question_linear = getattr(self, 'question_linear' + str(step))
        q_i = question_linear(self.linear_drop(query_node_emb))
        cq = self.cq_linear(self.linear_drop(torch.cat((relational_ins, q_i), dim=-1)))
        # batch_size, 1, entity_dim
        ca = self.ca_linear(self.linear_drop(cq * query_hidden_emb))
        # batch_size, max_local_entity, 1
        # cv = self.softmax_d1(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER)
        attn_weight = F.softmax(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER, dim=1)
        # batch_size, max_local_entity, 1
        relational_ins = torch.sum(attn_weight * query_hidden_emb, dim=1)
        return relational_ins, attn_weight

    def forward(self, query_text):
        self.init_reason(query_text)
        for i in range(self.num_step):
            relational_ins, attn_weight = self.get_instruction(self.relational_ins, step=i)
            self.instructions.append(relational_ins)
            self.attn_list.append(attn_weight)
            self.relational_ins = relational_ins
        return self.instructions, self.attn_list

    # def __repr__(self):
    #     return "LSTM + token-level attention"

class PLMInstruction(BaseInstruction):

    def __init__(self, args, plm_config, query_tokenizer, query_encoder):
        super(PLMInstruction, self).__init__(args)
        self.args = args
        self.plm_config, self.query_tokenizer, self.query_encoder = plm_config, query_tokenizer, query_encoder
        self.plm_hidden_size = self.plm_config.hidden_size
        self.encoder_def()
        entity_dim = self.entity_dim
        self.cq_linear = nn.Linear(in_features=2 * entity_dim, out_features=entity_dim)
        self.ca_linear = nn.Linear(in_features=entity_dim, out_features=1)
        if self.args["init_same_as_plm"]:
            self.init_weights(self.cq_linear)
            self.init_weights(self.ca_linear)
        for i in range(self.num_step):
            if self.args["simplify_model"]:
                self.add_module('question_linear' + str(i), nn.Linear(in_features=self.plm_hidden_size, out_features=entity_dim))
            else:
                self.add_module('question_linear' + str(i), nn.Linear(in_features=entity_dim, out_features=entity_dim))
            if self.args["init_same_as_plm"]:
                self.init_weights(getattr(self, 'question_linear' + str(i)))

    def encoder_def(self):
        # initialize entity embedding
        word_dim = self.word_dim
        kg_dim = self.kg_dim
        kge_dim = self.kge_dim
        entity_dim = self.entity_dim
        # self.node_encoder = nn.LSTM(input_size=word_dim, hidden_size=entity_dim,
        #                             batch_first=True, bidirectional=False)
        # self.query_convert = nn.Sequential(
        #     nn.Linear(in_features=self.hid_size,  out_features=int(self.hid_size/2)),
        #     nn.ReLU(),
        #     nn.Linear(in_features=int(self.hid_size/2), out_features=entity_dim))
        if self.args["simplify_model"]:
            self.query_convert = nn.Linear(in_features=self.plm_hidden_size, out_features=self.plm_hidden_size)
        else:
            self.query_convert = nn.Linear(in_features=self.plm_hidden_size, out_features=entity_dim)
        if self.args["init_same_as_plm"]:
            self.init_weights(self.query_convert)

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

    def encode_question(self, query_input):
        '''
        query_ids: one batch of tokenized input query (bs,msl)
        attention_mask: one batch of attention mask (bs,msl)
        query_mask: one batch of mask for query text, 0 for query text, 1 for other token ids (bs, msl)
        '''
        batch_size = query_input["input_ids"].size(0)
        # query_word_emb = self.word_embedding(query_ids)  # batch_size, max_query_word, word_dim
        # query_hidden_emb, (h_n, c_n) = self.node_encoder(self.lstm_drop(query_word_emb),
        #                                                  self.init_hidden(1, batch_size,
        #                                                                   self.entity_dim))  # 1, batch_size, entity_dim

        # self.instruction_hidden = h_n
        # self.instruction_mem = c_n
        # self.query_node_emb = h_n.squeeze(dim=0).unsqueeze(dim=1)  # batch_size, 1, entity_dim
        # self.query_hidden_emb = query_hidden_emb
        # self.query_mask = (query_text != self.num_word).float()
        # return query_hidden_emb, self.query_node_emb
        if "bert" in self.args["model_path"]:
            output = self.query_encoder(input_ids=query_input["input_ids"],
                                        attention_mask=query_input["attention_mask"],
                                        token_type_ids=query_input["token_type_ids"], output_hidden_states=True)
        elif "roberta" in self.args["model_path"]:
            output = self.query_encoder(input_ids=query_input["input_ids"],
                                        attention_mask=query_input["attention_mask"], output_hidden_states=True)
        else:
            # raise NotImplementedError
            output = self.query_encoder(input_ids=query_input["input_ids"],
                                        attention_mask=query_input["attention_mask"], output_hidden_states=True)
        query_hidden_emb = output["last_hidden_state"]  # (bs, ml, hid)
        if self.args["fixed_plm_for_query_encoding"]:
            query_hidden_emb = query_hidden_emb.detach()
        # query_hidden_emb = output["hidden_states"]  # ((bs,ml,hid), )
        if self.args["simplify_model"]:
            query_hidden_emb = query_hidden_emb
            # query_node_emb = query_hidden_emb[:, 0:1, :]
            if self.args["use_pooler_output"]:
                query_node_emb = output["pooler_output"]
            else:
                query_node_emb = query_hidden_emb[:, 0:1, :]
        else:
            query_hidden_emb = self.query_convert(query_hidden_emb)  # (bs, ml, entity_dim)
            query_node_emb = query_hidden_emb[:, 0:1, :]  # get the cls (bs, 1, hid) -> (bs, 1, entity_dim)
        # query_node_emb = output["pooler_output"].unsqueeze(dim=1)  # get the cls (bs, 1, hid)

        self.query_hidden_emb = query_hidden_emb
        self.query_node_emb = query_node_emb
        self.query_mask = query_input["query_mask"]
        return query_hidden_emb, self.query_node_emb

    def init_reason(self, query_input):
        batch_size = query_input["input_ids"].size(0)
        self.encode_question(query_input)
        self.relational_ins = torch.zeros(batch_size, self.entity_dim).to(self.device) # (bs, entity_dim)
        self.instructions = []
        self.attn_list = []

    def get_instruction(self, relational_ins, step=0, query_node_emb=None):
        query_hidden_emb = self.query_hidden_emb # (bs, ml, entity_dim)
        query_mask = self.query_mask
        if query_node_emb is None:
            query_node_emb = self.query_node_emb  # (bs, 1, entity_dim)
        if self.args["simplify_model"]:
            question_linear = getattr(self, 'question_linear' + str(step))
            if self.args["aggregate_token_representation"]:
                q_i = question_linear(self.linear_drop(query_node_emb))  # (bs, 1, entity_dim)
                q_i = q_i.permute(0, 2, 1) # (bs, entity_dim, 1)
                # batch_size, entity_dim, 1
                ca = torch.bmm(query_hidden_emb, q_i)  # (bs, msl, 1)
                # batch_size, max_local_entity, 1
                # cv = self.softmax_d1(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER)
                # attn_weight = F.softmax(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER, dim=1)
                attn_weight = F.softmax(ca + query_mask.unsqueeze(2) * VERY_NEG_NUMBER, dim=1)  # (bs, msl, 1)
                # batch_size, max_local_entity, 1
                relational_ins = torch.sum(attn_weight * query_hidden_emb, dim=1)  # (bs, entity_dim)
            else:
                q_i = question_linear(self.linear_drop(query_node_emb))  # (bs, entity_dim)
                relational_ins = q_i
            attn_weight = None
        else:
            relational_ins = relational_ins.unsqueeze(1)  # (bs, 1, entity_dim)
            question_linear = getattr(self, 'question_linear' + str(step))
            q_i = question_linear(self.linear_drop(query_node_emb))  # (bs, 1, entity_dim)
            cq = self.cq_linear(self.linear_drop(torch.cat((relational_ins, q_i), dim=-1))) # (bs, 1, entity_dim)
            # batch_size, 1, entity_dim
            ca = self.ca_linear(self.linear_drop(cq * query_hidden_emb)) # (bs, msl, 1)
            # batch_size, max_local_entity, 1
            # cv = self.softmax_d1(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER)
            # attn_weight = F.softmax(ca + (1 - query_mask.unsqueeze(2)) * VERY_NEG_NUMBER, dim=1)
            attn_weight = F.softmax(ca + query_mask.unsqueeze(2) * VERY_NEG_NUMBER, dim=1) # (bs, msl, 1)
            # batch_size, max_local_entity, 1
            relational_ins = torch.sum(attn_weight * query_hidden_emb, dim=1) # (bs, entity_dim)
            # relational_ins = self.query_hidden_emb[12-(self.num_step-step)][:, 0, :]  # (bs, hid_dim)
            attn_weight = None
        return relational_ins, attn_weight

    def forward(self, query_input):
        self.init_reason(query_input)
        for i in range(self.num_step):
            relational_ins, attn_weight = self.get_instruction(self.relational_ins, step=i)
            self.instructions.append(relational_ins)
            self.attn_list.append(attn_weight)
            self.relational_ins = relational_ins
        return self.instructions, self.attn_list

    # def __repr__(self):
    #     return "LSTM + token-level attention"