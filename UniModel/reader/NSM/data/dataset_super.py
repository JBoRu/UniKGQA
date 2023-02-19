import json
import numpy as np
import re
import os
from tqdm import tqdm
import torch
from collections import Counter
try:
    from NSM.data.basic_dataset import BasicDataLoader
except:
    from UniModel.reader.NSM.data.basic_dataset import BasicDataLoader
from transformers import AutoTokenizer
import logging

logger = logging.getLogger("NSM")


class SingleDataLoader(BasicDataLoader):
    def __init__(self, config, word2id, relation2id, entity2id, data_type="train"):
        super(SingleDataLoader, self).__init__(config, word2id, relation2id, entity2id, data_type)
        self._load_data()
        self.use_label = config['use_label']
        self.label_f1 = config['label_f1']
        if data_type == "train" and self.use_label:
            label_file = os.path.join(config['checkpoint_dir'], config['label_file'])
            self.load_label(label_file)

    def _build_graph(self, tp_graph):
        head_list, rel_list, tail_list = tp_graph
        length = len(head_list)
        out_degree = {}
        in_degree = {}
        for i in range(length):
            head = head_list[i]
            rel = rel_list[i]
            tail = tail_list[i]
            out_degree.setdefault(head, {})
            out_degree[head].setdefault(rel, set())
            out_degree[head][rel].add(tail)
            in_degree.setdefault(tail, {})
            in_degree[tail].setdefault(rel, set())
            in_degree[tail][rel].add(head)
        return in_degree, out_degree

    def backward_step(self, possible_heads, cur_action, target_tail, in_degree):
        '''
        input: graph_edge, cur answers, cur relation
        output: edges used, possible heads
        '''
        tp_list = []
        available_heads = set()
        flag = False
        if self.use_self_loop and cur_action == self.num_kb_relation - 1:
            for ent in target_tail:
                tp_list.append((ent, self.num_kb_relation - 1, ent))
            available_heads |= target_tail
            # print("self-loop")
        else:
            # print("non self-loop")
            # print(target_tail)
            for ent in target_tail:
                # print("have target")
                if ent in in_degree and cur_action in in_degree[ent]:
                    # print("enter case")
                    legal_set = in_degree[ent][cur_action] & possible_heads
                    for legal_head in legal_set:
                        tp_list.append((legal_head, cur_action, ent))
                        available_heads.add(legal_head)
                else:
                    flag = True
                    print("debug")
                    print(ent in in_degree)
                    if ent in in_degree:
                        print(cur_action in in_degree[ent])
        return available_heads, tp_list, flag

    def forward_step(self, hop_edge_list, tp_weight_dict):
        new_weight_dict = {}
        if len(hop_edge_list) == 0:
            return new_weight_dict
        # tp_weight_dict = hop_weight_dict[step]
        out_degree = {}
        for head, rel, tail in hop_edge_list:
            if head in tp_weight_dict:
                out_degree.setdefault(head, 0.0)
                out_degree[head] += 1.0
        for head, rel, tail in hop_edge_list:
            if head in tp_weight_dict:
                edge_weight = tp_weight_dict[head] / out_degree[head]
                new_weight_dict.setdefault(tail, 0.0)
                new_weight_dict[tail] += edge_weight
        return new_weight_dict

    def multi_hop_trace(self, tp_obj, acc_reason_answers, in_degree, seed_ent=0):
        hop_dict = {}
        tp_key = "seed_%d" % (seed_ent)
        pred_entities = set(tp_obj[tp_key][str(self.num_step - 1)]["answer"])
        common = pred_entities & acc_reason_answers
        hop_edges = {}
        if len(common) == 0:
            for step in range(self.num_step):
                hop_edges[step] = []
            return hop_edges, True
        action_list = []
        order_list = reversed(range(self.num_step))
        target_tail = acc_reason_answers
        # hop_dict[self.num_step] = target_tail
        exist_flag = False
        for step in order_list:
            # if step == self.num_step - 1:
            cur_action = int(tp_obj[tp_key][str(step)]["action"])
            action_list.append(cur_action)
            if step > 0:
                possible_heads = set(tp_obj[tp_key][str(step - 1)]["answer"])
            else:
                possible_heads = set([seed_ent])
            # print("step", step, possible_heads, cur_action)
            target_tail, tp_triple_list, flag = self.backward_step(possible_heads, cur_action, target_tail, in_degree)
            if flag or exist_flag:
                exist_flag = True
                # print(target_tail, tp_triple_list)
                # hop_dict[step] = target_tail
            # print(target_tail, tp_triple_list)
            # hop_dict[step] = target_tail
            hop_edges[step] = tp_triple_list
        # print(hop_edges)
        return hop_edges, exist_flag

    def load_label(self, label_file):
        if not self.use_label:
            return None
        if self.num_step == 1:
            return None
        label_dist = np.zeros((self.num_data, self.num_step, self.max_local_entity), dtype=float)
        label_valid = np.zeros((self.num_data, 1), dtype=float)
        index = 0
        num_labelled_case = 0
        with open(label_file) as f_in:
            for line in f_in:
                tp_obj = json.loads(line)
                hit = tp_obj['hit']
                f1 = tp_obj['f1']
                tp_seed_list = self.seed_list[index]
                tp_edge_list = self.kb_adj_mats[index]
                in_degree, out_degree = self._build_graph(tp_edge_list)
                real_answer_list = []
                g2l = self.global2local_entity_maps[index]
                for global_ent in self.answer_lists[index]:
                    if global_ent in g2l:
                        real_answer_list.append(g2l[global_ent])
                accurate_answer_set = set(real_answer_list)
                merge_result = tp_obj["merge_pred"]
                acc_reason_answers = set(merge_result) & accurate_answer_set
                num_seed = len(tp_seed_list)
                if hit > 0 and f1 >= self.label_f1:
                    label_valid[index, 0] = 1.0
                    num_labelled_case += 1
                    # good case, we will label it with care
                    label_flag = False
                    for seed_ent in tp_seed_list:
                        hop_edges, flag = self.multi_hop_trace(tp_obj, acc_reason_answers, in_degree, seed_ent=seed_ent)
                        tp_weight_dict = {seed_ent: 1.0 / len(tp_seed_list)}
                        if not flag:
                            label_flag = True
                        for i in range(self.num_step):
                            hop_edge_list = hop_edges[i]
                            curr_weight_dict = self.forward_step(hop_edge_list, tp_weight_dict)
                            for local_ent in curr_weight_dict:
                                label_dist[index, i, local_ent] += curr_weight_dict[local_ent]
                            tp_weight_dict = curr_weight_dict
                    if not label_flag:
                        print(index, "can't label")
                        num_labelled_case -= 1
                        # print(line.strip())
                        label_valid[index, 0] = 0.0
                        for i in range(self.num_step):
                            ent_ct = {}
                            for seed_ent in tp_seed_list:
                                tp_key = "seed_%d" % (seed_ent)
                                tp_answer_list = tp_obj[tp_key][str(i)]["answer"]
                                for local_ent in tp_answer_list:
                                    ent_ct.setdefault(local_ent, 0.0)
                                    ent_ct[local_ent] += 1.0 / len(tp_answer_list)
                            # for more detailed labeling, we can deduce it from final aggregated results
                            for local_ent in ent_ct:
                                label_dist[index, i, local_ent] = ent_ct[local_ent] / num_seed
                                # dist sum 1.0
                else:
                    # bad case, we will label it simple, because we don't use it
                    label_valid[index, 0] = 0.0
                    for i in range(self.num_step):
                        ent_ct = {}
                        for seed_ent in tp_seed_list:
                            tp_key = "seed_%d" % (seed_ent)
                            tp_answer_list = tp_obj[tp_key][str(i)]["answer"]
                            for local_ent in tp_answer_list:
                                ent_ct.setdefault(local_ent, 0.0)
                                ent_ct[local_ent] += 1.0 / len(tp_answer_list)
                        # for more detailed labeling, we can deduce it from final aggregated results
                        for local_ent in ent_ct:
                            label_dist[index, i, local_ent] = ent_ct[local_ent] / num_seed
                            # dist sum 1.0
                index += 1
        assert index == self.num_data
        self.label_dist = label_dist
        self.label_valid = label_valid
        print('--------------------------------')
        print("{} cases among {} cases can be labelled".format(num_labelled_case, self.num_data))
        print('--------------------------------')

    def get_label(self):
        if not self.use_label or self.num_step == 1:
            return None, None
        label_valid = self.label_valid[self.sample_ids]
        # print(label_valid)
        labeL_dist_list = []
        for i in range(self.num_step):
            label_dist = self.label_dist[self.sample_ids, i]
            labeL_dist_list.append(label_dist)
        return labeL_dist_list, label_valid

    def deal_multi_seed(self, sample_ids):
        true_sample_ids = []
        tp_seed_list = self.seed_list[sample_ids]
        true_batch_id = []
        true_seed_ids = []
        # multi_seed_maks = []
        for i, seed_list in enumerate(tp_seed_list):
            true_batch_id.append([])
            for seed_ent in seed_list:
                true_batch_id[i].append(len(true_sample_ids))
                true_sample_ids.append(sample_ids[i])
                true_seed_ids.append(seed_ent)
                # if len(seed_list) > 1:
                #     multi_seed_maks.append(1.0)
                # else:
                #     multi_seed_maks.append(0.0)
        # print(tp_seed_list)
        # print(true_sample_ids, len(true_sample_ids))
        seed_dist = np.zeros((len(true_sample_ids), self.max_local_entity), dtype=float)
        for j, local_ent in enumerate(true_seed_ids):
            seed_dist[j, local_ent] = 1.0
            # single seed entity
        return true_batch_id, true_sample_ids, seed_dist

    def get_batch(self, iteration, batch_size, fact_dropout, q_type=None, test=False):
        start = batch_size * iteration
        end = min(batch_size * (iteration + 1), self.num_data)
        sample_ids = self.batches[start: end]
        self.sample_ids = sample_ids
        # true_batch_id, sample_ids, seed_dist = self.deal_multi_seed(ori_sample_ids)
        # self.sample_ids = sample_ids
        # self.true_sample_ids = ori_sample_ids
        # self.batch_ids = true_batch_id
        true_batch_id = None
        seed_dist = self.seed_distribution[sample_ids]
        q_input = self.deal_q_type(q_type)
        if test:
            return self.candidate_entities[sample_ids], \
                   self.query_entities[sample_ids], \
                   (self._build_fact_mat(sample_ids, fact_dropout=fact_dropout)), \
                   q_input, \
                   seed_dist, \
                   true_batch_id, \
                   self.answer_dists[sample_ids], \
                   self.answer_lists[sample_ids],\

        return self.candidate_entities[sample_ids], \
               self.query_entities[sample_ids], \
               (self._build_fact_mat(sample_ids, fact_dropout=fact_dropout)), \
               q_input, \
               seed_dist, \
               true_batch_id, \
               self.answer_dists[sample_ids]


class SingleDataLoaderForPLM(BasicDataLoader):
    def __init__(self, config, word2id, relation2id, entity2id, data_type="train"):
        super(SingleDataLoaderForPLM, self).__init__(config, word2id, relation2id, entity2id, data_type)
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
        self._load_data()
        self.use_label = config['use_label']
        self.label_f1 = config['label_f1']
        if data_type == "train" and self.use_label:
            label_file = os.path.join(config['checkpoint_dir'], config['label_file'])
            self.load_label(label_file)

    def _load_data(self):
        logger.info('converting global to local entity index ...')
        self.global2local_entity_maps = self._build_global2local_entity_maps()

        if self.use_self_loop:
            self.max_facts = self.max_facts + self.max_local_entity

        self.question_id = []
        self.candidate_entities = np.full((self.num_data, self.max_local_entity), len(self.entity2id), dtype=int)
        self.kb_adj_mats = np.empty(self.num_data, dtype=object)
        self.q_adj_mats = np.empty(self.num_data, dtype=object)
        # self.kb_fact_rels = np.full((self.num_data, self.max_facts), self.num_kb_relation, dtype=int)
        self.query_entities = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.seed_list = np.empty(self.num_data, dtype=object)
        self.seed_distribution = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        # self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.answer_lists = np.empty(self.num_data, dtype=object)

        logger.info('preparing dep ...')
        self._prepare_dep()
        logger.info('preparing data ...')
        self._prepare_data()

    def _prepare_dep(self):
        max_count = 0
        for line in self.data:
            query = line["question"]
            query_ids = self.tokenizer.tokenize(query, add_special_tokens=True)
            max_count = max(max_count, len(query_ids))
        logger.info("Max length of query is %d"%(max_count))
        self.max_query_word = max_count
        self.query_ids = np.full((self.num_data, self.max_query_word), self.tokenizer.pad_token_id, dtype=int)
        self.attention_mask = np.full((self.num_data, self.max_query_word), 0, dtype=int)
        self.query_mask = np.full((self.num_data, self.max_query_word), 1, dtype=int)
        self.token_type_ids = np.full((self.num_data, self.max_query_word), 0, dtype=int)
        next_id = 0
        self.node2layer = []
        self.dep_parents = []
        self.dep_relations = []
        for sample in tqdm(self.data):
            query = sample["question"]
            input = self.tokenizer(query, add_special_tokens=True, truncation=True, padding="max_length",
                                   max_length=self.max_query_word, return_tensors='np', return_special_tokens_mask=True)
            self.query_ids[next_id, :] = input["input_ids"]
            self.attention_mask[next_id, :] = input["attention_mask"]
            self.query_mask[next_id, :] = input["special_tokens_mask"]
            if hasattr(input, "token_type_ids"):
                self.token_type_ids[next_id, :] = input["token_type_ids"]
            next_id += 1

    def _prepare_data(self):
        """
        global2local_entity_maps: a map from global entity id to local entity id
        adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
        """
        next_id = 0
        num_query_entity = {}
        for sample in tqdm(self.data):
            self.question_id.append(sample["id"])
            # get a list of local entities
            g2l = self.global2local_entity_maps[next_id]
            if len(g2l) == 0:
                logger.info(next_id)
                continue
            # build connection between question and entities in it
            tp_set = set()
            seed_list = []
            for j, entity in enumerate(sample['entities']):
                # if entity['text'] not in self.entity2id:
                #     continue
                global_entity = entity  # self.entity2id[entity['text']]
                if global_entity not in g2l:
                    continue
                local_ent = g2l[global_entity]
                self.query_entities[next_id, local_ent] = 1.0
                seed_list.append(local_ent)
                tp_set.add(local_ent)
            self.seed_list[next_id] = seed_list
            num_query_entity[next_id] = len(tp_set)
            for global_entity, local_entity in g2l.items():
                if local_entity not in tp_set:  # skip entities in question
                    self.candidate_entities[next_id, local_entity] = global_entity
                # if local_entity != 0:  # skip question node
                #     self.candidate_entities[next_id, local_entity] = global_entity

            # relations in local KB
            head_list = []
            rel_list = []
            tail_list = []
            # for i, tpl in enumerate(sample['subgraph']['new_tuples']):
            for i, tpl in enumerate(sample['subgraph']['tuples']):
                sbj, rel, obj = tpl
                # head = g2l[self.entity2id[sbj['text']]]
                # rel = self.relation2id[rel['text']]
                # tail = g2l[self.entity2id[obj['text']]]
                head = g2l[sbj]
                rel = int(rel)
                tail = g2l[obj]
                head_list.append(head)
                rel_list.append(rel)
                tail_list.append(tail)
                if self.use_inverse_relation:
                    head_list.append(tail)
                    rel_list.append(rel + len(self.relation2id))
                    tail_list.append(head)
            if len(tp_set) > 0:
                for local_ent in tp_set:
                    self.seed_distribution[next_id, local_ent] = 1.0 / len(tp_set)
            else:
                for index in range(len(g2l)):
                    self.seed_distribution[next_id, index] = 1.0 / len(g2l)
            try:
                assert np.sum(self.seed_distribution[next_id]) > 0.0
            except:
                logger.info(next_id, len(tp_set))
                exit(-1)

            # tokenize question
            # tokens = self.tokenize_sent(sample['question'])
            # tokens = sample['question'].split()
            # for j, word in enumerate(tokens):
            #     # if j < self.max_query_word:
            #     if word in self.word2id:
            #         self.query_texts[next_id, j] = self.word2id[word]
            #     else:
            #         self.query_texts[next_id, j] = len(self.word2id)# self.word2id['__unk__']

            # construct distribution for answers
            answer_list = []
            for answer in sample['answers']:
                keyword = 'kb_id'
                # answer_ent = self.entity2id[answer[keyword]]
                answer_ent = answer[keyword]
                answer_list.append(answer_ent)
                if answer_ent in g2l:
                    self.answer_dists[next_id, g2l[answer_ent]] = 1.0
                # else:
                #     print("answer ent not in subgraph!")
                #     logger.info("answer ent not in subgraph!")
            self.answer_lists[next_id] = answer_list
            self.kb_adj_mats[next_id] = (np.array(head_list, dtype=int),
                                         np.array(rel_list, dtype=int),
                                         np.array(tail_list, dtype=int))

            next_id += 1
        num_no_query_ent = 0
        num_one_query_ent = 0
        num_multiple_ent = 0
        for i in range(next_id):
            ct = num_query_entity[i]
            if ct == 1:
                num_one_query_ent += 1
            elif ct == 0:
                num_no_query_ent += 1
            else:
                num_multiple_ent += 1
        logger.info("{} cases in total, {} cases without query entity, {} cases with single query entity,"
                    " {} cases with multiple query entities".format(next_id, num_no_query_ent,
                                                                    num_one_query_ent, num_multiple_ent))

    def _build_graph(self, tp_graph):
        head_list, rel_list, tail_list = tp_graph
        length = len(head_list)
        out_degree = {}
        in_degree = {}
        for i in range(length):
            head = head_list[i]
            rel = rel_list[i]
            tail = tail_list[i]
            out_degree.setdefault(head, {})
            out_degree[head].setdefault(rel, set())
            out_degree[head][rel].add(tail)
            in_degree.setdefault(tail, {})
            in_degree[tail].setdefault(rel, set())
            in_degree[tail][rel].add(head)
        return in_degree, out_degree

    def get_batch(self, iteration, batch_size, fact_dropout, q_type=None, test=False):
        start = batch_size * iteration
        end = min(batch_size * (iteration + 1), self.num_data)
        sample_ids = self.batches[start: end]
        self.sample_ids = sample_ids
        # true_batch_id, sample_ids, seed_dist = self.deal_multi_seed(ori_sample_ids)
        # self.sample_ids = sample_ids
        # self.true_sample_ids = ori_sample_ids
        # self.batch_ids = true_batch_id
        true_batch_id = None
        seed_dist = self.seed_distribution[sample_ids]
        q_input = self.deal_q_type(q_type)
        kb_adj_mat = (self._build_fact_mat(sample_ids, fact_dropout=fact_dropout))
        if test:
            return self.candidate_entities[sample_ids], \
                   self.query_entities[sample_ids], \
                   kb_adj_mat, \
                   q_input, \
                   seed_dist, \
                   true_batch_id, \
                   self.answer_dists[sample_ids], \
                   self.answer_lists[sample_ids],\

        return self.candidate_entities[sample_ids], \
               self.query_entities[sample_ids], \
               kb_adj_mat, \
               q_input, \
               seed_dist, \
               true_batch_id, \
               self.answer_dists[sample_ids]

    def deal_q_type(self, q_type=None):
        sample_ids = self.sample_ids
        if q_type is None:
            q_type = self.q_type
        if q_type == "seq":
            q_input = {"input_ids": self.query_ids[sample_ids], "attention_mask": self.attention_mask[sample_ids],
                       "token_type_ids": self.token_type_ids[sample_ids], "query_mask": self.query_mask[sample_ids]}
        else:
            raise NotImplementedError
        return q_input

    def _build_fact_mat(self, sample_ids, fact_dropout):
        batch_heads = np.array([], dtype=int)
        batch_rels = np.array([], dtype=int)
        batch_tails = np.array([], dtype=int)
        batch_ids = np.array([], dtype=int)
        for i, sample_id in enumerate(sample_ids):
            index_bias = i * self.max_local_entity
            head_list, rel_list, tail_list = self.kb_adj_mats[sample_id]
            num_fact = len(head_list)
            num_keep_fact = int(np.floor(num_fact * (1 - fact_dropout)))
            mask_index = np.random.permutation(num_fact)[: num_keep_fact]

            real_head_list = head_list[mask_index] + index_bias
            real_tail_list = tail_list[mask_index] + index_bias
            real_rel_list = rel_list[mask_index]
            batch_heads = np.append(batch_heads, real_head_list)
            batch_rels = np.append(batch_rels, real_rel_list)
            batch_tails = np.append(batch_tails, real_tail_list)
            batch_ids = np.append(batch_ids, np.full(len(mask_index), i, dtype=int))
            if self.use_self_loop:
                num_ent_now = len(self.global2local_entity_maps[sample_id])
                ent_array = np.array(range(num_ent_now), dtype=int) + index_bias
                rel_array = np.array([self.num_kb_relation - 1] * num_ent_now, dtype=int)
                batch_heads = np.append(batch_heads, ent_array)
                batch_tails = np.append(batch_tails, ent_array)
                batch_rels = np.append(batch_rels, rel_array)
                batch_ids = np.append(batch_ids, np.full(num_ent_now, i, dtype=int))
        fact_ids = np.array(range(len(batch_heads)), dtype=int)
        head_count = Counter(batch_heads)
        # tail_count = Counter(batch_tails)
        weight_list = [1.0 / head_count[head] for head in batch_heads]
        # entity2fact_index = torch.LongTensor([batch_heads, fact_ids])
        # entity2fact_val = torch.FloatTensor(weight_list)
        # entity2fact_mat = torch.sparse.FloatTensor(entity2fact_index, entity2fact_val, torch.Size(
        #     [len(sample_ids) * self.max_local_entity, len(batch_heads)]))
        return batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list

    def backward_step(self, possible_heads, cur_action, target_tail, in_degree):
        '''
        input: graph_edge, cur answers, cur relation
        output: edges used, possible heads
        '''
        tp_list = []
        available_heads = set()
        flag = False
        if self.use_self_loop and cur_action == self.num_kb_relation - 1:
            for ent in target_tail:
                tp_list.append((ent, self.num_kb_relation - 1, ent))
            available_heads |= target_tail
            # print("self-loop")
        else:
            # print("non self-loop")
            # print(target_tail)
            for ent in target_tail:
                # print("have target")
                if ent in in_degree and cur_action in in_degree[ent]:
                    # print("enter case")
                    legal_set = in_degree[ent][cur_action] & possible_heads
                    for legal_head in legal_set:
                        tp_list.append((legal_head, cur_action, ent))
                        available_heads.add(legal_head)
                else:
                    flag = True
                    logger.info("debug")
                    logger.info(ent in in_degree)
                    if ent in in_degree:
                        logger.info(cur_action in in_degree[ent])
        return available_heads, tp_list, flag

    def forward_step(self, hop_edge_list, tp_weight_dict):
        new_weight_dict = {}
        if len(hop_edge_list) == 0:
            return new_weight_dict
        # tp_weight_dict = hop_weight_dict[step]
        out_degree = {}
        for head, rel, tail in hop_edge_list:
            if head in tp_weight_dict:
                out_degree.setdefault(head, 0.0)
                out_degree[head] += 1.0
        for head, rel, tail in hop_edge_list:
            if head in tp_weight_dict:
                edge_weight = tp_weight_dict[head] / out_degree[head]
                new_weight_dict.setdefault(tail, 0.0)
                new_weight_dict[tail] += edge_weight
        return new_weight_dict

    def multi_hop_trace(self, tp_obj, acc_reason_answers, in_degree, seed_ent=0):
        hop_dict = {}
        tp_key = "seed_%d" % (seed_ent)
        pred_entities = set(tp_obj[tp_key][str(self.num_step - 1)]["answer"])
        common = pred_entities & acc_reason_answers
        hop_edges = {}
        if len(common) == 0:
            for step in range(self.num_step):
                hop_edges[step] = []
            return hop_edges, True
        action_list = []
        order_list = reversed(range(self.num_step))
        target_tail = acc_reason_answers
        # hop_dict[self.num_step] = target_tail
        exist_flag = False
        for step in order_list:
            # if step == self.num_step - 1:
            cur_action = int(tp_obj[tp_key][str(step)]["action"])
            action_list.append(cur_action)
            if step > 0:
                possible_heads = set(tp_obj[tp_key][str(step - 1)]["answer"])
            else:
                possible_heads = set([seed_ent])
            # print("step", step, possible_heads, cur_action)
            target_tail, tp_triple_list, flag = self.backward_step(possible_heads, cur_action, target_tail, in_degree)
            if flag or exist_flag:
                exist_flag = True
                # print(target_tail, tp_triple_list)
                # hop_dict[step] = target_tail
            # print(target_tail, tp_triple_list)
            # hop_dict[step] = target_tail
            hop_edges[step] = tp_triple_list
        # print(hop_edges)
        return hop_edges, exist_flag

    def load_label(self, label_file):
        if not self.use_label:
            return None
        if self.num_step == 1:
            return None
        label_dist = np.zeros((self.num_data, self.num_step, self.max_local_entity), dtype=float)
        label_valid = np.zeros((self.num_data, 1), dtype=float)
        index = 0
        num_labelled_case = 0
        with open(label_file) as f_in:
            for line in f_in:
                tp_obj = json.loads(line)
                hit = tp_obj['hit']
                f1 = tp_obj['f1']
                tp_seed_list = self.seed_list[index]
                tp_edge_list = self.kb_adj_mats[index]
                in_degree, out_degree = self._build_graph(tp_edge_list)
                real_answer_list = []
                g2l = self.global2local_entity_maps[index]
                for global_ent in self.answer_lists[index]:
                    if global_ent in g2l:
                        real_answer_list.append(g2l[global_ent])
                accurate_answer_set = set(real_answer_list)
                merge_result = tp_obj["merge_pred"]
                acc_reason_answers = set(merge_result) & accurate_answer_set
                num_seed = len(tp_seed_list)
                if hit > 0 and f1 >= self.label_f1:
                    label_valid[index, 0] = 1.0
                    num_labelled_case += 1
                    # good case, we will label it with care
                    label_flag = False
                    for seed_ent in tp_seed_list:
                        hop_edges, flag = self.multi_hop_trace(tp_obj, acc_reason_answers, in_degree, seed_ent=seed_ent)
                        tp_weight_dict = {seed_ent: 1.0 / len(tp_seed_list)}
                        if not flag:
                            label_flag = True
                        for i in range(self.num_step):
                            hop_edge_list = hop_edges[i]
                            curr_weight_dict = self.forward_step(hop_edge_list, tp_weight_dict)
                            for local_ent in curr_weight_dict:
                                label_dist[index, i, local_ent] += curr_weight_dict[local_ent]
                            tp_weight_dict = curr_weight_dict
                    if not label_flag:
                        print(index, "can't label")
                        num_labelled_case -= 1
                        # print(line.strip())
                        label_valid[index, 0] = 0.0
                        for i in range(self.num_step):
                            ent_ct = {}
                            for seed_ent in tp_seed_list:
                                tp_key = "seed_%d" % (seed_ent)
                                tp_answer_list = tp_obj[tp_key][str(i)]["answer"]
                                for local_ent in tp_answer_list:
                                    ent_ct.setdefault(local_ent, 0.0)
                                    ent_ct[local_ent] += 1.0 / len(tp_answer_list)
                            # for more detailed labeling, we can deduce it from final aggregated results
                            for local_ent in ent_ct:
                                label_dist[index, i, local_ent] = ent_ct[local_ent] / num_seed
                                # dist sum 1.0
                else:
                    # bad case, we will label it simple, because we don't use it
                    label_valid[index, 0] = 0.0
                    for i in range(self.num_step):
                        ent_ct = {}
                        for seed_ent in tp_seed_list:
                            tp_key = "seed_%d" % (seed_ent)
                            tp_answer_list = tp_obj[tp_key][str(i)]["answer"]
                            for local_ent in tp_answer_list:
                                ent_ct.setdefault(local_ent, 0.0)
                                ent_ct[local_ent] += 1.0 / len(tp_answer_list)
                        # for more detailed labeling, we can deduce it from final aggregated results
                        for local_ent in ent_ct:
                            label_dist[index, i, local_ent] = ent_ct[local_ent] / num_seed
                            # dist sum 1.0
                index += 1
        assert index == self.num_data
        self.label_dist = label_dist
        self.label_valid = label_valid
        print('--------------------------------')
        print("{} cases among {} cases can be labelled".format(num_labelled_case, self.num_data))
        print('--------------------------------')

    def get_label(self):
        if not self.use_label or self.num_step == 1:
            return None, None
        label_valid = self.label_valid[self.sample_ids]
        # print(label_valid)
        labeL_dist_list = []
        for i in range(self.num_step):
            label_dist = self.label_dist[self.sample_ids, i]
            labeL_dist_list.append(label_dist)
        return labeL_dist_list, label_valid

    def deal_multi_seed(self, sample_ids):
        true_sample_ids = []
        tp_seed_list = self.seed_list[sample_ids]
        true_batch_id = []
        true_seed_ids = []
        # multi_seed_maks = []
        for i, seed_list in enumerate(tp_seed_list):
            true_batch_id.append([])
            for seed_ent in seed_list:
                true_batch_id[i].append(len(true_sample_ids))
                true_sample_ids.append(sample_ids[i])
                true_seed_ids.append(seed_ent)
                # if len(seed_list) > 1:
                #     multi_seed_maks.append(1.0)
                # else:
                #     multi_seed_maks.append(0.0)
        # print(tp_seed_list)
        # print(true_sample_ids, len(true_sample_ids))
        seed_dist = np.zeros((len(true_sample_ids), self.max_local_entity), dtype=float)
        for j, local_ent in enumerate(true_seed_ids):
            seed_dist[j, local_ent] = 1.0
            # single seed entity
        return true_batch_id, true_sample_ids, seed_dist