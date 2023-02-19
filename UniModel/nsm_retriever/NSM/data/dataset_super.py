import json
import numpy as np
import re
import os
from tqdm import tqdm
import torch
from collections import Counter
from NSM.data.basic_dataset import BasicDataLoader
from transformers import AutoTokenizer
import logging

logger = logging.getLogger("NSM")


class SingleDataLoader(BasicDataLoader):
    def __init__(self, config, word2id, relation2id, entity2id, data_type="train"):
        super(SingleDataLoader, self).__init__(config, word2id, relation2id, entity2id, data_type)
        self._load_data()

    def _load_data(self):
        logger.info('Convert global to local entity index ...')
        self.global2local_entity_maps = self._build_global2local_entity_maps()

        self.question_id = []
        self.candidate_entities = np.full((self.num_data, self.max_local_entity), len(self.entity2id), dtype=int)
        self.kb_adj_mats = np.empty(self.num_data, dtype=object)
        self.q_adj_mats = np.empty(self.num_data, dtype=object)
        self.query_entities = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.seed_list = np.empty(self.num_data, dtype=object)
        self.seed_distribution = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.answer_dists = np.zeros((self.num_data, self.max_local_entity), dtype=float)
        self.answer_lists = np.empty(self.num_data, dtype=object)

        logger.info('Prepare question and data ...')
        self._prepare_dep()
        self._prepare_data()

    def _prepare_dep(self):
        max_count = 0
        for line in self.dep:
            word_list = line["dep"]
            max_count = max(max_count, len(word_list))
        self.max_query_word = max_count
        self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        next_id = 0
        for sample in tqdm(self.dep):
            tp_dep = sample["dep"]
            tokens = [item[0] for item in tp_dep]
            for j, word in enumerate(tokens):
                if word in self.word2id:
                    self.query_texts[next_id, j] = self.word2id[word]
                else:
                    self.query_texts[next_id, j] = len(self.word2id)
            next_id += 1

    def _prepare_data(self):
        """
        global2local_entity_maps: a map from global entity id to local entity id
        adj_mats: a local adjacency matrix for each relation. relation 0 is reserved for self-connection.
        """
        next_id = 0
        num_query_entity = {}
        for sample in tqdm(self.data):
            self.question_id.append(sample["ID"] if 'ID' in sample else sample['id'])
            # get a list of local entities
            g2l = self.global2local_entity_maps[next_id]
            if len(g2l) == 0:
                logger.info("ID: %d doesn't have any mapping elements in g2l." % next_id)
                continue
            # build connection between question and entities in it
            tp_set = set()
            seed_list = []
            for j, entity in enumerate(sample['entities']):
                global_entity = entity
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

            # relations in local KB
            head_list = []
            rel_list = []
            tail_list = []
            for i, tpl in enumerate(sample['subgraph']['tuples']):
                sbj, rel, obj = tpl
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
                logger.info("ID: %d doesn't have any seed entity. It has %d topic entities." %
                            (next_id, len(tp_set)))
                exit(-1)

            # construct distribution for answers
            answer_list = []
            for answer in sample['answers']:
                keyword = 'kb_id'
                answer_ent = answer[keyword]
                answer_list.append(answer_ent)
                if answer_ent in g2l:
                    self.answer_dists[next_id, g2l[answer_ent]] = 1.0
                # else:
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

    def get_batch(self, iteration, batch_size, fact_dropout, q_type=None, test=False):
        start = batch_size * iteration
        end = min(batch_size * (iteration + 1), self.num_data)
        sample_ids = self.batches[start: end]
        self.sample_ids = sample_ids
        true_batch_id = None
        seed_dist = self.seed_distribution[sample_ids]
        q_input = self.query_texts[sample_ids]
        if test:
            return self.candidate_entities[sample_ids], \
                   self.query_entities[sample_ids], \
                   (self._build_fact_mat(sample_ids, fact_dropout=fact_dropout)), \
                   q_input, \
                   seed_dist, \
                   true_batch_id, \
                   self.answer_dists[sample_ids], \
                   self.answer_lists[sample_ids]
        else:
            return self.candidate_entities[sample_ids], \
                   self.query_entities[sample_ids], \
                   (self._build_fact_mat(sample_ids, fact_dropout=fact_dropout)), \
                   q_input, \
                   seed_dist, \
                   true_batch_id, \
                   self.answer_dists[sample_ids]

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
        weight_list = [1.0 / head_count[head] for head in batch_heads]
        return batch_heads, batch_rels, batch_tails, batch_ids, fact_ids, weight_list


class SingleDataLoaderForPLM(BasicDataLoader):
    def __init__(self, config, word2id, relation2id, entity2id, data_type="train"):
        super(SingleDataLoaderForPLM, self).__init__(config, word2id, relation2id, entity2id, data_type)
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
        self._load_data()

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

        logger.info('Prepare question and data ...')
        self._prepare_dep()
        self._prepare_data()

    def _prepare_dep(self):
        max_count = 0
        for line in self.data:
            query = line["question"]
            query_ids = self.tokenizer.tokenize(query, add_special_tokens=True)
            max_count = max(max_count, len(query_ids))
        logger.info("Max length of query is %d" % (max_count))
        self.max_query_word = max_count
        self.query_ids = np.full((self.num_data, self.max_query_word), self.tokenizer.pad_token_id, dtype=int)
        self.attention_mask = np.full((self.num_data, self.max_query_word), 0, dtype=int)
        self.query_mask = np.full((self.num_data, self.max_query_word), 1, dtype=int)
        self.token_type_ids = np.full((self.num_data, self.max_query_word), 0, dtype=int)
        next_id = 0
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
            self.question_id.append(sample["ID"])
            # get a list of local entities
            g2l = self.global2local_entity_maps[next_id]
            if len(g2l) == 0:
                logger.info(next_id)
                continue
            # build connection between question and entities in it
            tp_set = set()
            seed_list = []
            for j, entity in enumerate(sample['entities']):
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

            # relations in local KB
            head_list = []
            rel_list = []
            tail_list = []
            for i, tpl in enumerate(sample['subgraph']['tuples']):
                sbj, rel, obj = tpl
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
        q_input = {"input_ids": self.query_ids[sample_ids], "attention_mask": self.attention_mask[sample_ids],
                   "token_type_ids": self.token_type_ids[sample_ids], "query_mask": self.query_mask[sample_ids]}
        kb_adj_mat = (self._build_fact_mat(sample_ids, fact_dropout=fact_dropout))
        if test:
            return self.candidate_entities[sample_ids], \
                   self.query_entities[sample_ids], \
                   kb_adj_mat, \
                   q_input, \
                   seed_dist, \
                   true_batch_id, \
                   self.answer_dists[sample_ids], \
                   self.answer_lists[sample_ids]
        else:
            return self.candidate_entities[sample_ids], \
                   self.query_entities[sample_ids], \
                   kb_adj_mat, \
                   q_input, \
                   seed_dist, \
                   true_batch_id, \
                   self.answer_dists[sample_ids]

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
