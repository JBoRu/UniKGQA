import json
import numpy as np
import re
import os
from tqdm import tqdm
import torch
from collections import Counter
import logging

logger = logging.getLogger("NSM")


def load_dict(filename):
    word2id = dict()
    with open(filename, encoding='utf-8') as f_in:
        for line in f_in:
            word = line.strip()
            word2id[word] = len(word2id)
    return word2id


class BasicDataLoader(object):
    def __init__(self, config, word2id, relation2id, entity2id, data_type="train"):
        self._parse_args(config, word2id, relation2id, entity2id)
        self._load_file(config, data_type)

    def _load_file(self, config, data_type="train"):
        data_file = os.path.join(config['data_folder'], data_type + config["data_name"])
        logger.info('Loading data from %s' % (data_file))
        self.data = []
        self.dep = []
        skip_index = set()
        index = 0
        with open(data_file) as f_in:
            all_lines = f_in.readlines()
            if config['one_shot'] and data_type == 'train':
                sample_idx_path = config['sample_idx_path']
                with open(sample_idx_path, 'r') as f:
                    sample_idx = f.readlines()
                    sample_idx = [int(l.strip().strip("\n")) for l in sample_idx]
                    print("Load %d samples idx from %s" % (len(sample_idx), sample_idx_path))
                new_lines = []
                for idx in sample_idx:
                    new_lines.append(all_lines[idx])
                all_lines = new_lines
            for line in tqdm(all_lines):
                index += 1
                line = json.loads(line)
                if len(line['entities']) == 0:
                    skip_index.add(index)
                    continue
                self.data.append(line)
                question_dep = {'dep':
                                    [[w] for w in line['question'].replace('.', ' ').replace('|', ' ').split()]
                                }
                self.dep.append(question_dep)
                self.max_facts = max(self.max_facts, len(line['subgraph']['tuples']))
                # if index > 100:
                #     break
        logger.info("Skip index of [%s], which don't have any entities." % skip_index)
        logger.info('Max facts: [%d].' % self.max_facts)
        self.num_data = len(self.data)
        logger.info("%s data: %d." % (data_type, self.num_data))
        self.batches = np.arange(self.num_data)

    def _parse_args(self, config, word2id, relation2id, entity2id):
        self.use_inverse_relation = config['use_inverse_relation']
        self.use_self_loop = config['use_self_loop']
        self.num_step = config['num_step']
        self.max_local_entity = 0
        self.max_relevant_doc = 0
        self.max_facts = 0

        print('building word index ...')
        if word2id is not None:
            self.word2id = word2id
            self.id2word = {i: word for word, i in word2id.items()}
        self.relation2id = relation2id
        self.entity2id = entity2id
        self.id2entity = {i: entity for entity, i in entity2id.items()}

        if self.use_inverse_relation:
            self.num_kb_relation = 2 * len(relation2id)
        else:
            self.num_kb_relation = len(relation2id)
        if self.use_self_loop:
            self.num_kb_relation = self.num_kb_relation + 1
        print("Entity: {}, Relation in KB: {}, Relation in use: {} ".format(len(entity2id),
                                                                            len(self.relation2id),
                                                                            self.num_kb_relation))

    @staticmethod
    def tokenize_sent(question_text):
        question_text = question_text.strip().lower()
        question_text = re.sub('\'s', ' s', question_text)
        words = []
        for w_idx, w in enumerate(question_text.split(' ')):
            w = re.sub('^[^a-z0-9]|[^a-z0-9]$', '', w)
            if w == '':
                continue
            words += [w]
        return words

    def reset_batches(self, is_sequential=True):
        if is_sequential:
            self.batches = np.arange(self.num_data)
        else:
            self.batches = np.random.permutation(self.num_data)

    def _build_global2local_entity_maps(self):
        """Create a map from global entity id to local entity of each sample"""
        global2local_entity_maps = [None] * self.num_data
        total_local_entity = 0.0
        next_id = 0
        for sample in tqdm(self.data):
            g2l = dict()
            self._add_entity_to_map(self.entity2id, sample['entities'], g2l)
            if 'const_entities' in sample:
                self._add_entity_to_map(self.entity2id, sample['const_entities'], g2l)
            # construct a map from global entity id to local entity id
            self._add_entity_to_map(self.entity2id, sample['subgraph']['entities'], g2l)

            global2local_entity_maps[next_id] = g2l
            total_local_entity += len(g2l)
            self.max_local_entity = max(self.max_local_entity, len(g2l))
            next_id += 1
        logger.info('avg local entity: %.2f' % (total_local_entity / next_id))
        logger.info('max local entity: %d' % (self.max_local_entity))
        return global2local_entity_maps

    @staticmethod
    def _add_entity_to_map(entity2id, entities, g2l):
        for entity_global_id in entities:
            # entity_text = entity['text']
            # if entity_text not in entity2id:
            #     continue
            # entity_global_id = entity2id[entity_text]
            # print(entity_global_id)
            # print(entity_global_id)
            if entity_global_id not in g2l:
                g2l[entity_global_id] = len(g2l)
