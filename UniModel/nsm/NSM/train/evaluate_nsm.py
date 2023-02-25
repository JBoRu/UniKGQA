from collections import Counter

import torch
import numpy as np
from tqdm import tqdm
import random
import math
import json
import os
from transformers import AutoTokenizer
VERY_SMALL_NUMBER = 1e-10


def cal_accuracy(pred, answer_dist):
    """
    pred: batch_size
    answer_dist: batch_size, max_local_entity
    """
    num_correct = 0.0
    num_answerable = 0.0
    for i, l in enumerate(pred):
        num_correct += (answer_dist[i, l] != 0)
    for dist in answer_dist:
        if np.sum(dist) != 0:
            num_answerable += 1
    return num_correct / len(pred), num_answerable / len(pred)


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
            return 1.0, 1.0, 1.0, 1.0, 0, retrieved  # precision, recall, f1, hits
        else:
            return 0.0, 1.0, 0.0, 1.0, 1, retrieved  # precision, recall, f1, hits
    else:
        hits = float(best_ans in answers)
        if len(retrieved) == 0:
            return 1.0, 0.0, 0.0, hits, 2, retrieved  # precision, recall, f1, hits
        else:
            p, r = correct / len(retrieved), correct / len(answers)
            f1 = 2.0 / (1.0 / p + 1.0 / r) if p != 0 and r != 0 else 0.0
            return p, r, f1, hits, 3, retrieved


class Evaluator_nsm:

    def __init__(self, args, student, entity2id, relation2id, num_relation):
        self.student = student
        self.args = args
        self.eps = args['eps']
        self.num_step = args['num_step']
        self.use_inverse_relation = args['use_inverse_relation']
        self.use_self_loop = args['use_self_loop']
        id2entity = {idx: entity for entity, idx in entity2id.items()}
        self.id2entity = id2entity
        id2relation = {idx: relation for relation, idx in relation2id.items()}
        num_rel_ori = len(relation2id)
        if self.use_inverse_relation:
            for i in range(len(id2relation)):
                id2relation[i + num_rel_ori] = id2relation[i] + "_rev"
        if self.use_self_loop:
            id2relation[len(id2relation)] = "self.loop.edge"
        assert len(id2relation) == num_relation, (len(id2relation), num_relation)
        self.num_kb_relation = num_relation
        self.id2relation = id2relation
        self.relation2id = {relation: idx for idx, relation in id2relation.items()}
        self.file_write = None
        if self.args["model_path"]:
            self.tokenizer = AutoTokenizer.from_pretrained(self.args["model_path"])
        else:
            self.tokenizer = None

    def write_info(self, valid_data, tp_list):
        question_list = valid_data.get_quest()
        num_step = self.num_step
        obj_list = []
        if tp_list is not None:
            # attn_list = [tp[1] for tp in tp_list]
            action_list = [tp[0] for tp in tp_list]
        for i in range(len(question_list)):
            obj_list.append({})
        for j in range(num_step):
            if tp_list is None:
                actions = None
            else:
                actions = action_list[j]
                actions = actions.cpu().numpy()
            # if attn_list is not None:
            #     attention = attn_list[j].cpu().numpy()
            for i in range(len(question_list)):
                tp_obj = obj_list[i]
                q = question_list[i]
                # real_index = self.true_batch_id[i][0]
                tp_obj['question'] = q
                tp_obj[j] = {}
                # print(actions)
                if tp_list is not None:
                    action = actions[i]
                    rel_action = self.id2relation[action]
                    tp_obj[j]['rel_action'] = rel_action
                    tp_obj[j]['action'] = str(action)
                    # if attn_list is not None:
                    #     attention_tp = attention[i]
                    #     tp_obj[j]['attention'] = attention_tp.tolist()
        return obj_list

    def evaluate(self, valid_data, test_batch_size=20, write_info=False):
        self.student.eval()
        self.count = 0
        eps = self.eps
        id2entity = self.id2entity
        eval_loss, eval_acc, eval_max_acc = [], [], []
        f1s, hits, precisions, recalls = [], [], [], []
        valid_data.reset_batches(is_sequential=True)
        num_epoch = math.ceil(valid_data.num_data / test_batch_size)
        if write_info and self.file_write is None:
            filename = os.path.join(self.args['checkpoint_dir'],
                                    "{}_test.info".format(self.args['experiment_name']))
            self.file_write = open(filename, "w")
        case_ct = {}
        max_local_entity = valid_data.max_local_entity
        ignore_prob = (1 - eps) / max_local_entity
        for iteration in tqdm(range(num_epoch)):
            batch = valid_data.get_batch(iteration, test_batch_size, fact_dropout=0.0, test=True)
            with torch.no_grad():
                loss, extras, pred_dist, tp_list = self.student(batch[:-1])
                pred = torch.max(pred_dist, dim=1)[1]
            local_entity, query_entities, kb_adj_mat, query_text, \
            seed_dist, true_batch_id, answer_dist, answer_list = batch
            # self.true_batch_id = true_batch_id
            if write_info:
                obj_list = self.write_info(valid_data, tp_list)
                # pred_sum = torch.sum(pred_dist, dim=1)
                # print(pred_sum)
            candidate_entities = torch.from_numpy(local_entity).type('torch.LongTensor')
            true_answers = torch.from_numpy(answer_dist).type('torch.FloatTensor')
            query_entities = torch.from_numpy(query_entities).type('torch.LongTensor')
            # acc, max_acc = cal_accuracy(pred, true_answers.cpu().numpy())
            eval_loss.append(loss.item())
            # eval_acc.append(acc)
            # eval_max_acc.append(max_acc)
            batch_size = pred_dist.size(0)
            batch_answers = answer_list
            batch_candidates = candidate_entities
            pad_ent_id = len(id2entity)
            for batch_id in range(batch_size):
                answers = batch_answers[batch_id]
                candidates = batch_candidates[batch_id, :].tolist()
                probs = pred_dist[batch_id, :].tolist()
                seed_entities = query_entities[batch_id, :].tolist()
                candidate2prob = []
                for c, p, s in zip(candidates, probs, seed_entities):
                    if s == 1.0:
                        # ignore seed entities
                        continue
                    if c == pad_ent_id:
                        continue
                    if p < ignore_prob:
                        continue
                    candidate2prob.append((c, p))
                precision, recall, f1, hit, case, retrived = f1_and_hits_new(answers, candidate2prob, eps)
                # if hit < 1:
                #     example_id = iteration*test_batch_size+batch_id
                #     print("Example id-%s:%.2f"%(example_id,hit))
                if write_info:
                    tp_obj = obj_list[batch_id]
                    tp_obj['precison'] = precision
                    tp_obj['recall'] = recall
                    tp_obj['f1'] = f1
                    tp_obj['hit'] = hit
                    tp_obj['cand'] = retrived
                    self.file_write.write(json.dumps(tp_obj) + "\n")
                case_ct.setdefault(case, 0)
                case_ct[case] += 1
                f1s.append(f1)
                hits.append(hit)
                precisions.append(precision)
                recalls.append(recall)
        print('evaluation.......')
        print('how many eval samples......', len(f1s))
        # print('avg_f1', np.mean(f1s))
        print('avg_hits', np.mean(hits))
        print('avg_precision', np.mean(precisions))
        print('avg_recall', np.mean(recalls))
        print('avg_f1', np.mean(f1s))
        print(case_ct)
        if write_info:
            self.file_write.close()
            self.file_write = None
        return np.mean(f1s), np.mean(hits)

    def evaluate_single_sample(self, valid_data, test_batch_size=20, num_steps=1, write_info=False):
        self.student.eval()
        self.count = 0
        input = self.create_one_sample_input(valid_data)
        id2relation = {idx: relation for relation, idx in self.relation2id.items()}
        with torch.no_grad():
            loss, extras, pred_dist, tp_list = self.student(input[:-2], new_id2rel=id2relation)
        return pred_dist[0], input[-2]

    def create_one_sample_input(self, valid_data):
        num_data = 1
        batches = np.arange(num_data)
        data = [valid_data]
        # 1._build_global2local_entity_maps
        global2local_entity_maps = [None] * num_data
        total_local_entity = 0.0
        next_id = 0
        max_local_entity = 0
        for sample in data:
            g2l = dict()
            # for entity_global_id in sample['entities']:
            #     if entity_global_id not in g2l:
            #         g2l[entity_global_id] = len(g2l)
            for entity_global_id in sample['subgraph']['entities']:
                if entity_global_id not in g2l:
                    g2l[entity_global_id] = len(g2l)

            global2local_entity_maps[next_id] = g2l
            total_local_entity += len(g2l)
            max_local_entity = max(max_local_entity, len(g2l))
            next_id += 1
        self.max_local_entity = max_local_entity
        self.global2local_entity_maps = global2local_entity_maps

        self.question_id = []
        self.candidate_entities = np.full((num_data, max_local_entity), len(g2l), dtype=int)
        self.kb_adj_mats = np.empty(num_data, dtype=object)
        self.q_adj_mats = np.empty(num_data, dtype=object)
        # self.kb_fact_rels = np.full((self.num_data, self.max_facts), self.num_kb_relation, dtype=int)
        self.query_entities = np.zeros((num_data, max_local_entity), dtype=float)
        self.seed_list = np.empty(num_data, dtype=object)
        self.seed_distribution = np.zeros((num_data, max_local_entity), dtype=float)
        # self.query_texts = np.full((self.num_data, self.max_query_word), len(self.word2id), dtype=int)
        self.answer_dists = np.zeros((num_data, max_local_entity), dtype=float)
        self.answer_lists = np.empty(num_data, dtype=object)

        # 2.preprare question
        max_count = 0
        for line in data:
            query = line["question"]
            query_ids = self.tokenizer.tokenize(query, add_special_tokens=True)
            max_count = max(max_count, len(query_ids))
        self.max_query_word = max_count
        self.query_ids = np.full((num_data, self.max_query_word), self.tokenizer.pad_token_id, dtype=int)
        self.attention_mask = np.full((num_data, self.max_query_word), 0, dtype=int)
        self.query_mask = np.full((num_data, self.max_query_word), 1, dtype=int)
        self.token_type_ids = np.full((num_data, self.max_query_word), 0, dtype=int)
        next_id = 0
        self.node2layer = []
        self.dep_parents = []
        self.dep_relations = []
        for sample in data:
            query = sample["question"]
            input = self.tokenizer(query, add_special_tokens=True, truncation=True, padding="max_length",
                                   max_length=self.max_query_word, return_tensors='np', return_special_tokens_mask=True)
            self.query_ids[next_id, :] = input["input_ids"]
            self.attention_mask[next_id, :] = input["attention_mask"]
            self.query_mask[next_id, :] = input["special_tokens_mask"]
            if hasattr(input, "token_type_ids"):
                self.token_type_ids[next_id, :] = input["token_type_ids"]
            next_id += 1

        # 3.prepare kb adj
        next_id = 0
        num_query_entity = {}
        for sample in data:
            self.question_id.append(sample["ID"])
            # get a list of local entities
            g2l = global2local_entity_maps[next_id]
            if len(g2l) == 0:
                print(next_id)
                continue
            # build connection between question and entities in it
            tp_set = set()
            seed_list = []
            for j, entity in enumerate(sample['entities']):
                # if entity['text'] not in self.entity2id:
                #     continue
                global_entity = entity  # self.entity2id[entity['text']]
                # global_entity = 0  # self.entity2id[entity['text']]
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

            valid_rels = set()
            for tpl in sample['subgraph']['tuples']:
                h, r, t = tpl
                if r in self.relation2id:
                    valid_rels.add(r)
            not_used_rels = set(self.relation2id.keys()) - valid_rels
            for tpl in sample['subgraph']['tuples']:
                h, r, t = tpl
                if r not in self.relation2id:
                    replaced_r = not_used_rels.pop()
                    self.relation2id[r] = self.relation2id.pop(replaced_r)

            for i, tpl in enumerate(sample['subgraph']['tuples']):
                sbj, rel, obj = tpl
                # head = g2l[self.entity2id[sbj['text']]]
                # rel = self.relation2id[rel['text']]
                # tail = g2l[self.entity2id[obj['text']]]
                head = g2l[sbj]
                rel = self.relation2id[rel]
                tail = g2l[obj]
                head_list.append(head)
                rel_list.append(rel)
                tail_list.append(tail)
                if self.use_inverse_relation:
                    head_list.append(tail)
                    rel_list.append(rel + len(self.id2relation))
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
                print(next_id, len(tp_set))
                exit(-1)

            self.kb_adj_mats[next_id] = (np.array(head_list, dtype=int),
                                         np.array(rel_list, dtype=int),
                                         np.array(tail_list, dtype=int))

            next_id += 1

        # 4.construct one batch with size of 1
        sample_ids = batches
        seed_dist = self.seed_distribution[sample_ids]
        q_input = {"input_ids": self.query_ids[sample_ids], "attention_mask": self.attention_mask[sample_ids],
                   "token_type_ids": self.token_type_ids[sample_ids], "query_mask": self.query_mask[sample_ids]}
        kb_adj_mat = (self._build_fact_mat(sample_ids, fact_dropout=0.0))
        true_batch_id = None

        return self.candidate_entities[sample_ids], \
               self.query_entities[sample_ids], \
               kb_adj_mat, \
               q_input, \
               seed_dist, \
               true_batch_id, \
               self.answer_dists[sample_ids], \
               self.global2local_entity_maps[0], \
               self.answer_lists[sample_ids], \

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
