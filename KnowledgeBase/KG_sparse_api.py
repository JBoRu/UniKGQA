import math
import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
VERY_LARGT_NUM = 10**8
PATH_CUTOFF = 10**6
NODE_CUTOFF = 10**4


class KnowledgeGraphSparse(object):
    def __init__(self, triples_path: str, ent_type_path: str):
        self.triple = self._load_npy_file(triples_path)
        self.ent_type = self._load_npy_file(ent_type_path)
        self.bin_map = np.zeros_like(self.ent_type, dtype=np.int32)
        self.E = self.triple.shape[0]
        self.head2fact = csr_matrix(
            (np.ones(self.E), (self.triple[:, 0], np.arange(self.E)))).astype('bool')
        self.rel2fact = csr_matrix(
            (np.ones(self.E), (self.triple[:, 1], np.arange(self.E)))).astype('bool')
        self.tail2fact = csr_matrix(
            (np.ones(self.E), (self.triple[:, 2], np.arange(self.E)))).astype('bool')
        self.max_head = max(self.triple[:, 0])
        self.max_tail = max(self.triple[:, 2])

    @staticmethod
    def _load_npy_file(filename):
        return np.load(filename)

    @staticmethod
    def path_join(lhs, rhs, only_keep_rel=False, path_cutoff=PATH_CUTOFF):
        lhs_length = lhs.shape[1]
        rhs_length = rhs.shape[1]
        df_lhs = pd.DataFrame(lhs, columns=[str(i) for i in range(lhs_length)])
        df_rhs = pd.DataFrame(rhs, columns=[str(i) for i in range(lhs_length-1, lhs_length+rhs_length-1)])
        paths = pd.merge(df_lhs, df_rhs, on=str(lhs_length-1)) # consume time
        if only_keep_rel:
            rel_ids = [str(i) for i in range(1, lhs_length+rhs_length-1, 2)]
            rel_paths = paths.loc[:, rel_ids]
            rel_paths.drop_duplicates(inplace=True)
            paths = rel_paths.to_numpy().tolist()
        else:
            # paths.drop_duplicates(inplace=True)
            paths = paths.to_numpy()
        return paths[:path_cutoff]

    def _fetch_forward_triple(self, seed_set):
        seed_set = np.clip(seed_set, a_min=0, a_max=self.max_head)
        indices = self.head2fact[seed_set].indices
        return self.triple[indices]

    def _fetch_backward_triple(self, seed_set):
        seed_set = np.clip(seed_set, a_min=0, a_max=self.max_tail)
        indices = self.tail2fact[seed_set].indices
        return self.triple[indices]

    def get_out_relations(self, head_set):
        triples = self._fetch_forward_triple(head_set)
        return np.unique(triples[:, 1])

    def get_out_rt(self, head_set, cpes, answers):
        # start = time.time()
        triples = self._fetch_forward_triple(head_set)
        all_rels = set(triples[:, 1])
        # rel_2_tails_dict = defaultdict(set)
        # if time.time()-start > 1:
        #     print("55-57 consumes %.2f s"%(time.time()-start))
        # start = time.time()
        # for rel in all_rels:
        #     rel_indices = (triples[:, 1] == rel)
        #     tmp_triples = triples[rel_indices]
        #     tails = tmp_triples[:, 2]
        #     rel_2_tails_dict[rel] = set(tails)
        # if time.time()-start > 1:
        #     print("61-65 consumes %.2f s"%(time.time()-start))
        # return rel_2_tails_dict
        return triples, all_rels

    def get_tails(self, head_set, relation):
        triples = self._fetch_forward_triple(head_set)
        rel_indices = (triples[:, 1] == relation)
        triples = triples[rel_indices]
        if len(triples) != 0:
            return np.unique(triples[:, 2])
        else:
            return np.array([])

    def get_triples_along_relation(self, head_set, relation):
        triples = self._fetch_forward_triple(head_set)
        rel_indices = (triples[:, 1] == relation)
        triples = triples[rel_indices]
        return triples

    def get_triples_along_relation_path(self, head_set, tail_set, rel_path, next_hop_num):
        tgt_node = np.array(tail_set, dtype=np.int32).reshape(-1, 1)
        tail_set = np.array(tail_set, dtype=np.int32).reshape(-1)
        head_set = np.array(head_set, dtype=np.int32).reshape(-1)
        deduced_nodes = self.deduce_node_leaves_by_path(head_set, rel_path) # consume time
        if len(deduced_nodes) == 1 and head_set in deduced_nodes:
            return []
        else:
            forw_one_tris = self._fetch_forward_triple(deduced_nodes)
            forw_one_nodes = np.unique(forw_one_tris[:, 2]) # consume time

        if next_hop_num == 1:
            if np.intersect1d(forw_one_nodes, tail_set).size > 0:
                path_x_t = self.path_join(forw_one_tris, tgt_node, only_keep_rel=True)  # consume time
                return path_x_t
            else:
                return []
        elif next_hop_num == 2:
            back_one_tris = self._fetch_backward_triple(tail_set)
            back_one_nodes = np.unique(back_one_tris[:, 0])
            if np.intersect1d(forw_one_nodes, back_one_nodes).size > 0:
                path_x_c_t = self.path_join(forw_one_tris, back_one_tris, only_keep_rel=True) # consume time
                return path_x_c_t
            else:
                return []
        else:
            return []

    def get_tail_indices(self, head_set, relation):
        # triple = self._fetch_forward_triple(head_set)
        # rel_indices = (triple[:, 1] == relation)
        node_indices = self.head2fact[head_set]
        rel_indices = self.rel2fact[relation]
        return np.intersect1d(node_indices, rel_indices, assume_unique=True)

    def filter_cvt_nodes(self, seed_ary, CVT_TYPE=3):
        seed_type = self.ent_type[seed_ary]
        return seed_ary[seed_type == CVT_TYPE]

    def get_shortest_path_length(self, src_set, tgt_set):
        '''计算从src_set到tgt_set的最短路的长度'''
        # One Hop forward and backward
        forward_triple_one = self._fetch_forward_triple(src_set)
        backward_triple_one = self._fetch_backward_triple(tgt_set)
        forward_node_one = np.unique(forward_triple_one[:, 2])
        backward_node_one = np.unique(backward_triple_one[:, 0])
        if np.intersect1d(forward_node_one, tgt_set).size > 0:
            return 1
        if np.intersect1d(src_set, backward_node_one).size > 0:
            return 1
        if np.intersect1d(forward_node_one, backward_node_one).size > 0:
            return 2

        # Two Hop forward and backward
        forward_cvt_node = self.filter_cvt_nodes(forward_node_one)
        forward_triple_two = self._fetch_forward_triple(forward_cvt_node)
        forward_node_two = np.unique(forward_triple_two[:, 2])
        backward_cvt_node = self.filter_cvt_nodes(backward_node_one)
        backward_triple_two = self._fetch_backward_triple(backward_cvt_node)
        backward_node_two = np.unique(backward_triple_two[:, 0])

        if np.intersect1d(forward_node_two, tgt_set).size > 0:
            return 2
        if np.intersect1d(src_set, backward_node_two).size > 0:
            return 2
        if np.intersect1d(forward_node_two, backward_node_one).size > 0:
            return 3
        if np.intersect1d(forward_node_one, backward_node_two).size > 0:
            return 3
        if np.intersect1d(forward_node_two, backward_node_two).size > 0:
            return 4
        return -1

    def get_all_path_with_length_limit_ori(self, src_set, tgt_set, limit):
        assert 1 <= limit <= 4
        path_list = []
        forward_triple_one = self._fetch_forward_triple(src_set)
        backward_triple_one = self._fetch_backward_triple(tgt_set)
        forward_node_one = np.unique(forward_triple_one[:, 2])
        backward_node_one = np.unique(backward_triple_one[:, 0])
        forward_cvt_node = self.filter_cvt_nodes(forward_node_one)
        forward_triple_two = self._fetch_forward_triple(forward_cvt_node)
        forward_node_two = np.unique(forward_triple_two[:, 2])
        backward_cvt_node = self.filter_cvt_nodes(backward_node_one)
        backward_triple_two = self._fetch_backward_triple(backward_cvt_node)
        backward_node_two = np.unique(backward_triple_two[:, 0])

        src_node = np.array(src_set, dtype=np.int32).reshape(-1, 1)
        tgt_node = np.array(tgt_set, dtype=np.int32).reshape(-1, 1)
        if limit >= 1:
            local_path = []
            if np.intersect1d(forward_node_one, tgt_set).size > 0:
                path_s_t = self.path_join(forward_triple_one, tgt_node)
                path_list.append(
                    {'length': 1, 'path': path_s_t})
        if limit >= 2:
            if np.intersect1d(forward_node_one, backward_node_one).size > 0:
                path_s_m_t = self.path_join(
                    forward_triple_one, backward_triple_one)
                path_list.append(
                    {'length': 2, 'path': path_s_m_t})
        if limit >= 3:
            local_path = []
            if np.intersect1d(forward_node_two, backward_node_one).size > 0:
                common = np.intersect1d(forward_node_two, backward_node_one)
                back_for_common_triple = self._fetch_backward_triple(common)
                path_s_c_e = self.path_join(
                    forward_triple_one, back_for_common_triple)
                path_s_c_e_t = self.path_join(path_s_c_e, backward_triple_one)
                local_path.append(path_s_c_e_t)
            if np.intersect1d(forward_node_one, backward_node_two).size > 0:
                common = np.intersect1d(forward_node_one, backward_node_two)
                forw_for_common_triple = self._fetch_forward_triple(common)
                path_e_c_t = self.path_join(
                    forw_for_common_triple, backward_triple_one)
                path_s_e_c_t = self.path_join(forward_triple_one, path_e_c_t)
                local_path.append(path_s_e_c_t)
            if local_path:
                path_list.append(
                    {'length': 3, 'path': np.concatenate(local_path)})
        if limit >= 4:
            if np.intersect1d(forward_node_two, backward_node_two).size > 0:
                common = np.intersect1d(forward_node_two, backward_node_two)
                back_for_common_triple = self._fetch_backward_triple(common)
                forw_for_common_triple = self._fetch_forward_triple(common)
                path_e_c_t = self.path_join(
                    forw_for_common_triple, backward_triple_one)
                path_s_c_e = self.path_join(
                    forward_triple_one, back_for_common_triple)
                path_e_c_e_c_t = self.path_join(path_e_c_t, path_s_c_e)
                path_list.append({'length': 4, 'path': path_e_c_e_c_t})
        return path_list

    def get_all_path_with_length_limit(self, src_set, tgt_set, min_hop, max_hop):
        assert 2 <= min_hop <= max_hop <= 5
        src_node = np.array(src_set, dtype=np.int32).reshape(-1, 1)
        tgt_node = np.array(tgt_set, dtype=np.int32).reshape(-1, 1)
        path_list = []
        forward_triple_one = self._fetch_forward_triple(src_set)
        forward_node_one = np.unique(forward_triple_one[:, 2])
        backward_triple_one = self._fetch_backward_triple(tgt_set)
        backward_node_one = np.unique(backward_triple_one[:, 0])

        # forward_cvt_node = self.filter_cvt_nodes(forward_node_one)
        forward_triple_two = self._fetch_forward_triple(forward_node_one)
        forward_node_two = np.unique(forward_triple_two[:, 2])
        forward_node_two = np.setdiff1d(forward_node_two, src_node)

        # backward_cvt_node = self.filter_cvt_nodes(backward_node_one)
        backward_triple_two = self._fetch_backward_triple(backward_node_one)
        backward_node_two = np.unique(backward_triple_two[:, 0])
        backward_node_two = np.setdiff1d(backward_node_two, tgt_node)

        backward_triple_three = self._fetch_backward_triple(backward_node_two)
        backward_node_three = np.unique(backward_triple_three[:, 0])
        backward_node_three = np.setdiff1d(backward_node_three, backward_node_one)

        for limit in range(min_hop, max_hop+1):
            rel_paths = []
            if limit == 2:
                common = np.intersect1d(forward_node_one, backward_node_one)
                if common.size > 0:  # s e t
                    rel_paths = self.path_join(
                        forward_triple_one, backward_triple_one, only_keep_rel=True)
            elif limit == 3:
                common = np.intersect1d(forward_node_two, backward_node_one)
                if common.size > 0:  # s e m t
                    back_for_common_triple = self._fetch_backward_triple(common)
                    path_s_e_m = self.path_join(
                        forward_triple_one, back_for_common_triple)
                    rel_paths = self.path_join(path_s_e_m, backward_triple_one, only_keep_rel=True)
            elif limit == 4:
                common = np.intersect1d(forward_node_two, backward_node_two)
                if common.size > 0: # s e m e t
                    back_for_common_triple = self._fetch_backward_triple(common)
                    forw_for_common_triple = self._fetch_forward_triple(common)
                    path_m_e_t = self.path_join(
                        forw_for_common_triple, backward_triple_one)
                    path_s_e_m = self.path_join(
                        forward_triple_one, back_for_common_triple)
                    rel_paths = self.path_join(path_s_e_m, path_m_e_t, only_keep_rel=True)
            elif limit == 5:
                common = np.intersect1d(forward_node_two, backward_node_three)
                if common.size > 0:  # s e m e e t
                    back_for_common_triple = self._fetch_backward_triple(common) # e m
                    forw_for_common_triple = self._fetch_forward_triple(common) # m e
                    paths_m_e_e = self.path_join(forw_for_common_triple, backward_triple_two)
                    paths_m_e_e_t = self.path_join(paths_m_e_e, backward_triple_one)
                    paths_s_e_m = self.path_join(forward_triple_one, back_for_common_triple)
                    rel_paths = self.path_join(paths_s_e_m, paths_m_e_e_t, only_keep_rel=True)
            else:
                print("Not implement limit length more than 5")
            if len(rel_paths) > 0:
                path_list.extend(rel_paths)

        return path_list

    def search_paths(self, src_set, tgt_set, hop):
        src_node = np.array(src_set, dtype=np.int32).reshape(-1, 1)
        tgt_node = np.array(tgt_set, dtype=np.int32).reshape(-1, 1)

        forward_triple_one = self._fetch_forward_triple(src_set)
        forward_node_one = np.unique(forward_triple_one[:, 2])
        if hop == 1:
            local_path = []
            if np.intersect1d(forward_node_one, tgt_node).size > 0:
                paths_s_t = self.path_join(forward_triple_one, tgt_node, only_keep_rel=True)
                local_path.extend(paths_s_t)
            return local_path

        backward_triple_one = self._fetch_backward_triple(tgt_set)
        backward_node_one = np.unique(backward_triple_one[:, 0])
        if hop == 2:
            local_path = []
            if np.intersect1d(forward_node_one, backward_node_one).size > 0:
                paths_s_m_t = self.path_join(forward_triple_one, backward_triple_one, only_keep_rel=True)
                local_path.extend(paths_s_m_t)
            return local_path

        # forward_cvt_node = self.filter_cvt_nodes(forward_node_one)
        # forward_triple_two = self._fetch_forward_triple(forward_cvt_node)
        # forward_node_two = np.unique(forward_triple_two[:, 2])
        # backward_cvt_node = self.filter_cvt_nodes(backward_node_one)
        # backward_triple_two = self._fetch_backward_triple(backward_cvt_node)
        # backward_node_two = np.unique(backward_triple_two[:, 0])

        forward_triple_two = self._fetch_forward_triple(forward_node_one)
        forward_node_two = np.unique(forward_triple_two[:, 2])
        forward_node_two = np.setdiff1d(forward_node_two, src_node) # avoid reverse relation

        backward_triple_two = self._fetch_backward_triple(backward_node_one)
        backward_node_two = np.unique(backward_triple_two[:, 0])
        backward_node_two = np.setdiff1d(backward_node_two, tgt_node) # avoid reverse relation

        if hop == 3:
            local_path = []
            common = np.intersect1d(forward_node_two, backward_node_one)  # m
            if common.size > 0: # s e m t
                back_for_common_triple = self._fetch_backward_triple(common) # e m
                paths_s_e_m = self.path_join(forward_triple_one, back_for_common_triple)
                paths_s_e_m_t = self.path_join(paths_s_e_m, backward_triple_one, only_keep_rel=True)
                local_path.extend(paths_s_e_m_t)
            return local_path

        if hop == 4:
            local_path = []
            common = np.intersect1d(forward_node_two, backward_node_two)  # m
            if common.size > 0: # s e m e t
                back_for_common_triple = self._fetch_backward_triple(common) # e m
                forw_for_common_triple = self._fetch_forward_triple(common) # m e
                paths_m_e_t = self.path_join(forw_for_common_triple, backward_triple_one)
                paths_s_e_m = self.path_join(forward_triple_one, back_for_common_triple)
                paths_s_e_m_e_t = self.path_join(paths_s_e_m, paths_m_e_t, only_keep_rel=True)
                local_path.extend(paths_s_e_m_e_t)
            return local_path

        forward_triple_three = self._fetch_forward_triple(forward_node_two)
        forward_node_three = np.unique(forward_triple_three[:, 2])
        forward_node_three = np.setdiff1d(forward_node_three, forward_node_one)

        if hop == 5:
            local_path = []
            common = np.intersect1d(forward_node_three, backward_node_two)  # m
            if common.size > 0: # s e e m e t
                back_for_common_triple = self._fetch_backward_triple(common) # e m
                forw_for_common_triple = self._fetch_forward_triple(common) # m e
                paths_m_e_t = self.path_join(forw_for_common_triple, backward_triple_one)
                paths_e_e_m = self.path_join(forward_triple_two, back_for_common_triple)
                paths_e_e_m_e_t = self.path_join(paths_e_e_m, paths_m_e_t)
                paths_s_e_e_m_e_t = self.path_join(forward_triple_one, paths_e_e_m_e_t, only_keep_rel=True)
                local_path.extend(paths_s_e_e_m_e_t)
            return local_path

        if hop > 5:
            print("Not implement for more than 5 hop.")
            return []

    def deduce_subgraph_by_path(self, head, path):
        '''Return Edge from path'''
        # TODO: Implement this by performing small modifications to the method `deduce_node_leaves_by_path`
        seed_set = head
        all_triple = []
        for hop_idx, rel in path.items():
            chain_rels, const_rels = rel["chain"], rel["const"]
            for rel in const_rels:
                const_triple = self.get_triples_along_relation(seed_set, rel)
                all_triple.append(const_triple)
            if len(chain_rels) > 0:
                seed_triple = self.get_triples_along_relation(seed_set, chain_rels[0])
                all_triple.append(seed_triple)
                seed_set = np.unique(seed_triple[:, 2])
            else:
                seed_set = np.array([])
            if not seed_set.size:
                break
        if not all_triple:
            return np.array([head]), []
        all_triple = np.concatenate(all_triple)
        all_triple = np.unique(all_triple, axis=0)
        edges = all_triple
        nodes = np.unique(edges[:, [0, 2]])
        return tuple(nodes), tuple(edges)

    def deduce_node_leaves_by_path(self, src, path):
        seed_set = src
        for p in path:
            seed_set = self.get_tails(seed_set, p)
            if not seed_set.size:
                break
        return seed_set

    def deduce_node_leaves_by_path_wo_reverse(self, src, exclude_nodes, path):
        seed_set = src
        for p in path:
            seed_set = self.get_tails(seed_set, p)
            if not seed_set.size:
                break
        if len(path) == 0:
            seed_set = {seed_set} - exclude_nodes
            seed_set = np.array(list(seed_set))
        else:
            seed_set = np.setdiff1d(seed_set, exclude_nodes)
        return seed_set

    def deduce_relation_leaves_by_path(self, src, path):
        node_leaves = self.deduce_node_leaves_by_path(src, path)
        return self.get_out_relations(node_leaves)

    def deduce_relation_leaves_and_nodes_by_path(self, src, exclude_nodes, path):
        node_leaves = self.deduce_node_leaves_by_path_wo_reverse(src, exclude_nodes, path)
        triples = self._fetch_forward_triple(node_leaves)
        # all_rels = set(triples[:, 1])
        return triples
        # return self.get_out_rt(node_leaves, cpes, answers)

    def deduce_next_triples_by_path_wo_reverse(self, src, path):
        seed_set = src
        exclude_nodes = np.array([seed_set])
        previous_ndoes = np.array([])
        for i, p in enumerate(path):
            seed_set = self.get_tails(seed_set, p)
            if i > 0:
                seed_set = np.setdiff1d(seed_set, exclude_nodes)
                exclude_nodes = deepcopy(previous_ndoes)
            previous_ndoes = deepcopy(seed_set)
            if not seed_set.size:
                break
        if type(seed_set) is int or len(seed_set) > 0:
            triples = self._fetch_forward_triple(seed_set)
            return triples
        else:
            return []

    def get_relations_within_specified_hop(self, src, max_hop):
        rels_per_hop = defaultdict(set)
        for hop_idx in range(max_hop):
            try:
                hop_triples = self._fetch_forward_triple(src)
            except Exception as e:
                print(e)
                break
            hop_rels = np.unique(hop_triples[:, 1]).tolist()
            rels_per_hop[hop_idx].update(hop_rels)
            src = np.unique(hop_triples[:, 2])
            # src = np.clip(src, a_min=0, a_max=self.max_head)
            src = np.unique(src)
            if not src.size:
                break
        return rels_per_hop

    def deduce_subgraph_by_abstract_sg(self, topic_entities, max_deduced_triples, subgraph_paths):
        inst_triples = set()
        inst_ents = set()
        for path in subgraph_paths:
            # {0:{"chain":[], "const":[]}}
            real_ent, triples = self.deduce_subgraph_by_path(topic_entities[0], path)
            if len(triples) < max_deduced_triples:
                for tri in triples:
                    h, r, t = tri
                    inst_triples.add(tuple(tri))
                    inst_ents.add(h)
                    inst_ents.add(t)
        return (tuple(inst_ents), tuple(inst_triples))

    def get_subgraph_within_khop(self, qid, task_name, split, question, tpe, cpes, max_hop, weak_gold_rels_per_hop, id2rel, rel2id,
                                 tokenizer, model, topk, filter_score, not_filter_rels):
        triples_per_hop = defaultdict(list)
        seed_set = tpe
        exclude_nodes = np.array([])
        previous_ndoes = np.array([seed_set])
        abs_cpes_id = set()
        all_const_rels = set()
        abs_tpe_id = 0
        placeholder_id = 1
        last_abs_id = [(tpe, abs_tpe_id)]
        self.all_rel_and_score = defaultdict()
        self.reserved_rel_and_score = defaultdict()
        for i in range(max_hop):
            cur_abs_id = []
            cur_ins_tails = []
            for ins_abs in last_abs_id:
                ins_seed, pa_id = ins_abs
                triples = self._fetch_forward_triple(ins_seed)
                if len(triples) == 0:
                    continue

                if exclude_nodes.size > 0:
                    exclude_triples = self.path_join(triples, exclude_nodes.reshape((-1, 1)))
                    if len(exclude_triples) != 0:
                        # exclude_rels = set(exclude_triples[:, 1])
                        seed_set = np.setdiff1d(triples[:, 2], exclude_nodes)
                    else:
                        # exclude_rels = set()
                        seed_set = np.unique(triples[:, 2])
                    cur_ins_tails.append(seed_set)
                else:
                    # exclude_rels = set()
                    seed_set = np.unique(triples[:, 2])
                    cur_ins_tails.append(seed_set)
                exclude_rels = set()

                const_rels = self.get_const_rels(triples, cpes, exclude_rels)
                const_rels_str = [id2rel[rel] for rel in const_rels]
                all_const_rels.update(const_rels_str)
                candidate_rels = set(triples[:, 1].tolist()) - exclude_rels - const_rels
                candidate_rels_str = [id2rel[rel] for rel in candidate_rels]

                if len(candidate_rels_str) == 0:
                    continue

                if not_filter_rels:
                    filtered_rels_str = candidate_rels_str
                else:
                    filtered_rels_str = self.get_filtered_rels(question, candidate_rels_str, tokenizer, model,
                                                                   topk, filter_score)

                if split != 'test':  # This is a trick only used for train/dev set.
                    weak_label_rels = weak_gold_rels_per_hop[i] & set(candidate_rels_str)
                    for rel in weak_label_rels:
                        if rel not in filtered_rels_str:
                            filtered_rels_str.append(rel)

                exclude_nodes_set = set(exclude_nodes)
                for rel_str in filtered_rels_str:
                    rel_indices = (triples[:, 1] == rel2id[rel_str])
                    triples_for_rel = triples[rel_indices]
                    tails = set(triples_for_rel[:, 2]) - exclude_nodes_set
                    triples_per_hop[i].append((pa_id, rel_str, placeholder_id))
                    cur_abs_id.append((np.array(list(tails)), placeholder_id))
                    placeholder_id = placeholder_id + 1
                for rel_str in const_rels_str:
                    triples_per_hop[i].append((pa_id, rel_str, placeholder_id))
                    abs_cpes_id.add(placeholder_id)
                    placeholder_id = placeholder_id + 1

            exclude_nodes = deepcopy(previous_ndoes)
            if len(cur_ins_tails) > 0:
                previous_ndoes = np.concatenate(cur_ins_tails)
            else:
                break
            last_abs_id = deepcopy(cur_abs_id)
        return triples_per_hop, self.reserved_rel_and_score, all_const_rels, abs_cpes_id

    def get_const_rels(self, triples, cpes_id, exclude_rels):
        const_rels = set()
        cpes_node = np.array(cpes_id, dtype=np.int32).reshape(-1, 1)
        rel_paths_e_t = self.path_join(triples, cpes_node, only_keep_rel=True)
        for rp in rel_paths_e_t:
            for rel in rp:
                if rel not in exclude_rels:
                    const_rels.add(rel)
        return const_rels

    def get_filtered_rels(self, question, cur_relations, tokenizer, model, topk, filter_score):
        scored_rel_list, filtered_rel_scored_list = self.score_relations(question, cur_relations, tokenizer, model, filter_score)
        # 过滤关系和得分
        ordered_rels_scored = sorted(filtered_rel_scored_list, key=lambda x: x[1], reverse=True)
        # 过滤方法为topk和最少路径filter_method == "topk":
        reserved_rels = ordered_rels_scored[:topk]
        reserved_rels = [rel_score[0] for rel_score in reserved_rels]
        return reserved_rels

    def score_relations(self, query, cur_relations, tokenizer, model, min_filter_score: float = 0.0):
        import torch
        cur_relations = list(cur_relations)
        all_relation_list = cur_relations
        query_lined_list = [query]
        q_emb = self.get_texts_embeddings(query_lined_list, tokenizer, model).unsqueeze(1)  # (1,1,hid)
        target_emb = self.get_texts_embeddings(all_relation_list, tokenizer, model).unsqueeze(0)  # (1,bs,hid)
        sim_score = torch.cosine_similarity(q_emb, target_emb, dim=2)  # (1,bs)
        sim_score = sim_score.squeeze(dim=0)  # (bs)
        results = []
        filtered_results = []
        for idx, rel in enumerate(cur_relations):
            score = sim_score[idx]
            if score >= min_filter_score:
                results.append((rel, score))
                filtered_results.append((rel, score))
                self.reserved_rel_and_score[rel] = score
            else:
                results.append((rel, score))
            self.all_rel_and_score[rel] = score
        return results, filtered_results

    def get_texts_embeddings(self, texts, tokenizer, model):
        import torch
        max_bs = 300
        total = len(texts)
        steps = math.ceil(total / max_bs)
        all_embeddings = []
        for i in range(steps):
            texts_batch = texts[i * max_bs: (i + 1) * max_bs]
            inputs = tokenizer(texts_batch, padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
                all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0).to(model.device)  # (bs, hid)
        return all_embeddings
