import time
from collections import defaultdict
from copy import deepcopy

import logging
import numpy as np

from KnowledgeBase.KG_sparse_api import KnowledgeGraphSparse
from KnowledgeBase.KG_dense_api import KonwledgeGraphDense
import pickle

END_OF_HOP = "end.hop"
SEP = "[SEP]"

class KnowledgeGraph(object):
    def __init__(self, dense_kg_path, sparse_kg_path, ent2id_path, rel2id_path):
        # print("The dense KG instantiate via virtuoso from the %s" % (dense_kg_path))
        # self.virtuoso_kg = KonwledgeGraphDense.instantiate("virtuoso", dense_kg_path)
        # print("The dense KG instantiate over.")
        if sparse_kg_path is not None and ent2id_path is not None and rel2id_path is not None:
            triples_path, ent_type_path = sparse_kg_path
            print("The sparse KG instantiate via int triples from the %s" % (triples_path))
            self.sparse_kg = KnowledgeGraphSparse(triples_path=triples_path, ent_type_path=ent_type_path)
            self.ent2id = self._load_pickle_file(ent2id_path)
            self.id2ent = self._reverse_dict(self.ent2id)
            self.rel2id = self._load_pickle_file(rel2id_path)
            self.id2rel = self._reverse_dict(self.rel2id)
            print("The sparse KG instantiate over, all triples: %d, max head id: %d."%(self.sparse_kg.E, self.sparse_kg.max_head))

    @staticmethod
    def _load_pickle_file(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _reverse_dict(ori_dict):
        reversed_dict = {v: k for k, v in ori_dict.items()}
        return reversed_dict

    def get_out_relations(self, kg, h):
        """
        Get the out relations of the set of head entities.
        :param kg: The kg type used.
        :param h: the list of head entities or one head entity.
        :return: list of out relations.
        """
        if kg == "dense":
            relations_list = self.virtuoso_kg.get_out_relations(h)
        elif kg == "sparse":
            relations_set = set()
            h_id = self.ent2id[h]
            relations_id_ary = self.sparse_kg.get_out_relations(h_id)
            for rel_id in relations_id_ary:
                relations_set.add(self.id2rel[rel_id])
            relations_list = list(relations_set)
        else:
            raise NotImplementedError
        return relations_list

    def get_tails(self, kg, src, relation):
        """
        Get the tail entities from head entities and relations
        :param kg: The kg type used.
        :param src: the list of head entities or one head entity.
        :param relation: the list of relations or one relation corresponding to the src.
        :return: list of out tail entities.
        """
        if kg == "dense":
            tails_list = self.virtuoso_kg.get_tails(src, relation)
        elif kg == "sparse":
            tails_set = set()
            src_id = self.ent2id[src]
            rel_id = self.rel2id[relation]
            res_ary = self.sparse_kg.get_tails(src_id, rel_id)
            for tail in res_ary:
                tails_set.add(self.id2ent[tail])
            tails_list = list(tails_set)
        else:
            raise NotImplementedError
        return tails_list

    def deduce_subgraph_by_path(self, kg, src, path, no_hop_flag):
        """
        deduce the instantiate kg subgraph by one path
        :param src: head entity
        :param path: the path
        :param no_hop_flag: end of hop flag
        :return: list of nodes and list of triples in deduced subgraph
        """
        # 将子图实例化, 返回节点集合和边集合
        nodes, triples = set(), set()
        hop_nodes, next_hop_set = set(), set()
        hop_nodes.add(src)
        nodes.add(src)
        for relation in path:
            next_hop_set = set()
            if relation == no_hop_flag:
                continue
            for node in hop_nodes:
                for tail in self.get_tails(kg, node, relation):
                    next_hop_set.add(tail)
                    triples.add((node, relation, tail))
            hop_nodes = deepcopy(next_hop_set)
            nodes = nodes | hop_nodes
        return list(nodes), list(triples)

    def deduce_subgraph_by_all_paths(self, kg, srcs, paths):
        """
        deduce the instantiate kg subgraph by n*K paths corresponding to n topic entities.
        :param srcs: head entities
        :param paths: the paths
        :return: list of nodes and list of triples in deduced subgraph
        """
        # 将子图实例化, 返回节点集合和边集合
        assert len(srcs) == len(paths)  # n K paths corresponding to n topic entities
        total_nodes, total_triples = set(), set()
        for te, ps in zip(srcs, paths):  # [[[],score],]
            union_nodes, union_triples = set(), set()
            for p in ps:  # [[],score]
                # print(p[0])
                nodes4p, triples4p = self.deduce_subgraph_by_path(kg, te, p[0], END_OF_HOP)
                if len(nodes4p) > 100:
                    # print("One tree from one path num_ent: %d more than 100, we delete the path" % (len(nodes4p)))
                    continue
                union_nodes.update(nodes4p)
                union_triples.update(triples4p)
            total_nodes.update(union_nodes)
            total_triples.update(union_triples)
        print("Subgraph from K trees - num_ent: %d num_tri: %d" % (len(total_nodes), len(total_triples)))
        return (list(total_nodes), list(total_triples))

    def deduce_leaves_by_path(self, kg, src, path, no_hop_flag=END_OF_HOP):
        # 效率瓶颈，有待优化
        hop_nodes, next_hop_set = set(), set()
        hop_nodes.add(src)
        for relation in path:
            if relation == no_hop_flag:
                continue
            next_hop_set = set()
            for node in hop_nodes:
                for tail in self.get_tails(kg, node, relation):
                    next_hop_set.add(tail)
            hop_nodes = deepcopy(next_hop_set)
        return tuple(hop_nodes)

    def get_paths(self, kg, topic_ent, ans, hop, ans_type):
        """
        Search the paths from topic entity to answer entity within hop.
        :param topic_ent:
        :param ans:
        :param hop: specified hop
        :param ans_type:
        :return:
        """
        if kg == 'dense':
            return self.virtuoso_kg.get_paths(topic_ent, ans, hop, ans_type)
        elif kg == 'sparse':
            paths_list = []
            try:
                src_id = self.ent2id[topic_ent]
            except KeyError:
                print("Topic entity %s not in ent2id dict."%(topic_ent))
                return paths_list
            try:
                ans_id = self.ent2id[ans]
            except KeyError:
                print("Answer %s not in ent2id dict."%(ans))
                return paths_list
            try:
                rel_paths = self.sparse_kg.search_paths(src_id, ans_id, hop)
            except Exception as e:
                logging.exception(e)
                # print("Error in get paths between tpe and ans:\n", e)
                rel_paths = []
            for rp in rel_paths:
                path = [self.id2rel[r] for r in rp]
                paths_list.append(path)
            return paths_list

    def get_paths_kgc(self, kg, split, topic_ent, ans, hop, exclude_rel):
        """
        Search the paths from topic entity to answer entity within hop.
        :param topic_ent:
        :param ans:
        :param hop: specified hop
        :param ans_type:
        :return:
        """
        if kg == 'dense':
            pass
        elif kg == 'sparse':
            paths_list = []
            try:
                src_id = self.ent2id[topic_ent]
            except KeyError:
                print("Topic entity %s not in ent2id dict."%(topic_ent))
                return paths_list
            try:
                ans_id = self.ent2id[ans]
            except KeyError:
                print("Answer %s not in ent2id dict."%(ans))
                return paths_list
            try:
                rel_paths = self.sparse_kg.search_paths(src_id, ans_id, hop)
            except Exception as e:
                print("Exception in get paths between tpe and ans:\n", e)
                rel_paths = []
            for rp in rel_paths:
                path = [self.id2rel[r] for r in rp]
                if split == 'train' and path[0] == exclude_rel:
                    continue
                paths_list.append(path)
            return paths_list

    def get_paths_from_tpe_to_cpes(self, source, cpes, rels_path, task_name):
        rel_paths_list = []
        try:
            src_id = self.ent2id[source]
        except KeyError:
            print("Topic entity %s not in ent2id dict." % (source))
            return rel_paths_list

        cpes_id = []
        for cpe in cpes:
            try:
                e_id = self.ent2id[cpe]
                cpes_id.append(e_id)
            except:
                continue
        rels_path_id = [self.rel2id[r] for r in rels_path]

        # const rels before ?x:
        before_x_common_rel_path = rels_path_id[0:-1]
        if len(before_x_common_rel_path) > 0: # const rels must exist in two hop
            paths_y_t = self.sparse_kg.get_triples_along_relation_path(src_id, cpes_id, before_x_common_rel_path,
                                                                       next_hop_num=1)
            for pyt in paths_y_t:
                path_for_tpe_to_cpe = []
                path_for_tpe_to_cpe.extend(before_x_common_rel_path)
                path_for_tpe_to_cpe.extend(pyt)
                path_for_tpe_to_cpe = [self.id2rel[r] for r in path_for_tpe_to_cpe]
                if path_for_tpe_to_cpe not in rel_paths_list:
                    rel_paths_list.append(path_for_tpe_to_cpe)

        # const rels after ?x
        after_x_common_rel_path = rels_path_id
        paths_x_t = self.sparse_kg.get_triples_along_relation_path(src_id, cpes_id, after_x_common_rel_path,
                                                                   next_hop_num=1)
        for pxt in paths_x_t:
            path_for_tpe_to_cpe = []
            path_for_tpe_to_cpe.extend(after_x_common_rel_path)
            path_for_tpe_to_cpe.extend(pxt)
            path_for_tpe_to_cpe = [self.id2rel[r] for r in path_for_tpe_to_cpe]
            if path_for_tpe_to_cpe not in rel_paths_list:
                rel_paths_list.append(path_for_tpe_to_cpe)

        if task_name == "cwq":
            paths_x_c_t = self.sparse_kg.get_triples_along_relation_path(src_id, cpes_id, after_x_common_rel_path,
                                                                       next_hop_num=2)
            for pxct in paths_x_c_t:
                path_for_tpe_to_cpe = []
                path_for_tpe_to_cpe.extend(after_x_common_rel_path)
                path_for_tpe_to_cpe.extend(pxct)
                path_for_tpe_to_cpe = [self.id2rel[r] for r in path_for_tpe_to_cpe]
                if path_for_tpe_to_cpe not in rel_paths_list:
                    rel_paths_list.append(path_for_tpe_to_cpe)

        return rel_paths_list

    def get_limit_paths(self, topic_ent, ans, min_hop, max_hop):
        """
        Search the paths from topic entity to answer entity within hop.
        :param topic_ent:
        :param ans:
        :param hop: specified hop
        :param ans_type:
        :return:
        """

        paths_list = []
        try:
            src_id = self.ent2id[topic_ent]
        except KeyError:
            print("Topic entity %s not in ent2id dict."%(topic_ent))
            return paths_list
        try:
            ans_id = self.ent2id[ans]
        except KeyError:
            print("Answer %s not in ent2id dict."%(ans))
            return paths_list
        try:
            rel_paths = self.sparse_kg.get_all_path_with_length_limit(src_id, ans_id, min_hop, max_hop)
        except Exception as e:
            print("Exception in get paths between tpe and cpe:\n", e)
            rel_paths = []
        for rp in rel_paths:
            path = [self.id2rel[r] for r in rp]
            paths_list.append(path)
        return paths_list

    def get_tails_with_path(self, kg, path):
        if kg == "dense":
            return self.virtuoso_kg.get_tails_with_path(path)
        elif kg == "sparse":
            src = path[0]
            path = path[1:]
            src_id = self.ent2id[src]
            path_id = [self.rel2id[r] for r in path]
            tails = set()
            tails_ary = self.sparse_kg.deduce_node_leaves_by_path(src_id, path_id)
            for t in tails_ary:
                tails.add(self.id2ent[t])
            return list(tails)

    def get_relations_within_2hop(self, kg, entity):
        if kg == "dense":
            return self.virtuoso_kg.get_relations_within_2hop(entity)
        elif kg == "sparse":
            src_id = self.ent2id[entity]
            relations_set = set()
            relations_id_list = self.sparse_kg.get_relations_within_2hop(src_id)
            for rel_id in relations_id_list:
                rel = self.id2rel[rel_id]
                relations_set.add(rel)
            relations_list = list(relations_set)
            return relations_list

    def instaniate_and_execute(self, kg, path):
        if kg == "dense":
            return self.virtuoso_kg.instaniate_and_execute(path)
        elif kg == "sparse":
            src = path[0]
            path = path[1:]
            src_id = self.ent2id[src]
            path_id = [self.rel2id[r] for r in path]
            tails = set()
            tails_ary = self.sparse_kg.deduce_node_leaves_by_path(src_id, path_id)
            for t in tails_ary:
                tails.add(self.id2ent[t])
            return list(tails)

    def get_relations_within_specified_hop(self, kg, ent, max_hop):
        if kg == "dense":
            return self.virtuoso_kg.get_neibouring_relations(ent, max_hop)
        elif kg == "sparse":
            src_id = self.ent2id[ent]
            rels_per_hop = defaultdict(set)
            rels_id_per_hop = self.sparse_kg.get_relations_within_specified_hop(src_id, max_hop)
            for k, v in rels_id_per_hop.items():
                v_str = [self.id2rel[rid] for rid in v]
                rels_per_hop[k].update(v_str)
            return rels_per_hop

    def deduce_relation_leaves_by_path(self, kg, src, path_rels, hop_id):
        if kg == "dense":
            raise NotImplementedError
        elif kg == "sparse":
            rels_str = set()
            assert len(path_rels) == hop_id
            try:
                src_id = self.ent2id[src]
            except KeyError:
                print("%s not in ent2id!"%(src))
                return []
            path_rels_id = [self.rel2id[r] for r in path_rels]
            rels_ids = self.sparse_kg.deduce_relation_leaves_by_path(src_id, path_rels_id)
            for rid in rels_ids:
                rels_str.add(self.id2rel[rid])
            return list(rels_str)

    def deduce_relation_leaves_and_nodes_by_path(self, kg, src, cpes, path_rels, task_name):
        if kg == "dense":
            raise NotImplementedError
        elif kg == "sparse":
            try:
                src_id = self.ent2id[src]
            except Exception as e:
                print(e)
                print("Error in get entity id of %s from dict."%(src))
                return []
            cpes_id = []
            for cpe in cpes:
                try:
                    e_id = self.ent2id[cpe]
                    cpes_id.append(e_id)
                except Exception as e:
                    print(e)
                    print("Error in get entity id of %s from dict."%(cpe))
                    continue
            path_rels_id = [self.rel2id[r] for r in path_rels]

            triples = self.sparse_kg.deduce_next_triples_by_path_wo_reverse(src_id, path_rels_id)

            if len(triples) == 0:
                return [], []
            all_rels = np.unique(triples[:, 1])
            const_rels = []
            cpes_node = np.array(cpes_id, dtype=np.int32).reshape(-1, 1)
            rel_paths_e_t = self.sparse_kg.path_join(triples, cpes_node, only_keep_rel=True)
            for rp in rel_paths_e_t:
                for rel in rp:
                    const_rels.append(self.id2rel[rel])
            # all_tails = np.unique(triples[:, 2])
            # common = np.intersect1d(all_tails, cpes_id)
            # if common.size > 0:
            #     cpes_node = np.array(cpes_id, dtype=np.int32).reshape(-1, 1)
            #     rel_paths_e_t = self.sparse_kg.path_join(triples, cpes_node, only_keep_rel=True)
            #     for rp in rel_paths_e_t:
            #         for rel in rp:
            #             const_rels.append(self.id2rel[rel])
            inst_rels = [self.id2rel[r] for r in all_rels]
            return inst_rels, const_rels

    def deduce_subgraph_by_abstract_sg(self, kg, topic_entity, max_deduced_triples, subgraph_paths):
        if kg == "sparse":
            entities, triples = set(), set()
            topic_entity = [self.ent2id[topic_entity]]
            subgraph_paths = [{idx: {"chain": [self.rel2id[rel] for rel in rel_const["chain"]], "const": [self.rel2id[rel] for rel in rel_const["const"]]}
                              for idx, rel_const in path.items()} for path in subgraph_paths]
            sg = self.sparse_kg.deduce_subgraph_by_abstract_sg(topic_entity, max_deduced_triples, subgraph_paths)
            if sg is not None:
                entities_id, triples_id = sg
            else:
                return (entities, triples)
            for eid in entities_id:
                entities.add(self.id2ent[eid])
            for tri in triples_id:
                h, r, t = tri
                h = self.id2ent[h]
                t = self.id2ent[t]
                r = self.id2rel[r]
                triples.add((h, r, t))
            return (tuple(entities), tuple(triples))

    def get_subgraph_within_khop(self, qid, task_name, split, question, weak_gold_rels_per_hop, tpe, cpes, max_hop, tokenizer, model,
                                 topk, filter_score, not_filter_rels):
        try:
            tpe_id = self.ent2id[tpe]
        except Exception as e:
            logging.exception(e)
            print("Tpe %s not in ent2id dict"%(tpe))
            return defaultdict(set)
        cpes_id = set()
        for cpe in cpes:
            try:
                cpe_id = self.ent2id[cpe]
                cpes_id.add(cpe_id)
            except Exception as e:
                logging.exception(e)
                print("Qid_%s: cpe %s not in ent2id dict"%(cpe))
                continue
        abs_triples_per_hop, reserved_rel_and_score, all_const_rels, abs_cpes_id = self.sparse_kg.get_subgraph_within_khop(
                            qid, task_name, split, question, tpe_id, list(cpes_id), max_hop, weak_gold_rels_per_hop, self.id2rel, self.rel2id,
                            tokenizer, model, topk, filter_score, not_filter_rels)
        return abs_triples_per_hop, reserved_rel_and_score, all_const_rels, abs_cpes_id