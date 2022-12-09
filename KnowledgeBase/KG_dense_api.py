'''Provide API of KG'''
from typing import Dict, List, Set, Tuple
import networkx as nx
import pickle
from copy import deepcopy

from KnowledgeBase.sparql_executor import *

# from construct_contrastive_training_data import END_OF_HOP
END_OF_HOP = "end.hop"
SEP = "[SEP]"


class KonwledgeGraphDense(object):
    def __init__(self,
                 G: nx.DiGraph,
                 head2relation: Dict[str, Tuple[str]],
                 head_relation_2_tail: Dict[str, Dict[str, List[str]]],
                 type: str
                 ):
        self.G = G
        self.head2relation = head2relation
        self.head_relation_2_tail = head_relation_2_tail
        self.type = type

    @classmethod
    def instantiate(cls, source, path):
        if source == "virtuoso":
            return KonwledgeGraphDense.load_with_virtuoso()
        elif source == "triples":
            return KonwledgeGraphDense.load_from_triples(path)
        elif source == "ckpt":
            return KonwledgeGraphDense.load_from_ckpt(path)
        else:
            raise NotImplementedError("Not implement this source type:%s" % (source))

    @classmethod
    def load_from_ckpt(cls, ckpt_path):
        with open(ckpt_path, 'rb') as f:
            data_dict = pickle.load(f)
        return KonwledgeGraphDense(data_dict['G'], data_dict['head2relation'], data_dict['head_relation_2_tail'],
                              type='ckpt')

    @classmethod
    def load_from_triples(cls, path):
        with open(path, "r") as f:
            all_triples = f.readlines()
        G = nx.DiGraph()
        head2relation = defaultdict(set)
        head_relation_2_tail = defaultdict(lambda: defaultdict(list))
        for hrt in all_triples:
            hrt = hrt.strip("\n").split("\t")
            if "common.topic.description" in hrt:
                continue
            if len(hrt) != 3:
                print("Load bad triple case: %s" % (hrt))
                continue
            h, r, t = hrt
            G.add_edge(h, t, keyword=r)
            head2relation[h].add(r)
            head_relation_2_tail[h][r].append(t)
        head2relation = {k: tuple(v) for k, v in head2relation.items()}
        head_relation_2_tail = dict(head_relation_2_tail)
        out_path = path + ".subgraph.ckpt"
        with open(out_path, 'wb') as f:
            data_dict = {'G': G, 'head2relation': head2relation, 'head_relation_2_tail': head_relation_2_tail}
            pickle.dump(data_dict, f)
            print("Save the loaded DiGraph to %s" % (out_path))
        return KonwledgeGraphDense(G, head2relation, head_relation_2_tail, type='ckpt')

    @classmethod
    def load_with_virtuoso(cls):
        # initialize a null KG class
        G = nx.DiGraph()
        head2relation = {}
        head_relation_2_tail = {}
        return KonwledgeGraphDense(G, head2relation, head_relation_2_tail, type='virtuoso')

    def dump_to_ckpt(self, ckpt_path):
        data_dict = dict(G=self.G, head2relation=self.head2relation,
                         head_relation_2_tail=self.head_relation_2_tail)
        with open(ckpt_path, 'wb') as f:
            pickle.dump(data_dict, f)

    def get_out_relations(self, head):
        res = set()
        if self.type == "virtuoso":
            res_tmp = get_out_relations(head)
        else:
            res_tmp = self.head2relation.get(head, tuple())
        res.update(res_tmp)
        return list(res)

    def get_tails(self, src, relation):
        tails_set = set()
        if self.type == "virtuoso":
            res = list(get_out_entities(src, relation))
        else:
            res = self.head_relation_2_tail.get(src, dict()).get(relation, list())
        tails_set.update(res)
        tails_list = list(tails_set)
        return tails_list

    def get_all_path(self, src, tgt, cutoff: int = 3):
        paths = nx.all_simple_edge_paths(self.G, src, tgt, cutoff=cutoff)
        path_all = []
        for path in paths:
            # path_str = [src]
            path_str = []
            for ht in path:
                h, t = ht
                r = self.G[h][t]["keyword"]
                path_str.append(r)
            path_all.append(path_str)
        return path_all

    def get_shorted_path_limit(self, src, tgt):
        return nx.shortest_path_length(self.G, src, tgt)

    def deduce_subgraph_by_path(self, src: str, path: List[str], no_hop_flag: str) -> Tuple[
        List[str], List[Tuple[str, str, str]]]:
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
                for tail in self.get_tail(node, relation):
                    next_hop_set.add(tail)
                    triples.add((node, relation, tail))
            hop_nodes = deepcopy(next_hop_set)
            nodes = nodes | hop_nodes
        return list(nodes), list(triples)

    def deduce_subgraph_by_all_paths(self, src, paths):
        # 将子图实例化, 返回节点集合和边集合
        assert len(src) == len(paths)  # n K paths corresponding to n topic entities
        total_nodes, total_triples = set(), set()
        for te, ps in zip(src, paths):  # [[[],score],]
            union_nodes, union_triples = set(), set()
            for p in ps:  # [[],score]
                # print(p[0])
                nodes4p, triples4p = self.deduce_subgraph_by_path(te, p[0], END_OF_HOP)
                if len(nodes4p) > 100:
                    # print("One tree from one path num_ent: %d more than 100, we delete the path" % (len(nodes4p)))
                    continue
                union_nodes.update(nodes4p)
                union_triples.update(triples4p)
            total_nodes.update(union_nodes)
            total_triples.update(union_triples)
        # print("Subgraph from K trees - num_ent: %d num_tri: %d" % (len(total_nodes), len(total_triples)))
        return (list(total_nodes), list(total_triples))

    def deduce_leaves_by_path(self, src: str, path: List[str], no_hop_flag: str = 'NoHop') -> Tuple[str]:
        # 效率瓶颈，有待优化
        hop_nodes, next_hop_set = set(), set()
        hop_nodes.add(src)
        for relation in path:
            if relation == no_hop_flag:
                continue
            next_hop_set = set()
            for node in hop_nodes:
                for tail in self.get_tail(node, relation):
                    next_hop_set.add(tail)
            hop_nodes = deepcopy(next_hop_set)
        return tuple(hop_nodes)

    def get_paths(self, topic_ent, ans, hop, ans_type):
        if self.type == 'virtuoso':
            return search_paths(topic_ent, ans, hop, ans_type)
        else:
            return self.get_all_path(topic_ent, ans, cutoff=hop)

    def get_tails_with_path(self, path):
        if self.type == 'virtuoso':
            return instaniate_and_execute(path)
        else:
            src = path[0]
            hop_nodes = [src]
            next_hop_nodes = []
            for rel in path[1:]:
                for n in hop_nodes:
                    tail = self.head_relation_2_tail.get(n, dict()).get(rel, list())
                    next_hop_nodes.extend(tail)
                hop_nodes = deepcopy(next_hop_nodes)
                next_hop_nodes = []
            return list(set(hop_nodes))

    def get_relations_within_2hop(self, entity):
        if self.type == 'virtuoso':
            return get_relations_within_2hop(entity)
        else:
            within_2_hop_rels = set()
            one_hop_rels = self.head2relation.get(entity, set())
            within_2_hop_rels.update(one_hop_rels)
            for rel in one_hop_rels:
                one_hop_tails = self.head_relation_2_tail.get(entity, dict()).get(rel, list())
                for tail in one_hop_tails:
                    two_hop_rels = self.head2relation.get(tail, set())
                    within_2_hop_rels.update(two_hop_rels)
            return within_2_hop_rels

    def instaniate_and_execute(self, path):
        if self.type == 'virtuoso':
            return instaniate_and_execute(path)
        else:
            src = path[0]
            ans = set()
            if len(path) == 2:
                one_hop_rel = path[1]
                tails = self.head_relation_2_tail.get(src, dict()).get(one_hop_rel, list())
                ans.update(tails)
            elif len(path) == 3:
                one_hop_rel = path[1]
                one_hop_tails = self.head_relation_2_tail.get(src, dict()).get(one_hop_rel, list())
                two_hop_rel = path[2]
                for tail in one_hop_tails:
                    two_hop_tails = self.head_relation_2_tail.get(tail, dict()).get(two_hop_rel, list())
                    ans.update(two_hop_tails)
            return ans

    def get_two_hop_subgraph(self, src, src_type):
        return get_subgraph_within_2hop(src, src_type)

    def deduce_subgraph_by_abstract_sg(self, topic_entities, max_deduced_triples, subgraph_paths):
        abstract_id2real_mid = defaultdict(set)
        abstract_id2real_mid[0].add(topic_entities[0])
        abstract_hrt2real_hrt = defaultdict(set)
        inst_triples = set()
        inst_ents = set()
        for path in subgraph_paths:
            assert path[0] == 0
            if path[-2] == "end.hop":
                path = path[0:-2]
            if len(path) < 3:
                continue
            real_ent, triples = creat_query_and_execute(topic_entities[0], path)
            all_triples = set()
            for k, v in triples.items():
                all_triples.update(v)
            if len(all_triples) < max_deduced_triples:
                for tri in all_triples:
                    h, r, t = tri
                    inst_triples.add(tri)
                    inst_ents.add(h)
                    inst_ents.add(t)
        #     for re_idx, ae_idx in enumerate(range(2, len(path), 2)):
        #         if path[ae_idx] not in abstract_id2real_mid:
        #             abstract_id2real_mid[path[ae_idx]].update(real_ent[re_idx])
        #         else:
        #             abstract_id2real_mid[path[ae_idx]] &= real_ent[re_idx]
        #         abs_hrt = (path[ae_idx-2], path[ae_idx-1], path[ae_idx])
        #         abstract_hrt2real_hrt[abs_hrt].update(triples[re_idx])
        #
        # inst_triples = set()
        # inst_ents = set()
        # for tri in subgraph_triples:
        #     h, r, t = tri
        #     real_hrts = abstract_hrt2real_hrt[tri]
        #     rhs, rts = abstract_id2real_mid[h], abstract_id2real_mid[t]
        #     for real_hrt in real_hrts:
        #         rh, r, rt = real_hrt
        #         if rh in rhs and rt in rts:
        #             inst_triples.add(real_hrt)
        #             inst_ents.add(rh)
        #             inst_ents.add(rt)

        return (tuple(inst_ents), tuple(inst_triples))

    def get_neibouring_relations(self, ent, max_hop):
        return get_relations_within_2hop(ent)