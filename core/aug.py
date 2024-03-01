"""
    https://github.com/Shen-Lab/GraphCL
"""

import copy
import random

import dgl
import dhg
import numpy
import torch


def aug_drop_node(graph, root, drop_percent=0.2):
    """
        drop non-root nodes
    """
    num = graph.number_of_nodes()  # number of nodes of one graph
    drop_num = int(num * drop_percent)  # number of drop nodes
    aug_graph = copy.deepcopy(graph)
    all_node_list = [i for i in range(num) if i != root]
    drop_node_list = random.sample(all_node_list, drop_num)
    aug_graph.remove_nodes(drop_node_list)
    aug_graph = add_self_loop_if_not_in(aug_graph)
    return aug_graph


def aug_drop_node_hypergraph(graph, hypergraph: dhg.Hypergraph, root, drop_percent=0.2):

    num = hypergraph.num_v  # number of nodes of one graph
    drop_num = int(num * drop_percent)  # number of drop nodes
    all_node_list = [i for i in range(num) if i != root]
    drop_node_list = random.sample(all_node_list, drop_num)



    # 普通图结构
    aug_graph = copy.deepcopy(graph)
    aug_graph.remove_nodes(drop_node_list)
    aug_graph = add_self_loop_if_not_in(aug_graph)

    aug_hyperedges = []
    # 对超边内的节点进行reindex
    for hyperedge in hypergraph.e[0]:
        aug_hyperedge = []
        for v in hyperedge:
            decrease = 0
            if v not in drop_node_list:
                for v_drop in drop_node_list:
                    if v > v_drop:
                        decrease += 1
                aug_hyperedge.append(v - decrease)

        if len(aug_hyperedge) > 0:
            aug_hyperedges.append(aug_hyperedge)

    aug_hypergraph = dhg.Hypergraph(num-drop_num, aug_hyperedges)
    return aug_graph, aug_hypergraph


# 增强根因节点在超边种的权重
# aiops2 gaia1
def aug_add_root_node_hypergraph(graph, hypergraph: dhg.Hypergraph, root, add_percent=2):
    # 普通图结构
    aug_graph = copy.deepcopy(graph)

    aug_hyperedges = []
    aug_hyperedge_types = []
    #
    for hyperedge in hypergraph.e[0]:
        aug_hyperedge = list(hyperedge).copy()
        if root in hyperedge:
            add_root = [root for i in range(add_percent)]
            aug_hyperedge.extend(add_root)
        aug_hyperedges.append(aug_hyperedge)

    aug_hypergraph = dhg.Hypergraph(hypergraph.num_v, aug_hyperedges, e_types=hypergraph.e[2])
    return aug_graph, aug_hypergraph


# 随机去除不带根因节点的超边
def aug_drop_hyperedge_without_root(graph, hypergraph: dhg.Hypergraph, root, drop_percent=0.2):
    # 普通图结构
    aug_graph = copy.deepcopy(graph)

    aug_hyperedges_ori = hypergraph.e[0].copy()
    aug_hyperedges_withoot_root_list = [i for i in range(len(aug_hyperedges_ori)) if root not in aug_hyperedges_ori[i]]
    drop_hyperedge_indexs = sorted(random.sample(aug_hyperedges_withoot_root_list, int(drop_percent * len(aug_hyperedges_withoot_root_list))))

    aug_hyperedges = []
    aug_hyperedge_types = []
    for i in range(len(aug_hyperedges_ori)):
        if i not in drop_hyperedge_indexs:
            aug_hyperedges.append(list(aug_hyperedges_ori[i]).copy())
            aug_hyperedge_types.append(hypergraph.e[2][i])

    aug_hypergraph = dhg.Hypergraph(hypergraph.num_v, aug_hyperedges, e_types=aug_hyperedge_types)
    return aug_graph, aug_hypergraph


def aug_drop_node_list(graph_list, labels, drop_percent):
    graph_num = len(graph_list)  # number of graphs
    aug_list = []
    for i in range(graph_num):
        aug_graph = aug_drop_node(graph_list[i], labels[i], drop_percent)
        aug_list.append(aug_graph)
    return aug_list


def aug_random_walk(graph, root, drop_percent=0.2):
    """
        random walk from root
    """
    rg = dgl.reverse(graph, copy_ndata=False, copy_edata=False)
    num_edge = rg.number_of_edges()  # number of edges of one graph
    retain_num = num_edge - int(num_edge * drop_percent)  # number of retain edges
    trace = dgl.sampling.random_walk(rg, [root], length=retain_num, return_eids=True)[1]
    edges = trace.flatten()
    subgraph = dgl.edge_subgraph(graph, edges, store_ids=False)
    subgraph = add_self_loop_if_not_in(subgraph)
    return subgraph


def aug_random_walk_list(graph_list, labels, drop_percent):
    graph_num = len(graph_list)  # number of graphs
    aug_list = []
    for i in range(graph_num):
        sub_graph = aug_random_walk(graph_list[i], labels[i], drop_percent)
        aug_list.append(sub_graph)
    return aug_list


def add_self_loop_if_not_in(graph):
    in_degrees = graph.in_degrees()
    zero_indegree_nodes = [i for i in range(len(in_degrees)) if in_degrees[i].item() == 0]
    for node in zero_indegree_nodes:
        graph.add_edges(node, node)
    return graph