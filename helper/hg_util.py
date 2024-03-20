# 超图batch
from typing import List

import dhg
from dhg import Hypergraph


def batch_hypergraph(hypergraphs: List[Hypergraph], device):
    node_nums = [hypergraph.num_v for hypergraph in hypergraphs]
    hyperedges = hypergraphs[0].e[0].copy()
    hyperedge_types = hypergraphs[0].e[2].copy()

    bais = 0
    for i in range(len(hypergraphs)-1):
        hypergraph = hypergraphs[i+1]
        bais += node_nums[i]
        hyperedges.extend([[v + bais for v in hyperedge] for hyperedge in hypergraph.e[0]])
        hyperedge_types.extend(hypergraphs[i+1].e[2].copy())
    node_num = bais + node_nums[-1]

    return dhg.Hypergraph(node_num, hyperedges, device=device, e_types=hyperedge_types)