import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from core.aug import aug_drop_node
import dgl
from core.aug import *
import dhg

class MultiModalDataSet(Dataset):
    def __init__(self, metrics, traces, logs, instance_labels, type_labels, nodes, edges, device, deployment, cloudbed_index, aug=False):
        self.data = []
        node_num = len(nodes)
        for i in range(len(instance_labels)):

            hyperedges = []
            hyperedge_types = []
            hyperedges_set = set()

            hyperedge_type = -1

            # # 调用超边：根据被调用者将调用者进行聚集
            # # hyperedges_invocation = {nodes[i]: set() for i in range(node_num)}
            # hyperedges_invocation = [set() for i in range(node_num)]

            # 考虑被调用者不能通过调用超边进行更新，因此将被调用者加入超边节点
            hyperedges_invocation = [{i} for i in range(node_num)]

            for caller, callee in zip(edges[0], edges[1]):
                # hyperedges_invocation[nodes[callee]].add(nodes[caller])
                hyperedges_invocation[callee].add(caller)

            hyperedge_type += 1
            for hyperedge in hyperedges_invocation:
                # gaia上保留这种自调用超边
                # aiops上不需要这种自调用超边
                if len(hyperedge) > 0:
                    # 将被调用者加入到超边中，构成调用超边（防止没有调用关系形成的调用超边）
                    hyperedges.append(list(hyperedge))
                    hyperedge_types.append(hyperedge_type)
                    hyperedges_set.add("_".join(list(map(lambda item: str(item), sorted(hyperedge)))))


            # 将pod根据service进行聚集
            hyperedge_type += 1
            hyperedges_service = [[int(i * 2), int(i * 2 +1)] for i in range(node_num // 2)]
            for hyperedge in hyperedges_service:
                if len(hyperedge) > 0:
                    hyperedges.append(hyperedge)
                    hyperedge_types.append(hyperedge_type)
                    hyperedges_set.add("_".join(list(map(lambda item: str(item),sorted(hyperedge)))))


            # 将pod根据host进行聚集
            hyperedge_type += 1
            node_to_idx = {node: idx for idx, node in enumerate(nodes)}
            for hyperedge in deployment[cloudbed_index[i]].values():
                if len(hyperedge) > 0:
                    hyperedge = [node_to_idx[node] for node in hyperedge]
                    hyperedge_types.append(hyperedge_type)
                    hyperedges.append(hyperedge)
                    hyperedges_set.add("_".join(list(map(lambda item: str(item), sorted(hyperedge)))))

            # # 去掉重复的超边
            # hyperedges = []
            # for hyperedge in hyperedges_set:
            #     hyperedges.append(list(map(lambda item: int(item), hyperedge.split("_"))))


            hypergraph = dhg.Hypergraph(node_num, e_list=hyperedges, e_types=hyperedge_types)

            # # hypergraph.draw(v_label=nodes)
            # hypergraph.draw()
            # plt.show()
            graph = dgl.graph(edges, num_nodes=node_num)
            graph.ndata["metrics"] = torch.FloatTensor(metrics[i])
            # graph.ndata["metrics"] = torch.zeros(metrics[i].shape)
            graph.ndata["traces"] = torch.FloatTensor(traces[i])
            # graph.ndata["traces"] = torch.zeros(traces[i].shape)
            graph.ndata["logs"] = torch.FloatTensor(logs[i])
            # graph.ndata["logs"] = torch.zeros(logs[i].shape)

            root, type = instance_labels[i], type_labels[i]

            in_degrees = graph.in_degrees()
            zero_indegree_nodes = [i for i in range(len(in_degrees)) if in_degrees[i].item() == 0]
            for node in zero_indegree_nodes:
                graph.add_edges(node, node)

            self.data.append((graph.to(device), hypergraph, (root, type)))
            if aug:
                # 删除节点
                # aug_graph, aug_hypergraph = aug_drop_node_hypergraph(graph, hypergraph,root)
                # 增加根因节点权重
                aug_graph, aug_hypergraph = aug_add_root_node_hypergraph(graph, hypergraph, root)
                # aug_graph, aug_hypergraph = aug_drop_hyperedge_without_root(graph, hypergraph, root)
                self.data.append((aug_graph.to(device), aug_hypergraph, (root,type)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
