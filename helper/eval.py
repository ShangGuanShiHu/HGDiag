import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from dgl import DGLGraph


# def accuracy(output, target, batch_graph: DGLGraph, topk=(1, 5), eval=False):
#     maxk = max(topk)
#     batch_size = target.size(0)
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#     res = []
#     for k in topk:
#         if eval and k == 1:
#             res.append(pd.value_counts(target[correct[:k].reshape(-1)!=True].numpy().tolist()))
#         correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#         res.append((correct_k / batch_size).item())
#     return tuple(res)


def accuracy(output, target, batch_graphs: DGLGraph, topk=(1, 5), eval=False):
    maxk = max(topk)
    res = []
    start = 0
    topk_dict = [[] for k in range(maxk)]
    non_root = []
    for idx, num_node in enumerate(batch_graphs.batch_num_nodes()):
        end = start + num_node.item()
        rank_list = output[start:end].flatten().topk(maxk, 0)[1]
        for k in range(maxk):
            if rank_list[k].item() == target[idx].item():
                topk_dict[k].append(target[idx].item())
                break
        if k != 1:
            non_root.append(target[idx].item())
        start = end
    if eval:
        res.append(pd.value_counts(non_root))


    batch_size = batch_graphs.batch_size

    for k in topk:
        topK_num = 0
        for k_temp in range(k):
            topK_num += len(topk_dict[k_temp])
        res.append(topK_num/ batch_size)

    return res


def precision(output, target, k=5):
    _, pred = output.topk(k, 1, True, True)
    y_pred = pred.detach().numpy()
    y_true = target.detach().numpy().reshape(-1, 1)
    pre = precision_score(y_true, y_pred[:, 0], average='weighted')

    return pre


def recall(output, target, k=5):
    _, pred = output.topk(k, 1, True, True)
    y_pred = pred.detach().numpy()
    y_true = target.detach().numpy().reshape(-1, 1)
    rec = recall_score(y_true, y_pred[:, 0], average='weighted')

    return rec


def f1score(output, target, k=5):
    _, pred = output.topk(k, 1, True, True)
    y_pred = pred.detach().numpy()
    y_true = target.detach().numpy().reshape(-1, 1)
    f1 = f1_score(y_true, y_pred[:, 0], average='weighted')

    return f1