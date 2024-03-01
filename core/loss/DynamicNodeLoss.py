import torch.nn.functional as F


def dynamicNodeLoss(batch_graphs, root_logit, instance_labels):
    l_rcl = None
    start = 0
    for idx, num_node in enumerate(batch_graphs.batch_num_nodes()):
        end = start + num_node.item()
        if start == 0:
            l_rcl = F.cross_entropy(root_logit[start:end].H, instance_labels[idx].view(1))
        else:
            l_rcl += F.cross_entropy(root_logit[start:end].H, instance_labels[idx].view(1))
        start = end

    l_rcl = l_rcl / batch_graphs.batch_size
    return l_rcl


# def dynamicNodeLoss(batch_graphs, root_logit, instance_labels):
#     return F.cross_entropy(root_logit.view(-1,10), instance_labels)