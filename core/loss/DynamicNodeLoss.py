import dgl
import torch
import torch.nn.functional as F


def dynamicNodeLoss(batch_graphs, root_logit, instance_labels):
    l_rcl = None
    start = 0

    for idx, num_node in enumerate(batch_graphs.batch_num_nodes()):
        end = start + num_node.item()
        input = F.softmax(root_logit[start:end].H, dim=1)
        input = torch.clamp(input, min=1e-30)
        # 阈值
        # alp_normal = 1 * batch_graphs.batch_size / batch_graphs.number_of_nodes()
        # alp_abnormal = 0.9
        # input = torch.where(input < alp_normal, torch.tensor(1e-10, device=root_logit.device), input)
        # input = torch.where(input > alp_abnormal, torch.tensor(1.0, device=root_logit.device), input)
        # input = torch.where(input < alp_normal, torch.tensor(1e-10, device=root_logit.device), input)

        loss = F.nll_loss(torch.log(input), instance_labels[idx].view(1))
        if start == 0:
            l_rcl = loss
        else:
            l_rcl += loss
        start = end

    l_rcl = l_rcl / batch_graphs.batch_size
    return l_rcl


# def dynamicNodeLoss(batch_graphs, root_logit, instance_labels):
#     return F.cross_entropy(root_logit.view(-1,10), instance_labels)