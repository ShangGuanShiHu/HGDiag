
from collections import OrderedDict
import copy
import torch
import torch.nn.functional as F
from core.model.MainModel import MainModel
from .loss.DynamicNodeLoss import dynamicNodeLoss

# Adapted from https://github.com/google-research/google-research/blob/master/tag/celeba/CelebA.ipynb


def cal_task_affinity(model: MainModel, 
                      optimizer, 
                      batch_graphs,
                      batch_hypergraphs,
                      instance_labels, 
                      type_labels):
    model1 = copy.deepcopy(model)
    model2 = copy.deepcopy(model)

    lr = optimizer.param_groups[0]['lr']
    weight_decay = optimizer.param_groups[0]['weight_decay']
    opt1 = torch.optim.Adam(model1.parameters(), lr=lr, weight_decay=weight_decay)
    opt2 = torch.optim.Adam(model2.parameters(), lr=lr, weight_decay=weight_decay)

    ####################################
    ####   calculate the affinity   ####
    ####    RCL -> FTI              ####
    ####################################
    root_logit, type_logit = model1(batch_graphs, batch_hypergraphs)
    l_rcl = dynamicNodeLoss(batch_graphs, root_logit, instance_labels)
    l_fti = F.cross_entropy(type_logit, type_labels)
    # update model with l_rcl
    l_rcl.backward()
    opt1.step()
    _, type_logit2 = model1(batch_graphs,batch_hypergraphs)
    l_fti2 = F.cross_entropy(type_logit2, type_labels)
    Z_r2f = 1 - (l_fti2.detach().item()/l_fti.detach().item())

    ####################################
    ####   calculate the affinity   ####
    ####    FTI -> RCL              ####
    ####################################
    root_logit, type_logit = model2(batch_graphs,batch_hypergraphs)
    l_rcl = dynamicNodeLoss(batch_graphs, root_logit, instance_labels)
    l_fti = F.cross_entropy(type_logit, type_labels)
    # update model with l_fti
    l_fti.backward()
    opt2.step()
    root_logit2, _ = model1(batch_graphs,batch_hypergraphs)
    l_rcl2 = dynamicNodeLoss(batch_graphs, root_logit2, instance_labels)
    Z_f2r = 1 - (l_rcl2.detach().item()/l_rcl.detach().item())

    return Z_r2f, Z_f2r