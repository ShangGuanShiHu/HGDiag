import argparse
import random
import time
from typing import List

import dhg
from dhg import Hypergraph

from helper.logger import get_logger
import dgl
import torch
import numpy as np
from torch.utils.data import DataLoader
from core.TVDiag import TVDiag
from process.EventProcess import EventProcess
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

# common
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--log_step', type=int, default=20)
parser.add_argument('--eval_period', type=int, default=10)
parser.add_argument('--reconstruct', type=bool, default=False)

# # dataset gaia
# parser.add_argument('--dataset', type=str, default='gaia', help='name of dataset')
# parser.add_argument('--N_I', type=int, default=10, help='number of instances')
# parser.add_argument('--N_T', type=int, default=5, help='number of failure types')
# # TVDiag
# parser.add_argument('--feat_drop', type=float, default=0.5)
# parser.add_argument('--attn_drop', type=float, default=0.5)
# parser.add_argument('--lr', type=float, default=0.01)
# parser.add_argument('--batch_size', type=int, default=1024)
# parser.add_argument('--guide_weight', type=float, default=0.1)
# # Scorer
# parser.add_argument('--scorer_layers', type=int, default=3)
# parser.add_argument('--scorer_drop', type=float, default=0.3)

# dataset aiops22
parser.add_argument('--dataset', type=str, default='aiops22', help='name of dataset')
parser.add_argument('--N_I', type=int, default=40, help='number of instances')
parser.add_argument('--N_T', type=int, default=9, help='number of failure types')
# TVDiag
parser.add_argument('--feat_drop', type=float, default=0.5)
parser.add_argument('--attn_drop', type=float, default=0.5)
# parser.add_argument('--lr', type=float, default=0.0005)
# parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--guide_weight', type=float, default=0.1)
# Scorer
parser.add_argument('--scorer_layers', type=int, default=3)
parser.add_argument('--scorer_drop', type=float, default=0.3)


# TVDiag
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--seq_hidden', type=int, default=128)
parser.add_argument('--linear_hidden', type=list, default=[64])
parser.add_argument('--graph_hidden', type=int, default=64)
parser.add_argument('--graph_out', type=int, default=32)
parser.add_argument('--TO', type=bool, default=False)
parser.add_argument('--CM', type=bool, default=False)
parser.add_argument('--temperature', type=float, default=0.3)

parser.add_argument('--dynamic_weight', type=bool, default=False)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--aug', type=bool, default=False)
parser.add_argument('--aug_percent', type=float, default=0.2)
parser.add_argument('--aug_method', type=str, default='node_drop')

# Muti-UniGAT atten_e num
parser.add_argument('--e_type_num', type=int, default=1)

parser.add_argument('--device', type=str,default='cuda')

args = parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# 超图batch
def batch_hypergraph(hypergraphs: List[Hypergraph]):
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

    return dhg.Hypergraph(node_num, hyperedges, device=args.device, e_types=hyperedge_types)



# def collate(samples, ):
#     graphs, labels = map(list, zip(*samples))
#     batched_graph = dgl.batch(graphs)
#     batched_labels = torch.tensor(labels, device=args.device)
#     return batched_graph, batched_labels


def collate(samples, ):
    graphs, hypergrahps, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_hypergraph = batch_hypergraph(hypergrahps)
    batched_labels = torch.tensor(labels, device=args.device)
    return batched_graph, batched_hypergraph, batched_labels


def build_dataloader(args, logger, device='cpu'):
    reconstruct = args.reconstruct

    N_T = args.N_T
    N_I = args.N_I
    epochs = args.epochs
    embedding_dim = args.embedding_dim

    processor = EventProcess(args, logger)
    train_data, test_data = processor.process(reconstruct=reconstruct, device=device)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    return {'N_I': N_I,
            'N_T': N_T,
            'epochs': epochs,
            'embedding_dim': embedding_dim}, \
           train_dataloader, test_data


if __name__ == '__main__':
    logger = get_logger(f'logs/{args.dataset}', 'TVDiag')
    set_seed(args.seed)

    device = args.device
    logger.info("Load dataset")
    data_args, train_dl, test_data = build_dataloader(args, logger, device)
    # 将超边类型数量传下去，由于实验过程中，超边种类是在mutimodal_dataset中是变化的
    args.e_type_num = test_data[0][1].e_type_num


    logger.info("Training...")
    model = TVDiag(args, logger,device)

    model.train(train_dl, test_data)