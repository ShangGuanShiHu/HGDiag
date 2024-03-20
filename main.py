import argparse
import random

from helper.logger import get_logger
from helper.hg_util import batch_hypergraph
import dgl
import torch
import numpy as np
from torch.utils.data import DataLoader
from core.HGDiag import HGDiag
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
# parser.add_argument('--lr', type=float, default=0.005)
# parser.add_argument('--batch_size', type=int, default=1024)
# parser.add_argument('--guide_weight', type=float, default=0.1)
# parser.add_argument('--epochs', type=int, default=200)


# dataset aiops22
parser.add_argument('--dataset', type=str, default='aiops22', help='name of dataset')
parser.add_argument('--N_I', type=int, default=40, help='number of instances')
parser.add_argument('--N_T', type=int, default=9, help='number of failure types')
# TVDiag
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--guide_weight', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=1000)

# Scorer
parser.add_argument('--scorer_layers', type=int, default=3)

# HGDiag
parser.add_argument('--feat_drop', type=float, default=0.5)
parser.add_argument('--attn_drop', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--seq_hidden', type=int, default=128)
parser.add_argument('--linear_hidden', type=list, default=[64])
parser.add_argument('--graph_hidden', type=int, default=64)
parser.add_argument('--graph_out', type=int, default=32)
parser.add_argument('--temperature', type=float, default=0.3)

# Muti-task
parser.add_argument('--dynamic_weight', type=bool, default=False)


# Muti-UniGAT atten_e num
parser.add_argument('--e_type_num', type=int, default=1)

parser.add_argument('--device', type=str, default='cuda')

args = parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def collate(samples, ):
    graphs, hypergrahps, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_hypergraph = batch_hypergraph(hypergrahps, device=args.device)
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
    logger = get_logger(f'logs/{args.dataset}', 'HGDiag')
    set_seed(args.seed)

    device = args.device
    logger.info("Load dataset")
    data_args, train_dl, test_data = build_dataloader(args, logger, device)
    # 将超边类型数量传下去，由于实验过程中，超边种类是在mutimodal_dataset中是变化的
    args.e_type_num = test_data[0][1].e_type_num


    logger.info("Training...")
    model = HGDiag(args, logger,device)

    model.train(train_dl, test_data)