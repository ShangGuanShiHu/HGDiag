import torch
from torch import nn

from core.model.scorer import Scorer,Classifier
from core.model.Encoder import Encoder


class MainModel(nn.Module):
    def __init__(self, args):
        super(MainModel, self).__init__()

        self.args = args

        self.metric_encoder = Encoder(in_dim=args.embedding_dim,
                                      feat_drop=args.feat_drop,
                                      attn_drop=args.attn_drop,
                                      graph_hidden_dim=args.graph_hidden,                                
                                      out_dim=args.graph_out,
                                      e_type_num=args.e_type_num)
        self.trace_encoder = Encoder(in_dim=args.embedding_dim,
                                      feat_drop=args.feat_drop,
                                      attn_drop=args.attn_drop,
                                      graph_hidden_dim=args.graph_hidden,
                                      out_dim=args.graph_out,
                                     e_type_num=args.e_type_num)
        self.log_encoder = Encoder(in_dim=args.embedding_dim,
                                      feat_drop=args.feat_drop,
                                      attn_drop=args.attn_drop,
                                      graph_hidden_dim=args.graph_hidden,
                                      out_dim=args.graph_out,
                                   e_type_num=args.e_type_num
                                      )
        fuse_dim = 3 * args.graph_out

        self.locator = Scorer(in_dim=fuse_dim*2, num_residual_blocks=args.scorer_layers)

        self.typeClassifier = Classifier(in_dim=fuse_dim,
                                         hiddens=args.linear_hidden,
                                         out_dim=args.N_T)

    def forward(self, batch_graphs, batch_hypergraphs):

        x_m = batch_graphs.ndata['metrics']
        x_t = batch_graphs.ndata['traces']
        x_l = batch_graphs.ndata['logs']

        l_m, t_m = self.metric_encoder(batch_graphs, x_m, batch_hypergraphs)
        l_t, t_t = self.trace_encoder(batch_graphs, x_t, batch_hypergraphs)
        l_l, t_l = self.log_encoder(batch_graphs, x_l, batch_hypergraphs)

        l = torch.cat((l_m, l_t, l_l), dim=1)
        t = torch.cat((t_m, t_t, t_l), dim=1)
        # f = f_m

        root_logit = self.locator(l)
        type_logit = self.typeClassifier(t)

        return root_logit, type_logit