
from torch import nn
from .backone.unigat import UniGATEncoder


class Encoder(nn.Module):
    def __init__(self, 
                 in_dim, 
                 graph_hidden_dim, 
                 out_dim,
                 feat_drop=0.5,
                 attn_drop=0.5,
                 device='cpu',
                 e_type_num=1):
        super(Encoder, self).__init__()

        # word CNN
        # self.sequential_encoder = CNN1dEncoder(
        #     in_dim=1,
        #     hidden_dim=seq_hidden,
        #     kernel_size=3,
        #     dropout=0.2
        # ).to(device)

        # feature aggregation
        self.graph_encoder = UniGATEncoder(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=graph_hidden_dim,
            num_layers=2,
            heads=[8,1],
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            e_type_num=e_type_num
        ).to(device)


    def forward(self, g, x, hg):
        # h = self.sequential_encoder(x)
        l, t = self.graph_encoder(g, x, hg)
        return l, t