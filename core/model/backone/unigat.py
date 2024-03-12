import dgl
import dhg
import torch
import torch.nn as nn
from dhg.nn import MultiHeadWrapper, UniGATConv
from torch.functional import F
import dgl.nn.pytorch as dglnn


class UniGATEncoder(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers=3,
                 heads=[8,8,1],
                 feat_drop=0.5,
                 attn_drop=0.5,
                 e_type_num=1):
        super(UniGATEncoder, self).__init__()

        self.num_layers=num_layers
        self.gatv2_layers = nn.ModuleList()
        self.activation = F.elu
        # input projection (no residual)
        self.drop_layer = nn.Dropout(feat_drop)
        self.gatv2_layers.append(
            MultiHeadWrapper(
                heads[0],
                "concat",
                UniGATConv,
                in_channels=in_dim,
                out_channels=hidden_dim,
                drop_rate=attn_drop,
                e_type_num=e_type_num
            )
        )

        # hidden layers
        for l in range(0, num_layers-2):
            self.gatv2_layers.append(
                MultiHeadWrapper(
                    heads[l],
                    "concat",
                    UniGATConv,
                    in_channels=hidden_dim * heads[l + 1],
                    out_channels=hidden_dim,
                    drop_rate=attn_drop,
                    e_type_num=e_type_num
                )
            )
        # output projection
        self.gatv2_layers.append(
            UniGATConv(
                hidden_dim * heads[-2],
                out_dim,
                drop_rate=attn_drop,
                is_last=False,
                e_type_num=e_type_num
            )
        )
        self.pool = dglnn.MaxPooling()

    def forward(self, g, x, hg):
        h = x
        for l in range(self.num_layers):
            h = self.drop_layer(h)
            h = self.gatv2_layers[l](X=h,hg=hg)
        # output projection
        logits = self.pool(g, h)

        return torch.cat([h, dgl.broadcast_nodes(g, logits)], dim=1), logits

        # return h, logits
