import torch.nn as nn
import torch.nn.functional as F
from gcn_layer import GraphConvolution
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        out = F.relu(self.gc1(x, adj))
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.gc2(out, adj)
        # out = out.reshape(1, x.shape[0], 10)
        # for l in range(1, 64):
        #     out_temp = F.relu(self.gc1(x, adj))
        #     out_temp = F.dropout(out_temp, self.dropout, training=self.training)
        #     out_temp = self.gc2(out_temp, adj)
        #     out_temp = out_temp.reshape(1, x.shape[0], 10)
        #     out = torch.cat((out, out_temp), 0)
        return out
