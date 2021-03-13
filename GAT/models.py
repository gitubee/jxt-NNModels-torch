import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GATMutiHeadAttLayer




class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.MHAlayer1 = GATMutiHeadAttLayer(nfeat, nhid, nheads, dropout, alpha)
        self.out_att = GATMutiHeadAttLayer(nhid * nheads, nclass, 1, dropout, alpha, concat=False)

    def forward(self, x, adj):
        #print(x.size())
        #print(adj.size())
        x = F.dropout(x, self.dropout, training=self.training)
        #assert(0)
        x = self.MHAlayer1(x, adj)
        #print(x.size())
        #assert(0)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        #print(x.size())
        return F.log_softmax(x, dim=1)