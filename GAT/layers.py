import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class GATMutiHeadAttLayer(nn.Module):
    def __init__(self, in_features, out_features, heads, dropout=0.4, alpha=0.2, concat=True):
        super(GATMutiHeadAttLayer,self).__init__()
        self.dropout       = dropout        # drop prob = 0.6
        self.in_features   = in_features    # 
        self.out_features  = out_features   # 
        self.heads         = heads          #
        self.alpha         = alpha          # LeakyReLU with negative input slope, alpha = 0.2 
        self.concat        = concat         # concat all heads or calculate average
        #leaky relu
        self.gain = nn.init.calculate_gain('leaky_relu', self.alpha)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        # Xavier Initialization of Weights
        self.W = nn.Parameter(torch.zeros(size=(heads, in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, self.gain)
        self.a_1 = nn.Parameter(torch.zeros(size=(heads, out_features, 1)))
        nn.init.xavier_uniform_(self.a_1.data, self.gain)
        self.a_2 = nn.Parameter(torch.zeros(size=(heads, out_features, 1)))
        nn.init.xavier_uniform_(self.a_2.data, self.gain)
    
    def forward(self, input_seq, adj):
        if(len(input_seq.size())==2):
            input_seq = torch.unsqueeze(input_seq, 0)
            adj = torch.unsqueeze(adj, 0)
        input_seq = torch.unsqueeze(input_seq, 1)
        adj = torch.unsqueeze(adj, 1)
        in_size = input_seq.size()
        nbatchs = in_size[0]
        slen = in_size[2]
        #transform the input features into higher-level feature
        h = torch.matmul(input_seq, self.W)
        #calculate the attention,divide the a*(hi,hj) into [a_1*h]+[a_2*h]
        f_1 = torch.matmul(h, self.a_1)
        f_2 = torch.matmul(h, self.a_2)
        e = f_1.expand(nbatchs, self.heads, slen, slen) + f_2.expand(nbatchs, self.heads, slen, slen).transpose(2,3)
        e = self.leakyrelu(e)

        # softmax the attention and drop
        zero_vec  = -9e15*torch.ones_like(e)
        attention = torch.where(adj.expand(nbatchs, self.heads, slen, slen) > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # calculate the nodes vector
        node_out=torch.matmul(attention, h)
        # concate all heads or heads average
        if(self.concat):
            node_out=node_out.transpose(1,2).contiguous().view(nbatchs,slen,-1)
            node_out=F.elu(node_out)
        else:
            node_out=node_out.mean(1)
        return node_out.squeeze()
def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
        
