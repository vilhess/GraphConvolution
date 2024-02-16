import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GraphConvolutionBlock(nn.Module):

    # Convolution Block

    def __init__(self, in_features, out_features, adj, bias=True):
        super(GraphConvolutionBlock, self).__init__()
        if torch.is_tensor(adj)==False:
            adj = torch.Tensor(adj)
        adj = adj.fill_diagonal_(1)
        D = torch.diag(1/torch.sqrt(torch.sum(adj, dim=1)))
        self.new_adj = torch.mm(torch.mm(D, adj), D)
        self.in_features = in_features
        self.out_features = out_features
        lam = 1/np.sqrt(out_features)
        self.weight = Parameter(torch.Tensor(in_features, out_features).to(torch.float32).uniform_(-lam, lam))
        self.is_bias = bias
        if self.is_bias:
            self.bias = Parameter(torch.Tensor(out_features).uniform_(-lam, lam).to(torch.float32))
        
    def forward(self, X):
        z = torch.mm(X, self.weight)
        out = torch.spmm(self.new_adj, z)
        if self.is_bias:
            out = out + self.bias
        return z, out
    

class GCN(nn.Module):

    # 2 layers model using the Graph Convolution from above

    def __init__(self, in_channels, hidden_dim, out_channels, adj, return_hidden = False):
        super(GCN, self).__init__()
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(0.5)
        self.gc1 = GraphConvolutionBlock(in_channels, hidden_dim, adj)
        self.gc2 = GraphConvolutionBlock(hidden_dim, out_channels, adj)
        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.return_hidden = return_hidden

    def forward(self, x):
        z1, out = self.gc1(x)
        out = self.dp(self.relu((out)))
        z2, out = self.gc2(out)

        if self.return_hidden:
            return self.lsoftmax(out), z1
        
        return self.lsoftmax(out)