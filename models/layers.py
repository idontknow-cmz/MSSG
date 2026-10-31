"""Graph encoders, semantic projection heads, and expression decoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LLMHead(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim_ratio=0.5):
        super().__init__()
        hidden_dim = max(1, int(input_dim * hidden_dim_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = (
            torch.mm(adj, support)
            if adj.layout == torch.strided
            else torch.sparse.mm(adj, support)
        )
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):

    def __init__(self, nfeat, nhid, out, dropout, n_layers):
        super(GCN, self).__init__()
        if n_layers < 2:
            raise ValueError("GCN requires at least two graph-convolution layers")
        self.n_layers = int(n_layers)
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.hidden_layers = nn.ModuleList(
            (GraphConvolution(nhid, nhid) for _ in range(self.n_layers - 2))
        )
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        for layer in self.hidden_layers:
            x = F.relu(layer(x, adj))
            x = F.dropout(x, self.dropout, training=self.training)
        return self.gc2(x, adj)


class ZINBDecoder(torch.nn.Module):

    def __init__(self, nfeat, nhid1, nhid2):
        super(ZINBDecoder, self).__init__()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nhid2, nhid1), torch.nn.BatchNorm1d(nhid1), torch.nn.ReLU()
        )
        self.pi = torch.nn.Linear(nhid1, nfeat)
        self.disp = torch.nn.Linear(nhid1, nfeat)
        self.mean = torch.nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 0.0001, 10000.0)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-05, 1000000.0)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]
