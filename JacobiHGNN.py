# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 19:24:58 2023

@author: dell
"""

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


def JacobiConv(L, xs, adj, alphas, a=1.0, b=1.0, l=-1.0, r=1.0):  # a, b超参数，是否有影响？原文在主代码中由**kwargs来传递
    '''
    Jacobi Bases. Please refer to our paper for the form of the bases.
    '''
    if L == 0: return xs[0]
    if L == 1:
        coef1 = (a - b) / 2 - (a + b + 2) / 2 * (l + r) / (r - l)  # l,r 的作用？
        coef1 *= alphas[0]  # 原系数*γ_k
        coef2 = (a + b + 2) / (r - l)
        coef2 *= alphas[0]  # 原系数*γ_k
        return coef1 * xs[-1] + coef2 * (adj @ xs[-1])  # P_1(A)X
    coef_l = 2 * L * (L + a + b) * (2 * L - 2 + a + b)
    coef_lm1_1 = (2 * L + a + b - 1) * (2 * L + a + b) * (2 * L + a + b - 2)
    coef_lm1_2 = (2 * L + a + b - 1) * (a ** 2 - b ** 2)
    coef_lm2 = 2 * (L - 1 + a) * (L - 1 + b) * (2 * L + a + b)
    tmp1 = alphas[L - 1] * (coef_lm1_1 / coef_l)  # γ_k θ
    tmp2 = alphas[L - 1] * (coef_lm1_2 / coef_l)  # γ_k θ'
    tmp3 = alphas[L - 1] * alphas[L - 2] * (coef_lm2 / coef_l)  # γ_k γ_(k-1) θ''
    tmp1_2 = tmp1 * (2 / (r - l))
    tmp2_2 = tmp1 * ((r + l) / (r - l)) + tmp2
    nx = tmp1_2 * (adj @ xs[-1]) - tmp2_2 * xs[-1]
    nx -= tmp3 * xs[-2]
    return nx  # 公式(15)


class PolyConvFrame(nn.Module):
    '''
    A framework for polynomial graph signal filter.
    Args:
        conv_fn: the filter function, like PowerConv, LegendreConv,...
        depth (int): the order of polynomial.
        cached (bool): whether or not to cache the adjacency matrix.
        alpha (float):  the parameter to initialize polynomial coefficients.
        fixed (bool): whether or not to fix to polynomial coefficients.
    '''

    def __init__(self,
                 depth: int = 3,
                 alpha: float = 1.0,
                 a: float = 1.0,
                 b: float = 1.0,
                 h_feats: int = 64,
                 fixed: float = False):
        super().__init__()
        self.depth = depth
        self.a = a
        self.b = b
        self.h_feats = h_feats
        self.basealpha = alpha  # γ'
        self.alphas = nn.ParameterList([nn.Parameter(torch.tensor(float(min(1 / alpha, 1))),  # θ
                                                     requires_grad=not fixed) for i in range(depth + 1)])
        self.adj = None
        self.conv_fn = JacobiConv

        self.S_att = Spatial_Attention_Module(self.h_feats, self.depth+1)
        self.fc = nn.Linear(h_feats, h_feats)
        # self.FCk = nn.Linear(self.depth + 1, 1)

        # self.comb_weight = nn.Parameter(torch.FloatTensor(1, (self.depth + 1), self.h_feats))
        # nn.init.xavier_normal_(self.comb_weight, gain=1.414)

    def forward(self, x: Tensor, adj: Tensor):
        '''
        Args:
            x: node embeddings. of shape (number of nodes, node feature dimension)
            adj : sparse adjacency matrix
        '''
        alphas = [self.basealpha * torch.tanh(_) for _ in self.alphas]  # γ_k = γ'tanθ
        xs = [self.conv_fn(0, [x], adj, alphas, self.a, self.b)]
        for L in range(1, self.depth + 1):
            tx = self.conv_fn(L, xs, adj, alphas, self.a, self.b)
            xs.append(tx)
        '''
        node-level attention fusion
        '''
        xs = [x.unsqueeze(0) for x in xs]
        raw = torch.cat(xs, dim=0)  # [K,N,F]
        x_s = self.S_att(raw)       # [N,C]
        out = self.fc(x_s)
        '''
        FCk
        '''
        # xs = [x.unsqueeze(2) for x in xs]
        # raw = torch.cat(xs, dim=2)  # [N,C,K]
        # x_s = self.FCk(raw).squeeze(2)   # [N,C,1] -> [N,C]
        # out = self.fc(x_s)
        return out


class Spatial_Attention_Module(nn.Module):
    def __init__(self, h_feats, order):
        super(Spatial_Attention_Module, self).__init__()
        self.attn_fn = nn.Tanh()
        self.order = order
        self.linear = nn.ModuleList()
        for _ in range(self.order):
            self.linear.append(nn.Linear(h_feats, h_feats))

    def forward(self, x):
        '''
        x: [K, N, F]
        '''
        q = torch.mean(x, dim=1).unsqueeze(1)  # [K, 1, F]

        x_proj = torch.zeros_like(x).to(q.device)
        for i, lin in enumerate(self.linear):
            x_proj[i] = lin(x[i])

        x_proj = x_proj.permute(0, 2, 1)  # [K, F, N]

        att = torch.bmm(q, x_proj).squeeze(1)  # [K, 1, N] -> [K, N]
        att = self.attn_fn(att)
        att = torch.softmax(att, dim=0).unsqueeze(1)  # [K, 1, N]
        att = att.permute(2, 1, 0)  # [N, 1, K]

        x_proj = x_proj.permute(2, 0, 1)  # [N, K, F]
        out = torch.bmm(att, x_proj).squeeze(1)  # [N, 1, F] -> [N, F]

        return out
class JacobiGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, adj, depth=3, alpha=1.0, a=1.0, b=1.0, dropout=0.0, fixed=False):
        super(JacobiGNN, self).__init__()
        self.adj = adj
        self.depth = depth
        self.alpha = alpha
        self.a = a
        self.b = b
        self.fixed = fixed
        self.dropout = dropout

        self.act = nn.ReLU()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_feats, h_feats))
        self.fcs.append(nn.Linear(h_feats, h_feats))
        self.fcs.append(nn.Linear(h_feats, num_classes))

        self.frame_fn = PolyConvFrame(self.depth, self.alpha, self.a, self.b, h_feats, self.fixed)
        self.mlp = nn.Linear(h_feats, h_feats)

        self.params1 = list(self.frame_fn.parameters())
        self.params2 = list(self.fcs.parameters())

    def forward(self, features):
        h = self.fcs[0](features)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.act(h)
        h = self.fcs[1](h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.act(h)
        '''
        MLP
        '''
        # h = self.mlp(h)
        '''
        JacobiGNN
        '''
        h = self.frame_fn(h, self.adj)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.act(h)

        logit = self.fcs[-1](h)  # [N, h_feats] -> [N, 2]
        return h, logit


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=None):
        super(HGNN_conv, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, hg):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = 0.5 * hg.D_e.matmul(hg.MT.t().matmul(hg.D_v.matmul(hg.MT.matmul(x))))

        return x


class HGNN(nn.Module):
    def __init__(self, in_ft, out_ft, num_classes, nlayers, dropout=0.0, bias=None):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(HGNN_conv(in_ft, out_ft))
        self.act = nn.ReLU()
        self.fc = nn.Linear(out_ft, num_classes)

    def forward(self, x, hg):
        for i, conv in enumerate(self.convs):
            x = conv(x, hg)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class JacobiHGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, adj, depth=3, nlayers=2, alpha=1.0, a=1.0, b=1.0, drop1=0.0, drop2=0.0):
        super(JacobiHGNN, self).__init__()
        self.depth = depth
        self.nlayers = nlayers
        self.a = a
        self.b = b
        self.alpha = alpha
        self.dropout1 = drop1
        self.dropout2 = drop2
        self.jacobi = JacobiGNN(in_feats, h_feats, num_classes, adj, self.depth, self.alpha, self.a, self.b,
                                self.dropout1, fixed=False)
        self.hgnn = HGNN(h_feats, h_feats, num_classes, self.nlayers, self.dropout2, bias=None)
        self.fc = nn.Linear(h_feats, num_classes)

        self.params1 = list(self.jacobi.parameters())
        self.params2 = list(self.hgnn.parameters()) + list(self.fc.parameters())

    def forward(self, hg, features):
        x, logit1 = self.jacobi(features)
        x = self.hgnn(x, hg)
        logit = self.fc(x)
        return logit, logit1






