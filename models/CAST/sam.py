import torch
import torch.nn as nn
from math import sqrt

class StaticAttention(nn.Module):
    def __init__(self, scale=None, dropout=0.1, n_size=(96,96), nhead=8):
        super(StaticAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.scores = torch.nn.Parameter(torch.rand(size=(1, nhead, n_size[0], n_size[1])), requires_grad=True)

    def forward(self, values, B, E, need_weights=False):
        scale = self.scale or 1. / sqrt(E)
        scores = self.scores.repeat(B, 1, 1, 1)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if need_weights:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
