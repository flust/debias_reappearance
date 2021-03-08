import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_


class FM_model(nn.Module):
    def __init__(self, n, k):
        super(FM_model, self).__init__()
        self.n = n # len(items) + len(users)
        self.k = k
        self.linear = nn.Linear(self.n, 1, bias=True)
        self.v = nn.Parameter(torch.randn(self.k, self.n))

    def fm_layer(self, x):
        # x 属于 R^{batch*n}
        linear_part = self.linear(x)
        # 矩阵相乘 (batch*p) * (p*k)
        inter_part1 = torch.mm(x, self.v.t())  # out_size = (batch, k)
        # 矩阵相乘 (batch*p)^2 * (p*k)^2
        inter_part2 = torch.mm(torch.pow(x, 2), torch.pow(self.v, 2).t()) # out_size = (batch, k)
        output = linear_part + 0.5 * torch.sum(torch.pow(inter_part1, 2) - inter_part2)
        # 这里torch求和一定要用sum
        return output  # out_size = (batch, 1)

    def forward(self, x):
        output = self.fm_layer(x)
        return output


# class FFM(nn.Module):
#     def __init__(self, field_dims, embedding_size, device):
#         super().__init__()
#         self.device = device
#         self.num_fields = len(field_dims)
#         self.embeddings = nn.ModuleList([
#             nn.Embedding(sum(field_dims), embedding_size) for _ in range(self.num_fields)
#         ])
#         self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
#         for embedding in self.embeddings:
#             nn.init.xavier_uniform_(embedding.weight.data)
#
#     def forward(self, x):
#         x = x + x.new_tensor(self.offsets).unsqueeze(0)
#         xs = [self.embeddings[i](x) for i in range(self.num_fields)]
#         ix = list()
#         for i in range(self.num_fields-1):
#             for j in range(i+1, self.num_fields):
#                 ix.append(xs[j][:, j] * xs[i][:, j])
#         ix = torch.stack(ix, dim=1)
#         return ix
#
