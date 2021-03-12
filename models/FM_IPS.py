import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from .loss import IPSLoss
from .FM import FeaturesLinear, FeaturesEmbedding, FactorizationMachine


class FM_IPS(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, inverse_propensity, device='cpu'):
        super().__init__()
        self.inverse_propensity = inverse_propensity
        self.device = device
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.loss = IPSLoss(self.device)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x - 1  # user item begin with 1
        x = self.linear(x) + self.fm(self.embedding(x))
        return torch.sigmoid(x.squeeze(1))

    def calculate_loss(self, user_list, item_list, label_list):
        return self.loss(self.forward(torch.vstack([user_list, item_list]).t()), label_list, self.inverse_propensity)

    def predict(self, user, item):
        return self.forward(torch.vstack([user, item]).t())

    def get_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
