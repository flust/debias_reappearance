import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.init import normal_


class IPSLoss(nn.Module):
    # use propensity score to debias
    def __init__(self):
        super(IPSLoss, self).__init__()
        self.loss = 0.
        # self.MSELoss = nn.MSELoss()

    def forward(self, output, label, inverse_propensity):
        self.loss = torch.tensor(0.0)
        label0 = label.cpu().numpy()

        weight = torch.Tensor(
            list(map(lambda x: (inverse_propensity[int(label0[x]) - 1]), range(0, len(label0)))))

        # unweightedloss = F.binary_cross_entropy(output, torch.Tensor(label), reduce='none')
        unweightedloss = output - torch.Tensor(label)
        unweightedloss = torch.pow(unweightedloss, 2)
        weightedloss = unweightedloss * weight
        self.loss = torch.sum(weightedloss)
        return self.loss


class MF_IPS(nn.Module):
    def __init__(self, num_users, num_items, embedding_size, inverse_propensity):
        super(MF_IPS, self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.inverse_propensity = inverse_propensity

        self.user_e = nn.Embedding(self.num_users, embedding_size)
        self.item_e = nn.Embedding(self.num_items, embedding_size)
        self.user_b = nn.Embedding(self.num_users, 1)
        self.item_b = nn.Embedding(self.num_items, 1)

        self.loss = IPSLoss()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.1)

    def forward(self, user, item):
        user_embedding = self.user_e(user)
        item_embedding = self.item_e(item)

        preds = self.user_b(user)
        preds += self.item_b(item)
        preds += (user_embedding * item_embedding).sum(dim=1, keepdim=True)

        return preds.squeeze()

    def calculate_loss(self, user_list, item_list, label_list):
        return self.loss(self.forward(user_list, item_list), label_list, self.inverse_propensity)

    def predict(self, user, item):
        return self.forward(user, item)

    def get_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
