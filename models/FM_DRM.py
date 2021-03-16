import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from .loss import DRMLoss
from .FM import FeaturesLinear, FeaturesEmbedding, FactorizationMachine


class FM_DRM(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, inverse_propensity, impute_label, device='cpu'):
        super().__init__()
        self.inverse_propensity = inverse_propensity
        self.device = device
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.impute_label = impute_label
        self.loss = DRMLoss(self.impute_label, self.device)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x - 1  # user item begin with 1
        x = self.linear(x) + self.fm(self.embedding(x))
        return x.squeeze(1)

    def calculate_loss(self, user_list, item_list, label_list):
        user_id = {}
        user_count = 0
        item_id = {}
        item_count = 0

        user_num = len(set(user_list.tolist()))
        item_num = len(set(item_list.tolist()))
        id_user = torch.zeros(user_num)
        id_item = torch.zeros(item_num)
        label = torch.zeros(user_num, item_num)
        flag = torch.zeros(user_num, item_num)

        for i in range(len(user_list)):
            if user_list[i].item() not in user_id:
                user_id[user_list[i].item()] = user_count
                id_user[user_count] = user_list[i]
                user_count += 1
            if item_list[i].item() not in item_id:
                item_id[item_list[i].item()] = item_count
                id_item[item_count] = item_list[i]
                item_count += 1
            label[user_id[user_list[i].item()], item_id[item_list[i].item()]] = label_list[i]
            flag[user_id[user_list[i].item()], item_id[item_list[i].item()]] = 1

        label[label == 0] = self.impute_label
        label = label.reshape(-1)
        flag = flag.reshape(-1)
        grid_0 = id_user.repeat(item_count, 1).t().reshape(-1)
        grid_1 = id_item.repeat(user_count).reshape(-1)

        # grid = torch.meshgrid(user_list, item_list)
        # grid_0 = grid[0].reshape(-1)
        # grid_1 = grid[1].reshape(-1)
        # label = torch.diag(label_list)
        # flag = torch.eye(label_list.shape[0])
        # if isinstance(self.impute_label, np.float64):
        #     label[label == 0] = self.impute_label
        # # else:
        # #     for i in range(len(item_list)):
        # #         label[:, i] = self.impute_label[item_list[i]]
        # label = label.reshape(-1)
        # flag = flag.reshape(-1)
        # return self.loss(self.forward(torch.vstack([grid_0, grid_1]).t()), label, self.inverse_propensity, flag)

        output = self.forward(torch.vstack([grid_0.long(), grid_1.long()]).t())
        return self.loss(output, label, self.inverse_propensity, flag)

    def predict(self, user, item):
        return self.forward(torch.vstack([user, item]).t())

    def get_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
