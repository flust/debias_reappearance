import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.init import normal_


class FMLoss(nn.Module):
    def __init__(self):
        super(FMLoss, self).__init__()
        self.loss = 0.

    def forward(self, output, label):
        self.loss = torch.sum(torch.log(1 + torch.exp(- output * label)))
        return self.loss


class IPSLoss(nn.Module):
    # use propensity score to debias
    def __init__(self, device):
        super(IPSLoss, self).__init__()
        self.loss = 0.
        self.device = device
        # self.MSELoss = nn.MSELoss()

    def forward(self, output, label, inverse_propensity):
        self.loss = torch.tensor(0.0)
        label0 = label.cpu().numpy()

        weight = torch.Tensor(
            list(map(lambda x: (inverse_propensity[int(label0[x]) - 1]), range(0, len(label0))))).to(self.device)

        # unweightedloss = F.binary_cross_entropy(output, torch.Tensor(label), reduce='none')
        unweightedloss = output - label
        unweightedloss = torch.pow(unweightedloss, 2)
        weightedloss = unweightedloss * weight
        self.loss = torch.sum(weightedloss)
        return self.loss


class DRMLoss(nn.Module):
    # use propensity score to debias
    def __init__(self, impute_label, device):
        super(DRMLoss, self).__init__()
        self.loss = 0.
        self.impute_label = impute_label
        self.device = device
        # self.MSELoss = nn.MSELoss()

    def l_lowercase(self, true, pred):
        return torch.pow(true - pred, 2)

    def forward(self, output, label, inverse_propensity, flag):
        self.loss = torch.tensor(0.0)
        label0 = label.cpu().numpy()

        weight = torch.Tensor(
            list(map(lambda x: (inverse_propensity[int(label0[x]) - 1]) if flag[x] == 1 else 0,
                     range(0, len(label0))))).to(self.device)

        # unweightedloss = F.binary_cross_entropy(output, torch.Tensor(label), reduce='none')
        # unweightedloss = self.l_lowercase(label, output)
        # weightedloss = unweightedloss * weight
        # self.IPS = torch.sum(weightedloss)
        self.IPS = torch.sum(weight * self.l_lowercase(label, output))

        impute_loss = self.l_lowercase(label, output)
        self.loss1 = self.IPS - torch.sum(impute_loss)
        self.loss2 = torch.sum(impute_loss * (1 - flag))

        return self.loss1 + self.loss2
