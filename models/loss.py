import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.init import normal_


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
