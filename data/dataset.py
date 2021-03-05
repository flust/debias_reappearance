# coding:utf8
import os
from torch.utils import data
import numpy as np
import pandas as pd


class Yahoo(data.Dataset):
    def __init__(self, filename, train=True, test=False):
        self.train = train
        self.test = test
        raw_matrix = np.loadtxt(filename)
        self.users_num = int(max(raw_matrix[:, 0]))
        self.items_num = int(max(raw_matrix[:, 1]))
        self.data = raw_matrix

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]