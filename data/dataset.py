# coding:utf8
import os
from torch.utils import data
import numpy as np
import pandas as pd


class Yahoo(data.Dataset):
    def __init__(self, filename):
        raw_matrix = np.loadtxt(filename)
        self.users_num = int(max(raw_matrix[:, 0]))
        self.items_num = int(max(raw_matrix[:, 1]))
        self.data = raw_matrix

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]


class Yahoo2(data.Dataset):
    def __init__(self, s_c_data, s_t_data):
        raw_matrix_c = np.loadtxt(s_c_data)
        raw_matrix_t = np.loadtxt(s_t_data)
        self.s_c = raw_matrix_c[:, :3]
        self.s_t = raw_matrix_t[:, :3]
        raw_matrix = np.vstack((raw_matrix_c, raw_matrix_t))
        self.users_num = int(max(raw_matrix[:, 0]))
        self.items_num = int(max(raw_matrix[:, 1]))
        self.data = raw_matrix

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]