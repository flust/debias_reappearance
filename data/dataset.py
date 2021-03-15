# coding:utf8
import os
from torch.utils import data
import numpy as np
import pandas as pd


class Yahoo1(data.Dataset):
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


class Yahoo3(data.Dataset):
    def __init__(self, S_c_data, S_t_data, S_te_data, S_va_data):
        self.S_c = np.loadtxt(S_c_data)
        S_c_len = self.S_c.shape[0]
        self.S_t = np.loadtxt(S_t_data)
        S_t_len = self.S_t.shape[0]
        self.S_te = np.loadtxt(S_te_data)
        S_te_len = self.S_te.shape[0]
        self.S_va = np.loadtxt(S_va_data)
        S_va_len = self.S_va.shape[0]

        raw_matrix = np.vstack((self.S_c, self.S_t, self.S_te, self.S_va))
        self.users_num = int(max(raw_matrix[:, 0]))
        self.items_num = int(max(raw_matrix[:, 1]))

        # onehot_data = pd.get_dummies(pd.DataFrame(raw_matrix), columns=[0, 1]).values
        onehot_data = raw_matrix
        self.S_c = onehot_data[:S_c_len, :]
        self.S_t = onehot_data[S_c_len: S_c_len + S_t_len, :]
        self.S_te = onehot_data[S_c_len + S_t_len: S_c_len + S_t_len + S_te_len, :]
        self.S_va = onehot_data[S_c_len + S_t_len + S_te_len:, :]

        self.data = self.S_c

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]