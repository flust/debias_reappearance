import numpy as np
import pandas as pd
from collections import Counter
import torch

if __name__ == '__main__':
    print("calculate propensity score begin")
    # load origin data
    S_c_data = np.loadtxt('./S_c.txt').astype(int)
    S_t_data = np.loadtxt('./S_t.txt').astype(int)
    S_ct_data = np.vstack([S_c_data, S_t_data])

    # calculate num of user and item
    user_num = max(S_ct_data[:, 0])
    item_num = max(S_ct_data[:, 1])

    # calculate different rating num and normalize it

    # S_c & S_t for P_O_Y
    P_O_Y = np.bincount(S_ct_data[:, 2])
    P_O_Y = P_O_Y / P_O_Y.sum()

    # S_t for P_Y
    P_Y = np.bincount(S_t_data[:, 2])
    P_Y = P_Y / P_Y.sum()

    # calculate Propensity Score
    P = P_O_Y * S_ct_data.shape[0] / (user_num * item_num) / P_Y

    np.savetxt('propensity_score.txt', P)
    print('propensity score for different rating', P)
    print('calculate propensity score end')

