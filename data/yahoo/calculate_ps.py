import numpy as np
import pandas as pd
from collections import Counter
import torch

if __name__ == '__main__':
    print("calculate propensity score begin")
    # load origin data
    train_data = np.loadtxt('train.txt')
    train_data = train_data.astype(int)
    test_data = np.loadtxt('test.txt')
    test_data = test_data.astype(int)

    # calculate num of user and item
    user_num = max(train_data[:, 0])
    item_num = max(train_data[:, 1])

    # calculate different rating num and normalize it (train data)
    P_O_Y = np.bincount(train_data[:, 2])[1:]
    P_O_Y = P_O_Y / P_O_Y.sum()

    # sample 5% of test data to calculate P_Y
    np.random.seed(2021)
    df = pd.DataFrame(test_data)
    test1 = df.sample(frac=0.05, replace=False)
    test2 = df.append(test1)
    test2 = test2.drop_duplicates(keep=False)
    test1 = np.array(test1).astype(int)
    test2 = np.array(test2).astype(int)
    P_Y = np.bincount(test1[:, 2])[1:]
    P_Y = P_Y / P_Y.sum()

    # calculate Propensity Score with the formula
    P = P_O_Y * train_data.shape[0] / (user_num * item_num) / P_Y
    np.savetxt('propensity_score.txt', P)
    np.savetxt('test1.txt', test1, fmt="%d")
    np.savetxt('test2.txt', test2, fmt="%d")
    print('propensity score for different rating', P)
    print('calculate propensity score end')

