import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from torch.nn.init import normal_

TRAIN_DATA_PATH = './../raw_data/train.txt'
TEST_DATA_PATH = './../raw_data/test.txt'
S_c_FILE = 'S_c.txt'  #
S_t_FILE = 'S_t.txt'
S_va_FILE = 'S_va.txt'
S_te_FILE = 'S_te.txt'

TRAIN = True
PREPARE_S_c = True
PREPARE_S_t = True
DEVICE = 'cpu'
BATCH_SIZE = 128  # batch size
EMBEDDING_SIZE = 64
MAX_EPOCH = 4
VERBOSE = 1
LR = 0.002  # initial learning rate
LR_DECAY = 0.5  # when val_loss increase, lr = lr*lr_decay
WEIGHT_DECAY = 1e-5


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


class MF_Naive(nn.Module):
    def __init__(self, num_users, num_items, embedding_size):
        super(MF_Naive, self).__init__()

        self.num_users = num_users
        self.num_items = num_items

        self.user_e = nn.Embedding(self.num_users, embedding_size)
        self.item_e = nn.Embedding(self.num_items, embedding_size)
        self.user_b = nn.Embedding(self.num_users, 1)
        self.item_b = nn.Embedding(self.num_items, 1)

        self.apply(self._init_weights)

        self.loss = nn.MSELoss()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.1)

    def forward(self, user, item):
        user = user - 1
        item = item - 1
        user_embedding = self.user_e(user)
        item_embedding = self.item_e(item)

        preds = self.user_b(user)
        preds += self.item_b(item)
        preds += (user_embedding * item_embedding).sum(dim=1, keepdim=True)

        return preds.squeeze()

    def calculate_loss(self, user_list, item_list, label_list):
        return self.loss(self.forward(user_list, item_list), label_list)

    def predict(self, user, item):
        return self.forward(user, item)

    def get_optimizer(self, lr, weight_decay):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


def MAE(preds, true):
    absError = []
    for i in range(len(preds)):
        val = true[i] - preds[i]
        absError.append(abs(val))  # 误差绝对值
    return sum(absError) / len(absError)


def MSE(preds, true):
    squaredError = []
    for i in range(len(preds)):
        val = true[i] - preds[i]
        squaredError.append(val * val)  # target-prediction之差平方
    return sum(squaredError) / len(squaredError)


def RMSE(preds, true):
    squaredError = []
    absError = []
    for i in range(len(preds)):
        val = true[i] - preds[i]
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值
    from math import sqrt
    return sqrt(sum(squaredError) / len(squaredError))


def evaluate_model(model, val_data):
    true = val_data[:, 2]
    user = torch.LongTensor(val_data[:, 0]).to(DEVICE)
    item = torch.LongTensor(val_data[:, 1]).to(DEVICE)
    preds = model.predict(user, item).to(DEVICE)
    #print(preds)

    mae = MAE(preds, true)
    mse = MSE(preds, true)
    rmse = RMSE(preds, true)
    return mae, mse, rmse


if __name__ == '__main__':
    train_data = Yahoo(TRAIN_DATA_PATH)
    train_dataloader = DataLoader(train_data, BATCH_SIZE)

    model = MF_Naive(train_data.users_num, train_data.items_num, EMBEDDING_SIZE)
    model.to(DEVICE)

    optimizer = model.get_optimizer(LR, WEIGHT_DECAY)

    model.train()

    best_mse = 10000.
    best_mae = 10000.
    best_iter = 0

    if TRAIN:
        for epoch in range(MAX_EPOCH):
            print('epoch:', epoch)
            t1 = time()
            for i, data in tqdm(enumerate(train_dataloader)):
                user = data[:, 0].to(DEVICE)
                item = data[:, 1].to(DEVICE)
                label = data[:, 2].to(DEVICE)

                loss = model.calculate_loss(user.long(), item.long(), label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(loss.item())
            t2 = time()

            if epoch % VERBOSE == 0:
                (mae, mse, rmse) = evaluate_model(model, train_data)
                print('Iteration %d[%.1f s]: MAE = %.4f, MSE = %.4f, RMSE = %.4f [%.1f s]'
                      % (epoch, t2 - t1, mae, mse, rmse, time() - t2))
                if mae < best_mae:
                    best_mae, best_mse, best_iter = mae, mse, epoch
                    torch.save(model.state_dict(), str(type(model)) + "-dataprepare.pth")

    if PREPARE_S_c:
        model.load_state_dict(torch.load(str(type(model)) + "-dataprepare.pth"))

        interaction_num = 0
        with open(S_c_FILE, 'ab') as f:

            user_list = train_data.data[:, 0]
            item_list = train_data.data[:, 1]
            label_list = train_data.data[:, 2]
            pred_list = model.predict(torch.LongTensor(user_list), torch.LongTensor(item_list)).detach().numpy()
            # print(interaction.shape, label_list.shape)
            target = train_data.data[pred_list > 2.5, :]

            # target[target[:, 2] != 5, 2] = -1
            target[target[:, 2] != 5, 2] = 0
            target[target[:, 2] == 5, 2] = 1

            interaction_num += target.shape[0]
            # print(target.shape)
            np.savetxt(f, target, fmt=['%d', '%d', '%d'])
        print(interaction_num)

    if PREPARE_S_t:
        test_data = np.loadtxt(TEST_DATA_PATH)
        test_data = test_data.astype(int)
        df = pd.DataFrame(test_data)
        S_t = df.sample(n=2700, replace=False)

        df = df.append(S_t)
        df = df.drop_duplicates(keep=False)
        S_va = df.sample(n=2700, replace=False)

        df = df.append(S_va)
        S_te = df.drop_duplicates(keep=False)

        S_t = S_t.values
        # S_t[S_t[:, 2] != 5, 2] = -1
        S_t[S_t[:, 2] != 5, 2] = 0
        S_t[S_t[:, 2] == 5, 2] = 1
        S_va = S_va.values
        # S_va[S_va[:, 2] != 5, 2] = -1
        S_va[S_va[:, 2] != 5, 2] = 0
        S_va[S_va[:, 2] == 5, 2] = 1
        S_te = S_te.values
        # S_te[S_te[:, 2] != 5, 2] = -1
        S_te[S_te[:, 2] != 5, 2] = 0
        S_te[S_te[:, 2] == 5, 2] = 1

        print(S_t.shape)
        print(S_va.shape)
        print(S_te.shape)

        np.savetxt(S_t_FILE, S_t, fmt='%d')
        np.savetxt(S_va_FILE, S_va, fmt='%d')
        np.savetxt(S_te_FILE, S_te, fmt='%d')
