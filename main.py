from config import opt
import os
import torch as t
import models
from data.dataset import Yahoo, Yahoo2
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time
from evaluate import *
import numpy as np
import argparse


def train_MF_Naive_or_MF_IPS():
    # train model 'MF_Naive' or 'MF_IPS'
    print('train_MF_Naive_or_MF_IPS begin')

    # collect data
    train_data = Yahoo(opt.train_data)
    val_data = Yahoo(opt.test_data)
    if opt.model == 'MF_IPS':
        inverse_propensity = np.reciprocal(np.loadtxt(opt.propensity_score))
    else:
        inverse_propensity = np.ones(5)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True)

    # get model
    model = getattr(models, opt.model)(train_data.users_num + 1, train_data.items_num + 1,
                                       opt.embedding_size, inverse_propensity, opt.device)
    model.to(opt.device)
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    best_mse = 10000.
    best_mae = 10000.
    best_iter = 0

    # train
    model.train()
    for epoch in range(opt.max_epoch):
        t1 = time()
        for i, data in tqdm(enumerate(train_dataloader)):
            # train model
            user = data[:, 0].to(opt.device)
            item = data[:, 1].to(opt.device)
            label = data[:, 2].to(opt.device)

            loss = model.calculate_loss(user.long(), item.long(), label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Loss:', loss.item())
        t2 = time()
        if epoch % opt.verbose == 0:
            print('epoch', epoch)

            (mae, mse, rmse) = evaluate_model(model, val_data, inverse_propensity, opt)
            print('Iteration %d[%.1f s]: MAE = %.4f, MSE = %.4f, RMSE = %.4f [%.1f s]'
                  % (epoch, t2 - t1, mae, mse, rmse, time() - t2))
            if mae < best_mae:
                best_mae, best_mse, best_iter = mae, mse, epoch
                torch.save(model.state_dict(), str(type(model)) + "-model2.pth")

    print("End. Best Iteration %d:  MAE = %.4f, MSE = %.4f. " % (best_iter, best_mae, best_mse))
    return model


def test_1(model, test_data):
    # test model 'MF_Naive' or 'MF_IPS'
    test_data = Yahoo(opt.test_data)
    if opt.model == 'MF_IPS':
        inverse_propensity = np.reciprocal(np.loadtxt(opt.propensity_score))
    else:
        inverse_propensity = np.ones(5)
    (mae, mse, rmse) = evaluate_model(model, test_data, inverse_propensity, opt)
    print('MAE = %.4f, MSE = %.4f, RMSE = %.4f' % (mae, mse, rmse))


def train_CausEProd():
    # train model 'CausEProd'
    print('train_CausEProd begin')

    # collect data
    train_data = Yahoo2(opt.s_c_data, opt.s_t_data)
    val_data = Yahoo(opt.test_data)

    train_dataloader_s_c = DataLoader(train_data.s_c, opt.batch_size, shuffle=True)
    train_dataloader_s_t = DataLoader(train_data.s_t, opt.batch_size, shuffle=True)
    # get model
    model = getattr(models, opt.model)(train_data.users_num + 1, train_data.items_num + 1,
                                       opt.embedding_size, opt.reg_c, opt.reg_c, opt.reg_tc,
                                       train_data.s_c[:, :2].tolist(), train_data.s_t[:, :2].tolist())
    model.to(opt.device)
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    best_mse = 10000.
    best_mae = 10000.
    best_iter = 0

    # train
    model.train()
    for epoch in range(opt.max_epoch):
        t1 = time()
        for i, data in tqdm(enumerate(train_dataloader_s_c)):
            # train model
            user = data[:, 0].to(opt.device)
            item = data[:, 1].to(opt.device)
            label = data[:, 2].to(opt.device)

            loss = model.calculate_loss(user.long(), item.long(), label.float(), control=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Loss:', loss.item())
        for i, data in tqdm(enumerate(train_dataloader_s_t)):
            # train model
            user = data[:, 0].to(opt.device)
            item = data[:, 1].to(opt.device)
            label = data[:, 2].to(opt.device)

            loss = model.calculate_loss(user.long(), item.long(), label.float(), control=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t2 = time()
        print('Loss:', loss.item())
        if epoch % opt.verbose == 0:
            print('epoch', epoch)
            (mae, mse, rmse) = evaluate_model(model, val_data, None, opt)
            print('Iteration %d[%.1f s]: MAE = %.4f, MSE = %.4f, RMSE = %.4f [%.1f s]'
                  % (epoch, t2 - t1, mae, mse, rmse, time() - t2))
            if mae < best_mae:
                best_mae, best_mse, best_iter = mae, mse, epoch
                torch.save(model.state_dict(), str(type(model)) + "-model2.pth")

    print("End. Best Iteration %d:  MAE = %.4f, MSE = %.4f. " % (best_iter, best_mae, best_mse))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo of argparse")
    # parser.add_argument('--model', default='MF_Naive') # 1.0210 1.7074
    # parser.add_argument('--model', default='MF_IPS') # 0.8750 1.3433    0.8739 1.3451
    parser.add_argument('--model', default='CausEProd') # 1.0371 1.4658

    args = parser.parse_args()
    opt.model = args.model

    # print config
    print('\n'.join(['%s:%s' % item for item in opt.__dict__.items()]))

    if opt.model == 'MF_Naive' or opt.model == 'MF_IPS':
        model = train_MF_Naive_or_MF_IPS()
    elif opt.model == 'CausEProd':
        model = train_CausEProd()
    print('end')
