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
    print('train_1 begin')

    # collect data
    train_data = Yahoo(opt.train_data)
    val_data = Yahoo(opt.test_data)
    if opt.is_ips:
        inverse_propensity = np.reciprocal(np.loadtxt(opt.propensity_score))
    else:
        inverse_propensity = np.ones(5)
    train_dataloader = DataLoader(train_data, opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers)

    # get model
    model = getattr(models, opt.model)(train_data.users_num + 1, train_data.items_num + 1,
                                       opt.embedding_size, inverse_propensity)
    model.to(opt.device)
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    best_mse = 10000.
    best_mae = 10000.
    best_iter = 0

    # train
    model.train()
    for epoch in range(opt.max_epoch):
        print('epoch', epoch)
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

        if epoch == 100:
            print('hello')
        t2 = time()
        if epoch % opt.verbose == 0:
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
    if is_ips:
        inverse_propensity = np.reciprocal(np.loadtxt(opt.propensity_score))
    else:
        inverse_propensity = np.ones(5)
    (mae, mse, rmse) = evaluate_model(model, test_data, inverse_propensity, opt)
    print('MAE = %.4f, MSE = %.4f, RMSE = %.4f' % (mae, mse, rmse))


def train_CausEProd():
    # train model 'CausEProd'
    print('train_1 begin')

    # collect data
    train_data = Yahoo2(opt.s_c_data, opt.s_t_data)
    val_data = Yahoo(opt.test_data)

    train_dataloader = DataLoader(train_data, opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers)

    # get model
    model = getattr(models, opt.model)(train_data.users_num + 1, train_data.items_num + 1,
                                       opt.embedding_size, opt.reg_c, opt.reg_c, opt.reg_tc,
                                       train_data.s_c, train_data.s_t)
    model.to(opt.device)
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    best_mse = 10000.
    best_mae = 10000.
    best_iter = 0

    # train
    model.train()
    for epoch in range(opt.max_epoch):
        print('epoch', epoch)
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

        if epoch == 100:
            print('hello')
        t2 = time()
        if epoch % opt.verbose == 0:
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
    parser.add_argument('--model', default='CausEProd')
    parser.add_argument('--is_ips', default=False)
    args = parser.parse_args()
    opt.model = args.model
    if opt.model == 'CausEProd':
        args.is_ips = False
    opt.is_ips = args.is_ips

    # print config
    print(opt)

    if opt.model == 'MF_Naive' or opt.model == 'MF_IPS':
        model = train_MF_Naive_or_MF_IPS()
    elif opt.model == 'CausEProd':
        model = train_CausEProd()
    print('end')
