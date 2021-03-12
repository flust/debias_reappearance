from config import opt
import os
import torch as t
import models
from data.dataset import Yahoo1, Yahoo2, Yahoo3
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
    train_data = Yahoo1(opt.train_data)
    val_data = Yahoo1(opt.test_data)
    if opt.model == 'MF_IPS':
        inverse_propensity = np.reciprocal(np.loadtxt(opt.propensity_score_1))
    else:
        inverse_propensity = np.ones(5)
    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True)

    # get model
    model = getattr(models, opt.model)(train_data.users_num, train_data.items_num,
                                       opt.embedding_size, inverse_propensity, opt.device)
    model.to(opt.device)
    optimizer = model.get_optimizer(opt.lr, opt.weight_decay)

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
    test_data = Yahoo1(opt.test_data)
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
    val_data = Yahoo2(opt.test_data)

    train_dataloader_s_c = DataLoader(train_data.s_c, opt.batch_size, shuffle=True)
    train_dataloader_s_t = DataLoader(train_data.s_t, opt.batch_size, shuffle=True)
    # get model
    model = getattr(models, opt.model)(train_data.users_num, train_data.items_num,
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


def train_FFM(data, train_set, model='FM', inverse_propensity=None):
    # actually FM
    if train_set == 'S_c':
        train_dataloader = DataLoader(data.S_c, opt.batch_size, shuffle=True)
    elif train_set == 'S_t':
        train_dataloader = DataLoader(data.S_t, opt.batch_size, shuffle=True)
    elif train_set == 'S_ct':
        train_dataloader = DataLoader(np.vstack([data.S_t, data.S_c]), opt.batch_size, shuffle=True)

    model = getattr(models, 'FM')([data.users_num, data.items_num], opt.embedding_size,
                                  inverse_propensity, opt.device)
    model.to(opt.device)
    optimizer = model.get_optimizer(opt.lr, opt.weight_decay)
    model.train()
    best_NLL = 0.
    best_AUC = 0.
    for epoch in range(opt.max_epoch):
        t1 = time()
        # for i, data_i in tqdm(enumerate(train_dataloader)):
        for i, data_i in enumerate(train_dataloader):
            # train model
            user = data_i[:, 0].to(opt.device)
            item = data_i[:, 1].to(opt.device)
            label = data_i[:, 2].to(opt.device)
            loss = model.calculate_loss(user.long(), item.long(), label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t2 = time()
        if epoch % opt.verbose == 0:
            print('epoch', epoch)
            true = data.S_te[:, 2]
            preds = model.predict(torch.LongTensor(data.S_te[:, 0]),
                                  torch.LongTensor(data.S_te[:, 1])).detach().numpy()

            nll, auc = evaluate_model_3(true, preds)
            print('Iteration %d[%.1f s]: NLL = %.4f, AUC = %.4f [%.1f s]'
                  % (epoch, t2 - t1, nll, auc, time() - t2))

            if auc > best_AUC:
                best_NLL, best_AUC = nll, auc
                torch.save(model.state_dict(), str(type(model)) + "-model3.pth")

    # print("End. Best Iteration %d:  MAE = %.4f, MSE = %.4f. " % (best_iter, best_mae, best_mse))
    return best_NLL, best_AUC


def train_New():
    def calculate_CTR(col):
        return (len(col) + sum(col)) / 2 / len(col)

    data = Yahoo3(opt.S_c_data, opt.S_t_data, opt.S_te_data, opt.S_va_data)

    # ------ average(S_c) ------
    print("==================== average(S_c) ====================")
    ave_S_c = calculate_CTR(data.S_c[:, 2])
    print("average(S_c): CTR:", ave_S_c)

    ave_S_c_NLL, ave_S_c_AUC = evaluate_model_3(data.S_te[:, -1], np.repeat(ave_S_c, data.S_te.shape[0]))
    print('average(S_c): NLL:', ave_S_c_NLL, 'AUC: ', ave_S_c_AUC)

    # ------ average(S_t) ------
    print("==================== average(S_t) ====================")
    ave_S_t = calculate_CTR(data.S_t[:, 2])
    print("average(S_t): CTR:", ave_S_t)

    ave_S_t_NLL, ave_S_t_AUC = evaluate_model_3(data.S_te[:, -1], np.repeat(ave_S_t, data.S_te.shape[0]))
    print('average(S_t): NLL:', ave_S_t_NLL, 'AUC: ', ave_S_t_AUC)

    # ------ FFM(S_c) ------
    print("==================== FFM(S_c) ====================")
    FFM_S_c_NLL, FFM_S_c_AUC = train_FFM(data, 'S_c', 'FM')
    print('FFM(S_c): NLL:', FFM_S_c_NLL, 'AUC: ', FFM_S_c_AUC)

    # ------ FFM(S_t) ------
    print("==================== FFM(S_t) ====================")
    FFM_S_t_NLL, FFM_S_t_AUC = train_FFM(data, 'S_t', 'FM')
    print('FFM(S_t): NLL:', FFM_S_t_NLL, 'AUC: ', FFM_S_t_AUC)

    # ------ FFM(S_c & S_t) ------
    print("==================== FFM(S_ct) ====================")
    FFM_S_ct_NLL, FFM_S_ct_AUC = train_FFM(data, 'S_ct', 'FM')
    print('FFM(S_ct): NLL:', FFM_S_ct_NLL, 'AUC: ', FFM_S_ct_AUC)

    # ------ IPS ------
    print("==================== IPS ====================")
    inverse_propensity = np.reciprocal(np.loadtxt(opt.propensity_score_3))
    IPS_NLL, IPS_AUC = train_FFM(data, 'S_ct', 'FM_IPS', inverse_propensity)
    print('IPS: NLL:', IPS_NLL, 'AUC: ', IPS_AUC)

    # ------ CausE ------


    # ------ New(avg) ------


    # ------ New(item-avg) ------


    # ------ New(complex) ------



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo of argparse")
    # parser.add_argument('--model', default='MF_Naive') # 1.0210 1.7074
    # parser.add_argument('--model', default='MF_IPS') # 0.8750 1.3433    0.8739 1.3451
    # parser.add_argument('--model', default='CausEProd') # 1.0371 1.4658
    parser.add_argument('--model', default='New')

    args = parser.parse_args()
    opt.model = args.model

    # print config
    print('\n'.join(['%s:%s' % item for item in opt.__dict__.items()]))

    if opt.model == 'MF_Naive' or opt.model == 'MF_IPS':
        opt.train_data = './data/yahoo/1/train.txt'
        opt.test_data = './data/yahoo/1/test2.txt'
        model = train_MF_Naive_or_MF_IPS()
    elif opt.model == 'CausEProd':
        opt.s_c_data = './data/yahoo/2/train.txt'
        opt.s_t_data = './data/yahoo/2/test1.txt'
        model = train_CausEProd()
    elif opt.model == 'New':
        model = train_New()
    print('end')
