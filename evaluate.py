from metrics import MSE, MAE, RMSE, MSE_ips, MAE_ips, RMSE_ips, AUC, NLL
import torch


def evaluate_model(model, val_data, inverse_propensity, opt):
    true = val_data[:, 2]
    user = torch.LongTensor(val_data[:, 0]).to(opt.device)
    user_num = max(user)
    item = torch.LongTensor(val_data[:, 1]).to(opt.device)
    item_num = max(item)
    preds = model.predict(user, item).to(opt.device)
    #print(preds)

    if opt.is_eval_ips:
        mae = MAE_ips(preds, true, user_num, item_num, inverse_propensity)
        mse = MSE_ips(preds, true, user_num, item_num, inverse_propensity)
        rmse = RMSE_ips(preds, true, user_num, item_num, inverse_propensity)
    else:
        mae = MAE(preds, true)
        mse = MSE(preds, true)
        rmse = RMSE(preds, true)
    return mae, mse, rmse


def evaluate_model_3(true, preds):
    return NLL(true, preds), AUC(true, preds)