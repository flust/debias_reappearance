# coding:utf8
import warnings
import torch


class DefaultConfig(object):
    model = 'MF_Naive'  # 使用的模型，与models/__init__.py中的名字一致
    is_eval_ips = False

    train_data = './data/yahoo/1/train.txt'
    test_data = './data/yahoo/1/test2.txt'

    s_c_data = './data/yahoo/2/train.txt'
    s_t_data = './data/yahoo/2/test1.txt'

    S_c_data = './data/yahoo/3/S_c.txt'
    S_t_data = './data/yahoo/3/S_t.txt'
    S_te_data = './data/yahoo/3/S_te.txt'
    S_va_data = './data/yahoo/3/S_va.txt'

    reg_c = 0.001
    reg_t = 0.001
    reg_tc = 0.001

    propensity_score_1 = './data/yahoo/1/propensity_score.txt'
    propensity_score_3 = './data/yahoo/3/propensity_score.txt'

    device = 'cpu'
    batch_size = 128  # batch size
    embedding_size = 16

    # debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    # result_file = 'result.csv'

    max_epoch = 2
    verbose = 10
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-5


opt = DefaultConfig()
