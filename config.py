# coding:utf8
import warnings
import torch


class DefaultConfig(object):
    model = 'MF_Naive'  # 使用的模型，与models/__init__.py中的名字一致
    is_eval_ips = False
    train_data = './data/yahoo/train.txt'  # 训练集存放路径
    test_data = './data/yahoo/test2.txt'  # 测试集存放路径
    s_c_data = './data/yahoo/train.txt'
    s_t_data = './data/yahoo/test1.txt'
    reg_c = 0.001
    reg_t = 0.001
    reg_tc = 0.001

    propensity_score = './data/yahoo/propensity_score.txt'

    device = 'cpu'
    batch_size = 128  # batch size
    embedding_size = 64

    # debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    # result_file = 'result.csv'

    max_epoch = 50
    verbose = 2
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-5


opt = DefaultConfig()
