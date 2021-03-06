# coding:utf8
import warnings
import torch


class DefaultConfig(object):
    model = 'MF_IPS'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    is_ips = True
    train_data = './data/yahoo/train.txt'  # 训练集存放路径
    test_data = './data/yahoo/test2.txt'  # 测试集存放路径

    propensity_score = './data/yahoo/propensity_score.txt'

    device = 'cpu'
    batch_size = 32  # batch size
    use_gpu = False  # user GPU or not
    num_workers = 0  # how many workers for loading data
    embedding_size = 64

    # debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    # result_file = 'result.csv'

    max_epoch = 100
    verbose = 10
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-5  # 损失函数

opt = DefaultConfig()
