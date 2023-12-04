import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd

# Parameters
HIDDEN_SIZE = 256
BATCH_SIZE = 64
NUM_STEPS = 100
N_LAYER = 2
N_EPOCHS = 100
N_FEATURE = 128
USE_GPU = False

# Reading and preprocessing Data
def Reading_and_Processing(filenames, num_steps):

    # 从txt文件中读取数据
    def read_data(filename):
        """载入“nnds”数据集"""
        return np.loadtxt(open(filename,"rb"),delimiter=",")

    # 预处理数据 将读入的数据按行切割成列表
    def preprocess(data, num_examples=None):
        return torch.tensor(data, dtype=torch.float)

    raw_data = read_data(filenames)
    x_raw = preprocess(raw_data[:, :-1])
    y_raw = preprocess(raw_data[:, -1])

    return x_raw, (y_raw).long()-1   # 计算损失时label值要求是long


# Split
# 分层采样 划分数据集：保证训练集中既包含一定比例的正样本又要包含一定比例的负样本
def split_dataset(X_all, Y_all, train_rate=0.6, eval_rate=0.2):
    # n_splits: the number of splitting iterations in the cross-validator
    split_train = StratifiedShuffleSplit(n_splits=1, test_size=1-train_rate, random_state=0)
    split_eval_test = StratifiedShuffleSplit(n_splits=1, test_size=1-train_rate-eval_rate, random_state=0)

    for train_index, eval_test_index in split_train.split(X_all, Y_all):
        train_data, eval_test_data = X_all[train_index], X_all[eval_test_index]
        train_label, eval_test_label = Y_all[train_index], Y_all[eval_test_index]
    for eval_index, test_index in split_eval_test.split(eval_test_data, eval_test_label):
        eval_data, test_data = eval_test_data[eval_index], eval_test_data[test_index]
        eval_label, test_label = eval_test_label[eval_index], eval_test_label[test_index]

    return [[train_data, train_label],
            [eval_data, eval_label],
            [test_data, test_label]]


# Preparing Data
class ParkinsonDataset(Dataset):
    def __init__(self, num_steps, data):
        x_raw, y_raw = data
        self.num_steps = num_steps
        self.len = y_raw.shape[0]
        self.x_data = self.norm_reshape(x_raw, num_steps)
        self.y_data = y_raw

    def __getitem__(self, index):
        # Save countries and its index in list and dictionary
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    # 对将特征数据转换为 sample_num, num_step, num_feature的维度
    # 对特征值做归一化
    def norm_reshape(self, raw_data, num_steps):
        """shape of raw_data: sample_num , num_step * num_features"""
        # 计算   特征维度
        num_features = int(raw_data.shape[1] / num_steps)
        # 转换raw_data维度， 取出样本特征的第一维（当前时刻）
        data = raw_data.reshape(-1, num_features)
        # 对特征做归一化处理
        # 将所有特征放在一个共同的尺度上， 我们(通过将特征重新缩放到零均值和单位方差来标准化数据)
        data = (data - torch.mean(data, dim=0)) / torch.std(data, dim=0)
        # 转换维度：sample_num, num_step, num_feature的维度
        return data.reshape(raw_data.shape[0], -1, num_features)

# Prepare Dataset and DataLoader
def data_load_iter(num_steps, batch_size, is_to_split=True, train_rate=0.6, eval_rate=0.2):
    filenames = r'../dataset/Gait_dataset.csv'
    x_all, y_all = Reading_and_Processing(filenames, num_steps)

    datasets = split_dataset(x_all, y_all)
    train_data, eval_data, test_data = [ParkinsonDataset(num_steps, data) for data in datasets]

    train_iter = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=True)
    eval_iter = DataLoader(dataset=eval_data,
                           batch_size=batch_size,
                           shuffle=False)
    test_iter = DataLoader(dataset=eval_data,
                           batch_size=batch_size,
                           shuffle=False)
    all_iter = DataLoader(dataset=ParkinsonDataset(num_steps, [x_all, y_all]),
                         batch_size=batch_size,
                         shuffle=False)

    num_features = train_data.x_data.shape[-1]
    return train_iter, eval_iter, test_iter, all_iter, num_features


if __name__ == '__main__':
    train_iter, eval_iter, test_iter, all_iter, num_features = data_load_iter(num_steps=100, batch_size=64)

    for x, y in all_iter:
        print(x.shape, y.shape)
        print(y[:10])
        print(y.dtype)