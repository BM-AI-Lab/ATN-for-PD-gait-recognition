import torch
import numpy as np
from d2l import torch as d2l
from sklearn.model_selection import StratifiedShuffleSplit
from data import data_load_iter
from model import ATN
from tools import train_attention, test_accuracy, plot_confusion_matrix


if __name__ == "__main__":
    num_classes = 2
    num_hiddens, num_layers, dropout, batch_size, num_steps = 64, 2, 0.4, 256, 100
    lr, num_epochs, device = 0.003, 100, d2l.try_gpu()
    ffn_num_input, ffn_num_hiddens, num_heads = 64, 128, 4
    key_size, query_size, value_size = 64, 64, 64
    norm_shape = [64]

    train_iter, eval_iter, test_iter, num_features = data_load_iter(num_steps, batch_size)
    net = ATN(num_features, key_size, query_size, value_size, num_hiddens, num_classes,
                             norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                             num_layers, dropout)
    train_attention(net, train_iter, eval_iter, num_epochs, lr, device)
    ModelPath = f'{net.Model_Name}_layer{num_layers}_{num_hiddens}.pt'
    torch.save(net.state_dict(), ModelPath)

    # 预测
    accuracy, Scorelist, _, conf_matrix = test_accuracy(net, test_iter, device=None)
    print(accuracy)
    print(conf_matrix.numpy())

    # 计算评价指标 画混淆矩阵
    cm = conf_matrix.numpy().astype('float') / conf_matrix.numpy().sum(axis=1)[:, np.newaxis]
    # plot_confusion_matrix(cm)