import math
import os.path
import time
import xlsxwriter
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pdb

import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
from Data_iter import data_load_iter
from Attention import TransformerEncoder

train_recording_path = './train_recordings'
if not os.path.isdir(train_recording_path):
    os.mkdir(train_recording_path)
timestamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
ModelPath = os.path.join(train_recording_path, f'{timestamp}.pt')


# 更新混淆矩阵
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def evaluate_accuracy(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            valid_len = (torch.ones([X.shape[0]]) * X.shape[1]).to(device)
            metric.add(d2l.accuracy(net(X, valid_len), y), d2l.size(y))
    return metric[0] / metric[1]


def train_attention(net, train_iter, eval_iter, num_epochs, lr, device):
    """注意力时间序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    net.train()

    # 创建保存结果的目录和文件
    file_path = os.path.join(train_recording_path, timestamp+r'.xlsx')
    recordexcel = xlsxwriter.Workbook(file_path)
    worksheet = recordexcel.add_worksheet()
    worksheet.write_row(0, 0, ('epoch', 'train loss', 'train acc', 'test acc'))

    best_acc = 0.1
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['0', 'train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # 训练损失总和，词元数量
        for i, batch in enumerate(train_iter):
            optimizer.zero_grad()
            X, Y = [x.to(device) for x in batch]
            valid_len = (torch.ones([X.shape[0]]) * X.shape[1]).to(device)
            Y_hat = net(X, valid_len)
            l = loss(Y_hat, Y.long())
            l.backward()	# 损失函数的标量进行“反向传播”
            d2l.grad_clipping(net, theta=1)     # 梯度裁剪
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(Y_hat, Y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

        eval_acc = evaluate_accuracy(net, eval_iter)
        # 保存训练loss和acc到excel
        worksheet.write_row(epoch + 1, 0, (epoch, train_l, train_acc, eval_acc))
        # 保存模型
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(net.state_dict(), ModelPath)

        if (epoch + 1) % 5 == 0:
            print(f'epoch {epoch+1}: '
                  f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
                  f'eval acc {eval_acc:.3f}')
            animator.add(epoch + 1, (train_l, train_acc, eval_acc))
            d2l.plt.pause(0.1)

    recordexcel.close()

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'eval acc {eval_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
    d2l.plt.savefig(r'result.png')
    d2l.plt.show()


def test_accuracy(net, data_iter, device=None):
    """Compute the Evaluation indicators for a model on a dataset using a GPU."""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    net.to(device)
    metric = d2l.Accumulator(2)
    Scorelist = []
    FeatureList = []
    conf_martix = torch.zeros(net.num_classes, net.num_classes)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            valid_len = (torch.ones([X.shape[0]]) * X.shape[1]).to(device)
            # softmax计算每个class得分   并计算每个样本的预测结果
            features, score = net(X, valid_len, getFeature=True)
            score = nn.functional.softmax(score, dim=1)
            # pdb.set_trace()
            Scorelist.append(score)
            FeatureList.append(features)
            conf_martix = confusion_matrix(score.argmax(dim=1, keepdim=True), y, conf_martix)
            metric.add(d2l.accuracy(net(X, valid_len), y), d2l.size(y))
        Scorelist = torch.cat(Scorelist, dim=0)
        FeatureList = torch.cat(FeatureList, dim=0)
        conf_martix = conf_martix.long()
        # Confusion Matrix


    return metric[0] / metric[1], Scorelist, FeatureList, conf_martix

def plot_confusion_matrix(cm, classes=["Healthy Co", "Severity2", "Severity2.5", "Severity3"],
                          name='confusion',
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    #    plt.title(title,fontsize=12)
    #    plt.colorbar()
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # 通过绘制格网，模拟每个单元格的边框
    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 将x轴上的lables旋转45度
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
             rotation_mode="anchor", fontsize=8)

    #    tick_marks = np.arange(len(classes))
    #    ax.xticks(tick_marks, classes, rotation=45,fontsize=12)
    #    ax.yticks(np.arange(0,len(classes),1), classes,fontsize=12)
    a = []
    cm_normalize = cm
    thresh = cm.max() * 0.5
    #
    for i, j in itertools.product(range(cm_normalize.shape[0]), range(cm_normalize.shape[1])):
        ax.text(j, i, str(round(cm_normalize[i, j] * 100, 2)) + '%',
                horizontalalignment="center", verticalalignment="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=8)
        if i == j:
            a.append(round(cm_normalize[i, j] * 100, 2))

    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    fig.tight_layout()

    plt.savefig("{}.png".format(name))
    plt.show()
    return (np.sum(a) / len(classes) / 100)

