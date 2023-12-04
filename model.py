import math
import torch
from torch import nn
from d2l import torch as d2l
from Data_iter import data_load_iter


## 组成transformer编码器的基础组件
# 基于位置的前馈网络：对序列中的所有位置变换时使用的是同一个多层感知机（MLP）
# 输入 X 形状（批量大小， 时间步长或序列长度， 隐单元数或序列维度）
# 将北与各两层的感知机转成形状维（批量大小， 时间步数， ffn_num_outputs）的输出张量。
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


# 残差连接和层规范化
class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


## 编码器
# 编码器中的一个层 即一个【transformer block】
# transformer编码器中的任何层都不会改变其输入的形状
class EncoderBlock(nn.Module):
    """transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


# 【transformer编码器】
# 堆叠了 num_Layers 个 EncoderBlock 类的实例。
# 这里将embddding层改成一个全连接层  映射到隐层大小：因为输入的不是词向量，而是一个表示特征的向量
class ATN(d2l.Encoder):
    """transformer编码器"""
    def __init__(self, num_feature, key_size, query_size, value_size,
                 num_hiddens, num_classes, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(ATN, self).__init__(**kwargs)
        self.Model_Name = 'Attention'
        self.num_classes = num_classes
        self.num_hiddens = num_hiddens
        self.embedding = nn.Linear(num_feature, num_hiddens)         # 将embedding改成linear层  因为输入的数据本身就带有特征值
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)     # 计算位置编码，和embedding code相加，再做一个dropout
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

        self.to_latent = nn.Identity()    # 占位 输出结构 较 输出结构不变： 便于后续扩展网络结构

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, num_classes)
        )

    def forward(self, X, valid_lens, getFeature=False):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)    # 为了可视化每个block的注意力权重
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        # meaning pooling
        X = X.mean(dim=1)
        X = self.to_latent(X)

        if (getFeature):
            features = self.mlp_head[0](X)
            return features, self.mlp_head[1](X)

        return self.mlp_head(X)


if __name__ == '__main__':
    # Parameters
    num_classes = 4
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 4, 0.1, 64, 100
    lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size = 32, 32, 32
    norm_shape = [32]


    train_iter, eval_iter, test_iter, num_features = data_load_iter(num_steps, batch_size)
    net = ATN(
        num_features, key_size, query_size, value_size, num_hiddens, num_classes,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)


    loss = nn.CrossEntropyLoss()
    for X, Y in train_iter:
        valid_len = torch.ones([batch_size]) * 100
        X_out = net(X, valid_len)
        print(X.shape, X_out.shape)
        l = loss(X_out, Y.long())
        break