import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn import preprocessing

dtype = torch.FloatTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # 位置编码层
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # # 使用正弦函数来表示奇数编号的子向量
        pe[:, 0::2] = torch.sin(position * div_term)
        # # 使用余弦函数来表示偶数编号的子向量
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class scMoAnno(nn.Module):
    def __init__(self, input_rna, input_atac, gap, dropout, num_classes, n_head):
        super(scMoAnno, self).__init__()
        self.input_rna = input_rna
        self.input_atac = input_atac
        self.gap = gap
        self.num_classes = num_classes
        self.n_head = n_head
        self.dropout = nn.Dropout(p=dropout)

        # 具有自注意力的 Transformer 编码器层
        self.encoder_layer_rna = nn.TransformerEncoderLayer(
            d_model=gap, dim_feedforward=1024, nhead=n_head, dropout=dropout
        )

        self.encoder_layer_atac = nn.TransformerEncoderLayer(
            d_model=gap, dim_feedforward=1024, nhead=n_head, dropout=dropout
        )

        # 带有自注意力的位置编码
        self.positionalEncoding_rna = PositionalEncoding(d_model=gap, dropout=dropout)
        self.positionalEncoding_atac = PositionalEncoding(d_model=gap, dropout=dropout)

        # 分类层
        # self.pred_layer = nn.Sequential(
        #     nn.Linear(gap + gap, int(gap + gap)),
        #     nn.ReLU(),
        #     nn.Linear(int(gap + gap), num_classes)
        # )

        self.pred_layer = nn.Sequential(
            nn.Linear(gap, int(gap)),
            nn.ReLU(),
            nn.Linear(int(gap), num_classes)
        )

    def forward(self, x_rna, x_atac):
        out_x_rna = x_rna.permute(1, 0, 2)
        out_x_atac = x_atac.permute(1, 0, 2)
        # 位置编码
        out_x_rna = self.positionalEncoding_rna(out_x_rna)
        out_x_atac = self.positionalEncoding_atac(out_x_atac)
        # Transformer编码器层
        out_x_rna = self.encoder_layer_rna(out_x_rna)
        out_x_atac = self.encoder_layer_atac(out_x_atac)
        out_x_rna = out_x_rna.transpose(0, 1)
        out_x_atac = out_x_atac.transpose(0, 1)
        # 平均池化层
        out_x_rna = out_x_rna.mean(dim=1)
        out_x_atac = out_x_atac.mean(dim=1)
        # out = torch.cat((out_x_rna, out_x_atac), dim=1)
        out = torch.add(out_x_rna * 0.9, out_x_atac * 0.1)
        # 预测分类层
        pred = self.pred_layer(out)
        # 返回直接利用拼接模态预测的细胞类型结果(cell, num_classes)
        return pred


class scMoAnnoPretrain(nn.Module):
    def __init__(self, input_rna, input_atac, gap, dropout, num_classes, n_head):
        super(scMoAnnoPretrain, self).__init__()
        self.input_rna = input_rna
        self.input_atac = input_atac
        self.gap = gap
        self.dropout = dropout
        self.num_classes = num_classes
        self.n_head = n_head
        self.mix_attention_head = self.n_head * 2
        self.attention_dim = self.input_rna

        # 定义交叉注意力层
        self.cross_attention_layer = nn.MultiheadAttention(self.attention_dim, self.mix_attention_head)

        # 具有自注意力的 Transformer 编码器层
        self.encoder_layer_rna = nn.TransformerEncoderLayer(
            d_model=self.gap, dim_feedforward=1024, nhead=self.n_head, dropout=dropout
        )

        self.encoder_layer_atac = nn.TransformerEncoderLayer(
            d_model=self.gap, dim_feedforward=1024, nhead=self.n_head, dropout=dropout
        )

        # 带有自注意力的位置编码
        self.positionalEncoding_rna = PositionalEncoding(d_model=self.gap, dropout=dropout)
        self.positionalEncoding_atac = PositionalEncoding(d_model=self.gap, dropout=dropout)

        # 分类层
        self.pred_layer = nn.Sequential(
            nn.Linear(self.gap, self.gap),
            nn.ReLU(),
            nn.Linear(self.gap, num_classes)
        )

    def pretrain(self, x_rna, x_atac):
        x_rna_qkv, x_atac_qkv = x_rna, x_atac
        # 交叉注意力(qkv的三个线性层在nn.MultiheadAttention已经内置了)
        x_rna_att = self.cross_attention_layer(x_rna_qkv, x_atac_qkv, x_atac_qkv)[0]
        x_atac_att = self.cross_attention_layer(x_atac_qkv, x_rna_qkv, x_rna_qkv)[0]

        # 消融实验消去交叉注意力
        # x_rna_att = self.cross_attention_layer(x_rna_qkv, x_rna_qkv, x_rna_qkv)[0]
        # x_atac_att = self.cross_attention_layer(x_atac_qkv, x_atac_qkv, x_atac_qkv)[0]

        # 新的特征
        f_rna = x_rna * 0.5 + x_rna_att * 0.5
        f_atac = x_atac * 0.5 + x_atac_att * 0.5

        # 进行预测的时候需要进行reshape
        # [n_cell, n_feature] -> [n_cell, n_sub_num, n_new_feature=gap]
        f_rna = f_rna.reshape(-1, int(self.input_rna / self.gap), self.gap)
        f_atac = f_atac.reshape(-1, int(self.input_atac / self.gap), self.gap)

        f_rna = f_rna.permute(1, 0, 2)
        f_atac = f_atac.permute(1, 0, 2)
        # 位置编码
        f_rna = self.positionalEncoding_rna(f_rna)
        f_atac = self.positionalEncoding_atac(f_atac)
        # Transformer编码器层
        f_rna = self.encoder_layer_rna(f_rna)
        f_atac = self.encoder_layer_atac(f_atac)
        f_rna = f_rna.transpose(0, 1)
        f_atac = f_atac.transpose(0, 1)
        # 平均池化层
        f_rna = f_rna.mean(dim=1)
        f_atac = f_atac.mean(dim=1)
        # 分别用两种模态数据进行预测
        pred_rna = self.pred_layer(f_rna)
        pred_atac = self.pred_layer(f_atac)
        # 返回两种模态数据进行预测的细胞类型结果
        # [n_cell, num_classes]
        return pred_rna, pred_atac

    def extraction_feature(self, x_rna=None, x_atac=None):
        # 为了运行交叉注意力模型，并且为了满足在缺失模态时也能运用预训练模型的参数
        x_1 = x_rna if x_rna is not None else x_atac
        x_2 = x_atac if x_atac is not None else x_rna

        # 交叉注意力
        x_1_att = self.cross_attention_layer(x_1, x_2, x_2)[0]
        x_2_att = self.cross_attention_layer(x_2, x_1, x_1)[0]

        # 新的特征
        f_1 = x_1 * 0.5 + x_1_att * 0.5
        f_2 = x_2 * 0.5 + x_2_att * 0.5

        # 返回交叉注意力预训练模型提取的双模态数据
        # [n_cell, n_feature]
        return f_1, f_2

    def forward(self, x_rna=None, x_atac=None, ope='pretrain'):
        # ope是选择进行预训练(pretrain)还是选择进行特征提取(extraction_feature)
        if ope == 'pretrain':
            return self.pretrain(x_rna, x_atac)
        if ope == 'extraction_feature':
            if x_rna is None:
                return self.extraction_feature(x_rna=None, x_atac=x_atac)
            if x_atac is None:
                return self.extraction_feature(x_rna=x_rna, x_atac=None)
            # 若rna和atac数据均不为空，则提取两者的特征
            return self.extraction_feature(x_rna=x_rna, x_atac=x_atac)
