# # Import the required packages
import sklearn.model_selection
import torch
import torch.nn as nn
import warnings

from scipy.sparse import csr_matrix

warnings.filterwarnings('ignore')
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import math
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm
from torch.utils.data import (DataLoader, Dataset)

# torch.set_default_tensor_type(torch.DoubleTensor)
# torch.set_default_tensor_type(torch.FloatTensor)
import numpy as np
import random
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn import preprocessing

from scMoAnno import scMoAnno, scMoAnnoPretrain

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import scipy


##Set random seeds
def same_seeds(seed):
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.mps.benchmark = True


seed = 20020130
same_seeds(seed)
root_path = '../data/'


# # 取用于融合scRNA-seq和scATAC-seq表达量的数据集
class Dataset_FusionRNATAC(Dataset):
    def __init__(self, data_rna, data_atac, label):
        self.data_rna = data_rna
        self.data_atac = data_atac
        self.label = label
        self.length = len(data_rna)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data_rna = torch.from_numpy(self.data_rna)
        data_atac = torch.from_numpy(self.data_atac)
        label = torch.from_numpy(self.label)

        return data_rna[index], data_atac[index], label[index]


# # 读取细胞类型的真实标签
def read_label(label_path):
    y_train = pd.read_csv(label_path)
    y_train = y_train.T
    y_train = y_train.values[0]

    # 所有细胞种类清单（type1: A, type2: B, type3: C...）
    cell_types = []
    labels = []
    for i in y_train:
        i = str(i).upper()
        if not cell_types.__contains__(i):
            cell_types.append(i)
        labels.append(cell_types.index(i))

    # 所有细胞的种类（cell1: A, cell2: B, cell3: A...）
    return np.asarray(labels), cell_types


# # 做多模态交叉注意力预训练
def pretrain(rna_path, atac_path, save_path, label_path):
    global labels, cell_types
    adata_rna = sc.read_h5ad(rna_path)
    adata_atac = sc.read_h5ad(atac_path)

    # # 获取对齐的scRNA-seq和scATAC-seq数据的细胞类型标签
    if label_path is not None:
        # 所有细胞的种类（cell1: A, cell2: B, cell3: A...）
        labels, cell_types = read_label(label_path)

    # 读取scRNA-seq和scATAC-seq数据
    # adata_rna = adata_rna.X
    adata_rna = csr_matrix(adata_rna.X).toarray()
    # adata_atac = adata_atac.X
    adata_atac = csr_matrix(adata_atac.X).toarray()

    print('读完数据了')

    # # 设置FusionRNATAC数据融合预训练模型的超参数
    # 获取读入的scRNA-seq和scATAC-seq数据的特征维度(cell, feature)，所以shape[1]
    rna_input_size = adata_rna.shape[1]
    atac_input_size = adata_atac.shape[1]
    rna_output_size = rna_input_size
    atac_output_size = atac_input_size
    # 数据融合后的特征维度
    gap = 2048

    lr = 0.000068  # 学习率
    dp = 0.1  # 丢弃率
    n_epochs = 30  # 预训练轮数
    n_head = 8

    adata_rna = np.asarray(adata_rna)
    adata_atac = np.asarray(adata_atac)

    print('rna: ' + str(rna_input_size))
    print('atac: ' + str(atac_input_size))

    # 创建交叉注意力模型
    model = scMoAnnoPretrain(
        input_rna=rna_input_size,
        input_atac=atac_input_size,
        gap=gap,
        dropout=dp,
        num_classes=len(cell_types),
        n_head=n_head
    ).float().to(device)

    # 设置损失函数，因为是重构出原输入数据，所以使用均方差损失函数
    criterion = nn.CrossEntropyLoss()
    # 设置优化器函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 根据细胞数量设置不同批次大小
    # print(type(adata_rna))
    batch_sizes = 512

    # 模型设置5-KFold
    skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    fold = 0

    for train_index, test_index in skf.split(adata_atac, labels):
        fold = fold + 1
        # X_rna_train, X_rna_test = adata_rna[train_index], adata_rna[test_index]
        # X_atac_train, X_atac_test = adata_atac[train_index], adata_atac[test_index]
        # X_rna_train = np.asarray(X_rna_train)
        # X_rna_test = np.asarray(X_rna_test)
        # X_atac_train = np.asarray(X_atac_train)
        # X_atac_test = np.asarray(X_atac_test)
        #
        # y_train, y_test = labels[train_index], labels[test_index]
        #
        # y_train = np.asarray(y_train)
        # y_test = np.asarray(y_test)

        X_rna_train, X_rna_test, X_atac_train, X_atac_test, y_train, y_test = train_test_split(
            adata_rna, adata_atac, labels, test_size=0.2, random_state=seed, shuffle=True, stratify=labels
        )
        X_rna_train = np.asarray(X_rna_train)
        X_rna_test = np.asarray(X_rna_test)
        X_atac_train = np.asarray(X_atac_train)
        X_atac_test = np.asarray(X_atac_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        # 设置训练数据集
        train_dataset = Dataset_FusionRNATAC(data_rna=X_rna_train, data_atac=X_atac_train, label=y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True, pin_memory=True)

        # 设置测试数据集
        test_dataset = Dataset_FusionRNATAC(data_rna=X_rna_test, data_atac=X_atac_test, label=y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False, pin_memory=True)

        # # 开始训练
        model.train()
        for index, epoch in enumerate(range(n_epochs)):
            train_loss = []
            train_accs_rna = []
            train_accs_atac = []
            train_f1s_rna = []
            train_f1s_atac = []
            for batch in tqdm(train_loader):
                data_rna, data_atac, labels = batch

                data_rna = data_rna.to(device)
                data_atac = data_atac.to(device)
                labels = labels.to(device)

                # 设置成预训练模式，预训练模式会返回分别使用两种模态数据的预测结果
                logits_rna, logits_atac = model(x_rna=data_rna, x_atac=data_atac, ope='pretrain')
                labels = torch.tensor(labels, dtype=torch.long)
                loss_rna = criterion(logits_rna, labels)
                loss_atac = criterion(logits_atac, labels)
                optimizer.zero_grad()
                loss = loss_rna + loss_atac
                loss.backward()
                optimizer.step()
                # 获取使用rna模态数据的预测结果
                preds_rna = logits_rna.argmax(1)
                preds_rna = preds_rna.cpu().numpy()
                # 获取使用atac模态数据的预测结果
                preds_atac = logits_atac.argmax(1)
                preds_atac = preds_atac.cpu().numpy()

                labels = labels.cpu().numpy()

                # 计算评价指标
                acc_rna = accuracy_score(labels, preds_rna)
                acc_atac = accuracy_score(labels, preds_atac)
                f1_rna = f1_score(labels, preds_rna, average='macro')
                f1_atac = f1_score(labels, preds_atac, average='macro')
                train_loss.append(loss.item())
                train_accs_rna.append(acc_rna)
                train_accs_atac.append(acc_atac)
                train_f1s_rna.append(f1_rna)
                train_f1s_atac.append(f1_atac)
            train_loss = sum(train_loss) / len(train_loss)
            train_acc_rna = sum(train_accs_rna) / len(train_accs_rna)
            train_acc_atac = sum(train_accs_atac) / len(train_accs_atac)
            train_f1_rna = sum(train_f1s_rna) / len(train_f1s_rna)
            train_f1_atac = sum(train_f1s_atac) / len(train_f1s_atac)

            print(
                f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] "
                f"loss = {train_loss:.5f}, "
                f"acc_rna = {train_acc_rna:.5f}, "
                f"f1_rna = {train_f1_rna:.5f}, "
                f"acc_atac = {train_acc_atac:.5f}, "
                f"f1_atac = {train_f1_atac:.5f}"
            )

            # # 开始验证模型
            model.eval()
            test_accs_rna = []
            test_accs_atac = []
            test_f1s_rna = []
            test_f1s_atac = []
            pred_rna = []
            pred_atac = []
            ground_truth = []
            for batch in tqdm(test_loader):
                data_rna, data_atac, labels = batch

                data_rna = data_rna.to(device)
                data_atac = data_atac.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    # 设置成预训练模式，该模式会返回分别使用两种模态数据的预测结果
                    logits_rna, logits_atac = model(x_rna=data_rna, x_atac=data_atac, ope='pretrain')

                # 获取分别使用两个模态数据的测试预测结果
                preds_rna = logits_rna.argmax(1)
                preds_rna = preds_rna.cpu().numpy().tolist()
                preds_atac = logits_atac.argmax(1)
                preds_atac = preds_atac.cpu().numpy().tolist()
                labels = labels.cpu().numpy().tolist()

                # 计算评价指标
                acc_rna = accuracy_score(labels, preds_rna)
                acc_atac = accuracy_score(labels, preds_atac)
                f1_rna = f1_score(labels, preds_rna, average='macro')
                f1_atac = f1_score(labels, preds_atac, average='macro')
                test_accs_rna.append(acc_rna)
                test_accs_atac.append(acc_atac)
                test_f1s_rna.append(f1_rna)
                test_f1s_atac.append(f1_atac)

                pred_rna.extend(preds_rna)
                pred_atac.extend(preds_atac)
                ground_truth.extend(labels)
            test_acc_rna = sum(test_accs_rna) / len(test_accs_rna)
            test_f1_rna = sum(test_f1s_rna) / len(test_f1s_rna)
            test_acc_atac = sum(test_accs_atac) / len(test_accs_atac)
            test_f1_atac = sum(test_f1s_atac) / len(test_f1s_atac)
            print("---------------------------------------------end test---------------------------------------------")

            print(
                f"[ Test | {epoch + 1:03d}/{n_epochs:03d} ] "
                f"test_acc_rna = {test_acc_rna:.5f}, "
                f"test_f1_rna = {test_f1_rna:.5f}, "
                f"test_acc_atac = {test_acc_atac:.5f}, "
                f"test_f1_atac = {test_f1_atac:.5f}"
            )

            # 之前的pred_rna、pred_atac以及ground_truth保存的是细胞类型的数字编号
            # 这里需要转换成真实的细胞类型的名称字符串
            pred_rna_str = []
            pred_atac_str = []
            ground_truth_str = []
            for i in ground_truth:
                ground_truth_str.append(cell_types[i])
            for i in pred_rna:
                pred_rna_str.append(cell_types[i])
            for i in pred_atac:
                pred_atac_str.append(cell_types[i])

            # 保存预测的细胞类型和scMoAnno模型
            log_dir = save_path + 'log/'
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
                os.makedirs(log_dir + "ground_truth/")
                os.makedirs(log_dir + "pred/")
                os.makedirs(log_dir + "pred_rna/")
                os.makedirs(log_dir + "pred_atac/")

            model_dir = save_path + 'pretrained/'
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

            np.save(log_dir + 'ground_truth/epoch_' + str(index + 1) + '_true_label_val.npy', ground_truth_str)
            np.save(log_dir + 'pred_rna/epoch_' + str(index + 1) + '_pred_label_val_RNA.npy', pred_rna_str)
            np.save(log_dir + 'pred_atac/epoch_' + str(index + 1) + '_pred_label_val_ATAC.npy', pred_atac_str)
            torch.save(model.state_dict(), model_dir + 'epoch_' + str(index + 1) + '_scMoAnno_pretrained.pth')

            with open(log_dir + "train_validation_log.txt", "a") as f:
                f.writelines('epoch_pred:' + str(index + 1) + '\n')
                f.writelines("acc_rna:" + str(test_acc_rna) + "\n")
                f.writelines('f1_rna:' + str(test_f1_rna) + "\n")
                f.writelines("acc_atac:" + str(test_acc_atac) + "\n")
                f.writelines('f1_atac:' + str(test_f1_atac) + "\n")

        if fold == 1:
            break


# # 提取多模态特征的函数
def extraction_feature(rna_path, atac_path, pretrained_path, save_path, label_path):
    global labels, cell_types
    adata_rna = sc.read_h5ad(rna_path)
    adata_atac = sc.read_h5ad(atac_path)

    # # 获取对齐的scRNA-seq和scATAC-seq数据的细胞类型标签
    if label_path is not None:
        # 所有细胞的种类（cell1: A, cell2: B, cell3: A...）
        labels, cell_types = read_label(label_path)

    # 读取scRNA-seq和scATAC-seq数据
    adata_rna = adata_rna.X
    adata_atac = adata_atac.X

    if not isinstance(adata_rna, np.ndarray):
        adata_rna = adata_rna.toarray()
    if not isinstance(adata_atac, np.ndarray):
        adata_atac = adata_atac.toarray()

    print('读完数据了')

    # # 设置FusionRNATAC数据融合预训练模型的超参数
    # 获取读入的scRNA-seq和scATAC-seq数据的特征维度(cell, feature)，所以shape[1]
    rna_input_size = adata_rna.shape[1]
    atac_input_size = adata_atac.shape[1]
    rna_output_size = rna_input_size
    atac_output_size = atac_input_size
    # 数据融合后的特征维度
    pretrained_gap = 2048
    gap = 2048

    lr = 0.00008  # 学习率
    dp = 0.15  # 丢弃率
    n_epochs = 30  # 预测模型训练轮数
    n_head = 64

    adata_rna = np.asarray(adata_rna)
    adata_atac = np.asarray(adata_atac)

    print('rna: ' + str(rna_input_size))
    print('atac: ' + str(atac_input_size))

    # 创建scMoAnnoPretrain交叉注意力特征提取预训练模型
    model_pretrained = scMoAnnoPretrain(
        input_rna=rna_input_size,
        input_atac=atac_input_size,
        gap=pretrained_gap,
        dropout=dp,
        num_classes=len(cell_types),
        n_head=n_head
    ).float().to(device)

    # 加载预训练模型参数
    checkpoint = torch.load(pretrained_path)
    model_pretrained.load_state_dict(checkpoint)

    # 将预训练模型设置成评估模式
    model_pretrained.eval()

    # 创建scMoAnno预测模型
    model = scMoAnno(
        input_rna=rna_input_size,
        input_atac=atac_input_size,
        gap=gap,
        dropout=dp,
        num_classes=len(cell_types),
        n_head=n_head
    ).float().to(device)

    # 预测模型使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 设置优化器函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 根据细胞数量设置不同批次大小
    # batch_sizes = 256
    batch_sizes = 256 if len(adata_rna) > 5000 else 128

    # 模型设置5-KFold
    skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    fold = 0

    for train_index, test_index in skf.split(adata_atac, labels):
        fold = fold + 1
        # X_rna_train, X_rna_test = adata_rna[train_index], adata_rna[test_index]
        # X_atac_train, X_atac_test = adata_atac[train_index], adata_atac[test_index]
        # X_rna_train = np.asarray(X_rna_train)
        # X_rna_test = np.asarray(X_rna_test)
        # X_atac_train = np.asarray(X_atac_train)
        # X_atac_test = np.asarray(X_atac_test)
        #
        # y_train, y_test = labels[train_index], labels[test_index]
        #
        # y_train = np.asarray(y_train)
        # y_test = np.asarray(y_test)



        X_rna_train, X_rna_test, X_atac_train, X_atac_test, y_train, y_test = train_test_split(
            adata_rna, adata_atac, labels, test_size=0.2, random_state=seed, shuffle=True, stratify=labels
        )
        X_rna_train = np.asarray(X_rna_train)
        X_rna_test = np.asarray(X_rna_test)
        X_atac_train = np.asarray(X_atac_train)
        X_atac_test = np.asarray(X_atac_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        # 设置训练数据集
        train_dataset = Dataset_FusionRNATAC(data_rna=X_rna_train, data_atac=X_atac_train, label=y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True, pin_memory=True)

        # 设置测试数据集
        test_dataset = Dataset_FusionRNATAC(data_rna=X_rna_test, data_atac=X_atac_test, label=y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False, pin_memory=True)

        # # 开始提取训练数据的新特征并进行预测模型的训练
        model.train()
        for index, epoch in enumerate(range(n_epochs)):
            # 创建预测模型的训练指标
            train_loss = []
            train_accs = []
            train_f1s = []
            for batch in tqdm(train_loader):
                # 加载批次数据
                data_rna, data_atac, labels = batch
                data_rna = data_rna.to(device)
                data_atac = data_atac.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    # 设置成提取特征模式，该模式会返回两种模态数据被交叉注意力预训练后提取新的特征
                    f_rna, f_atac = model_pretrained(x_rna=data_rna, x_atac=data_atac, ope='extraction_feature')
                    f_rna = f_rna.reshape(-1, int(rna_input_size / gap), gap)
                    f_atac = f_atac.reshape(-1, int(atac_input_size / gap), gap)

                # # 用新提取的特征去做训练
                data_rna, data_atac = f_rna, f_atac
                # 训练预测结果
                logits = model(data_rna, data_atac)
                labels = torch.tensor(labels, dtype=torch.long)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # 细胞类型预测结果
                preds = logits.argmax(1)
                preds = preds.cpu().numpy()
                labels = labels.cpu().numpy()
                # 评价指标
                acc = accuracy_score(labels, preds)
                f1 = f1_score(labels, preds, average='macro')
                train_loss.append(loss.item())
                train_accs.append(acc)
                train_f1s.append(f1)
            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)
            train_f1 = sum(train_f1s) / len(train_f1s)

            print(
                f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] "
                f"loss = {train_loss:.5f}, "
                f"acc = {train_acc:.5f}, "
                f"f1 = {train_f1:.5f}"
            )

            # # 开始评估scMoAnno预测模型
            model.eval()
            test_accs = []
            test_f1s = []
            pred = []
            ground_truth = []
            for batch in tqdm(test_loader):
                data_rna, data_atac, labels = batch

                data_rna = data_rna.to(device)
                data_atac = data_atac.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    # 设置成提取特征模式，该模式会返回两种模态数据被交叉注意力预训练后提取新的特征
                    f_rna, f_atac = model_pretrained(x_rna=data_rna, x_atac=data_atac, ope='extraction_feature')
                    f_rna = f_rna.reshape(-1, int(rna_input_size / gap), gap)
                    f_atac = f_atac.reshape(-1, int(atac_input_size / gap), gap)

                # # 用新提取的特征去做测试
                data_rna, data_atac = f_rna, f_atac

                with torch.no_grad():
                    logits = model(data_rna, data_atac)

                # 获取细胞预测结果
                preds = logits.argmax(1)
                preds = preds.cpu().numpy().tolist()
                labels = labels.cpu().numpy().tolist()

                # 计算评价指标
                acc = accuracy_score(labels, preds)
                f1 = f1_score(labels, preds, average='macro')
                test_f1s.append(f1)
                test_accs.append(acc)

                pred.extend(preds)
                ground_truth.extend(labels)
            test_acc = sum(test_accs) / len(test_accs)
            test_f1 = sum(test_f1s) / len(test_f1s)
            print("---------------------------------------------end test---------------------------------------------")

            print(
                f"[ Test | {epoch + 1:03d}/{n_epochs:03d} ] "
                f"test_acc = {test_acc:.5f}, "
                f"test_f1 = {test_f1:.5f}"
            )

            # 之前的pred以及ground_truth保存的是细胞类型的数字编号
            # 这里需要转换成真实的细胞类型的名称字符串
            pred_str = []
            ground_truth_str = []
            for i in ground_truth:
                ground_truth_str.append(cell_types[i])
            for i in pred:
                pred_str.append(cell_types[i])

            # 保存预测的细胞类型和scMoAnno模型
            log_dir = save_path + "log/"
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
                os.makedirs(log_dir + "ground_truth/")
                os.makedirs(log_dir + "pred/")
                os.makedirs(log_dir + "pred_rna/")
                os.makedirs(log_dir + "pred_atac/")

            model_dir = save_path + 'pretrained/'
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

            # np.save(log_dir + 'ground_truth/epoch_' + str(index + 1) + '_true_label_val.npy', ground_truth_str)
            # np.save(log_dir + 'pred_rna/epoch_' + str(index + 1) + '_pred_label_val.npy', pred_str)
            # torch.save(model.state_dict(), model_dir + 'epoch_' + str(index + 1) + '_scMoAnno_prediction.pth')

            with open(log_dir + "train_validation_log.txt", "a") as f:
                f.writelines('epoch_pred:' + str(index + 1) + '\n')
                f.writelines("acc:" + str(test_acc) + "\n")
                f.writelines('f1:' + str(test_f1) + "\n")

        if fold == 1:
            break


# # 设定不同数据集名称，方便切换数据集进行训练
def selection_dataset(dataset_name):
    global rna_path, atac_path, pretrained_path, save_path_pretrain, save_path_extraction_feature, label_path

    rna_path = '../../data/' + dataset_name + '/rna.h5ad'
    atac_path = '../../data/' + dataset_name + '/atac.h5ad'
    pretrained_path = '../../data/' + dataset_name + '/results/pretrain/pretrained/epoch_1_scMoAnno_pretrained.pth'
    save_path_pretrain = '../../data/' + dataset_name + '/results/pretrain/'
    save_path_extraction_feature = '../../data/' + dataset_name + '/results/extraction_feature/'
    label_path = '../../data/' + dataset_name + '/Label.csv'

    return rna_path, atac_path, pretrained_path, save_path_pretrain, save_path_extraction_feature, label_path


# # 直接在这里设定要进行训练的数据集名称：
dataset_name = 'bmmc'


# # 控制执行预训练的函数入口
def execute_pretrain():
    (rna_path, atac_path, pretrained_path, save_path_pretrain,
     save_path_extraction_feature, label_path) = selection_dataset(dataset_name)

    pretrain(rna_path, atac_path, save_path_pretrain, label_path)


# # 控制执行提取多模态特征的函数入口
def execute_extraction_feature():
    (rna_path, atac_path, pretrained_path, save_path_pretrain,
     save_path_extraction_feature, label_path) = selection_dataset(dataset_name)

    extraction_feature(rna_path, atac_path, pretrained_path, save_path_extraction_feature, label_path)


if __name__ == '__main__':
    # execute_pretrain()
    execute_extraction_feature()
