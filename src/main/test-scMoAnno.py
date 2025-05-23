# # Import the required packages
import torch
import torch.nn as nn
import warnings

warnings.filterwarnings('ignore')
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import math
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm
from torch.utils.data import (DataLoader, Dataset)

torch.set_default_tensor_type(torch.DoubleTensor)
import numpy as np
import random
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import preprocessing

from scMoAnno import scMoAnno

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import scipy

from scipy.sparse import csr_matrix
from thop import profile


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


# # Dataset for fusing scRNA-seq and scATAC-seq expression values
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


def pretrain_fusion_feature(rna_path, atac_path, save_path, label_path):
    global cell_types, labels
    adata_rna = sc.read_h5ad(rna_path)
    adata_atac = sc.read_h5ad(atac_path)

    # # Get aligned cell type labels for scRNA-seq and scATAC-seq data
    if label_path != None:
        y_train = pd.read_csv(label_path)
        y_train = y_train.T
        y_train = y_train.values[0]

        # List of all cell types (type1: A, type2: B, type3: C...)
        cell_types = []
        labels = []
        for i in y_train:
            i = str(i).upper()
            if not cell_types.__contains__(i):
                cell_types.append(i)
            labels.append(cell_types.index(i))

        # Types for all cells (cell1: A, cell2: B, cell3: A...)
        labels = np.asarray(labels)

        # Read scRNA-seq and scATAC-seq data
        # adata_rna = adata_rna.X
        adata_rna = csr_matrix(adata_rna.X).toarray()
        # adata_atac = adata_atac.X
        adata_atac = csr_matrix(adata_atac.X).toarray()

    print('Finished reading data')

    # # Set hyperparameters for FusionRNATAC data fusion pretraining model
    # Get feature dimensions of input scRNA-seq and scATAC-seq data (cell, feature), so shape[1]
    rna_input_size = adata_rna.shape[1]
    atac_input_size = adata_atac.shape[1]
    rna_output_size = rna_input_size
    atac_output_size = atac_input_size
    # Feature dimension after data fusion
    gap = 1024

    lr = 0.0006  # Learning rate
    dp = 0.1  # Dropout rate
    n_epochs = 50  # Pretraining epochs
    n_head = 32

    print('rna: ' + str(rna_input_size))
    print('atac: ' + str(atac_input_size))

    # Convert 2D matrix to 3D tensor for subsequent Transformer encoder feature extraction and average pooling to remove gap_num dimension
    adata_rna_list = []
    for single_cell in adata_rna:
        feature = []
        length = len(single_cell)
        # Split scATAC-seq gene activity vector into sub-vectors of length 'gap'
        for k in range(0, length, gap):
            if (k + gap > length):
                a = single_cell[length - gap:length]
            else:
                a = single_cell[k:k + gap]

            # Scale each sub-vector
            a = preprocessing.scale(a, axis=0, with_mean=True, with_std=True, copy=True)
            feature.append(a)

        feature = np.asarray(feature)
        adata_rna_list.append(feature)

    adata_rna_list = np.asarray(adata_rna_list)  # (n_cells, gap_num, gap)

    adata_atac_list = []
    for single_cell in adata_atac:
        feature = []
        length = len(single_cell)
        # Split scATAC-seq gene activity vector into sub-vectors of length 'gap'
        for k in range(0, length, gap):
            if (k + gap > length):
                a = single_cell[length - gap:length]
            else:
                a = single_cell[k:k + gap]

            # Scale each sub-vector
            a = preprocessing.scale(a, axis=0, with_mean=True, with_std=True, copy=True)
            feature.append(a)

        feature = np.asarray(feature)
        adata_atac_list.append(feature)

    adata_atac_list = np.asarray(adata_atac_list)  # (n_cells, gap_num, gap)

    # Create model for data fusion pretraining
    model = scMoAnno(input_rna=rna_input_size, input_atac=atac_input_size, gap=gap, dropout=dp,
                     num_classes=len(cell_types), n_head=n_head).float().to(device)

    torch.set_default_tensor_type(torch.FloatTensor)

    # Create example input
    gap_num = math.ceil(rna_input_size / gap)  # Calculate sequence_length
    example_rna_input = torch.randn(1, gap_num, gap).to(device)  # (batch_size, sequence_length, feature_dim)
    example_atac_input = torch.randn(1, gap_num, gap).to(device)  # (batch_size, sequence_length, feature_dim)

    # Calculate FLOPS and number of parameters
    flops, params = profile(model, inputs=(example_rna_input, example_atac_input))

    # Print results
    print(f"FLOPS: {flops / 1e9:.5f} GFLOPs")
    print(f"Number of parameters: {params / 1e6:.5f} M")

    # Set loss function (using cross entropy since we're doing classification)
    criterion = nn.CrossEntropyLoss()
    # Set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Set batch size based on number of cells
    len_all_data = adata_rna.shape[0]
    if len_all_data > 5000:
        batch_sizes = 512
    else:
        batch_sizes = 256

    # Model verification by 5-KFold
    skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    fold = 0

    for train_index, test_index in skf.split(adata_rna_list, labels):
        fold = fold + 1
        X_rna_train, X_rna_test = adata_rna_list[train_index], adata_rna_list[test_index]
        X_atac_train, X_atac_test = adata_atac_list[train_index], adata_atac_list[test_index]
        X_rna_train = np.asarray(X_rna_train)
        X_rna_test = np.asarray(X_rna_test)
        X_atac_train = np.asarray(X_atac_train)
        X_atac_test = np.asarray(X_atac_test)

        y_train, y_test = labels[train_index], labels[test_index]

        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        # Setting the training dataset
        train_dataset = Dataset_FusionRNATAC(data_rna=X_rna_train, data_atac=X_atac_train, label=y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True, pin_memory=True)

        # Setting the test dataset
        test_dataset = Dataset_FusionRNATAC(data_rna=X_rna_test, data_atac=X_atac_test, label=y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False, pin_memory=True)

        # n_epochs: the times of Training
        model.train()
        for index, epoch in enumerate(range(n_epochs)):
            # model.train()
            # These are used to record information in training.
            train_loss = []
            train_accs = []
            train_f1s = []
            for batch in tqdm(train_loader):
                # A batch consists of scRNA-seq data and corresponding cell type annotations.
                data_rna, data_atac, labels = batch

                data_rna = data_rna.to(device)  # ...
                data_atac = data_atac.to(device)  # ...
                labels = labels.to(device)

                logits = model(data_rna, data_atac)
                labels = torch.tensor(labels, dtype=torch.long)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Getting the predicted cell type
                preds = logits.argmax(1)
                preds = preds.cpu().numpy()
                labels = labels.cpu().numpy()
                # Metrics
                acc = accuracy_score(labels, preds)
                f1 = f1_score(labels, preds, average='macro')
                train_loss.append(loss.item())
                train_accs.append(acc)
                train_f1s.append(f1)
            train_loss = sum(train_loss) / len(train_loss)
            train_acc = sum(train_accs) / len(train_accs)
            train_f1 = sum(train_f1s) / len(train_f1s)

            print(
                f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}, f1 = {train_f1:.5f}"
            )

            ##Start the validation model, which predicts the cell types in the test dataset
            model.eval()
            test_accs = []
            test_f1s = []
            y_predict = []
            labelss = []
            for batch in tqdm(test_loader):
                # A batch consists of scRNA-seq data and corresponding cell type annotations.
                data_rna, data_atac, labels = batch

                data_rna = data_rna.to(device)  # ...
                data_atac = data_atac.to(device)  # ...
                labels = labels.to(device)

                with torch.no_grad():
                    logits = model(data_rna, data_atac)

                # Getting the predicted cell type
                preds = logits.argmax(1)
                preds = preds.cpu().numpy().tolist()
                labels = labels.cpu().numpy().tolist()

                # Metrics
                acc = accuracy_score(labels, preds)
                f1 = f1_score(labels, preds, average='macro')
                test_f1s.append(f1)
                test_accs.append(acc)

                y_predict.extend(preds)
                labelss.extend(labels)
            test_acc = sum(test_accs) / len(test_accs)
            test_f1 = sum(test_f1s) / len(test_f1s)
            print("---------------------------------------------end test---------------------------------------------")
            # Metrics
            all_acc = accuracy_score(labelss, y_predict)
            all_f1 = f1_score(labelss, y_predict, average='macro')
            print("all_acc:", all_acc, "all_f1:", all_f1)

            labelsss = []
            y_predicts = []
            for i in labelss:
                labelsss.append(cell_types[i])
            for i in y_predict:
                y_predicts.append(cell_types[i])

            # Save predicted cell types and scMoAnno model
            log_dir = save_path + "log/"
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)

            model_dir = save_path + 'pretrained/'
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

            # np.save(log_dir + 'ground_truth/epoch_' + str(index + 1) + '_true_label_val.npy', labelsss)
            # np.save(log_dir + 'pred/epoch_' + str(index + 1) + '_pred_label_val.npy', y_predicts)
            # torch.save(model.state_dict(), model_dir + 'epoch_' + str(index + 1) + '_scMoAnno.tar')

            with open(log_dir + "train_validation_log.txt", "w") as f:
                f.writelines('epoch_pred:' + str(index + 1) + '\n')
                f.writelines("acc:" + str(all_acc) + "\n")
                f.writelines('f1:' + str(all_f1) + "\n")

        if fold == 5:
            break


def execute_pretrain_fusion_feature():
    rna_path = '../../data/bmmc/rna.h5ad'
    atac_path = '../../data/bmmc/atac.h5ad'
    save_path = '../../data/bmmc/results/'
    label_path = '../../data/bmmc/Label.csv'

    pretrain_fusion_feature(rna_path, atac_path, save_path, label_path)


if __name__ == '__main__':
    execute_pretrain_fusion_feature()
