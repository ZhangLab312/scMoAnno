# Import the required packages
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


## Set random seeds
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


# Dataset for fusing scRNA-seq and scATAC-seq expression data
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


# Read the true labels of cell types
def read_label(label_path):
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

    # Types of all cells (cell1: A, cell2: B, cell3: A...)
    return np.asarray(labels), cell_types


# Perform multimodal cross-attention pretraining
def pretrain(rna_path, atac_path, save_path, label_path):
    global labels, cell_types
    adata_rna = sc.read_h5ad(rna_path)
    adata_atac = sc.read_h5ad(atac_path)

    # Get the cell type labels of aligned scRNA-seq and scATAC-seq data
    if label_path is not None:
        # Types of all cells (cell1: A, cell2: B, cell3: A...)
        labels, cell_types = read_label(label_path)

    # Read scRNA-seq and scATAC-seq data
    adata_rna = csr_matrix(adata_rna.X).toarray()
    adata_atac = csr_matrix(adata_atac.X).toarray()

    print('Data reading completed')

    # Set hyperparameters for the FusionRNATAC data fusion pretraining model
    rna_input_size = adata_rna.shape[1]
    atac_input_size = adata_atac.shape[1]
    rna_output_size = rna_input_size
    atac_output_size = atac_input_size
    gap = 2048  # Dimensionality of fused features

    lr = 0.000068  # Learning rate
    dp = 0.1  # Dropout rate
    n_epochs = 30  # Number of pretraining epochs
    n_head = 8

    adata_rna = np.asarray(adata_rna)
    adata_atac = np.asarray(adata_atac)

    print('rna: ' + str(rna_input_size))
    print('atac: ' + str(atac_input_size))

    # Create a cross-attention model
    model = scMoAnnoPretrain(
        input_rna=rna_input_size,
        input_atac=atac_input_size,
        gap=gap,
        dropout=dp,
        num_classes=len(cell_types),
        n_head=n_head
    ).float().to(device)

    # Set the loss function, using cross-entropy loss for classification tasks
    criterion = nn.CrossEntropyLoss()
    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Set different batch sizes according to the number of cells
    batch_sizes = 512

    # 5-KFold cross-validation setup
    skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    fold = 0

    for train_index, test_index in skf.split(adata_atac, labels):
        fold = fold + 1
        X_rna_train, X_rna_test, X_atac_train, X_atac_test, y_train, y_test = train_test_split(
            adata_rna, adata_atac, labels, test_size=0.2, random_state=seed, shuffle=True, stratify=labels
        )
        X_rna_train = np.asarray(X_rna_train)
        X_rna_test = np.asarray(X_rna_test)
        X_atac_train = np.asarray(X_atac_train)
        X_atac_test = np.asarray(X_atac_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        # Set up the training dataset
        train_dataset = Dataset_FusionRNATAC(data_rna=X_rna_train, data_atac=X_atac_train, label=y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True, pin_memory=True)

        # Set up the test dataset
        test_dataset = Dataset_FusionRNATAC(data_rna=X_rna_test, data_atac=X_atac_test, label=y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False, pin_memory=True)

        # Start training
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

                # Set to pretrain mode, which returns prediction results using both modal data
                logits_rna, logits_atac = model(x_rna=data_rna, x_atac=data_atac, ope='pretrain')
                labels = torch.tensor(labels, dtype=torch.long)
                loss_rna = criterion(logits_rna, labels)
                loss_atac = criterion(logits_atac, labels)
                optimizer.zero_grad()
                loss = loss_rna + loss_atac
                loss.backward()
                optimizer.step()
                # Get prediction results using RNA modal data
                preds_rna = logits_rna.argmax(1)
                preds_rna = preds_rna.cpu().numpy()
                # Get prediction results using ATAC modal data
                preds_atac = logits_atac.argmax(1)
                preds_atac = preds_atac.cpu().numpy()

                labels = labels.cpu().numpy()

                # Calculate evaluation metrics
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

            # Start validating the model
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
                    # Set to pretrain mode, which returns prediction results using both modal data
                    logits_rna, logits_atac = model(x_rna=data_rna, x_atac=data_atac, ope='pretrain')

                # Get test prediction results using both modal data
                preds_rna = logits_rna.argmax(1)
                preds_rna = preds_rna.cpu().numpy().tolist()
                preds_atac = logits_atac.argmax(1)
                preds_atac = preds_atac.cpu().numpy().tolist()
                labels = labels.cpu().numpy().tolist()

                # Calculate evaluation metrics
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

            # The previous pred_rna, pred_atac, and ground_truth store the numerical labels of cell types
            # Here they need to be converted to the string names of the actual cell types
            pred_rna_str = []
            pred_atac_str = []
            ground_truth_str = []
            for i in ground_truth:
                ground_truth_str.append(cell_types[i])
            for i in pred_rna:
                pred_rna_str.append(cell_types[i])
            for i in pred_atac:
                pred_atac_str.append(cell_types[i])

            # Save the predicted cell types and the scMoAnno model
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

        if fold == 5:
            break

# Function to extract multimodal features
def extraction_feature(rna_path, atac_path, pretrained_path, save_path, label_path):
    global labels, cell_types
    adata_rna = sc.read_h5ad(rna_path)
    adata_atac = sc.read_h5ad(atac_path)

    # Get the cell type labels of aligned scRNA-seq and scATAC-seq data
    if label_path is not None:
        # Types of all cells (cell1: A, cell2: B, cell3: A...)
        labels, cell_types = read_label(label_path)

    # Read scRNA-seq and scATAC-seq data
    adata_rna = adata_rna.X
    adata_atac = adata_atac.X

    if not isinstance(adata_rna, np.ndarray):
        adata_rna = adata_rna.toarray()
    if not isinstance(adata_atac, np.ndarray):
        adata_atac = adata_atac.toarray()

    print('Data reading completed')

    # Set hyperparameters for the FusionRNATAC data fusion pretraining model
    # Get the feature dimensions of the input scRNA-seq and scATAC-seq data (cell, feature), so use shape[1]
    rna_input_size = adata_rna.shape[1]
    atac_input_size = adata_atac.shape[1]
    rna_output_size = rna_input_size
    atac_output_size = atac_input_size
    # Dimensionality of fused features
    pretrained_gap = 2048
    gap = 2048

    lr = 0.00008  # Learning rate
    dp = 0.15  # Dropout rate
    n_epochs = 30  # Number of training epochs for the prediction model
    n_head = 64

    adata_rna = np.asarray(adata_rna)
    adata_atac = np.asarray(adata_atac)

    print('rna: ' + str(rna_input_size))
    print('atac: ' + str(atac_input_size))

    # Create the scMoAnnoPretrain cross-attention feature extraction pretrained model
    model_pretrained = scMoAnnoPretrain(
        input_rna=rna_input_size,
        input_atac=atac_input_size,
        gap=pretrained_gap,
        dropout=dp,
        num_classes=len(cell_types),
        n_head=n_head
    ).float().to(device)

    # Load pretrained model parameters
    checkpoint = torch.load(pretrained_path)
    model_pretrained.load_state_dict(checkpoint)

    # Set the pretrained model to evaluation mode
    model_pretrained.eval()

    # Create the scMoAnno prediction model
    model = scMoAnno(
        input_rna=rna_input_size,
        input_atac=atac_input_size,
        gap=gap,
        dropout=dp,
        num_classes=len(cell_types),
        n_head=n_head
    ).float().to(device)

    # The prediction model uses cross-entropy loss function
    criterion = nn.CrossEntropyLoss()
    # Set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Set different batch sizes according to the number of cells
    batch_sizes = 256 if len(adata_rna) > 5000 else 128

    # 5-KFold cross-validation setup
    skf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    fold = 0

    for train_index, test_index in skf.split(adata_atac, labels):
        fold = fold + 1
        X_rna_train, X_rna_test, X_atac_train, X_atac_test, y_train, y_test = train_test_split(
            adata_rna, adata_atac, labels, test_size=0.2, random_state=seed, shuffle=True, stratify=labels
        )
        X_rna_train = np.asarray(X_rna_train)
        X_rna_test = np.asarray(X_rna_test)
        X_atac_train = np.asarray(X_atac_train)
        X_atac_test = np.asarray(X_atac_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        # Set up the training dataset
        train_dataset = Dataset_FusionRNATAC(data_rna=X_rna_train, data_atac=X_atac_train, label=y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True, pin_memory=True)

        # Set up the test dataset
        test_dataset = Dataset_FusionRNATAC(data_rna=X_rna_test, data_atac=X_atac_test, label=y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_sizes, shuffle=False, pin_memory=True)

        # Start extracting new features from training data and training the prediction model
        model.train()
        for index, epoch in enumerate(range(n_epochs)):
            # Create training metrics for the prediction model
            train_loss = []
            train_accs = []
            train_f1s = []
            for batch in tqdm(train_loader):
                # Load batch data
                data_rna, data_atac, labels = batch
                data_rna = data_rna.to(device)
                data_atac = data_atac.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    # Set to feature extraction mode, which returns new features extracted from both modal data
                    # after cross-attention pretraining
                    f_rna, f_atac = model_pretrained(x_rna=data_rna, x_atac=data_atac, ope='extraction_feature')
                    f_rna = f_rna.reshape(-1, int(rna_input_size / gap), gap)
                    f_atac = f_atac.reshape(-1, int(atac_input_size / gap), gap)

                # Use the newly extracted features for training
                data_rna, data_atac = f_rna, f_atac
                # Training predictions
                logits = model(data_rna, data_atac)
                labels = torch.tensor(labels, dtype=torch.long)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Cell type prediction results
                preds = logits.argmax(1)
                preds = preds.cpu().numpy()
                labels = labels.cpu().numpy()
                # Evaluation metrics
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

            # Start evaluating the scMoAnno prediction model
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
                    # Set to feature extraction mode, which returns new features extracted from both modal data
                    # after cross-attention pretraining
                    f_rna, f_atac = model_pretrained(x_rna=data_rna, x_atac=data_atac, ope='extraction_feature')
                    f_rna = f_rna.reshape(-1, int(rna_input_size / gap), gap)
                    f_atac = f_atac.reshape(-1, int(atac_input_size / gap), gap)

                # Use the newly extracted features for testing
                data_rna, data_atac = f_rna, f_atac

                with torch.no_grad():
                    logits = model(data_rna, data_atac)

                # Get cell prediction results
                preds = logits.argmax(1)
                preds = preds.cpu().numpy().tolist()
                labels = labels.cpu().numpy().tolist()

                # Calculate evaluation metrics
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

            # The previous pred and ground_truth store the numerical labels of cell types
            # Here they need to be converted to the string names of the actual cell types
            pred_str = []
            ground_truth_str = []
            for i in ground_truth:
                ground_truth_str.append(cell_types[i])
            for i in pred:
                pred_str.append(cell_types[i])

            # Save the predicted cell types and the scMoAnno model
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

            # The following lines are commented out as per the original code
            # np.save(log_dir + 'ground_truth/epoch_' + str(index + 1) + '_true_label_val.npy', ground_truth_str)
            # np.save(log_dir + 'pred_rna/epoch_' + str(index + 1) + '_pred_label_val.npy', pred_str)
            # torch.save(model.state_dict(), model_dir + 'epoch_' + str(index + 1) + '_scMoAnno_prediction.pth')

            with open(log_dir + "train_validation_log.txt", "a") as f:
                f.writelines('epoch_pred:' + str(index + 1) + '\n')
                f.writelines("acc:" + str(test_acc) + "\n")
                f.writelines('f1:' + str(test_f1) + "\n")

        if fold == 1:
            break


# Set different dataset names to facilitate switching between datasets for training
def selection_dataset(dataset_name):
    global rna_path, atac_path, pretrained_path, save_path_pretrain, save_path_extraction_feature, label_path

    rna_path = '../../data/' + dataset_name + '/rna.h5ad'
    atac_path = '../../data/' + dataset_name + '/atac.h5ad'
    pretrained_path = '../../data/' + dataset_name + '/results/pretrain/pretrained/epoch_30_scMoAnno_pretrained.pth'
    save_path_pretrain = '../../data/' + dataset_name + '/results/pretrain/'
    save_path_extraction_feature = '../../data/' + dataset_name + '/results/extraction_feature/'
    label_path = '../../data/' + dataset_name + '/Label.csv'

    return rna_path, atac_path, pretrained_path, save_path_pretrain, save_path_extraction_feature, label_path


# Directly set the name of the dataset to be trained here:
dataset_name = 'bmmc'


# Entry function to control the execution of pretraining
def execute_pretrain():
    (rna_path, atac_path, pretrained_path, save_path_pretrain,
     save_path_extraction_feature, label_path) = selection_dataset(dataset_name)

    pretrain(rna_path, atac_path, save_path_pretrain, label_path)


# Entry function to control the execution of multimodal feature extraction
def execute_extraction_feature():
    (rna_path, atac_path, pretrained_path, save_path_pretrain,
     save_path_extraction_feature, label_path) = selection_dataset(dataset_name)

    extraction_feature(rna_path, atac_path, pretrained_path, save_path_extraction_feature, label_path)


if __name__ == '__main__':
    execute_pretrain()
    execute_extraction_feature()
