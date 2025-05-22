# scMoAnno: Supervised Pre-training for Feature Extraction in Cell Type Annotation of Single-cell Multi-omics Data

## 📌 Overview

**scMoAnno** is a deep learning framework for accurate cell type annotation using paired single-cell multi-omics data (scRNA-seq + scATAC-seq). It addresses the limitations of single-omics data and improves generalization, especially in identifying rare cell types, through a two-stage supervised learning strategy.

## 🧠 Core Algorithm Strategy

scMoAnno consists of two major stages:

### 🔁 1. Pre-training with Cross-Attention Fusion

- **Input**: Paired scRNA-seq gene expression matrix and scATAC-seq peak count matrix.
- **Feature Extraction**:
  - Apply HVG selection and PCA to both modalities.
  - Feed processed data into a **Cross-Attention Network**.
  - Mutually learn omics-specific features and extract fused cross-omics representations.
- **Objective**: Optimize classification loss on both modalities simultaneously to train a fusion feature extractor.

### 🎯 2. Cell Type Prediction with Transformer Encoder

- **Input**: Fused multi-omics features from Stage 1.
- **Model**:
  - Apply vector splitting and a Transformer encoder to learn high-level representations.
  - Aggregate representations via average pooling.
  - Use a linear classifier for multi-class cell type prediction.
- **Loss**: Cross-entropy loss optimized on ground-truth cell labels.

## 🔍 Model Highlights

- **Multi-head Cross-Attention**: Explicitly models genetic distribution relationships between modalities.
- **Rare Cell Generalization**: Improves rare cell type identification via enhanced feature fusion.
- **Ablation Validated**: Outperforms direct/self-attention fusion strategies in accuracy and balanced accuracy.
- **Efficient & Lightweight**: Achieves competitive performance with fewer parameters than many baselines.

## 📊 Evaluation

- Outperforms 9 state-of-the-art methods on 4 benchmark datasets: 10X-Multiome, ISSAAC-seq, SHARE-seq, Multiome-BMMC.
- Demonstrates superior performance in:
  - **Accuracy / Balanced Accuracy**
  - **Rare cell type detection**
  - **Clustering metrics (NMI, ARI, Purity)**
  - **Modality-missing scenarios**

## 📁 Dataset Download

The datasets used in this project can be accessed via ZENODO (https://doi.org/10.5281/zenodo.15487954):

👉 [Download Benchmark Datasets](https://doi.org/10.5281/zenodo.15487954) ← _Please click the download link._

