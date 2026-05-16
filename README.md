
# Alzheimer's MRI Classification & Progression Prediction

This repository contains multiple deep learning and hybrid machine learning pipelines for Alzheimer's Disease classification and progression analysis using MRI scans and clinical data from the OASIS dataset.

---

## Repository Structure

```text
alzheimers-mri-classification-repo/
│
├── feature-extraction/
│   └── OASIS_FeatureExtraction.ipynb
│
├── cnn-classification/
│   └── CNNClassification.ipynb
│
├── hybrid-cnn-xgboost/
│   └── cnn_xgboost_pipeline.ipynb
│
├── tlstm/
│   ├── TLSTM_Train.ipynb
│   └── TLSTM_Test.ipynb
│
└── README.md
```

---

# Project Overview

This project explores several approaches for Alzheimer's Disease classification and longitudinal progression prediction:

- 3D CNN-based MRI classification
- Feature extraction using pretrained 3D ResNet-18
- Hybrid CNN + XGBoost pipeline
- Time-aware LSTM (TLSTM) for longitudinal patient progression modeling

The workflows use MRI scans along with clinical metadata such as MMSE, eTIV, nWBV, ASF, SES, and EDUC.

---

# 1. Feature Extraction Pipeline

## File
`feature-extraction/OASIS_FeatureExtraction.ipynb`

## Purpose
Extracts deep MRI embeddings using a pretrained 3D ResNet-18 model from MONAI.

The notebook:
- Loads preprocessed MRI scans
- Uses transfer learning with 3D ResNet-18
- Freezes early layers
- Generates compact feature embeddings
- Saves embeddings for downstream models

## Output
- `.npy` feature embedding files
- Train/test embedding folders

## Main Libraries
- MONAI
- PyTorch
- Nibabel
- NumPy

---

# 2. CNN Classification Pipeline

## File
`cnn-classification/CNNClassification.ipynb`

## Purpose
Performs direct Alzheimer's classification using MRI scans with a CNN-based architecture.

## Features
- MRI preprocessing
- Transfer learning
- Weighted loss handling
- Validation metrics
- Cross-validation
- Confusion matrix visualization

## Outputs
- Trained CNN model
- Validation metrics
- Classification reports
- Confusion matrices

---

# 3. Hybrid CNN + XGBoost Pipeline

## File
`hybrid-cnn-xgboost/cnn_xgboost_pipeline.ipynb`

## Purpose
Combines deep MRI embeddings with XGBoost for improved classification performance.

## Pipeline
1. Extract embeddings from MRI scans
2. Concatenate with clinical features
3. Train XGBoost classifier
4. Evaluate fold-wise performance

## Features
- Hybrid deep learning + ML approach
- Faster training compared to full CNN finetuning
- Clinical + imaging feature fusion
- Fold-wise embedding saving

## Outputs
- Saved embeddings
- Trained XGBoost model
- Cross-validation scores
- Metrics and reports

---

# 4. TLSTM Longitudinal Modeling

## Files
- `tlstm/TLSTM_Train.ipynb`
- `tlstm/TLSTM_Test.ipynb`

## Purpose
Uses Time-aware LSTM (TLSTM) for modeling longitudinal disease progression across patient visits.

## Features
- Sequential patient visit modeling
- Temporal interval handling
- Clinical feature integration
- PCA + StandardScaler preprocessing

## Workflow
### Training Notebook
- Loads training embeddings
- Preprocesses sequential data
- Trains TLSTM model
- Saves trained weights

### Testing Notebook
- Loads trained model
- Runs inference on test data
- Generates predictions and evaluation metrics

---

# Dataset Requirements

Expected inputs include:
- Preprocessed MRI `.nii` or `.nii.gz` files
- Clinical CSV metadata
- Extracted embedding folders

Example directory structure:

```text
data/
├── train/
├── test/
├── clinical.csv
└── embeddings/
```

---

# Installation

Install dependencies:

```bash
pip install monai nibabel xgboost torch torchvision scikit-learn pandas numpy matplotlib seaborn openpyxl
```

---

# Running the Pipelines

## Step 1 — Feature Extraction
Run:

```text
feature-extraction/OASIS_FeatureExtraction.ipynb
```

This generates MRI feature embeddings.

---

## Step 2 — Choose a Modeling Approach

### Option A: Pure CNN Classification
Run:

```text
cnn-classification/CNNClassification.ipynb
```

### Option B: CNN + XGBoost Hybrid
Run:

```text
hybrid-cnn-xgboost/cnn_xgboost_pipeline.ipynb
```

### Option C: Longitudinal TLSTM
Run:

```text
tlstm/TLSTM_Train.ipynb
tlstm/TLSTM_Test.ipynb
```

---

# Notes

- Most notebooks are configured for Google Colab or Kaggle.
- Update dataset paths before running.
- GPU acceleration is strongly recommended.
- Ensure MRI preprocessing is completed before feature extraction.

---

# Future Improvements

- Add attention-based temporal models
- Add Grad-CAM explainability
- Deploy with FastAPI backend
- Create inference API for clinical usage
- Add experiment tracking with Weights & Biases

---

# License

This project is intended for academic and research purposes.
