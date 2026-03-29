# Skin Condition Classifier
### Clinical AI for Dermatological Diagnosis — Built for L'Oreal Dermatological Beauty

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This project builds and evaluates two machine learning models — a **Random Forest** and a **Neural Network (MLP)** — to classify six clinically distinct skin conditions using 34 dermatological features per patient.

The workflow mirrors the diagnostic logic behind tools like the **SKINSCOPE LED** used at SkinCeuticals SkinLab: structured skin assessment, feature extraction, condition classification, treatment context.

**Dataset:** [UCI Dermatology Dataset](https://archive.ics.uci.edu/ml/datasets/Dermatology) — 366 patients, 34 clinical features, peer-reviewed and citable.
**Citation:** Ilter, N. and Guvenir, H.A. (1998). Differentiating Erythemato-Squamous Diseases.

---

## Conditions Classified

| Code | Condition | L'Oreal Relevance |
|------|-----------|-------------------|
| 1 | Psoriasis | La Roche-Posay Lipikar, Cicaplast lines |
| 2 | Seborrhoeic Dermatitis | Vichy Dercos, CeraVe Scalp range |
| 3 | Lichen Planus | Dermatological Beauty clinical pipeline |
| 4 | Pityriasis Rosea | La Roche-Posay Toleriane range |
| 5 | Chronic Dermatitis | CeraVe Moisturizing Cream core use case |
| 6 | Pityriasis Rubra Pilaris | Rare condition, research pipeline relevance |

---

## Results

| Model | Test Accuracy | Macro ROC-AUC |
|-------|--------------|---------------|
| Random Forest | 94.6% | 0.9982 |
| Neural Network MLP | 89.2% | 0.9953 |

Random Forest is the production model. ROC-AUC of 0.9982 indicates near-perfect discriminative ability across all six conditions.

---

## Project Structure
```
skin-condition-classifier/
├── src/
│   ├── load_data.py
│   ├── explore.py
│   ├── train.py
│   └── evaluate.py
├── results/
│   ├── class_distribution.png
│   ├── feature_importance_eda.png
│   ├── rf_confusion_matrix.png
│   ├── rf_feature_importance.png
│   ├── mlp_confusion_matrix.png
│   └── mlp_loss_curve.png
├── data/
├── requirements.txt
└── README.md
```

---

## How to Run
```bash
git clone https://github.com/qriyanka/Skin-Condition-Classifier-Project.git
cd Skin-Condition-Classifier-Project
conda create -n skin-classifier python=3.10
conda activate skin-classifier
pip install -r requirements.txt
python src/load_data.py
python src/explore.py
python src/train.py
```

---

## Key Technical Decisions

**Why Random Forest over deep learning?**
With 366 patients and 34 tabular features, tree-based methods consistently outperform neural networks. Random Forest also provides native feature importance — clinically interpretable output mapping directly to which skin markers drive each diagnosis.

**Why class weights?**
The dataset has a 5.6x imbalance (112 Psoriasis vs 20 Pityriasis Rubra Pilaris). Balanced class weights ensure the model learns equally from rare conditions.

**Why median imputation?**
8 missing values in the age column only. Median imputation is robust to outliers and preserves the clinical distribution.

---

## Clinical Features Used

34 features across two layers:

**Clinical markers:** erythema, scaling, definite borders, itching, koebner phenomenon, polygonal papules, follicular papules, oral mucosal involvement, knee and elbow involvement, scalp involvement, family history, age

**Histopathological markers:** melanin incontinence, eosinophil infiltrate, PNL infiltrate, fibrosis of papillary dermis, exocytosis, acanthosis, hyperkeratosis, parakeratosis, and 14 additional biopsy-level markers

This dual-layer feature set mirrors the structured assessment protocol used in physician-led clinical skin evaluations — analogous to what SKINSCOPE LED captures at the surface level.

---

## Industry Connection

L'Oreal Dermatological Beauty (La Roche-Posay, CeraVe, Vichy, SkinCeuticals) is actively investing in AI-powered skin assessment. This project demonstrates end-to-end ML pipeline ownership, dermatology domain knowledge, interpretable model design appropriate for clinical contexts, and reproducible documented code.

---

## Author

Priyanka — Data Scientist
GitHub: https://github.com/qriyanka

---

This project is for research and educational purposes. Not a medical diagnostic tool.