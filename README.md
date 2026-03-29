python3 << 'PYEOF'
code = """# Skin Condition Classifier
A machine learning classifier for six dermatological skin conditions, built on clinical and histopathological data.

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)](https://scikit-learn.org)

---

## About this project

I have a background in Clinical Laboratory Sciences and I am currently completing a 900-hour Master Esthetician program at Atelier Esthetique. I know what eosinophil infiltrate looks like under a microscope and I know what erythema looks like on a client. This project is about using that background to actually build something.

I trained two machine learning models to classify six dermatological skin conditions using 34 clinical and histopathological features. I wanted to build an end-to-end pipeline that reflects how skin conditions are actually differentiated in clinical practice, not just run a model on a dataset I found online.

---

## What it classifies

Six erythemato-squamous conditions that come up in both clinical lab work and esthetic practice:

| Condition | Clinical presentation |
|-----------|----------------------|
| Psoriasis | Chronic, immune-mediated, disrupts the skin barrier |
| Seborrhoeic Dermatitis | Scalp and facial, driven by yeast-related inflammation |
| Lichen Planus | T-cell mediated, polygonal papule presentation |
| Pityriasis Rosea | Herald patch, self-resolving, commonly misread |
| Chronic Dermatitis | Compromised barrier, persistent inflammation |
| Pityriasis Rubra Pilaris | Rare, frequently misdiagnosed, important edge case |

**Dataset:** [UCI Dermatology Dataset](https://archive.ics.uci.edu/ml/datasets/Dermatology) (366 patients, 34 clinical features, peer-reviewed)
**Citation:** Ilter, N. and Guvenir, H.A. (1998). Differentiating Erythemato-Squamous Diseases.

---

## Results

| Model | Test Accuracy | Macro ROC-AUC |
|-------|--------------|---------------|
| Random Forest | **94.6%** | **0.9982** |
| Neural Network (MLP) | 89.2% | 0.9953 |

The Random Forest performed better and it was the right choice for this dataset. 366 patients and 34 features is not a deep learning problem. The feature importance output also maps directly back to clinical markers, so you can read why the model made a prediction, not just that it did.

ROC-AUC of 0.9982 means near-perfect separation across all six conditions. Random guessing on 6 classes gives you 0.5.

---

## How to run it
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

There are three scripts (python src/load_data.py, python src/explore.py, python src/train.py), run them in order and everything generates itself. The charts and model outputs all save to results/ automatically.

---

## Technical decisions

**Random Forest over deep learning:** Small clinical datasets and tree-based methods are a natural fit. Neural networks would overfit here. Random Forest also gives you feature importance natively, which matters when the features have real clinical meaning.

**Class weights:** The data has a 5.6x imbalance between the most and least common condition. Without correction the model would underperform on rare conditions, which in a clinical context are often the most important ones to catch.

**Median imputation:** Eight missing values, all in the age column. Median imputation is standard for clinical data with small amounts of missingness and does not distort the distribution.

**On the features:** The 34 features split into clinical observations you would make during a skin assessment, erythema, scaling, itching, border definition, koebner phenomenon, and histopathological markers you would see under a microscope, acanthosis, hyperkeratosis, parakeratosis, eosinophil infiltrate, PNL infiltrate. My CLS training means I can read both layers of this dataset fluently, which shaped how I approached the analysis.

---

## About me

I'm Priyanka. Clinical Laboratory Scientist, Master Esthetician candidate at Atelier Esthetique, and data scientist in progress. I build projects where clinical science and machine learning overlap because that is where my background actually lives.

[GitHub](https://github.com/qriyanka)

---

