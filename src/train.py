import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score)
from sklearn.utils.class_weight import compute_class_weight
import joblib

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

COLUMNS = [
    "erythema","scaling","definite_borders","itching","koebner_phenomenon",
    "polygonal_papules","follicular_papules","oral_mucosal_involvement",
    "knee_elbow_involvement","scalp_involvement","family_history","melanin_incontinence",
    "eosinophils_infiltrate","PNL_infiltrate","fibrosis_papillary_dermis",
    "exocytosis","acanthosis","hyperkeratosis","parakeratosis","clubbing_rete_ridges",
    "elongation_rete_ridges","thinning_suprapapillary_epidermis","spongiform_pustule",
    "munro_microabcess","focal_hypergranulosis","disappearance_granular_layer",
    "vacuolisation_damage_basal_layer","spongiosis","saw_tooth_appearance_retes",
    "follicular_horn_plug","perifollicular_parakeratosis","inflammatory_mononuclear_infiltrate",
    "band_like_infiltrate","age","class"
]

CLASS_NAMES = {
    1: "Psoriasis",
    2: "Seborrhoeic Dermatitis",
    3: "Lichen Planus",
    4: "Pityriasis Rosea",
    5: "Chronic Dermatitis",
    6: "Pityriasis Rubra Pilaris"
}

print("=== LOADING DATA ===")
df = pd.read_csv(DATA_DIR / "dermatology.csv", header=None, names=COLUMNS, na_values="?")
print(f"Shape: {df.shape}")

features = [c for c in COLUMNS if c != "class"]
X = df[features]
y = df["class"]

print(f"Features: {X.shape[1]}  |  Samples: {len(y)}")
print(f"Classes: {sorted(y.unique())}")

# ── Preprocessing ─────────────────────────────────────────
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
print(f"Missing values after imputation: {np.isnan(X_imputed).sum()}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ── Train/test split — stratified ─────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, stratify=y, random_state=42
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── Class weights — handle imbalance ─────────────────────
classes = np.array(sorted(y.unique()))
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
print(f"Class weights: { {CLASS_NAMES[k]: round(v,2) for k,v in class_weight_dict.items()} }")

# ══════════════════════════════════════════════════════════
# MODEL 1: Random Forest
# ══════════════════════════════════════════════════════════
print()
print("=== MODEL 1: RANDOM FOREST ===")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    class_weight=class_weight_dict,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_acc   = accuracy_score(y_test, rf_preds)
rf_proba = rf.predict_proba(X_test)
rf_auc   = roc_auc_score(pd.get_dummies(y_test), rf_proba, average="macro")

cv_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring="accuracy")
print(f"Test accuracy : {rf_acc:.4f} ({rf_acc*100:.1f}%)")
print(f"Macro ROC-AUC : {rf_auc:.4f}")
print(f"5-fold CV     : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print()
print("Classification Report:")
target_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
print(classification_report(y_test, rf_preds, target_names=target_names))

joblib.dump(rf, RESULTS_DIR / "random_forest_model.pkl")
joblib.dump(imputer, RESULTS_DIR / "imputer.pkl")
joblib.dump(scaler,  RESULTS_DIR / "scaler.pkl")
print("Saved: results/random_forest_model.pkl")

# ── Random Forest confusion matrix ────────────────────────
cm = confusion_matrix(y_test, rf_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=[n[:12] for n in target_names],
            yticklabels=[n[:12] for n in target_names], ax=ax)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
ax.set_title("Random Forest — Normalized Confusion Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "rf_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/rf_confusion_matrix.png")

# ── Feature importance ────────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=features)
top20 = importances.sort_values(ascending=True).tail(20)
fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#c0392b" if v > top20.quantile(0.75) else "#2980b9" for v in top20.values]
ax.barh(top20.index, top20.values, color=colors)
ax.set_title("Random Forest — Top 20 Feature Importances\n(red = highest impact clinical markers)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("Importance score")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "rf_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/rf_feature_importance.png")

# ══════════════════════════════════════════════════════════
# MODEL 2: Neural Network (MLP)
# ══════════════════════════════════════════════════════════
print()
print("=== MODEL 2: NEURAL NETWORK (MLP) ===")
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation="relu",
    solver="adam",
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
)
mlp.fit(X_train, y_train)
mlp_preds = mlp.predict(X_test)
mlp_acc   = accuracy_score(y_test, mlp_preds)
mlp_proba = mlp.predict_proba(X_test)
mlp_auc   = roc_auc_score(pd.get_dummies(y_test), mlp_proba, average="macro")

print(f"Test accuracy : {mlp_acc:.4f} ({mlp_acc*100:.1f}%)")
print(f"Macro ROC-AUC : {mlp_auc:.4f}")
print()
print("Classification Report:")
print(classification_report(y_test, mlp_preds, target_names=target_names))

joblib.dump(mlp, RESULTS_DIR / "mlp_model.pkl")
print("Saved: results/mlp_model.pkl")

# ── MLP confusion matrix ──────────────────────────────────
cm2 = confusion_matrix(y_test, mlp_preds)
cm2_norm = cm2.astype(float) / cm2.sum(axis=1, keepdims=True)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(cm2_norm, annot=True, fmt=".2f", cmap="Purples",
            xticklabels=[n[:12] for n in target_names],
            yticklabels=[n[:12] for n in target_names], ax=ax)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
ax.set_title("Neural Network (MLP) — Normalized Confusion Matrix",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "mlp_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/mlp_confusion_matrix.png")

# ── MLP training loss curve ───────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(mlp.loss_curve_, label="Train loss", color="#c0392b")
if hasattr(mlp, "validation_scores_"):
    ax.plot(mlp.validation_scores_, label="Val accuracy", color="#2980b9")
ax.set_title("Neural Network — Training Loss Curve", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(alpha=0.3)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "mlp_loss_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/mlp_loss_curve.png")

# ══════════════════════════════════════════════════════════
# FINAL COMPARISON
# ══════════════════════════════════════════════════════════
print()
print("=== FINAL COMPARISON ===")
print(f"{'Model':<25} {'Accuracy':>10} {'ROC-AUC':>10}")
print("-" * 47)
print(f"{'Random Forest':<25} {rf_acc*100:>9.1f}% {rf_auc:>10.4f}")
print(f"{'Neural Network (MLP)':<25} {mlp_acc*100:>9.1f}% {mlp_auc:>10.4f}")
print()
winner = "Random Forest" if rf_acc >= mlp_acc else "Neural Network"
print(f"Best model: {winner}")
print()
print("All results saved to results/")