import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

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

COLORS = ["#c0392b","#e67e22","#f1c40f","#27ae60","#2980b9","#8e44ad"]

df = pd.read_csv(DATA_DIR / "dermatology.csv", header=None, names=COLUMNS, na_values="?")
df["class_name"] = df["class"].map(CLASS_NAMES)

print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Missing values: {df.isnull().sum().sum()}")

# ── Plot 1: Class distribution ─────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
counts = df["class_name"].value_counts().sort_values()
bars = ax.barh(counts.index, counts.values,
               color=[COLORS[int(df[df["class_name"]==n]["class"].iloc[0])-1]
                      for n in counts.index])
ax.bar_label(bars, padding=4, fontsize=10)
ax.set_xlabel("Number of patients", fontsize=11)
ax.set_title("Patient distribution across 6 skin conditions\nUCI Dermatology Dataset (n=366)",
             fontsize=13, fontweight="bold")
ax.set_xlim(0, 130)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/class_distribution.png")

# ── Plot 2: Top 15 most discriminative features ────────────
features = [c for c in COLUMNS if c not in ["class","class_name","age"]]
df_clean = df.dropna(subset=features)
means = df_clean.groupby("class_name")[features].mean()
variance = means.var(axis=0).sort_values(ascending=False)
top15 = variance.head(15).index.tolist()

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(top15))
width = 0.13

for i, (cls_id, cls_name) in enumerate(CLASS_NAMES.items()):
    subset = df_clean[df_clean["class"]==cls_id][top15].mean()
    ax.bar(x + i*width, subset.values, width, label=cls_name, color=COLORS[i-1], alpha=0.85)

ax.set_xticks(x + width*2.5)
ax.set_xticklabels([f.replace("_"," ") for f in top15],
                   rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Mean severity score (0-3)", fontsize=10)
ax.set_title("Top 15 discriminative clinical features by skin condition",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=8)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "feature_importance_eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/feature_importance_eda.png")

# ── Plot 3: Missing value summary ─────────────────────────
missing = df.isnull().sum()
missing = missing[missing > 0]
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(missing.index, missing.values, color="#c0392b", alpha=0.8)
ax.set_title("Missing values by feature", fontsize=11, fontweight="bold")
ax.set_ylabel("Count")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "missing_values.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/missing_values.png")

print()
print("=== EDA SUMMARY ===")
print(f"Total patients   : {len(df)}")
print(f"Clinical features: {len(features)}")
print(f"Missing values   : {df.isnull().sum().sum()} (in age column only)")
print(f"Class imbalance  : max={counts.max()} min={counts.min()} ratio={counts.max()/counts.min():.1f}x")
print("EDA complete.")