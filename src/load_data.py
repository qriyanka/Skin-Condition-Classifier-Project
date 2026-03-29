import pandas as pd
import urllib.request
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RAW_CSV  = DATA_DIR / "dermatology.csv"

# 34 clinical features + 1 class label
# Source: UCI ML Repository — Dermatology Dataset
# Citation: Ilter & Guvenir, 1998
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

print("Downloading UCI Dermatology dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data"
urllib.request.urlretrieve(url, RAW_CSV)
print(f"Saved to: {RAW_CSV}")

# Verify file on disk immediately
size = RAW_CSV.stat().st_size
print(f"File size on disk: {size} bytes")
assert size > 10000, "File too small — download may have failed"

# Load and inspect
df = pd.read_csv(RAW_CSV, header=None, names=COLUMNS, na_values="?")
print(f"Shape: {df.shape}")
print(f"Rows: {len(df)}  |  Columns: {len(df.columns)}")
print(f"Missing values: {df.isnull().sum().sum()}")
print()
print("Class distribution:")
for k, v in df["class"].value_counts().sort_index().items():
    print(f"  {k} — {CLASS_NAMES[k]}: {v} patients")
print()
print("VERIFIED: Dataset loaded correctly.")