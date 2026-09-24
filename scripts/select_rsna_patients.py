"""Sélectionne 100 patients RSNA équilibrés (50 normal + 50 pneumonia)."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
labels_path = ROOT / "data" / "rsna_raw" / "stage_2_train_labels.csv"

print(f" Lecture des labels : {labels_path}")
df = pd.read_csv(labels_path)
print(f" {len(df)} lignes au total")
print(f"   Colonnes : {list(df.columns)}")
print(f"   Aperçu :")
print(df.head())

# Un patient peut avoir plusieurs boxes → on déduplique
df_unique = df.drop_duplicates(subset="patientId", keep="first")
print(f"\n {len(df_unique)} patients uniques")
print(f"   Distribution :")
print(df_unique["Target"].value_counts())

# Sélection équilibrée (50 normal + 50 pneumonie)
import random
random.seed(42)  # reproductibilité

normal_ids = df_unique[df_unique["Target"] == 0]["patientId"].tolist()
pneumo_ids = df_unique[df_unique["Target"] == 1]["patientId"].tolist()

selected_normal = random.sample(normal_ids, 50)
selected_pneumo = random.sample(pneumo_ids, 50)

selected = pd.DataFrame({
    "patientId": selected_normal + selected_pneumo,
    "rsna_target": [0] * 50 + [1] * 50,
})
selected["our_label"] = selected["rsna_target"].map({0: "normal", 1: "suspected_opacity"})

# Sauvegarde
out_path = ROOT / "data" / "rsna_raw" / "selected_100_patients.csv"
selected.to_csv(out_path, index=False)
print(f"\n Sélection sauvée dans : {out_path.name}")
print(f"   • 50 normal → label THORAXIA 'normal'")
print(f"   • 50 pneumonie → label THORAXIA 'suspected_opacity'")
print(f"\n Aperçu des sélectionnés :")
print(selected.head(10))