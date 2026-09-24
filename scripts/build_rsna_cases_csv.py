"""Construit data/rsna_png/rsna_cases.csv au format THORAXIA."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
selection = pd.read_csv(ROOT / "data" / "rsna_raw" / "selected_100_patients.csv")
png_dir = ROOT / "data" / "rsna_png"

# On garde uniquement les patients dont le PNG existe
existing_pngs = {p.stem for p in png_dir.glob("*.png")}
selection = selection[selection["patientId"].isin(existing_pngs)].reset_index(drop=True)

print(f"{len(selection)} cas RSNA avec PNG dispos")

# Construction du CSV au format THORAXIA
cases = pd.DataFrame({
    "case_id": [f"RSNA_{i+1:03d}" for i in range(len(selection))],
    "image_path": [f"data/rsna_png/{pid}.png" for pid in selection["patientId"]],
    "source": "rsna_pneumonia_2018",
    "label": selection["our_label"],
    "split": "dev",
    "quality": "good",
    "notes": "RSNA Pneumonia — converted from DICOM 512x512",
    "patientId": selection["patientId"],
})

# Sauvegarde
out_path = ROOT / "data" / "rsna_png" / "rsna_cases.csv"
cases.to_csv(out_path, index=False)
print(f"Sauvé dans : {out_path}")
print(f"\nDistribution :")
print(cases["label"].value_counts())
print(f"\nAperçu :")
print(cases.head())