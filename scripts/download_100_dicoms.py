"""Télécharge uniquement les 100 DICOM sélectionnés."""
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
selection = pd.read_csv(ROOT / "data" / "rsna_raw" / "selected_100_patients.csv")
out_dir = ROOT / "data" / "rsna_raw" / "dicoms"
out_dir.mkdir(parents=True, exist_ok=True)

print(f"  Téléchargement de {len(selection)} DICOM ciblés...")

for patient_id in tqdm(selection["patientId"]):
    dicom_path = f"stage_2_train_images/{patient_id}.dcm"
    out_file = out_dir / f"{patient_id}.dcm"

    if out_file.exists():
        continue  # déjà téléchargé

    try:
        subprocess.run([
            "kaggle", "competitions", "download",
            "-c", "rsna-pneumonia-detection-challenge",
            "-f", dicom_path,
            "-p", str(out_dir),
            "--force",
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f" Échec pour {patient_id} : {e.stderr[:200]}")

print(f"\n Téléchargement terminé : {len(list(out_dir.glob('*.dcm')))} fichiers")