"""
DAG THORAXIA — Pipeline d'ingestion RSNA Pneumonia depuis Kaggle.

Architecture en 4 tâches séquentielles :
  1. kaggle_download   → télécharge le ZIP RSNA depuis Kaggle
  2. unzip_dataset     → décompresse le ZIP
  3. convert_to_png    → convertit les DICOM en PNG 512x512
  4. build_catalog     → génère le CSV récapitulatif (case_id, patientId, label)

Auteur : Équipe THORAXIA (EFREI MasterCamp 2026)

"""
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import zipfile
import subprocess

from airflow.decorators import dag, task

# ====================================================================
# CONFIGURATION
# ====================================================================
THORAXIA_ROOT = Path("/opt/thoraxia")
DATA_DIR = THORAXIA_ROOT / "data" / "rsna_full"
RAW_DIR = DATA_DIR / "raw"
DICOM_DIR = DATA_DIR / "stage_2_train_images"
PNG_DIR = DATA_DIR / "png"
CATALOG_PATH = DATA_DIR / "rsna_full_catalog.csv"

KAGGLE_COMPETITION = "rsna-pneumonia-detection-challenge"

default_args = {
    "owner": "thoraxia",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="thoraxia_kaggle_pipeline",
    description="Pipeline d'ingestion RSNA Pneumonia (download → unzip → convert → catalog)",
    schedule=None,
    start_date=datetime(2026, 6, 1),
    catchup=False,
    default_args=default_args,
    tags=["thoraxia", "rsna", "ingestion", "medical-ai"],
    doc_md=__doc__,
)
def thoraxia_kaggle_pipeline():

    @task(task_id="kaggle_download")
    def kaggle_download() -> str:
        """Télécharge le ZIP du dataset RSNA depuis Kaggle."""
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        zip_path = RAW_DIR / f"{KAGGLE_COMPETITION}.zip"

        if zip_path.exists() and zip_path.stat().st_size > 1_000_000_000:
            print(f" ZIP déjà présent : {zip_path} ({zip_path.stat().st_size / 1e9:.2f} Go) — skip")
            return str(zip_path)

        print(f" Téléchargement Kaggle : {KAGGLE_COMPETITION}")
        print(f" Destination : {RAW_DIR}")

        result = subprocess.run(
            ["kaggle", "competitions", "download",
             "-c", KAGGLE_COMPETITION,
             "-p", str(RAW_DIR)],
            capture_output=True, text=True, check=False
        )

        if result.returncode != 0:
            raise RuntimeError(f"Échec téléchargement Kaggle : {result.stderr}")

        print(f"Téléchargement terminé : {zip_path}")
        return str(zip_path)

    @task(task_id="unzip_dataset")
    def unzip_dataset(zip_path: str) -> str:
        """Décompresse le ZIP RSNA."""
        zip_p = Path(zip_path)
        if not zip_p.exists():
            raise FileNotFoundError(f"ZIP introuvable : {zip_p}")

        if DICOM_DIR.exists() and len(list(DICOM_DIR.glob("*.dcm"))) > 100:
            print(f"DICOM déjà décompressés : {DICOM_DIR} — skip")
            return str(DICOM_DIR)

        print(f"Décompression : {zip_p}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_p, "r") as zf:
            members = zf.namelist()
            print(f"{len(members)} fichiers dans le ZIP")
            zf.extractall(DATA_DIR)

        dicom_count = len(list(DICOM_DIR.glob("*.dcm")))
        print(f"{dicom_count} fichiers DICOM extraits dans {DICOM_DIR}")
        return str(DICOM_DIR)

    @task(task_id="convert_to_png")
    def convert_to_png(dicom_dir: str) -> dict:
        """Convertit tous les DICOM en PNG 512x512."""
        import pydicom
        from PIL import Image
        import numpy as np
        from tqdm import tqdm

        src = Path(dicom_dir)
        PNG_DIR.mkdir(parents=True, exist_ok=True)

        dicom_files = list(src.glob("*.dcm"))
        if not dicom_files:
            raise RuntimeError(f"Aucun DICOM trouvé dans {src}")

        print(f"{len(dicom_files)} DICOM à convertir")
        converted, skipped, errors = 0, 0, 0

        for dcm_path in tqdm(dicom_files, desc="DICOM → PNG"):
            png_path = PNG_DIR / f"{dcm_path.stem}.png"
            if png_path.exists():
                skipped += 1
                continue

            try:
                ds = pydicom.dcmread(str(dcm_path))
                arr = ds.pixel_array.astype(np.float32)
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
                arr = arr.astype(np.uint8)
                img = Image.fromarray(arr).convert("L").resize((512, 512), Image.LANCZOS)
                img.save(png_path)
                converted += 1
            except Exception as e:
                print(f"Erreur {dcm_path.name}: {e}")
                errors += 1

        stats = {"converted": converted, "skipped": skipped, "errors": errors,
                 "png_dir": str(PNG_DIR)}
        print(f"{converted} convertis · {skipped} skipés · {errors} erreurs")
        return stats

    @task(task_id="build_catalog")
    def build_catalog(stats: dict) -> str:
        """Génère le CSV catalogue (case_id, patientId, label, image_path)."""
        import pandas as pd

        png_files = sorted(Path(stats["png_dir"]).glob("*.png"))
        if not png_files:
            raise RuntimeError(f"Aucun PNG dans {stats['png_dir']}")

        labels_csv = DATA_DIR / "stage_2_train_labels.csv"
        if not labels_csv.exists():
            print(f" Labels CSV introuvable : {labels_csv} — catalogue minimal")
            labels_df = None
        else:
            labels_df = pd.read_csv(labels_csv)
            labels_df = labels_df.drop_duplicates(subset=["patientId"], keep="first")

        rows = []
        for i, png in enumerate(png_files, start=1):
            patient_id = png.stem
            row = {
                "case_id": f"RSNA_FULL_{i:05d}",
                "patientId": patient_id,
                "image_path": str(png.relative_to(THORAXIA_ROOT)),
            }
            if labels_df is not None:
                match = labels_df[labels_df["patientId"] == patient_id]
                if len(match):
                    target = int(match.iloc[0]["Target"])
                    row["label"] = "suspected_opacity" if target == 1 else "normal"
                else:
                    row["label"] = "unknown"
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(CATALOG_PATH, index=False)
        print(f"Catalogue généré : {CATALOG_PATH}")
        print(f"{len(df)} cas indexés")
        if "label" in df.columns:
            print(df["label"].value_counts())
        return str(CATALOG_PATH)

    # ===== DAG dependencies =====
    zip_path = kaggle_download()
    dicom_dir = unzip_dataset(zip_path)
    stats = convert_to_png(dicom_dir)
    build_catalog(stats)


thoraxia_kaggle_pipeline()