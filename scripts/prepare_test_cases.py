"""Prépare 30 cas de test pour la page dédiée dans l'app Streamlit.

Sélectionne 30 cas parmi les 50 déjà évalués (notebook 05) :
  - 4 cas uncertain garantis (RSNA_020, 007, 016, 088)
  - 13 cas normal
  - 13 cas suspected_opacity

Sortie :
  app/test_cases/*.png                   (30 images copiées)
  app/test_cases/test_predictions.json   (predictions + baseline + vérité terrain)
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CASES_CSV = ROOT / "data" / "rsna_png" / "rsna_cases.csv"
IMPROVED_CSV = ROOT / "eval" / "outputs" / "medgemma_rsna_improved_predictions.csv"
BASELINE_CSV = ROOT / "eval" / "outputs" / "medgemma_rsna_baseline_predictions.csv"
PNG_DIR = ROOT / "data" / "rsna_png"
DEST_DIR = ROOT / "app" / "test_cases"
DEST_JSON = DEST_DIR / "test_predictions.json"


UNCERTAIN_CASES = ["RSNA_020", "RSNA_007", "RSNA_016", "RSNA_088"]

N_NORMAL = 13
N_OPACITY = 13
SEED = 42


def load_predictions(csv_path: Path) -> pd.DataFrame:
    """Charge les prédictions avec gestion des colonnes variables."""
    df = pd.read_csv(csv_path)
    # Normalisation des noms de colonnes possibles
    if "true_label" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "true_label"})
    if "predicted_class" not in df.columns and "prediction" in df.columns:
        df = df.rename(columns={"prediction": "predicted_class"})
    return df


def main():
    print("=" * 60)
    print("THORAXIA — Préparation des 30 cas de test")
    print("=" * 60)

    # Vérifications
    if not IMPROVED_CSV.exists():
        print(f" Introuvable : {IMPROVED_CSV}")
        print("   Lance d'abord le notebook 05.")
        return

    if not PNG_DIR.exists():
        print(f" Dossier images introuvable : {PNG_DIR}")
        return

    # Charger les prédictions
    improved_df = load_predictions(IMPROVED_CSV)
    print(f" {len(improved_df)} cas dans improved_predictions.csv")

    baseline_df = None
    if BASELINE_CSV.exists():
        baseline_df = load_predictions(BASELINE_CSV)
        print(f" {len(baseline_df)} cas dans baseline_predictions.csv")
    else:
        print(" Pas de baseline_predictions.csv — utilisation d'improved seulement")

    # Charger le mapping case_id → patientId
    if CASES_CSV.exists():
        cases_df = pd.read_csv(CASES_CSV)
        print(f" {len(cases_df)} cas dans rsna_cases.csv")
    else:
        cases_df = None
        print(" rsna_cases.csv introuvable — les images seront cherchées par case_id")

    # === SÉLECTION DES 30 CAS ===
    selected_ids = []

    # 1. Les 4 uncertain garantis
    for uid in UNCERTAIN_CASES:
        if uid in improved_df["case_id"].values:
            selected_ids.append(uid)
            print(f" Uncertain sélectionné : {uid}")
        else:
            print(f" {uid} introuvable dans improved (skip)")

    # 2. Cas normal (bien classés en improved)
    normal_correct = improved_df[
        (improved_df["true_label"] == "normal") &
        (improved_df["predicted_class"] == "normal") &
        (~improved_df["case_id"].isin(selected_ids))
    ]
    normal_selected = normal_correct.sample(
        n=min(N_NORMAL, len(normal_correct)),
        random_state=SEED
    )
    selected_ids.extend(normal_selected["case_id"].tolist())
    print(f" {len(normal_selected)} cas normal ajoutés")

    # 3. Cas suspected_opacity (bien classés en improved)
    opacity_correct = improved_df[
        (improved_df["true_label"] == "suspected_opacity") &
        (improved_df["predicted_class"] == "suspected_opacity") &
        (~improved_df["case_id"].isin(selected_ids))
    ]
    opacity_selected = opacity_correct.sample(
        n=min(N_OPACITY, len(opacity_correct)),
        random_state=SEED
    )
    selected_ids.extend(opacity_selected["case_id"].tolist())
    print(f" {len(opacity_selected)} cas suspected_opacity ajoutés")

    print(f"\n Total sélectionné : {len(selected_ids)} cas")

    # === PRÉPARATION DU DOSSIER DESTINATION ===
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    # Nettoyer les anciens fichiers PNG et JSON
    for old in DEST_DIR.glob("*.png"):
        old.unlink()
    if DEST_JSON.exists():
        DEST_JSON.unlink()

    # === COPIE DES IMAGES + CONSTRUCTION DU JSON ===
    predictions = {}
    copied = 0
    skipped = 0

    for case_id in selected_ids:
        # Récupérer les infos improved
        imp_row = improved_df[improved_df["case_id"] == case_id].iloc[0]
        true_label = imp_row.get("true_label", "unknown")
        pred_class_improved = imp_row.get("predicted_class", "?")
        conf_improved = float(imp_row.get("confidence", 0))

        # Récupérer les infos baseline (si dispo)
        baseline_info = None
        if baseline_df is not None:
            bl_rows = baseline_df[baseline_df["case_id"] == case_id]
            if len(bl_rows) > 0:
                bl_row = bl_rows.iloc[0]
                baseline_info = {
                    "predicted_class": bl_row.get("predicted_class", "?"),
                    "confidence": float(bl_row.get("confidence", 0)),
                }

        # Trouver l'image source
        # Priorité 1 : chercher par patientId via rsna_cases.csv
        img_src = None
        if cases_df is not None:
            case_row = cases_df[cases_df["case_id"] == case_id]
            if len(case_row) > 0:
                patient_id = case_row.iloc[0].get("patientId", "")
                if patient_id:
                    candidate = PNG_DIR / f"{patient_id}.png"
                    if candidate.exists():
                        img_src = candidate

        # Priorité 2 : chercher par case_id direct
        if img_src is None:
            candidate = PNG_DIR / f"{case_id}.png"
            if candidate.exists():
                img_src = candidate

        # Priorité 3 : chercher un patientId dans imp_row
        if img_src is None and "patientId" in imp_row:
            candidate = PNG_DIR / f"{imp_row['patientId']}.png"
            if candidate.exists():
                img_src = candidate

        if img_src is None:
            print(f" Image introuvable pour {case_id} — skip")
            skipped += 1
            continue

        # Copier avec un nom uniforme
        dst_name = f"{case_id}.png"
        dst_path = DEST_DIR / dst_name
        shutil.copy(img_src, dst_path)

        # Déterminer correctness
        is_correct = (pred_class_improved == true_label)
        is_uncertain = (pred_class_improved == "uncertain")

        # Construction du dict de prédiction (format compatible app Streamlit)
        prediction = {
            "predicted_class": pred_class_improved,
            "confidence": conf_improved,
            "image_quality": imp_row.get("image_quality", "good"),
            "visual_evidence": [],  # Non stocké dans le CSV
            "justification": f"Analyse de la radiographie {case_id} — mode improved.",
            "limitations": [],
            "model_name": "google/medgemma-4b-it",
            "prompt_version": "improved-v1",
        }

        predictions[dst_name] = {
            "case_id": case_id,
            "true_label": true_label,
            "prediction": prediction,
            "baseline": baseline_info,
            "is_correct": is_correct,
            "is_uncertain": is_uncertain,
            "measured_latency": float(imp_row.get("latency_seconds", 3.0)),
        }
        copied += 1

    # Sauvegarder le JSON
    with open(DEST_JSON, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    # === RÉCAP ===
    print(f"\n{'=' * 60}")
    print(f" Terminé !")
    print(f"    {copied} images copiées dans {DEST_DIR}")
    print(f"    {skipped} skippées")
    print(f"    Prédictions dans {DEST_JSON}")

    # Stats de composition
    n_uncertain = sum(1 for p in predictions.values() if p["prediction"]["predicted_class"] == "uncertain")
    n_normal = sum(1 for p in predictions.values() if p["prediction"]["predicted_class"] == "normal")
    n_opacity = sum(1 for p in predictions.values() if p["prediction"]["predicted_class"] == "suspected_opacity")
    n_correct = sum(1 for p in predictions.values() if p["is_correct"])

    print(f"\n Composition :")
    print(f"   Normal      : {n_normal}")
    print(f"   Suspicion   : {n_opacity}")
    print(f"   Uncertain   : {n_uncertain}")
    print(f"   Corrects    : {n_correct}/{copied}")


if __name__ == "__main__":
    main()