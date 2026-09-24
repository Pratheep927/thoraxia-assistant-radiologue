"""Copie 4 cas RSNA représentatifs dans app/demo_images pour la galerie."""
import pandas as pd
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
preds_path = ROOT / "eval" / "outputs" / "medgemma_rsna_baseline_predictions.csv"
preds = pd.read_csv(preds_path)

# On prend 2 normal correctement classés + 2 pneumonie correctement classés
normal_ok = preds[(preds["label"] == "normal") &
                  (preds["predicted_class"] == "normal")].head(2)
pneumo_ok = preds[(preds["label"] == "suspected_opacity") &
                  (preds["predicted_class"] == "suspected_opacity")].head(2)

demo_dir = ROOT / "app" / "demo_images"
demo_dir.mkdir(exist_ok=True)

selection = pd.concat([normal_ok, pneumo_ok])
for i, row in enumerate(selection.itertuples(), start=1):
    src = ROOT / "data" / "rsna_png" / f"{row.patientId}.png"
    dst_name = f"demo_{i}_{row.label}.png"
    dst = demo_dir / dst_name
    shutil.copy(src, dst)
    print(f"✓ {dst_name} (case_id={row.case_id}, label={row.label})")

print(f"\n 4 images copiées dans : {demo_dir}")