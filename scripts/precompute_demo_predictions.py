"""Pré-calcule les prédictions MedGemma pour les images de démo.



Utilité : rendre la démo Streamlit instantanée sur les exemples.

"""
from __future__ import annotations
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.medgemma_inference import medgemma_predict
from src.guardrails import apply_safety_guardrails


def main():
    demo_dir = ROOT / "app" / "demo_images"
    output_path = demo_dir / "precomputed_predictions.json"

    demo_files = sorted(demo_dir.glob("*.png"))
    if not demo_files:
        print(f" Aucune image trouvée dans {demo_dir}")
        return

    print(f" {len(demo_files)} images à traiter dans {demo_dir}")
    print(f" Sortie : {output_path}")
    print(f" Temps estimé : ~{len(demo_files) * 3} min (~3 min par image sur CPU)\n")

    results = {}

    for i, img_path in enumerate(demo_files, start=1):
        print(f"[{i}/{len(demo_files)}] {img_path.name}...")
        t0 = time.perf_counter()

        try:
            pred = apply_safety_guardrails(
                medgemma_predict(img_path, mode="improved")
            )
            latency = round(time.perf_counter() - t0, 1)

            # On sauvegarde la latence RÉELLE mesurée pour l'afficher
            # dans l'UI comme si c'était live
            results[img_path.name] = {
                "prediction": pred,
                "measured_latency": latency,
                "computed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            print(f"   {pred['predicted_class']} (conf {pred.get('confidence', 0):.2f}) — {latency}s")

        except Exception as e:
            print(f"   Erreur : {e}")
            results[img_path.name] = {"error": str(e)}

    # Sauvegarde
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n Terminé ! {len(results)} prédictions sauvegardées dans :")
    print(f"   {output_path}")
    


if __name__ == "__main__":
    main()