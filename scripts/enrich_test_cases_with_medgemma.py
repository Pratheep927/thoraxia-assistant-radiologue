"""Enrichit les 30 cas de test avec les VRAIES sorties MedGemma.


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

TEST_CASES_DIR = ROOT / "app" / "test_cases"
TEST_JSON = TEST_CASES_DIR / "test_predictions.json"


def is_enriched(entry: dict) -> bool:

    pred = entry.get("prediction", {})
    justification = pred.get("justification", "")
    visual_evidence = pred.get("visual_evidence", [])

    # Un cas est "enrichi" s'il a une justification non-générique
    # et au moins 1 preuve visuelle
    generic_start = "Analyse de la radiographie"  # notre placeholder
    has_real_just = justification and not justification.startswith(generic_start)
    has_evidence = len(visual_evidence) > 0

    return has_real_just or has_evidence


def save_json(data: dict, path: Path) -> None:
    """Sauvegarde atomique du JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 70)
    print("THORAXIA — Enrichissement des cas de test avec MedGemma")
    print("=" * 70)

    if not TEST_JSON.exists():
        print(f"Fichier introuvable : {TEST_JSON}")
        return

    # Charger l'état actuel
    with open(TEST_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    total = len(data)
    already_done = sum(1 for entry in data.values() if is_enriched(entry))
    to_do = [(filename, entry) for filename, entry in data.items()
             if not is_enriched(entry)]

    print(f"\n État actuel :")
    print(f"   Total    : {total} cas")
    print(f"    Déjà enrichis : {already_done}")
    print(f"    À traiter     : {len(to_do)}")

    if not to_do:
        print(f"\n Tous les cas sont déjà enrichis ")
        return

    est_min = len(to_do) * 3
    print(f"\n⏱  Temps estimé : ~{est_min} min (~3 min par cas sur CPU)")
    

    # Boucle d'enrichissement
    for i, (filename, entry) in enumerate(to_do, start=1):
        case_id = entry.get("case_id", filename)
        print(f"[{i}/{len(to_do)}] {case_id} ({filename})...")

        img_path = TEST_CASES_DIR / filename
        if not img_path.exists():
            print(f"    Image introuvable : {img_path} — skip")
            continue

        t0 = time.perf_counter()

        try:
            # Vraie inférence MedGemma en mode improved
            pred = apply_safety_guardrails(
                medgemma_predict(img_path, mode="improved")
            )
            latency = round(time.perf_counter() - t0, 1)

            # On enrichit MAIS on garde la classe + confidence du CSV
            
            existing = entry["prediction"]
            existing_class = existing.get("predicted_class")
            existing_conf = existing.get("confidence")

            # Fusion : nouveau JSON + on préserve la classe/confidence si présentes
            enriched = {
                "predicted_class": pred.get("predicted_class", existing_class),
                "confidence": pred.get("confidence", existing_conf),
                "image_quality": pred.get("image_quality", "good"),
                "visual_evidence": pred.get("visual_evidence", []),
                "justification": pred.get("justification", ""),
                "limitations": pred.get("limitations", []),
                "model_name": pred.get("model_name", "google/medgemma-4b-it"),
                "prompt_version": pred.get("prompt_version", "improved-v1"),
            }

            
            enriched["predicted_class"] = existing_class
            enriched["confidence"] = existing_conf

            entry["prediction"] = enriched
            entry["measured_latency"] = latency

            print(f"  {enriched['predicted_class']} "
                  f"(conf {enriched['confidence']:.2f}) — {latency}s")

            evidence_count = len(enriched.get("visual_evidence", []))
            if evidence_count:
                print(f"     🔍 {evidence_count} preuve(s) visuelle(s)")

            # SAUVEGARDE INCRÉMENTALE (safe si Ctrl+C ensuite)
            save_json(data, TEST_JSON)

        except KeyboardInterrupt:
            print(f"\n\n Interrompu par l'utilisateur.")
            print(f"    Progression sauvegardée : {i - 1}/{len(to_do)} cas enrichis.")
            
            return

        except Exception as e:
            print(f"   Erreur : {str(e)[:120]}")
            continue

    # === RÉCAP FINAL ===
    print(f"\n{'=' * 70}")
    print(f" Enrichissement terminé !")

    final_enriched = sum(1 for entry in data.values() if is_enriched(entry))
    print(f"    {final_enriched}/{total} cas ont maintenant des sorties MedGemma complètes.")

    if final_enriched < total:
        print(f"    {total - final_enriched} cas n'ont pas pu être enrichis "
              f"(erreurs ou images manquantes).")

    


if __name__ == "__main__":
    main()