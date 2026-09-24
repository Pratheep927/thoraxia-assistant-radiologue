"""MedGemma inference module for THORAXIA — étape S4.

Wraps the google/medgemma-4b-it model behind the same contract as toy_predict:
takes an image path + a mode, returns a dict matching the JSON schema.
"""
from __future__ import annotations
import json
import re
import time
from pathlib import Path
from functools import lru_cache

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID = "google/medgemma-4b-it"
WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."

# Mapping mode → fichier de prompt
PROMPT_FILES = {
    "baseline": "prompts/baseline_prompt.txt",
    "improved": "prompts/improved_prompt.txt",
}


@lru_cache(maxsize=1)
def _load_model():
    """Charge MedGemma une seule fois et le garde en cache mémoire."""
    print(f" Chargement de {MODEL_ID} (1ère fois = ~4 Go à télécharger)...")
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,  # CPU = float32 (pas de fp16/bf16)
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f" MedGemma chargé en {time.perf_counter() - t0:.1f} s")
    return processor, model


def _read_prompt(mode: str, project_root: Path) -> str:
    """Lit le contenu du fichier prompt correspondant au mode."""
    prompt_path = project_root / PROMPT_FILES[mode]
    return prompt_path.read_text(encoding="utf-8")


def _parse_json_from_response(text: str) -> dict:
    """Extrait le 1er bloc JSON valide de la réponse texte du modèle."""
    # Cherche un bloc {...} équilibré
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def medgemma_predict(image_path: str | Path, mode: str = "baseline") -> dict:
    """Prédit la classe d'une radiographie avec MedGemma.

    Args:
        image_path: chemin vers l'image PNG
        mode: 'baseline' ou 'improved'

    Returns:
        dict respectant le contrat JSON THORAXIA (image_quality, predicted_class,
        confidence, visual_evidence, justification, limitations, warning)
    """
    if mode not in PROMPT_FILES:
        raise ValueError(f"mode doit être l'un de {list(PROMPT_FILES.keys())}")

    image_path = Path(image_path)
    project_root = Path(__file__).resolve().parent.parent

    # 1. Charger le modèle (cache après 1er appel)
    processor, model = _load_model()

    # 2. Lire le prompt depuis le fichier
    prompt_text = _read_prompt(mode, project_root)

    # 3. Charger l'image
    image = Image.open(image_path).convert("RGB")

    # 4. Construire le message multimodal
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }]

    # 5. Inférence
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,  # déterministe → reproductible
        )

    # Garde uniquement la partie générée (pas le prompt)
    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_len:]
    response_text = processor.decode(generated_ids, skip_special_tokens=True)

    # 6. Parser le JSON
    raw = _parse_json_from_response(response_text)

    # 7. Adapter au contrat THORAXIA (avec valeurs de secours)
    valid_classes = {"normal", "suspected_opacity", "uncertain"}
    valid_qualities = {"good", "limited", "poor"}

    predicted_class = raw.get("predicted_class", "uncertain")
    if predicted_class not in valid_classes:
        predicted_class = "uncertain"

    image_quality = raw.get("image_quality", "good")
    if image_quality not in valid_qualities:
        image_quality = "good"

    try:
        confidence = float(raw.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    return {
        "image_quality": image_quality,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 3),
        "visual_evidence": raw.get("visual_evidence", []),
        "justification": raw.get("justification", "no justification returned"),
        "limitations": raw.get("limitations", ["model output limited"]),
        "warning": WARNING,
        "model_name": MODEL_ID,
        "prompt_version": f"{mode}_v1",
        "raw_response": response_text[:500],  # utile pour debug, tronqué
    }