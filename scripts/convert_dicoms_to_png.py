"""Convertit les DICOM RSNA en PNG 512x512 utilisables par MedGemma."""
import pydicom
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
in_dir = ROOT / "data" / "rsna_raw" / "dicoms"
out_dir = ROOT / "data" / "rsna_png"
out_dir.mkdir(parents=True, exist_ok=True)

dicom_files = sorted(in_dir.glob("*.dcm"))
print(f" {len(dicom_files)} DICOM à convertir vers {out_dir}")

success = 0
errors = []

for dcm_path in tqdm(dicom_files, desc="Conversion"):
    try:
        # 1. Lire le DICOM
        ds = pydicom.dcmread(str(dcm_path))

        # 2. Récupérer les pixels (matrice numpy)
        pixel_array = ds.pixel_array.astype(np.float32)

        # 3. Normaliser en 0-255 (uint8) pour PNG
        p_min, p_max = pixel_array.min(), pixel_array.max()
        if p_max > p_min:
            pixel_array = (pixel_array - p_min) / (p_max - p_min) * 255
        pixel_array = pixel_array.astype(np.uint8)

        # 4. Redimensionner à 512x512 (cohérence avec dataset synthétique)
        img = Image.fromarray(pixel_array).convert("L")  # L = niveaux de gris
        img = img.resize((512, 512), Image.LANCZOS)
        img = img.convert("RGB")  # MedGemma attend RGB

        # 5. Sauvegarder
        out_path = out_dir / f"{dcm_path.stem}.png"
        img.save(out_path, "PNG")
        success += 1

    except Exception as e:
        errors.append((dcm_path.name, str(e)[:100]))

print(f"\n {success}/{len(dicom_files)} images converties")
if errors:
    print(f" Erreurs ({len(errors)}) :")
    for name, msg in errors[:5]:
        print(f"   • {name} : {msg}")