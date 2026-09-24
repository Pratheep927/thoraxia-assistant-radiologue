Données — THORAXIA

Ce dossier contient les données utilisées pour tester et évaluer le pipeline THORAXIA.


Images synthétiques (sample_images/)

Jeu de données jouet généré pour valider l'architecture, les logs, les métriques et l'interface Streamlit. Ces images imitent grossièrement une radiographie thoracique uniquement pour vérifier les flux de code. Elles ne doivent pas être utilisées pour évaluer une performance médicale.

Colonnes de synthetic_cases.csv

ColonneDescriptioncase_idIdentifiant unique du casimage_pathChemin relatif vers l'imagesourceOrigine de l'imagelabelClasse réelle (normal, suspected_opacity, uncertain)splitPartition (train, test)qualityQualité estimée de l'imagenotesRemarques éventuelles

Radiographies réelles (rsna_png/)

Images issues du RSNA Pneumonia Detection Challenge (Kaggle), un dataset public de radiographies thoraciques annotées par des radiologues experts. Utilisées pour l'évaluation comparative baseline vs improved sur 50 cas.


Source : RSNA Pneumonia Detection Challenge — Kaggle
Format original : DICOM → converti en PNG via scripts/convert_dicoms_to_png.py
Téléchargement : scripts/download_100_dicoms.py (API Kaggle)
Sélection : scripts/select_rsna_patients.py (50 cas équilibrés)
Licence : Soumise aux conditions d'utilisation Kaggle/RSNA — ne pas redistribuer



 Les images RSNA ne sont pas incluses dans ce dépôt. Pour les reproduire, suivez les instructions dans docs/ et utilisez vos propres credentials Kaggle.



Autres datasets compatibles

Pour étendre le projet, les datasets suivants sont compatibles avec le pipeline THORAXIA :


CheXpert (Stanford) — 224 000 radiographies
MIMIC-CXR (MIT) — accès sur PhysioNet
NIH ChestXray14 — 112 000 images publiques



Respectez toujours les licences et conditions d'accès de chaque dataset.
