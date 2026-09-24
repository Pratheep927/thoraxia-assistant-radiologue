"""THORAXIA — Application complète avec auth, dossiers patients, documents,
multi-image, export PDF, page profil, signalement de problème.

Améliorations :
- Navbar corrigée (tout sur une ligne, y compris pré-connexion)
- Accès à "À propos" sans être connecté
- Badge profil cliquable → page "Mon compte" avec édition
- Bouton "Signaler un problème" en bas de page → modal → mailto
- Échappement HTML des commentaires (anti-XSS)
- Confirmation avant suppression multiple
- Export PDF d'un dossier patient
- Dashboard statistiques dans Documents
"""
from __future__ import annotations
import json, time, hashlib, uuid, html, io, urllib.parse
from datetime import datetime, date
from pathlib import Path
import sys

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.medgemma_inference import medgemma_predict
from src.guardrails import apply_safety_guardrails

# ── Pré-calculs démo ──
_PRECOMPUTED_PATH = ROOT / "app" / "demo_images" / "precomputed_predictions.json"
def load_precomputed() -> dict:
    if not _PRECOMPUTED_PATH.exists(): return {}
    try: return json.loads(_PRECOMPUTED_PATH.read_text(encoding="utf-8"))
    except: return {}
_PRECOMPUTED = load_precomputed()

# ── Cas de test (30 cas RSNA pré-calculés) ──
_TEST_CASES_PATH = ROOT / "app" / "test_cases" / "test_predictions.json"
_TEST_CASES_DIR = ROOT / "app" / "test_cases"

def load_test_cases() -> dict:
    if not _TEST_CASES_PATH.exists():
        return {}
    try:
        return json.loads(_TEST_CASES_PATH.read_text(encoding="utf-8"))
    except:
        return {}

_TEST_CASES = load_test_cases()

# ── Logo ──
import base64
_LOGO_PATH = ROOT / "app" / "assets" / "thoraxia_logo.png"

def _logo_data_uri() -> str:
    """Encode le logo en base64 pour l'inline dans le HTML."""
    if not _LOGO_PATH.exists():
        return ""
    try:
        with open(_LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{b64}"
    except:
        return ""

_LOGO_URI = _logo_data_uri()

# ── Cache MedGemma ──
@st.cache_resource(show_spinner="Chargement de MedGemma 4B (~1 min au premier lancement)...")
def _ensure_model_loaded():
    from src.medgemma_inference import _load_model
    _load_model()
    return True

# ── Config page ──
_favicon_path = ROOT / "app" / "assets" / "thoraxia_logo.png"
st.set_page_config(
    page_title="THORAXIA",
    page_icon=str(_favicon_path) if _favicon_path.exists() else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Palette ──
S9="#0F172A"; S7="#334155"; S5="#64748B"; S3="#CBD5E1"; S2="#E2E8F0"
S1="#F1F5F9"; BG="#FAFAF9"; W="#FFFFFF"; AC="#C2410C"; OK="#059669"; AM="#B45309"

# E-mail du support technique
SUPPORT_EMAIL = "servicetechThoraxia@gmail.com"

# ═══════════════════ HELPERS SÉCURITÉ ═══════════════════
def _esc(text) -> str:
    """Échappe HTML pour éviter les injections XSS dans le contenu utilisateur."""
    return html.escape(str(text or ""))

# ═══════════════════ AUTHENTIFICATION ═══════════════════
USERS_FILE = ROOT / "data" / "users.json"
import os
SALT = os.getenv("THORAXIA_SALT", "thoraxia_efrei_mastercamp_2026")

def _hp(p): return hashlib.sha256((SALT+p).encode()).hexdigest()

def load_users():
    if not USERS_FILE.exists(): return {}
    try: return json.loads(USERS_FILE.read_text("utf-8"))
    except: return {}

def save_users(u):
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(u,indent=2,ensure_ascii=False),"utf-8")

def register_user(email,pw,name):
    e=email.strip().lower()
    if not e or "@" not in e: return False,"Adresse e-mail invalide."
    if len(pw)<6: return False,"Le mot de passe doit contenir au moins 6 caractères."
    if not name.strip(): return False,"Veuillez entrer votre nom."
    u=load_users()
    if e in u: return False,"Un compte existe déjà pour cet e-mail."
    u[e]={"password_hash":_hp(pw),"full_name":name.strip(),
          "created_at":datetime.utcnow().isoformat(timespec="seconds")+"Z"}
    save_users(u); return True,"Compte créé."

def verify_creds(email,pw):
    e=email.strip().lower(); u=load_users()
    if e not in u: return False,"Aucun compte pour cet e-mail.",None
    if u[e]["password_hash"]!=_hp(pw): return False,"Mot de passe incorrect.",None
    return True,"OK",{"email":e,**u[e]}

# ─── NOUVEAU : édition du profil ───
def update_user_info(old_email, new_email, new_full_name):
    """Met à jour le nom complet et/ou l'e-mail d'un utilisateur."""
    old_email = old_email.strip().lower()
    new_email = new_email.strip().lower()

    if not new_full_name.strip():
        return False, "Le nom ne peut pas être vide."
    if "@" not in new_email or "." not in new_email.split("@")[-1]:
        return False, "Adresse e-mail invalide."

    users = load_users()
    if old_email not in users:
        return False, "Utilisateur introuvable."

    # Si l'e-mail change, vérifier que le nouveau n'est pas déjà pris
    if new_email != old_email:
        if new_email in users:
            return False, "Cette adresse e-mail est déjà utilisée."
        # Déplacer l'entrée sous la nouvelle clé
        users[new_email] = users[old_email]
        del users[old_email]

    users[new_email]["full_name"] = new_full_name.strip()
    save_users(users)
    return True, "Informations mises à jour."

def change_password(email, old_pw, new_pw):
    """Change le mot de passe après vérification de l'ancien."""
    email = email.strip().lower()
    if len(new_pw) < 6:
        return False, "Le nouveau mot de passe doit contenir au moins 6 caractères."

    users = load_users()
    if email not in users:
        return False, "Utilisateur introuvable."
    if users[email]["password_hash"] != _hp(old_pw):
        return False, "Mot de passe actuel incorrect."

    users[email]["password_hash"] = _hp(new_pw)
    save_users(users)
    return True, "Mot de passe modifié."

# ═══════════════════ DOSSIERS PATIENTS ═══════════════════
RECORDS_DIR = ROOT / "data" / "patient_records"
RECORDS_IDX = RECORDS_DIR / "index.json"
IMAGES_DIR  = RECORDS_DIR / "images"

def _load_db():
    if not RECORDS_IDX.exists(): return {"records":{},"directories":["/"]}
    try: return json.loads(RECORDS_IDX.read_text("utf-8"))
    except: return {"records":{},"directories":["/"]}

def _save_db(db):
    RECORDS_DIR.mkdir(parents=True,exist_ok=True)
    RECORDS_IDX.write_text(json.dumps(db,indent=2,ensure_ascii=False),"utf-8")

def save_patient_record(info, analysis, latency, img_bytes, directory, user_email):
    db=_load_db(); rid=f"rec_{uuid.uuid4().hex[:8]}"
    IMAGES_DIR.mkdir(parents=True,exist_ok=True)
    (IMAGES_DIR/f"{rid}.png").write_bytes(img_bytes)
    if directory not in db["directories"]:
        parts=directory.strip("/").split("/")
        for i in range(len(parts)):
            p="/"+"/".join(parts[:i+1])
            if p not in db["directories"]: db["directories"].append(p)
    db["records"][rid]={**info,"image_filename":f"{rid}.png","analysis":analysis,
        "latency":latency,"directory":directory,
        "created_at":datetime.utcnow().isoformat(timespec="seconds")+"Z",
        "created_by":user_email}
    _save_db(db); return rid

def update_record(rid, updates):
    db=_load_db()
    if rid in db["records"]: db["records"][rid].update(updates); _save_db(db); return True
    return False

def delete_records(rids):
    db=_load_db()
    for rid in rids:
        if rid in db["records"]:
            f=IMAGES_DIR/db["records"][rid].get("image_filename","")
            if f.exists(): f.unlink()
            del db["records"][rid]
    _save_db(db)

def create_directory(path):
    db=_load_db(); p=("/"+path.strip("/")) if path.strip("/") else "/"
    if p not in db["directories"]:
        parts=p.strip("/").split("/")
        for i in range(len(parts)):
            pp="/"+"/".join(parts[:i+1])
            if pp not in db["directories"]: db["directories"].append(pp)
        _save_db(db)
    return p

def delete_directory(path):
    db=_load_db()
    to_del=[d for d in db["directories"] if d==path or d.startswith(path+"/")]
    for d in to_del:
        if d in db["directories"]: db["directories"].remove(d)
    rids=[r for r,v in db["records"].items() if v.get("directory","/")==path or v.get("directory","/").startswith(path+"/")]
    for rid in rids:
        f=IMAGES_DIR/db["records"][rid].get("image_filename","")
        if f.exists(): f.unlink()
        del db["records"][rid]
    _save_db(db)

def list_directory(path):
    db=_load_db(); path=path.rstrip("/") or "/"
    subdirs=set()
    prefix = path+"/" if path!="/" else "/"
    for d in db["directories"]:
        if d==path: continue
        if d.startswith(prefix):
            rest=d[len(prefix):]
            top=rest.split("/")[0]
            if top: subdirs.add(top)
    recs=[{"id":r,**v} for r,v in db["records"].items() if v.get("directory","/")==path]
    return sorted(subdirs), sorted(recs, key=lambda x:x.get("created_at",""), reverse=True)

def search_records(q):
    db=_load_db(); q=q.lower()
    return [{"id":r,**v} for r,v in db["records"].items()
            if q in " ".join([v.get("patient_lastname",""),v.get("patient_firstname",""),
                              v.get("patient_number",""),v.get("directory","")]).lower()]

def get_all_stats():
    """Retourne les stats globales pour le dashboard Documents."""
    db=_load_db()
    total=len(db["records"])
    if total==0: return {"total":0,"normal":0,"opacity":0,"uncertain":0,"dirs":0}
    counts={"normal":0,"suspected_opacity":0,"uncertain":0}
    for r in db["records"].values():
        pc=r.get("analysis",{}).get("predicted_class","")
        if pc in counts: counts[pc]+=1
    return {"total":total,"normal":counts["normal"],"opacity":counts["suspected_opacity"],
            "uncertain":counts["uncertain"],"dirs":len(db["directories"])-1}

# ═══════════════════ EXPORT PDF ═══════════════════
def generate_patient_pdf(rec, img_path):
    """Génère un PDF résumé du dossier patient."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
    except ImportError:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('title', parent=styles['Title'], fontSize=18,
                                  textColor=HexColor(S9), spaceAfter=6)
    subtitle_style = ParagraphStyle('sub', parent=styles['Normal'], fontSize=10,
                                     textColor=HexColor(S5), spaceAfter=20)
    h2_style = ParagraphStyle('h2', parent=styles['Heading2'], fontSize=13,
                               textColor=HexColor(S9), spaceAfter=8, spaceBefore=12)
    body = ParagraphStyle('body', parent=styles['Normal'], fontSize=10,
                          textColor=HexColor(S7), spaceAfter=4, leading=14)
    warn_style = ParagraphStyle('warn', parent=styles['Normal'], fontSize=9,
                                 textColor=HexColor(AC), alignment=TA_CENTER,
                                 spaceBefore=20, spaceAfter=6)

    story = []
    story.append(Paragraph("THORAXIA — Rapport d'analyse", title_style))
    story.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", subtitle_style))

    story.append(Paragraph("Informations du patient", h2_style))
    patient_data = [
        ["Nom", _esc(rec.get("patient_lastname", "—"))],
        ["Prénom", _esc(rec.get("patient_firstname", "—"))],
        ["Date de naissance", _esc(rec.get("patient_dob", "—"))],
        ["Numéro patient", _esc(rec.get("patient_number", "—"))],
        ["Date du traitement", _esc(rec.get("treatment_date", "—"))],
        ["Dossier", _esc(rec.get("directory", "/"))],
    ]
    t = Table(patient_data, colWidths=[5*cm, 10*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), HexColor(S1)),
        ('TEXTCOLOR', (0,0), (0,-1), HexColor(S9)),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor(S2)),
    ]))
    story.append(t)

    if img_path and img_path.exists():
        story.append(Paragraph("Radiographie analysée", h2_style))
        try:
            story.append(RLImage(str(img_path), width=8*cm, height=8*cm))
        except:
            story.append(Paragraph("<i>Image non affichable</i>", body))

    analysis = rec.get("analysis", {})
    story.append(Paragraph("Résultat de l'analyse MedGemma", h2_style))
    pc = analysis.get("predicted_class", "?")
    cd = {"normal":"Normal","suspected_opacity":"Suspicion d'opacité",
          "uncertain":"Incertain"}.get(pc, pc)
    analysis_data = [
        ["Classification", _esc(cd)],
        ["Confiance", f"{analysis.get('confidence', 0):.2f}"],
        ["Qualité d'image", _esc(analysis.get("image_quality", "—"))],
        ["Latence", f"{rec.get('latency', 0)} s"],
        ["Modèle", _esc(analysis.get("model_name", "google/medgemma-4b-it"))],
    ]
    t = Table(analysis_data, colWidths=[5*cm, 10*cm])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,-1), HexColor(S1)),
        ('TEXTCOLOR', (0,0), (0,-1), HexColor(S9)),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('GRID', (0,0), (-1,-1), 0.5, HexColor(S2)),
    ]))
    story.append(t)

    evidence = analysis.get("visual_evidence", [])
    if evidence:
        story.append(Paragraph("Observations visuelles", h2_style))
        for ev in evidence:
            story.append(Paragraph(f"• {_esc(ev)}", body))

    just = analysis.get("justification", "")
    if just:
        story.append(Paragraph("Justification du modèle", h2_style))
        story.append(Paragraph(_esc(just), body))

    comments = rec.get("comments", [])
    if comments:
        story.append(Paragraph("Commentaires du radiologue", h2_style))
        for cm in comments:
            story.append(Paragraph(f"<b>{cm.get('date','')[:10]}</b> — {_esc(cm.get('text',''))}", body))
            story.append(Spacer(1, 4))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "AVERTISSEMENT — Prototype pédagogique. Ce rapport ne constitue pas un diagnostic médical. "
        "Toute interprétation clinique doit être validée par un professionnel qualifié.",
        warn_style))

    doc.build(story)
    buf.seek(0)
    return buf.read()

# ═══════════════════ CSS ═══════════════════
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com/" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
html, body, [class*="css"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-feature-settings: 'cv11', 'ss01', 'ss03';
}}
.nav-wordmark{{font-size:1rem;font-weight:500;letter-spacing:.22em;color:{S9};}}
.user-badge{{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;
  background:{W};border:.5px solid {S2};border-radius:6px;font-size:.8rem;color:{S9};white-space:nowrap;}}
.user-avatar{{width:22px;height:22px;background:{S9};color:{W};border-radius:50%;
  display:flex;align-items:center;justify-content:center;font-size:.65rem;font-weight:500;}}

.stButton>button{{background:{W};border:.5px solid {S2};color:{S9};border-radius:6px;
  font-weight:500;font-size:.82rem;padding:6px 10px;white-space:nowrap!important;}}
.stButton>button:hover{{border-color:{S7};background:{S1};}}
.stButton>button[kind="primary"]{{background:{S9};color:{W};border-color:{S9};}}
.stButton>button[kind="primary"]:hover{{background:{S7};border-color:{S7};}}
.stButton>button:disabled{{background:{S1};color:{S5};border-color:{S2};}}

.page-title{{font-size:1.5rem;font-weight:500;color:{S9};letter-spacing:-.015em;margin:0 0 4px;}}
.page-subtitle{{font-size:.88rem;color:{S5};margin:0 0 20px;}}
.section-label{{font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;color:{S5};
  font-weight:500;margin:22px 0 10px;}}

.auth-shell{{max-width:400px;margin:40px auto 0;background:{W};border:.5px solid {S2};
  border-radius:10px;padding:36px;}}
.auth-title{{font-size:1.3rem;font-weight:500;color:{S9};margin:0 0 6px;}}
.auth-sub{{font-size:.85rem;color:{S5};margin:0 0 24px;}}
.auth-switch{{text-align:center;font-size:.82rem;color:{S5};margin-top:20px;
  padding-top:18px;border-top:.5px solid {S2};}}

.metric-card{{background:{W};border:.5px solid {S2};border-radius:8px;padding:14px 16px;height:100%;}}
.metric-label{{font-size:.72rem;color:{S5};margin-bottom:6px;}}
.metric-value{{font-size:1.6rem;font-weight:500;color:{S9};letter-spacing:-.02em;
  font-variant-numeric:tabular-nums;line-height:1.1;}}
.metric-sub{{font-size:.7rem;color:{OK};margin-top:4px;}}
.metric-sub.neutral{{color:{S5};}}

.disclaimer{{background:#FEF9F5;border:.5px solid #FCD9B6;border-left:2px solid {AC};
  color:#7C2D12;padding:10px 14px;border-radius:6px;font-size:.82rem;margin:8px 0 22px;
  display:flex;align-items:center;gap:10px;line-height:1.5;}}
.disclaimer strong{{font-weight:500;}}

.confirm-box{{background:#FEF2F2;border:.5px solid #FECACA;border-left:2px solid #DC2626;
  color:#7F1D1D;padding:12px 16px;border-radius:6px;font-size:.85rem;margin:12px 0;
  line-height:1.5;}}
.confirm-box strong{{color:#991B1B;font-weight:500;}}

.result-card{{background:{W};border:.5px solid {S2};border-radius:8px;padding:18px;}}
.result-header{{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:4px;}}
.result-label{{font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:{S5};font-weight:500;}}
.result-class{{font-size:1.3rem;font-weight:500;color:{S9};margin-top:2px;}}
.result-conf-val{{font-size:1.3rem;font-weight:500;color:{S9};margin-top:2px;
  font-variant-numeric:tabular-nums;text-align:right;}}
.result-bar-track{{height:4px;background:{S1};border-radius:2px;margin:14px 0 18px;overflow:hidden;}}
.result-bar-fill{{height:100%;background:{S9};}}
.result-bar-fill.opacity{{background:{AC};}}
.result-bar-fill.normal{{background:{OK};}}
.result-bar-fill.uncertain{{background:{AM};}}
.evidence-block{{font-size:.85rem;color:{S7};line-height:1.6;padding-left:12px;
  border-left:1.5px solid {S2};margin:8px 0 4px;}}
.result-meta{{display:flex;gap:24px;margin-top:16px;padding-top:14px;border-top:.5px solid {S2};}}
.result-meta-item .k{{font-size:.68rem;color:{S5};text-transform:uppercase;letter-spacing:.08em;}}
.result-meta-item .v{{font-size:.82rem;color:{S9};font-family:'JetBrains Mono',monospace;margin-top:2px;}}

.state-banner{{padding:8px 14px;border-radius:6px;font-size:.82rem;font-weight:500;
  margin:12px 0 4px;display:flex;align-items:center;gap:8px;}}
.state-analyzing{{background:#FFFBEB;color:#78350F;border:.5px solid #FDE68A;}}
.state-analyzing::before{{content:'';width:8px;height:8px;background:{AM};border-radius:50%;
  animation:pulse 1.4s ease-in-out infinite;}}
.state-done{{background:#F0FDF4;color:#14532D;border:.5px solid #BBF7D0;}}
.state-done::before{{content:'';width:8px;height:8px;background:{OK};border-radius:50%;}}
@keyframes pulse{{0%,100%{{opacity:1;}}50%{{opacity:.35;}}}}

.about-hero{{padding:20px 0 32px;border-bottom:.5px solid {S2};margin-bottom:32px;}}
.about-hero .eyebrow{{font-size:.72rem;text-transform:uppercase;letter-spacing:.15em;color:{S5};margin-bottom:12px;}}
.about-hero h1{{font-size:2rem;font-weight:500;color:{S9};letter-spacing:-.02em;margin:0 0 12px;line-height:1.2;}}
.about-hero p{{font-size:1rem;color:{S7};line-height:1.65;max-width:680px;margin:0;}}
.about-section{{padding:24px 0;border-bottom:.5px solid {S2};}}
.about-section:last-child{{border-bottom:none;}}
.about-section .num{{font-family:'JetBrains Mono',monospace;font-size:.75rem;color:{S5};margin-bottom:8px;}}
.about-section h2{{font-size:1.3rem;font-weight:500;color:{S9};margin:0 0 12px;}}
.about-section p{{font-size:.92rem;color:{S7};line-height:1.7;max-width:720px;margin:0 0 12px;}}
.about-section ul{{font-size:.92rem;color:{S7};line-height:1.75;padding-left:20px;margin:8px 0 0;max-width:720px;}}
.about-section ul li{{margin-bottom:4px;}}
.about-section ul li strong{{color:{S9};font-weight:500;}}
.team-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:16px;}}
.team-item{{background:{W};border:.5px solid {S2};border-radius:6px;padding:12px 14px;}}
.team-item .name{{font-size:.88rem;color:{S9};font-weight:500;margin:0 0 2px;}}
.team-item .role{{font-size:.78rem;color:{S5};margin:0;}}
.step-list{{counter-reset:step;padding:0;margin:12px 0 0;list-style:none;max-width:720px;}}
.step-list li{{counter-increment:step;padding:12px 0 12px 44px;border-top:.5px solid {S2};
  position:relative;font-size:.9rem;color:{S7};line-height:1.6;}}
.step-list li:last-child{{border-bottom:.5px solid {S2};}}
.step-list li::before{{content:counter(step,decimal-leading-zero);font-family:'JetBrains Mono',monospace;
  font-size:.72rem;color:{S5};position:absolute;left:0;top:14px;}}
.step-list li strong{{color:{S9};font-weight:500;}}

.breadcrumb{{font-size:.82rem;color:{S5};margin-bottom:16px;}}
.breadcrumb a{{color:{S9};text-decoration:none;font-weight:500;}}
.stats-grid{{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:22px;}}
.stat-mini{{background:{W};border:.5px solid {S2};border-radius:6px;padding:10px 12px;}}
.stat-mini .k{{font-size:.68rem;color:{S5};text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;}}
.stat-mini .v{{font-size:1.3rem;font-weight:500;color:{S9};font-variant-numeric:tabular-nums;}}
.comment-box{{background:{S1};border:.5px solid {S2};border-radius:6px;padding:10px 14px;margin-bottom:6px;}}
.comment-text{{font-size:.82rem;color:{S7};line-height:1.5;max-height:150px;overflow-y:auto;
  white-space:pre-wrap;word-break:break-word;}}
.comment-date{{font-size:.68rem;color:{S5};margin-top:6px;}}

/* Profile page */
.profile-card{{background:{W};border:.5px solid {S2};border-radius:8px;padding:24px;margin-bottom:16px;}}
.profile-avatar-large{{width:80px;height:80px;background:{S9};color:{W};border-radius:50%;
  display:flex;align-items:center;justify-content:center;font-size:1.6rem;font-weight:500;
  margin:0 auto 16px;letter-spacing:.02em;}}
.profile-name{{font-size:1.2rem;font-weight:500;color:{S9};text-align:center;margin:0 0 2px;}}
.profile-email{{font-size:.85rem;color:{S5};text-align:center;margin:0;}}

.stTextInput input,.stTextArea textarea{{background:{W}!important;border:.5px solid {S2}!important;
  border-radius:6px!important;font-size:.88rem!important;color:{S9}!important;}}
.stTextInput input:focus,.stTextArea textarea:focus{{border-color:{S7}!important;
  box-shadow:0 0 0 3px {S1}!important;}}
.stTextInput label,.stDateInput label,.stTextArea label{{font-size:.78rem!important;
  color:{S7}!important;font-weight:400!important;}}
.stTextArea textarea{{max-height:200px!important;resize:vertical!important;}}

[data-testid="stExpander"]{{border:.5px solid {S2}!important;border-radius:6px!important;
  margin-bottom:8px!important;background:{W}!important;box-shadow:none!important;}}
[data-testid="stExpander"] details summary{{background:{W}!important;color:{S9}!important;
  padding:10px 14px!important;font-size:.85rem!important;font-weight:500!important;}}
[data-testid="stExpander"] details summary:hover{{background:{S1}!important;}}
[data-testid="stExpander"] details summary p,
[data-testid="stExpander"] details summary span,
[data-testid="stExpander"] details summary strong{{color:{S9}!important;font-weight:500!important;margin:0!important;}}
[data-testid="stExpander"] details[open]>div{{background:{W}!important;padding:12px 16px!important;
  border-top:.5px solid {S2}!important;}}
[data-testid="stExpander"] details>div p,
[data-testid="stExpander"] details>div li,
[data-testid="stExpander"] details>div span,
[data-testid="stExpander"] details>div strong{{color:{S7}!important;font-size:.85rem!important;line-height:1.6!important;}}
[data-testid="stExpander"] details>div code{{background:{S1}!important;color:{S9}!important;
  padding:1px 6px!important;border-radius:4px!important;font-size:.78rem!important;
  font-family:'JetBrains Mono',monospace!important;}}
[data-testid="stJson"]{{background:{S1}!important;border-radius:6px!important;padding:12px!important;
  border:.5px solid {S2}!important;font-family:'JetBrains Mono',monospace!important;font-size:.78rem!important;}}
[data-testid="stJson"] *{{color:{S9}!important;background:transparent!important;}}
[data-testid="stFileUploader"] section{{background:{W};border:1px dashed {S2};border-radius:6px;padding:20px;}}

.footer{{color:{S5};font-size:.75rem;padding:24px 0 8px;border-top:.5px solid {S2};margin-top:32px;line-height:1.7;}}
.footer code{{background:{S1};color:{S7};padding:1px 6px;border-radius:3px;
  font-family:'JetBrains Mono',monospace;font-size:.72rem;}}

#MainMenu,footer{{visibility:hidden;}}
.stDeployButton{{display:none;}}
[data-testid="stHeader"]{{background:transparent;}}
[data-testid="stSidebar"]{{display:none;}}
</style>
""", unsafe_allow_html=True)

# ═══════════════════ SESSION STATE ═══════════════════
_DS = {
    "current_user":None,"session_accounts":[],"auth_page":"login",
    "current_page":"app","show_switch_menu":False,
    "selected_demo":None,"active_demo_idx":None,
    "analyzing":False,"last_pred":None,"last_latency":None,"last_error":None,"last_source":None,
    "multi_results":{},"multi_analyzing":False,
    "save_open":None,"save_success":None,
    "docs_dir":"/","docs_record":None,"docs_editing":False,
    "docs_selected":[],"docs_new_dir":False,
    "confirm_bulk_del":False,"confirm_dir_del":None,"confirm_rec_del":None,
    "tests_filter": "all",
    "tests_selected": None,
}
for k,v in _DS.items():
    if k not in st.session_state: st.session_state[k]=v

def _ini(name):
    p=[x for x in name.strip().split() if x]
    if not p: return "?"
    if len(p)==1: return p[0][:2].upper()
    return (p[0][0]+p[-1][0]).upper()

# ═══════════════════ SVG ICONS ═══════════════════
_WARN_SVG = f'<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="{AC}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="flex-shrink:0"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'

# ═══════════════════ NAVBAR ═══════════════════
def render_navbar():
    u = st.session_state.current_user
    if u:
        # Post-connexion : 6 colonnes
        c1, c2, c3, c4, c5, c6 = st.columns([2, 1, 1, 1, 1, 2.5])
    else:
        # Pré-connexion : 4 colonnes (logo + spacer + À propos + Connexion)
        c1, spacer_col, c_about, c_login = st.columns([2, 5, 1.2, 1.2])

    with c1:
        if _LOGO_URI:
            st.markdown(
                f'<div style="padding-top:4px;display:flex;align-items:center;">'
                f'<img src="{_LOGO_URI}" style="height:50px;width:auto;object-fit:contain;" alt="THORAXIA"/>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown('<div style="padding-top:6px"><span class="nav-wordmark">THORAXIA</span></div>',
                        unsafe_allow_html=True)

    if u:
        # Onglets de navigation
        for col, label, page, key in [
            (c2, "Analyse", "app", "nav_app"),
            (c3, "Cas de test", "tests", "nav_tests"),
            (c4, "Documents", "documents", "nav_docs"),
            (c5, "À propos", "about", "nav_about")
        ]:
            with col:
                btype = "primary" if st.session_state.current_page == page else "secondary"
                if st.button(label, key=key, use_container_width=True, type=btype):
                    st.session_state.current_page = page
                    st.session_state.show_switch_menu = False
                    st.session_state.docs_record = None
                    st.rerun()

        # Zone compte (c6)
        with c6:
            a, b, c = st.columns([1.8, 1, 1])
            with a:
                # Badge cliquable → page profil
                initials = _ini(u["full_name"])
                first_name = u["full_name"].split()[0]
                is_profile = st.session_state.current_page == "profile"
                btype = "primary" if is_profile else "secondary"
                if st.button(f"● {initials}  {first_name}", key="nav_profile",
                             use_container_width=True, type=btype,
                             help="Voir mon compte"):
                    st.session_state.current_page = "profile"
                    st.session_state.show_switch_menu = False
                    st.rerun()
            with b:
                if st.button("⇄ Changer", key="nav_sw", use_container_width=True):
                    st.session_state.show_switch_menu = not st.session_state.show_switch_menu
                    st.rerun()
            with c:
                if st.button("Quitter", key="nav_out", use_container_width=True):
                    for k in ["current_user", "selected_demo", "last_pred",
                              "analyzing", "active_demo_idx", "multi_results", "multi_analyzing"]:
                        st.session_state[k] = _DS.get(k)
                    st.session_state.auth_page = "login"
                    st.session_state.show_switch_menu = False
                    st.session_state.current_page = "app"
                    st.rerun()
    else:
        # NAVBAR PRÉ-CONNEXION : À propos + Connexion
        with c_about:
            is_about = st.session_state.current_page == "about"
            btype = "primary" if is_about else "secondary"
            if st.button("À propos", key="nav_about_prelog", use_container_width=True, type=btype):
                st.session_state.current_page = "about"
                st.rerun()
        with c_login:
            is_login = st.session_state.current_page != "about"
            btype = "primary" if is_login else "secondary"
            if st.button("Connexion", key="nav_in", use_container_width=True, type=btype):
                st.session_state.current_page = "app"
                st.session_state.auth_page = "login"
                st.rerun()

    st.markdown(f'<div style="height:1px;background:{S2};margin:8px 0 22px"></div>', unsafe_allow_html=True)

    # Menu changement de compte
    if u and st.session_state.show_switch_menu:
        others = [a for a in st.session_state.session_accounts if a["email"] != u["email"]]
        st.markdown(f'<div style="background:{W};border:.5px solid {S2};border-radius:8px;'
                    f'padding:14px 18px;margin-bottom:18px">'
                    f'<div style="font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;'
                    f'color:{S5};margin-bottom:8px">Changer de compte</div>', unsafe_allow_html=True)
        if others:
            for i, acc in enumerate(others):
                x1, x2 = st.columns([4, 1])
                with x1:
                    st.markdown(f'<div style="font-size:.85rem;color:{S9};font-weight:500">{_esc(acc["full_name"])}</div>'
                                f'<div style="font-size:.72rem;color:{S5}">{_esc(acc["email"])}</div>', unsafe_allow_html=True)
                with x2:
                    if st.button("Utiliser", key=f"sw_{i}", use_container_width=True):
                        st.session_state.current_user = acc
                        st.session_state.show_switch_menu = False
                        st.session_state.last_pred = None
                        st.session_state.multi_results = {}
                        st.rerun()
        else:
            st.markdown(f'<p style="font-size:.82rem;color:{S5}">Aucun autre compte dans cette session.</p>',
                        unsafe_allow_html=True)
        if st.button("Se connecter avec un autre compte", key="sw_new"):
            st.session_state.current_user = None
            st.session_state.auth_page = "login"
            st.session_state.show_switch_menu = False
            st.session_state.current_page = "app"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════ LOGIN ═══════════════════
def render_login():
    st.markdown('<div class="auth-shell">', unsafe_allow_html=True)
    if _LOGO_URI:
        st.markdown(
            f'<div style="text-align:center;margin-bottom:20px;">'
            f'<img src="{_LOGO_URI}" style="height:60px;width:auto;" alt="THORAXIA"/>'
            f'</div>',
            unsafe_allow_html=True
        )
    st.markdown('<div class="auth-title">Connexion à THORAXIA</div>', unsafe_allow_html=True)
    st.markdown("<div class=\"auth-sub\">Accédez à l'assistant en radiologie.</div>", unsafe_allow_html=True)
    email = st.text_input("E-mail", key="li_e", placeholder="nom@example.com")
    pw = st.text_input("Mot de passe", key="li_p", type="password", placeholder="Au moins 6 caractères")
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    if st.button("Se connecter", type="primary", use_container_width=True, key="li_go"):
        ok, msg, ud = verify_creds(email, pw)
        if ok:
            st.session_state.current_user = ud
            emails = [a["email"] for a in st.session_state.session_accounts]
            if ud["email"] not in emails:
                st.session_state.session_accounts.append(ud)
            st.session_state.current_page = "app"
            st.rerun()
        else:
            st.error(msg)
    st.markdown('<div class="auth-switch">Pas encore de compte ?</div>', unsafe_allow_html=True)
    if st.button("Créer un compte", use_container_width=True, key="li_reg"):
        st.session_state.auth_page = "register"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════ REGISTER ═══════════════════
def render_register():
    st.markdown('<div class="auth-shell">', unsafe_allow_html=True)
    if _LOGO_URI:
        st.markdown(
            f'<div style="text-align:center;margin-bottom:20px;">'
            f'<img src="{_LOGO_URI}" style="height:60px;width:auto;" alt="THORAXIA"/>'
            f'</div>',
            unsafe_allow_html=True
        )
    st.markdown('<div class="auth-title">Créer votre compte</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-sub">Un compte local pour la démo THORAXIA.</div>', unsafe_allow_html=True)
    name = st.text_input("Nom complet", key="rg_n", placeholder="Jean Dupont")
    email = st.text_input("E-mail", key="rg_e", placeholder="nom@example.com")
    pw = st.text_input("Mot de passe", key="rg_p", type="password", placeholder="Au moins 6 caractères")
    pw2 = st.text_input("Confirmer", key="rg_p2", type="password")
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    if st.button("Créer le compte", type="primary", use_container_width=True, key="rg_go"):
        if pw != pw2:
            st.error("Les mots de passe ne correspondent pas.")
        else:
            ok, msg = register_user(email, pw, name)
            if ok:
                _, _, ud = verify_creds(email, pw)
                st.session_state.current_user = ud
                st.session_state.session_accounts.append(ud)
                st.session_state.current_page = "about"
                st.success("Compte créé. Bienvenue !")
                time.sleep(.8)
                st.rerun()
            else:
                st.error(msg)
    st.markdown('<div class="auth-switch">Déjà un compte ?</div>', unsafe_allow_html=True)
    if st.button("Se connecter", use_container_width=True, key="rg_li"):
        st.session_state.auth_page = "login"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════ PROFILE ═══════════════════
def render_profile():
    user = st.session_state.current_user
    if user is None:
        st.error("Non connecté.")
        return

    st.markdown('<h1 class="page-title">Mon compte</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Gérez vos informations personnelles et votre mot de passe.</p>',
                unsafe_allow_html=True)

    # === CARTE PROFIL ===
    initials = _ini(user["full_name"])
    st.markdown(f'''
<div class="profile-card">
    <div class="profile-avatar-large">{initials}</div>
    <div class="profile-name">{_esc(user["full_name"])}</div>
    <div class="profile-email">{_esc(user["email"])}</div>
</div>''', unsafe_allow_html=True)

    # === INFORMATIONS PERSONNELLES ===
    st.markdown('<div class="section-label">Informations personnelles</div>', unsafe_allow_html=True)

    with st.form("profile_info_form"):
        col1, col2 = st.columns(2)
        with col1:
            new_name = st.text_input("Nom complet", value=user["full_name"], key="prof_name")
        with col2:
            new_email = st.text_input("Adresse e-mail", value=user["email"], key="prof_email")

        submitted = st.form_submit_button("Sauvegarder les modifications", type="primary",
                                          use_container_width=True)

        if submitted:
            ok, msg = update_user_info(user["email"], new_email, new_name)
            if ok:
                # Mettre à jour la session
                st.session_state.current_user["full_name"] = new_name.strip()
                st.session_state.current_user["email"] = new_email.strip().lower()
                # Mettre à jour aussi le pool de comptes de la session
                for a in st.session_state.session_accounts:
                    if a["email"] == user["email"]:
                        a["email"] = new_email.strip().lower()
                        a["full_name"] = new_name.strip()
                st.success(msg)
                time.sleep(0.8)
                st.rerun()
            else:
                st.error(msg)

    # === CHANGER LE MOT DE PASSE ===
    st.markdown('<div class="section-label">Changer le mot de passe</div>', unsafe_allow_html=True)

    with st.form("profile_pw_form"):
        old_pw = st.text_input("Mot de passe actuel", type="password", key="prof_old_pw",
                                placeholder="Entrez votre mot de passe actuel")
        col1, col2 = st.columns(2)
        with col1:
            new_pw = st.text_input("Nouveau mot de passe", type="password",
                                    key="prof_new_pw", placeholder="Au moins 6 caractères")
        with col2:
            new_pw2 = st.text_input("Confirmer", type="password",
                                     key="prof_new_pw2", placeholder="Confirmez le nouveau")

        submitted_pw = st.form_submit_button("Changer le mot de passe", type="primary",
                                              use_container_width=True)

        if submitted_pw:
            if not old_pw:
                st.error("Veuillez entrer votre mot de passe actuel.")
            elif new_pw != new_pw2:
                st.error("Les nouveaux mots de passe ne correspondent pas.")
            elif not new_pw:
                st.error("Le nouveau mot de passe ne peut pas être vide.")
            else:
                ok, msg = change_password(user["email"], old_pw, new_pw)
                if ok:
                    st.success(msg)
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.error(msg)

    # === INFORMATIONS DU COMPTE ===
    st.markdown('<div class="section-label">Informations du compte</div>', unsafe_allow_html=True)
    created = user.get("created_at", "—")
    created_display = created[:10] if isinstance(created, str) and len(created) >= 10 else created
    st.markdown(f'''
<div style="background:{W};border:.5px solid {S2};border-radius:6px;padding:14px 18px;
            font-size:.85rem;color:{S7}">
    <strong style="color:{S9};font-weight:500">Compte créé le</strong> : {_esc(created_display)}
</div>''', unsafe_allow_html=True)

# ═══════════════════ ABOUT ═══════════════════
def render_about():
    st.markdown("""
<div class="about-hero">
    <div class="eyebrow">À propos du projet</div>
    <h1>Un assistant en radiologie par IA, conçu de manière responsable.</h1>
    <p>THORAXIA est un prototype pédagogique développé à l'EFREI Paris dans le cadre du MasterCamp 2026
    de la majeure Big Data &amp; IA. Il explore comment un modèle de vision-langage médical peut être intégré
    dans un flux de travail clinique tout en maintenant le jugement humain au centre de chaque décision.</p>
</div>""", unsafe_allow_html=True)
    st.markdown("""
<div class="about-section"><div class="num">01 &mdash; Qui sommes-nous</div>
<h2>Notre équipe</h2>
<p>Six étudiants de l'EFREI Paris, majeure Big Data &amp; IA (2025–2026), encadrés par Badr Tajini.
Notre objectif : maquetter un flux de radiologie assisté par une IA responsable — un second lecteur,
pas un remplacement.</p></div>""", unsafe_allow_html=True)
    st.markdown('<div class="team-grid">', unsafe_allow_html=True)
    for n, r in [("Souhaib Sghaier", "Chef de projet"), ("Kevin Sivaharan", "Analyste business"),
                 ("Sajin Sivasaranam", "Ingénierie"), ("Pratheep Parthepan", "Communication"),
                 ("Laurent Qiu", "Assurance qualité"), ("Yanick Shan", "Documentation")]:
        st.markdown(f'<div class="team-item"><div class="name">{n}</div><div class="role">{r}</div></div>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="about-section"><div class="num">02 &mdash; Ce que fait THORAXIA</div>
<h2>Fonctionnement</h2>
<p>THORAXIA analyse des radiographies thoraciques de face et renvoie une prédiction structurée et auditable.
Le pipeline repose sur <code>google/medgemma-4b-it</code>, un modèle vision-langage médical de Google.</p>
<ul>
<li><strong>Classification</strong> — trois classes : normal, suspicion d'opacité, incertain.</li>
<li><strong>Preuves visuelles</strong> — le modèle décrit ce qu'il observe en langage clair.</li>
<li><strong>Score de confiance</strong> — valeur calibrée entre 0 et 1.</li>
<li><strong>Garde-fous</strong> — seuil 0,60 et sortie JSON validée.</li>
<li><strong>Reproductibilité</strong> — modèle, version du prompt et latence tracés.</li>
<li><strong>Gestion des dossiers patients</strong> — sauvegarde, recherche et export PDF.</li>
</ul></div>""", unsafe_allow_html=True)
    st.markdown("""
<div class="about-section"><div class="num">03 &mdash; Mode d'emploi</div>
<h2>Comment utiliser THORAXIA</h2>
<p>L'outil complète, sans remplacer, la lecture d'un radiologue.</p>
<ol class="step-list">
<li><strong>Ouvrez la page Analyse</strong> depuis la navigation.</li>
<li><strong>Choisissez un exemple ou importez vos images.</strong> La galerie contient des cas RSNA Pneumonia.
Vous pouvez aussi importer une ou plusieurs radiographies (PNG, JPG, JPEG).</li>
<li><strong>Lancez l'analyse.</strong> L'image est envoyée à MedGemma (quelques secondes).</li>
<li><strong>Lisez la sortie structurée</strong> : classe, confiance, preuves visuelles, justification, limites.</li>
<li><strong>Sauvegardez si besoin</strong> les données du patient dans un dossier organisé.</li>
<li><strong>Retrouvez, éditez et exportez</strong> les dossiers via la page Documents.</li>
<li><strong>Vérifiez avec un professionnel.</strong> Aucun résultat ne constitue un diagnostic.</li>
</ol></div>""", unsafe_allow_html=True)
    st.markdown("""
<div class="about-section"><div class="num">04 &mdash; Limites</div>
<h2>Ce que THORAXIA n'est pas</h2>
<p>THORAXIA est un prototype pédagogique évalué sur 50 cas RSNA (macro-F1 0,83). Ce n'est pas un dispositif
médical marqué CE ou approuvé FDA. Il ne doit jamais servir à prendre des décisions cliniques.</p></div>""",
                unsafe_allow_html=True)

# ═══════════════════ RENDER RESULT CARD ═══════════════════
def _render_result(pred, latency, with_expanders=True):
    pc=pred.get("predicted_class","?"); co=pred.get("confidence",0)
    cd="suspicion d'opacité" if pc=="suspected_opacity" else ("incertain" if pc=="uncertain" else pc)
    bc={"normal":"normal","suspected_opacity":"opacity","uncertain":"uncertain"}.get(pc,"")
    ev=pred.get("visual_evidence",[])
    et=" ".join([_esc(x) for x in ev]) if ev else "Aucune observation spécifique."
    st.markdown(f"""
<div class="result-card"><div class="result-header">
<div><div class="result-label">Classe</div><div class="result-class">{_esc(cd)}</div></div>
<div><div class="result-label" style="text-align:right">Confiance</div>
<div class="result-conf-val">{co:.2f}</div></div></div>
<div class="result-bar-track"><div class="result-bar-fill {bc}" style="width:{co*100:.0f}%"></div></div>
<div class="result-label">Preuves visuelles</div><div class="evidence-block">{et}</div>
<div class="result-meta">
<div class="result-meta-item"><div class="k">Latence</div><div class="v">{latency}s</div></div>
<div class="result-meta-item"><div class="k">Qualité</div><div class="v">{_esc(pred.get('image_quality','—'))}</div></div>
<div class="result-meta-item"><div class="k">Prompt</div><div class="v">{_esc(pred.get('prompt_version','—'))}</div></div>
</div></div>""", unsafe_allow_html=True)
    if with_expanders:
        with st.expander("Justification du modèle"): st.markdown(_esc(pred.get("justification","—")))
        with st.expander("Limites identifiées"):
            lims=pred.get("limitations",[])
            if lims:
                for l in lims: st.markdown(f"- {_esc(l)}")
            else: st.caption("Aucune limite explicite.")
        with st.expander("Métadonnées techniques"):
            st.markdown(f"- Modèle : `{_esc(pred.get('model_name','?'))}`")
            st.markdown(f"- Prompt : `{_esc(pred.get('prompt_version','?'))}`")
            st.markdown(f"- Latence : `{latency}s`")
            st.markdown(f"- Classe : `{_esc(pc)}`  ·  Confiance : `{co:.4f}`")
        with st.expander("Sortie JSON brute"): st.json(pred)

# ═══════════════════ SAVE FORM ═══════════════════
def render_save_form(img_bytes, pred, latency, key_prefix):
    with st.expander("Sauvegarder les données du patient"):
        c1,c2=st.columns(2)
        with c1: ln=st.text_input("Nom",key=f"{key_prefix}_ln",placeholder="Dupont")
        with c2: fn=st.text_input("Prénom",key=f"{key_prefix}_fn",placeholder="Jean")
        c3,c4=st.columns(2)
        with c3: dob=st.date_input("Date de naissance",key=f"{key_prefix}_dob",
                                    value=date(1980,1,1),
                                    min_value=date(1900,1,1),
                                    max_value=date.today())
        with c4: td=st.date_input("Date du traitement",key=f"{key_prefix}_td",value=date.today())
        pn=st.text_input("Numéro de patient",key=f"{key_prefix}_pn",placeholder="P-2026-001")
        comment=st.text_area("Commentaire",key=f"{key_prefix}_cm",placeholder="Notes cliniques...",
                             max_chars=5000,height=100)
        db=_load_db()
        dirs=sorted(db["directories"])
        dir_opts=dirs+["+ Nouveau dossier"]
        chosen=st.selectbox("Dossier de destination",dir_opts,key=f"{key_prefix}_dir")
        target_dir=chosen
        if chosen=="+ Nouveau dossier":
            target_dir="/"+st.text_input("Nom du nouveau dossier",key=f"{key_prefix}_nd",
                                          placeholder="Urgences/Janvier").strip("/")
        if st.button("Sauvegarder",type="primary",use_container_width=True,key=f"{key_prefix}_save"):
            if not ln.strip() or not fn.strip():
                st.error("Le nom et le prénom sont requis.")
            else:
                info={"patient_lastname":ln.strip(),"patient_firstname":fn.strip(),
                      "patient_dob":str(dob),"patient_number":pn.strip(),
                      "treatment_date":str(td),
                      "comments":[{"text":comment.strip(),"date":datetime.utcnow().isoformat(timespec="seconds")+"Z"}] if comment.strip() else []}
                rid=save_patient_record(info,pred,latency,img_bytes,target_dir,
                                        st.session_state.current_user["email"])
                st.success(f"Dossier sauvegardé (réf. {rid}).")

# ═══════════════════ DOCUMENTS ═══════════════════
def render_documents():
    if st.session_state.docs_record:
        render_record_detail(); return

    st.markdown('<h1 class="page-title">Documents</h1>',unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Dossiers patients sauvegardés.</p>',unsafe_allow_html=True)

    stats = get_all_stats()
    st.markdown(f'''
<div class="stats-grid">
<div class="stat-mini"><div class="k">Total dossiers</div><div class="v">{stats["total"]}</div></div>
<div class="stat-mini"><div class="k">Normal</div><div class="v" style="color:{OK}">{stats["normal"]}</div></div>
<div class="stat-mini"><div class="k">Suspicion</div><div class="v" style="color:{AC}">{stats["opacity"]}</div></div>
<div class="stat-mini"><div class="k">Incertain</div><div class="v" style="color:{AM}">{stats["uncertain"]}</div></div>
<div class="stat-mini"><div class="k">Dossiers créés</div><div class="v">{stats["dirs"]}</div></div>
</div>''',unsafe_allow_html=True)

    q=st.text_input("Rechercher un patient (nom, prénom, numéro)",key="docs_search",
                    placeholder="Rechercher...",label_visibility="collapsed")
    if q.strip():
        results=search_records(q.strip())
        st.markdown(f'<div class="section-label">{len(results)} résultat(s)</div>',unsafe_allow_html=True)
        if not results:
            st.info("Aucun résultat pour cette recherche.")
        else:
            _render_records_list(results,"search")
        return

    cur=st.session_state.docs_dir
    parts=cur.strip("/").split("/") if cur!="/" else []
    bc_html='<div class="breadcrumb">Racine'
    for p in parts:
        bc_html+=f' / {_esc(p)}'
    bc_html+='</div>'
    st.markdown(bc_html,unsafe_allow_html=True)

    if cur!="/":
        n1,n2,_=st.columns([1,1,4])
        with n1:
            if st.button("← Racine",key="bc_root",use_container_width=True):
                st.session_state.docs_dir="/";st.session_state.docs_selected=[];st.rerun()
        if len(parts)>1:
            with n2:
                parent="/"+"/".join(parts[:-1])
                if st.button("← Remonter",key="bc_up",use_container_width=True):
                    st.session_state.docs_dir=parent;st.session_state.docs_selected=[];st.rerun()

    t1,t2,_=st.columns([1.2,1,4])
    with t1:
        if st.button("Nouveau dossier",key="docs_mkdir",use_container_width=True):
            st.session_state.docs_new_dir=not st.session_state.docs_new_dir;st.rerun()
    if st.session_state.docs_new_dir:
        nd=st.text_input("Nom du dossier",key="docs_dirname",placeholder="Nom du dossier")
        if nd.strip():
            new_path=cur.rstrip("/")+"/"+nd.strip()
            if st.button("Créer",key="docs_mkdir_go",type="primary"):
                create_directory(new_path)
                st.session_state.docs_new_dir=False;st.rerun()

    subdirs,records=list_directory(cur)

    if subdirs:
        st.markdown('<div class="section-label">Dossiers</div>',unsafe_allow_html=True)
        cols=st.columns(min(len(subdirs),4))
        for i,sd in enumerate(subdirs):
            with cols[i%4]:
                full_path=cur.rstrip("/")+"/"+sd
                if st.button(f"📁 {_esc(sd)}",key=f"dir_{sd}",use_container_width=True):
                    st.session_state.docs_dir=full_path;st.session_state.docs_selected=[];st.rerun()

                if st.session_state.confirm_dir_del==full_path:
                    st.markdown(f'<div class="confirm-box"><strong>Supprimer ?</strong> Le dossier et tout son contenu seront perdus.</div>',
                                unsafe_allow_html=True)
                    cc1,cc2=st.columns(2)
                    with cc1:
                        if st.button("Confirmer",key=f"dirdel_ok_{sd}",type="primary",use_container_width=True):
                            delete_directory(full_path)
                            st.session_state.confirm_dir_del=None;st.rerun()
                    with cc2:
                        if st.button("Annuler",key=f"dirdel_no_{sd}",use_container_width=True):
                            st.session_state.confirm_dir_del=None;st.rerun()
                else:
                    if st.button("Supprimer",key=f"dirdel_{sd}",use_container_width=True):
                        st.session_state.confirm_dir_del=full_path;st.rerun()

    if records:
        st.markdown(f'<div class="section-label">Dossiers patients ({len(records)})</div>',unsafe_allow_html=True)

        sel=st.session_state.docs_selected
        sa=st.checkbox("Tout sélectionner",key="docs_selall",
                       value=len(sel)==len(records) and len(records)>0)
        if sa and len(sel)!=len(records):
            st.session_state.docs_selected=[r["id"] for r in records];st.rerun()
        elif not sa and len(sel)==len(records) and len(records)>0:
            st.session_state.docs_selected=[];st.rerun()

        if st.session_state.docs_selected:
            if st.session_state.confirm_bulk_del:
                st.markdown(f'<div class="confirm-box"><strong>Confirmer la suppression ?</strong> '
                            f'{len(st.session_state.docs_selected)} dossier(s) patient(s) seront définitivement perdus.</div>',
                            unsafe_allow_html=True)
                cc1,cc2=st.columns(2)
                with cc1:
                    if st.button("Oui, supprimer",type="primary",use_container_width=True,key="bulk_ok"):
                        delete_records(st.session_state.docs_selected)
                        st.session_state.docs_selected=[]
                        st.session_state.confirm_bulk_del=False;st.rerun()
                with cc2:
                    if st.button("Annuler",use_container_width=True,key="bulk_no"):
                        st.session_state.confirm_bulk_del=False;st.rerun()
            else:
                if st.button(f"Supprimer la sélection ({len(st.session_state.docs_selected)})",
                             key="docs_bulkdel",type="primary"):
                    st.session_state.confirm_bulk_del=True;st.rerun()

        _render_records_list(records,"dir")
    elif not subdirs:
        st.info("Ce dossier est vide.")

def _render_records_list(records,prefix):
    for r in records:
        rid=r["id"]
        c1,c2,c3,c4,c5=st.columns([.3,1.5,1.5,1,1])
        with c1:
            checked=st.checkbox("",key=f"sel_{prefix}_{rid}",
                                value=rid in st.session_state.docs_selected,label_visibility="collapsed")
            if checked and rid not in st.session_state.docs_selected:
                st.session_state.docs_selected.append(rid)
            elif not checked and rid in st.session_state.docs_selected:
                st.session_state.docs_selected.remove(rid)
        with c2:
            pc=r.get("analysis",{}).get("predicted_class","?")
            cd="suspicion d'opacité" if pc=="suspected_opacity" else ("incertain" if pc=="uncertain" else pc)
            st.markdown(f'**{_esc(r.get("patient_lastname",""))} {_esc(r.get("patient_firstname",""))}**')
            st.caption(f'N° {_esc(r.get("patient_number","—"))} · {_esc(cd)}')
        with c3:
            st.caption(f'Traitement : {_esc(r.get("treatment_date","—"))}')
            st.caption(f'Naissance : {_esc(r.get("patient_dob","—"))}')
        with c4:
            co=r.get("analysis",{}).get("confidence",0)
            st.caption(f'Confiance : {co:.2f}')
            st.caption(f'{_esc(r.get("created_at","")[:10])}')
        with c5:
            if st.button("Ouvrir",key=f"open_{prefix}_{rid}",use_container_width=True):
                st.session_state.docs_record=rid;st.session_state.docs_editing=False;st.rerun()

            if st.session_state.confirm_rec_del==rid:
                if st.button("Confirmer",key=f"del_ok_{prefix}_{rid}",type="primary",use_container_width=True):
                    delete_records([rid])
                    st.session_state.docs_selected=[s for s in st.session_state.docs_selected if s!=rid]
                    st.session_state.confirm_rec_del=None;st.rerun()
                if st.button("Annuler",key=f"del_no_{prefix}_{rid}",use_container_width=True):
                    st.session_state.confirm_rec_del=None;st.rerun()
            else:
                if st.button("Supprimer",key=f"del_{prefix}_{rid}",use_container_width=True):
                    st.session_state.confirm_rec_del=rid;st.rerun()

# ═══════════════════ RECORD DETAIL ═══════════════════
def render_record_detail():
    rid=st.session_state.docs_record
    db=_load_db()
    if rid not in db["records"]:
        st.error("Dossier introuvable.");st.session_state.docs_record=None;return

    rec=db["records"][rid]

    b1,b2,_=st.columns([1.2,1.2,4])
    with b1:
        if st.button("← Retour aux documents",key="rec_back",use_container_width=True):
            st.session_state.docs_record=None;st.session_state.docs_editing=False;st.rerun()
    with b2:
        img_path=IMAGES_DIR/rec.get("image_filename","")
        pdf_bytes = generate_patient_pdf(rec, img_path)
        if pdf_bytes:
            filename = f"rapport_{rec.get('patient_lastname','patient')}_{rec.get('patient_firstname','')}_{rid}.pdf"
            st.download_button("📄 Exporter en PDF", data=pdf_bytes, file_name=filename,
                              mime="application/pdf", use_container_width=True, key="rec_pdf")
        else:
            st.caption("Installez reportlab pour l'export PDF")

    st.markdown(f'<h1 class="page-title">{_esc(rec.get("patient_lastname",""))} {_esc(rec.get("patient_firstname",""))}</h1>',
                unsafe_allow_html=True)
    st.markdown(f'<p class="page-subtitle">N° {_esc(rec.get("patient_number","—"))} · '
                f'Dossier : {_esc(rec.get("directory","/"))} · Créé le {_esc(rec.get("created_at","")[:10])}</p>',
                unsafe_allow_html=True)

    col_img,col_res=st.columns([1,1.2],gap="medium")

    with col_img:
        if img_path.exists():
            st.image(str(img_path),use_container_width=True)
        else:
            st.warning("Image introuvable.")

    with col_res:
        pred=rec.get("analysis",{})
        lat=rec.get("latency",0)
        _render_result(pred,lat,with_expanders=True)

    st.markdown(f'<div style="height:1px;background:{S2};margin:20px 0"></div>',unsafe_allow_html=True)

    editing=st.session_state.docs_editing

    if not editing:
        st.markdown('<div class="section-label">Informations du patient</div>',unsafe_allow_html=True)
        if st.button("Modifier les informations",key="rec_edit"):
            st.session_state.docs_editing=True;st.rerun()
        i1,i2=st.columns(2)
        with i1:
            st.markdown(f"**Nom** : {_esc(rec.get('patient_lastname','—'))}")
            st.markdown(f"**Prénom** : {_esc(rec.get('patient_firstname','—'))}")
            st.markdown(f"**Date de naissance** : {_esc(rec.get('patient_dob','—'))}")
        with i2:
            st.markdown(f"**N° patient** : {_esc(rec.get('patient_number','—'))}")
            st.markdown(f"**Date du traitement** : {_esc(rec.get('treatment_date','—'))}")
            st.markdown(f"**Dossier** : {_esc(rec.get('directory','/'))}")
    else:
        st.markdown('<div class="section-label">Modifier les informations</div>',unsafe_allow_html=True)
        e1,e2=st.columns(2)
        with e1:
            new_ln=st.text_input("Nom",value=rec.get("patient_lastname",""),key="ed_ln")
            new_fn=st.text_input("Prénom",value=rec.get("patient_firstname",""),key="ed_fn")
            new_dob=st.text_input("Date de naissance",value=rec.get("patient_dob",""),key="ed_dob")
        with e2:
            new_pn=st.text_input("N° patient",value=rec.get("patient_number",""),key="ed_pn")
            new_td=st.text_input("Date du traitement",value=rec.get("treatment_date",""),key="ed_td")
            db2=_load_db()
            dirs=sorted(db2["directories"])
            cur_dir=rec.get("directory","/")
            idx=dirs.index(cur_dir) if cur_dir in dirs else 0
            new_dir=st.selectbox("Dossier",dirs,index=idx,key="ed_dir")
        b1,b2=st.columns(2)
        with b1:
            if st.button("Sauvegarder",type="primary",use_container_width=True,key="ed_save"):
                update_record(rid,{"patient_lastname":new_ln,"patient_firstname":new_fn,
                    "patient_dob":new_dob,"patient_number":new_pn,"treatment_date":new_td,
                    "directory":new_dir})
                st.session_state.docs_editing=False;st.success("Informations mises à jour.");st.rerun()
        with b2:
            if st.button("Annuler",use_container_width=True,key="ed_cancel"):
                st.session_state.docs_editing=False;st.rerun()

    st.markdown('<div class="section-label">Commentaires</div>',unsafe_allow_html=True)
    comments=rec.get("comments",[])
    if comments:
        for i,cm in enumerate(comments):
            st.markdown(f'<div class="comment-box">'
                        f'<div class="comment-text">{_esc(cm.get("text",""))}</div>'
                        f'<div class="comment-date">{_esc(cm.get("date","")[:10])}</div>'
                        f'</div>',unsafe_allow_html=True)
    else:
        st.caption("Aucun commentaire.")

    new_cm=st.text_area("Ajouter un commentaire",key="rec_newcm",placeholder="Notes cliniques...",
                        max_chars=5000,height=100)
    if st.button("Ajouter",key="rec_addcm"):
        if new_cm.strip():
            comments.append({"text":new_cm.strip(),
                            "date":datetime.utcnow().isoformat(timespec="seconds")+"Z"})
            update_record(rid,{"comments":comments})
            st.success("Commentaire ajouté.");st.rerun()

    st.markdown(f'<div style="height:1px;background:{S2};margin:24px 0"></div>',unsafe_allow_html=True)
    if st.session_state.confirm_rec_del==rid:
        st.markdown(f'<div class="confirm-box"><strong>Confirmer la suppression ?</strong> Ce dossier patient sera définitivement perdu.</div>',
                    unsafe_allow_html=True)
        cc1,cc2=st.columns(2)
        with cc1:
            if st.button("Oui, supprimer",type="primary",use_container_width=True,key="rec_del_ok"):
                delete_records([rid])
                st.session_state.docs_record=None
                st.session_state.confirm_rec_del=None;st.rerun()
        with cc2:
            if st.button("Annuler",use_container_width=True,key="rec_del_no"):
                st.session_state.confirm_rec_del=None;st.rerun()
    else:
        if st.button("Supprimer ce dossier patient",key="rec_delete"):
            st.session_state.confirm_rec_del=rid;st.rerun()

# ═══════════════════ CAS DE TEST ═══════════════════
def render_tests():
    st.markdown('<h1 class="page-title">Cas de test — 30 radiographies évaluées</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Résultats MedGemma pré-calculés sur 30 cas RSNA Pneumonia. '
                'Cliquez sur une vignette pour voir le détail.</p>', unsafe_allow_html=True)

    if not _TEST_CASES:
        st.warning("Aucun cas de test disponible. Lancez d'abord : "
                   "`python scripts/prepare_test_cases.py`")
        return

    all_cases = list(_TEST_CASES.values())
    n_total = len(all_cases)
    n_normal = sum(1 for c in all_cases if c["prediction"]["predicted_class"] == "normal")
    n_opacity = sum(1 for c in all_cases if c["prediction"]["predicted_class"] == "suspected_opacity")
    n_uncertain = sum(1 for c in all_cases if c["prediction"]["predicted_class"] == "uncertain")
    n_correct = sum(1 for c in all_cases if c["is_correct"])
    n_errors = n_total - n_correct - n_uncertain

    st.markdown(f'''
<div class="stats-grid">
<div class="stat-mini"><div class="k">Total</div><div class="v">{n_total}</div></div>
<div class="stat-mini"><div class="k">Normal</div><div class="v" style="color:{OK}">{n_normal}</div></div>
<div class="stat-mini"><div class="k">Suspicion</div><div class="v" style="color:{AC}">{n_opacity}</div></div>
<div class="stat-mini"><div class="k">Incertain</div><div class="v" style="color:{AM}">{n_uncertain}</div></div>
<div class="stat-mini"><div class="k">Corrects</div><div class="v" style="color:{OK}">{n_correct}/{n_total}</div></div>
</div>''', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Filtrer</div>', unsafe_allow_html=True)
    f1,f2,f3,f4,f5 = st.columns(5)
    for col,label,val in [(f1,"Tous","all"),(f2,"Normal","normal"),
                          (f3,"Suspicion","opacity"),(f4,"Incertain","uncertain"),
                          (f5,"Erreurs","errors")]:
        with col:
            btype = "primary" if st.session_state.tests_filter == val else "secondary"
            if st.button(label, key=f"filter_{val}", use_container_width=True, type=btype):
                st.session_state.tests_filter = val
                st.session_state.tests_selected = None
                st.rerun()

    def matches_filter(case_data):
        f = st.session_state.tests_filter
        pc = case_data["prediction"]["predicted_class"]
        if f == "all": return True
        if f == "normal": return pc == "normal"
        if f == "opacity": return pc == "suspected_opacity"
        if f == "uncertain": return pc == "uncertain"
        if f == "errors": return not case_data["is_correct"] and pc != "uncertain"
        return True

    filtered = {k: v for k, v in _TEST_CASES.items() if matches_filter(v)}

    if not filtered:
        st.info("Aucun cas correspondant au filtre.")
        return

    if st.session_state.tests_selected:
        _render_test_case_detail(st.session_state.tests_selected)
        return

    st.markdown(f'<div class="section-label">{len(filtered)} cas affiché(s)</div>',
                unsafe_allow_html=True)

    case_items = sorted(filtered.items(), key=lambda x: x[1]["case_id"])
    cols_per_row = 6

    for row_idx in range(0, len(case_items), cols_per_row):
        row_items = case_items[row_idx:row_idx + cols_per_row]
        cols = st.columns(cols_per_row)
        for i, (filename, data) in enumerate(row_items):
            with cols[i]:
                img_path = _TEST_CASES_DIR / filename
                if img_path.exists():
                    st.image(str(img_path), use_container_width=True)

                pc = data["prediction"]["predicted_class"]
                is_correct = data["is_correct"]
                is_unc = data["is_uncertain"]

                if is_unc:
                    badge = f'<span style="color:{AM};font-size:.7rem;font-weight:500">⚠ Incertain</span>'
                elif is_correct:
                    badge = f'<span style="color:{OK};font-size:.7rem;font-weight:500">✓ Correct</span>'
                else:
                    badge = f'<span style="color:#DC2626;font-size:.7rem;font-weight:500">✗ Erreur</span>'

                st.markdown(f'<div style="text-align:center;font-size:.72rem;color:{S5};'
                            f'margin:4px 0 2px">{_esc(data["case_id"])}</div>'
                            f'<div style="text-align:center;margin-bottom:6px">{badge}</div>',
                            unsafe_allow_html=True)

                if st.button("Voir", key=f"test_{filename}", use_container_width=True):
                    st.session_state.tests_selected = filename
                    st.rerun()


def _render_test_case_detail(filename):
    data = _TEST_CASES.get(filename)
    if not data:
        st.error("Cas introuvable.")
        return

    if st.button("← Retour à la grille", key="back_tests"):
        st.session_state.tests_selected = None
        st.rerun()

    case_id = data["case_id"]
    true_label = data["true_label"]
    pred_improved = data["prediction"]
    pred_baseline = data.get("baseline")
    is_correct = data["is_correct"]
    is_unc = data["is_uncertain"]

    if is_unc:
        status = "⚠ Cas incertain — Basculement par la règle 0.60"
        status_color = AM
    elif is_correct:
        status = "✓ Prédiction correcte"
        status_color = OK
    else:
        status = "✗ Erreur de classification"
        status_color = "#DC2626"

    st.markdown(f'<h1 class="page-title">{_esc(case_id)}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="page-subtitle" style="color:{status_color};font-weight:500">'
                f'{status}</p>', unsafe_allow_html=True)

    col_img, col_res = st.columns([1, 1.2], gap="medium")

    with col_img:
        img_path = _TEST_CASES_DIR / filename
        if img_path.exists():
            st.image(str(img_path), use_container_width=True)
        st.markdown(f'<p style="font-size:.72rem;color:{S5};text-align:center;margin-top:8px">'
                    f'Vérité terrain : <strong style="color:{S9}">{_esc(true_label)}</strong></p>',
                    unsafe_allow_html=True)

    with col_res:
        st.markdown('<div class="section-label">Résultat MedGemma (mode improved)</div>',
                    unsafe_allow_html=True)
        _render_result(pred_improved, data.get("measured_latency", 0), with_expanders=True)

    if pred_baseline:
        st.markdown(f'<div style="height:1px;background:{S2};margin:20px 0"></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="section-label">Comparaison Baseline vs Improved</div>',
                    unsafe_allow_html=True)

        cb, ci = st.columns(2)

        with cb:
            bl_pc = pred_baseline["predicted_class"]
            bl_cd = "suspicion d'opacité" if bl_pc == "suspected_opacity" else ("incertain" if bl_pc == "uncertain" else bl_pc)
            bl_correct = (bl_pc == true_label)
            bl_status = "✓ Correct" if bl_correct else ("⚠ Incertain" if bl_pc == "uncertain" else "✗ Erreur")
            bl_color = OK if bl_correct else (AM if bl_pc == "uncertain" else "#DC2626")

            st.markdown(f'''
<div style="background:{W};border:.5px solid {S2};border-radius:8px;padding:14px">
<div style="font-size:.7rem;color:{S5};text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Baseline</div>
<div style="font-size:1.1rem;color:{S9};font-weight:500">{_esc(bl_cd)}</div>
<div style="font-size:.85rem;color:{S5};margin-top:4px">Confiance : {pred_baseline["confidence"]:.2f}</div>
<div style="font-size:.75rem;color:{bl_color};font-weight:500;margin-top:8px">{bl_status}</div>
</div>''', unsafe_allow_html=True)

        with ci:
            imp_pc = pred_improved["predicted_class"]
            imp_cd = "suspicion d'opacité" if imp_pc == "suspected_opacity" else ("incertain" if imp_pc == "uncertain" else imp_pc)
            imp_status = "✓ Correct" if is_correct else ("⚠ Incertain" if is_unc else "✗ Erreur")
            imp_color = OK if is_correct else (AM if is_unc else "#DC2626")

            st.markdown(f'''
<div style="background:{W};border:.5px solid {S2};border-radius:8px;padding:14px">
<div style="font-size:.7rem;color:{S5};text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px">Improved</div>
<div style="font-size:1.1rem;color:{S9};font-weight:500">{_esc(imp_cd)}</div>
<div style="font-size:.85rem;color:{S5};margin-top:4px">Confiance : {pred_improved.get("confidence",0):.2f}</div>
<div style="font-size:.75rem;color:{imp_color};font-weight:500;margin-top:8px">{imp_status}</div>
</div>''', unsafe_allow_html=True)

        if bl_pc != imp_pc:
            st.markdown(f'<div style="margin-top:14px;padding:10px 14px;background:{S1};'
                        f'border-left:2px solid {S9};border-radius:6px;font-size:.85rem;color:{S7}">'
                        f'<strong>Changement observé :</strong> le mode improved a modifié la prédiction '
                        f'grâce au prompt enrichi et à la règle de seuil 0.60.</div>',
                        unsafe_allow_html=True)

# ═══════════════════ APP (ANALYSE) ═══════════════════
def render_app():
    st.markdown(f'''
<div style="display:flex;align-items:baseline;justify-content:space-between;margin-bottom:4px">
<h1 class="page-title">Classification de radiographies thoraciques</h1>
<div style="font-family:'JetBrains Mono',monospace;font-size:.72rem;color:{S5}">
v1.0 · google/medgemma-4b-it</div></div>''', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Prototype pédagogique pour un dépistage radiologique assisté par IA.</p>',
                unsafe_allow_html=True)
    _ensure_model_loaded()

    for col,(lb,val,sub,cls) in zip(st.columns(4),[
        ("Précision","0.80","Cible 0.70",""),("Macro F1","0.83","Cible 0.68",""),
        ("JSON valide","100%","Cible 95%",""),("Hallucinations","0","Garde-fous activés","neutral")]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{lb}</div>'
                        f'<div class="metric-value">{val}</div>'
                        f'<div class="metric-sub {cls}">{sub}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:.75rem;color:{S5};margin:8px 0 0">Mesuré sur 50 cas RSNA Pneumonia · mode amélioré</p>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="disclaimer">{_WARN_SVG}<span><strong>Prototype pédagogique.</strong> '
                f'Non destiné au diagnostic médical.</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Exemples de cas</div>', unsafe_allow_html=True)
    demo_dir = ROOT/"app"/"demo_images"
    demo_files = sorted(demo_dir.glob("*.png")) if demo_dir.exists() else []
    if demo_files:
        cols = st.columns(len(demo_files))
        for i,df in enumerate(demo_files):
            with cols[i]:
                label = "normal" if "normal" in df.stem else "suspicion d'opacité"
                st.image(str(df), use_container_width=True)
                ia = (st.session_state.active_demo_idx == i)
                if ia and st.session_state.analyzing:
                    bl,bt,bd = "Analyse en cours...", "secondary", True
                elif ia and st.session_state.last_pred and not st.session_state.analyzing:
                    bl,bt,bd = "Analyse terminée", "secondary", True
                else:
                    bl,bt,bd = f"Lancer — {label}", "primary" if ia else "secondary", False
                if st.button(bl, key=f"demo_{i}", use_container_width=True, type=bt, disabled=bd):
                    st.session_state.selected_demo = df
                    st.session_state.active_demo_idx = i
                    st.session_state.analyzing = True
                    st.session_state.last_pred = None
                    st.session_state.last_error = None
                    st.session_state.multi_results = {}
                    st.rerun()
    else:
        st.info("Aucun exemple. Lancez `python scripts/prepare_demo_images.py`.")

    if st.session_state.selected_demo and st.session_state.analyzing:
        img_path = st.session_state.selected_demo
        precomputed = _PRECOMPUTED.get(img_path.name)
        with st.spinner("MedGemma examine la radiographie..."):
            if precomputed and "prediction" in precomputed:
                time.sleep(1.5)
                st.session_state.last_pred = precomputed["prediction"]
                st.session_state.last_latency = precomputed.get("measured_latency", 0)
            else:
                t0 = time.perf_counter()
                try:
                    pred = apply_safety_guardrails(medgemma_predict(img_path, mode="improved"))
                    st.session_state.last_pred = pred
                    st.session_state.last_latency = round(time.perf_counter()-t0, 1)
                except Exception as e:
                    st.session_state.last_error = str(e)[:200]
        st.session_state.analyzing = False
        st.rerun()

    if st.session_state.selected_demo and st.session_state.last_pred:
        st.markdown('<div class="section-label">Résultat de l\'analyse</div>', unsafe_allow_html=True)
        col_img,col_res = st.columns([1,1.2], gap="medium")
        with col_img:
            st.image(str(st.session_state.selected_demo), use_container_width=True)
            st.markdown('<div class="state-banner state-done">Analyse terminée</div>', unsafe_allow_html=True)
            if st.button("Nouvelle analyse", use_container_width=True, key="reset_demo"):
                st.session_state.selected_demo = None
                st.session_state.active_demo_idx = None
                st.session_state.last_pred = None
                st.rerun()
        with col_res:
            _render_result(st.session_state.last_pred, st.session_state.last_latency or 0)

        img_bytes = st.session_state.selected_demo.read_bytes()
        render_save_form(img_bytes, st.session_state.last_pred, st.session_state.last_latency or 0, "demo")

    st.markdown('<div class="section-label">Ou importez vos radiographies</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("PNG, JPG, JPEG", type=["png","jpg","jpeg"],
                                accept_multiple_files=True, label_visibility="collapsed")

    if uploaded:
        if st.session_state.selected_demo:
            st.session_state.selected_demo = None
            st.session_state.active_demo_idx = None
            st.session_state.last_pred = None
            st.session_state.analyzing = False

        not_analyzed = [f for f in uploaded if f.name not in st.session_state.multi_results]
        if not_analyzed:
            if st.button(f"Analyser {len(not_analyzed)} image(s)", type="primary",
                         use_container_width=True, key="multi_go"):
                st.session_state.multi_analyzing = True
                st.rerun()

        if st.session_state.multi_analyzing:
            with st.spinner(f"Analyse de {len(not_analyzed)} radiographie(s)..."):
                for uf in not_analyzed:
                    tmp = ROOT/"data"/"_temp_upload.png"
                    tmp.parent.mkdir(parents=True, exist_ok=True)
                    img_bytes = uf.getbuffer()
                    with open(tmp, "wb") as f: f.write(img_bytes)
                    t0 = time.perf_counter()
                    try:
                        pred = apply_safety_guardrails(medgemma_predict(tmp, mode="improved"))
                        lat = round(time.perf_counter()-t0, 1)
                        st.session_state.multi_results[uf.name] = {"pred":pred, "latency":lat,
                            "error":None, "bytes":bytes(img_bytes)}
                    except Exception as e:
                        st.session_state.multi_results[uf.name] = {"pred":None,
                            "latency":0, "error":str(e)[:200], "bytes":bytes(img_bytes)}
            st.session_state.multi_analyzing = False
            st.rerun()

        analyzed = [f for f in uploaded if f.name in st.session_state.multi_results]
        if analyzed:
            st.markdown('<div class="section-label">Résultats</div>', unsafe_allow_html=True)
            for uf in analyzed:
                r = st.session_state.multi_results[uf.name]
                with st.expander(f"📄 {_esc(uf.name)}", expanded=False):
                    if r.get("error"):
                        st.error(f"Erreur : {_esc(r['error'])}")
                    else:
                        col_img,col_res = st.columns([1,1.2], gap="medium")
                        with col_img:
                            st.image(io.BytesIO(r["bytes"]), use_container_width=True)
                        with col_res:
                            _render_result(r["pred"], r["latency"], with_expanders=True)
                        render_save_form(r["bytes"], r["pred"], r["latency"], f"multi_{uf.name}")

# ═══════════════════ DIALOG SIGNALER UN PROBLÈME ═══════════════════
@st.dialog("Signaler un problème")
def report_issue_dialog():
    st.markdown(f'<div style="font-size:.85rem;color:{S7};margin-bottom:14px">'
                f'Décrivez le problème rencontré. Un e-mail sera envoyé à notre équipe technique.</div>',
                unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:.78rem;color:{S5};margin-bottom:14px">'
                f'<strong style="color:{S9};font-weight:500">Destinataire</strong> : '
                f'<code style="background:{S1};padding:1px 6px;border-radius:3px">{SUPPORT_EMAIL}</code></div>',
                unsafe_allow_html=True)

    user = st.session_state.get("current_user")
    sender_info = ""
    if user:
        sender_info = f"De : {user['full_name']} ({user['email']})\n\n"

    subject = st.text_input("Sujet", key="report_subject",
                            placeholder="Ex : Erreur lors de l'analyse d'une image")
    body = st.text_area("Description du problème", key="report_body",
                        placeholder="Décrivez le problème en détail (étapes pour reproduire, "
                                    "message d'erreur, contexte, navigateur utilisé...).",
                        height=180)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    if subject.strip() and body.strip():
        full_body = sender_info + body.strip()
        params = urllib.parse.urlencode(
            {"subject": subject.strip(), "body": full_body},
            quote_via=urllib.parse.quote
        )
        mailto_url = f"mailto:{SUPPORT_EMAIL}?{params}"
        st.link_button("📧 Envoyer ",
                       mailto_url, use_container_width=True, type="primary")
        st.caption("Cela ouvrira votre application de messagerie par défaut avec l'e-mail pré-rempli.")
    else:
        st.button("📧 Envoyer ",
                  disabled=True, use_container_width=True, type="primary",
                  help="Veuillez remplir le sujet et la description")

    if st.button("Annuler", use_container_width=True, key="report_cancel"):
        st.rerun()

# ═══════════════════ ROUTER ═══════════════════
render_navbar()

if st.session_state.current_user is None:
    # Non connecté : accès uniquement à login / register / about
    if st.session_state.current_page == "about":
        render_about()
    elif st.session_state.auth_page == "register":
        render_register()
    else:
        render_login()
else:
    p = st.session_state.current_page
    if p == "profile":
        render_profile()
    elif p == "documents":
        render_documents()
    elif p == "tests":
        render_tests()
    elif p == "about":
        render_about()
    else:
        render_app()

# ═══════════════════ FOOTER ═══════════════════
st.markdown(f"""
<div class="footer">
<strong style="color:{S9};font-weight:500;letter-spacing:.15em">THORAXIA</strong>
&nbsp;·&nbsp; EFREI MasterCamp 2026 &nbsp;·&nbsp; Big Data &amp; IA<br>
Architecture <code>google/medgemma-4b-it</code>
&nbsp;·&nbsp; Tuteur Badr Tajini<br>
<span style="color:{S5}">Prompt amélioré → MedGemma → Garde-fous · Dossiers patients · Export PDF</span>
</div>
""", unsafe_allow_html=True)

# Bouton "Signaler un problème" en bas de page
_fc1, _fc2, _fc3 = st.columns([2, 1.5, 2])
with _fc2:
    if st.button("🛟 Signaler un problème", key="footer_report",
                 use_container_width=True):
        report_issue_dialog()