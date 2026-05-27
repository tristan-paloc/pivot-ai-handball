"""Code monolithique d'origine - REFERENCE UNIQUEMENT.

Ce fichier contient le pipeline Colab d'origine de Tristan, avant refactoring.
La logique metier qui marche (homographie, tracking, classification couleur,
rendu radar) doit etre conservee et migree dans les modules `pivot_ai/`.

NE PAS executer ce fichier directement. C'est une reference de specification.

Dependance Roboflow Cloud A SUPPRIMER lors du refactor : remplacer par
inference locale via pivot_ai.detection.DetecteurLocal.
"""

# ============================================================
# BOOTSTRAP COMPLET PIVOT.AI - handball pipeline autonome
# ============================================================
import os
import pickle
import base64
import requests
from collections import defaultdict, Counter
import numpy as np
import cv2

# --- 1. Drive + paths ---
# from google.colab import drive
# drive.mount('/content/drive', force_remount=False)
DRIVE_ROOT  = "/content/drive/MyDrive/PIVOT_AI"
CLIPS_DIR   = f"{DRIVE_ROOT}/raw_clips"
OUTPUTS_DIR = f"{DRIVE_ROOT}/outputs"
# os.makedirs(OUTPUTS_DIR, exist_ok=True)

# --- 2. Installs ---
# !pip install -q "supervision==0.25.1" tqdm scikit-learn plotly
import supervision as sv
from sklearn.cluster import KMeans
from tqdm import tqdm

# --- 3. API Roboflow (A REMPLACER PAR INFERENCE LOCALE) ---
# from google.colab import userdata
# ROBOFLOW_API_KEY = userdata.get('ROBOFLOW_API_KEY')
ROBOFLOW_API_KEY = ""  # placeholder
os.environ['ROBOFLOW_API_KEY'] = ROBOFLOW_API_KEY

# --- 4. Constantes modele ---
MODEL_ID = "handball-detection-fj8rc/2"
CLASSES_HANDBALL = {"players": 0, "goalkeeper": 1, "referees": 2, "ball": 3}
CLASSES_INVERSE = {v: k for k, v in CLASSES_HANDBALL.items()}

# --- 5. Constantes radar ---
RADAR_W, RADAR_H = 960, 540
BROADCAST_W, BROADCAST_H = 960, 540
PX_PER_M = 22
TERRAIN_PX_W = 40 * PX_PER_M
TERRAIN_PX_H = 20 * PX_PER_M
ORIGIN_X = (RADAR_W - TERRAIN_PX_W) // 2
ORIGIN_Y = (RADAR_H - TERRAIN_PX_H) // 2


def m2px(x_m, y_m):
    return (ORIGIN_X + int(round(x_m * PX_PER_M)),
            ORIGIN_Y + int(round((20 - y_m) * PX_PER_M)))


# --- 6. detecter_handball : API cloud Roboflow (A SUPPRIMER, voir pivot_ai/detection.py) ---
def detecter_handball(frame, api_key, model_id=MODEL_ID, timeout=30):
    _, jpg = cv2.imencode(".jpg", frame)
    img_b64 = base64.b64encode(jpg).decode("ascii")
    r = requests.post(f"https://detect.roboflow.com/{model_id}",
                      params={"api_key": api_key}, data=img_b64,
                      headers={"Content-Type": "application/x-www-form-urlencoded"},
                      timeout=timeout)
    r.raise_for_status()
    preds = r.json().get("predictions", [])
    if not preds:
        return sv.Detections.empty()
    xyxy = np.array([[p["x"]-p["width"]/2, p["y"]-p["height"]/2,
                      p["x"]+p["width"]/2, p["y"]+p["height"]/2] for p in preds], dtype=np.float32)
    conf = np.array([p["confidence"] for p in preds], dtype=np.float32)
    cid  = np.array([CLASSES_HANDBALL.get(p["class"], -1) for p in preds], dtype=int)
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cid)


# --- 7. extraire_couleur_torse ---
def extraire_couleur_torse(frame, bbox):
    x1, y1, x2, y2 = bbox.astype(int)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0:
        return np.zeros(3, dtype=np.float64)
    cx = (x1+x2)//2
    half_w = max(1, int(w*0.20))
    y_h = y1 + int(h*0.15)
    y_b = y1 + int(h*0.40)
    crop = frame[y_h:y_b, cx-half_w:cx+half_w]
    if crop.size == 0:
        return np.zeros(3, dtype=np.float64)
    return np.median(crop.reshape(-1, 3), axis=0).astype(np.float64)


# --- 8. dessiner_radar ---
def dessiner_radar(positions):
    radar = np.full((RADAR_H, RADAR_W, 3), 35, dtype=np.uint8)
    cv2.rectangle(radar, m2px(0,0), m2px(40,20), (200,160,110), -1)
    cv2.rectangle(radar, m2px(0,0), m2px(40,20), (255,255,255), 2)
    cv2.line(radar, m2px(20,0), m2px(20,20), (255,255,255), 1)
    cd, cg = m2px(40,10), m2px(0,10)
    r6, r9 = 6*PX_PER_M, 9*PX_PER_M
    cv2.ellipse(radar, cd, (r6,r6), 0, 90, 270, (255,255,255), 2)
    cv2.ellipse(radar, cd, (r9,r9), 0, 90, 270, (255,255,255), 1, cv2.LINE_AA)
    cv2.ellipse(radar, cg, (r6,r6), 0, -90, 90, (255,255,255), 2)
    cv2.ellipse(radar, cg, (r9,r9), 0, -90, 90, (255,255,255), 1, cv2.LINE_AA)
    cv2.line(radar, m2px(0,8.5), m2px(0,11.5), (0,0,220), 4)
    cv2.line(radar, m2px(40,8.5), m2px(40,11.5), (0,0,220), 4)
    cv2.putText(radar, "MHB", (15,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(radar, "AIX", (15,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
    cv2.putText(radar, "GK",  (80,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,215,255), 2, cv2.LINE_AA)
    cv2.putText(radar, "REF", (80,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2, cv2.LINE_AA)
    for x_m, y_m, color, tag, tid in positions:
        x_c = float(np.clip(x_m, 0, 40))
        y_c = float(np.clip(y_m, 0, 20))
        cx, cy = m2px(x_c, y_c)
        cv2.circle(radar, (cx,cy), 9, color, -1)
        cv2.circle(radar, (cx,cy), 9, (255,255,255), 2)
        cv2.putText(radar, str(tid), (cx+11,cy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,255,255), 1, cv2.LINE_AA)
    return radar


# NOTE pour Claude Code :
# Le fichier complet d'origine contient aussi traiter_clip_v2 avec subsample + cache Drive
# + interpolation lineaire. Voir l'historique du chat Claude pour la version complete.
# Reproduire la logique dans pivot_ai/pipeline.py en utilisant les modules separes.
