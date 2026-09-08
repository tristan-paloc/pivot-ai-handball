"""Video annotee de verification du tracking.

Dessine sur le broadcast une couleur STABLE par tracker_id + un gros label ID,
et un bandeau aux changements de plan. Objectif : verifier a l'oeil si un ID
tient au contact / apres occlusion (un ID qui saute = une couleur qui change
sur le meme joueur). Aucun radar, aucune stat : juste le controle visuel.
"""

from __future__ import annotations

import colorsys
import logging
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)


def couleur_id(tracker_id: int) -> tuple[int, int, int]:
    """Couleur BGR vive et STABLE pour un tracker_id (repartition par nombre d'or)."""
    teinte = (tracker_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(teinte, 0.75, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def generer_video_annotee(
    chemin_video: str | Path,
    detections_trackees: dict[int, sv.Detections],
    chemin_sortie: str | Path,
    subsample: int = 1,
    frames_coupure: tuple[int, ...] = (),
    largeur_max: int = 1280,
) -> Path:
    """Ecrit une video annotee (boites colorees par ID + coupures).

    Seules les frames analysees (1 sur `subsample`) sont ecrites, a fps/subsample :
    video plus legere et sans clignotement des boites.

    Args:
        chemin_video: video source.
        detections_trackees: dict frame_idx -> Detections (avec tracker_id).
        chemin_sortie: chemin du MP4 de sortie.
        subsample: 1 frame sur N (doit matcher le tracking).
        frames_coupure: indices des changements de plan (bandeau affiche).
        largeur_max: redimensionne si la video est plus large.

    Returns:
        Path du fichier ecrit.

    Raises:
        RuntimeError: si la source ou le writer ne s'ouvrent pas.
    """
    chemin_video = str(chemin_video)
    chemin_sortie = Path(chemin_sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(chemin_video)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la video : {chemin_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    echelle = min(1.0, largeur_max / w) if w else 1.0
    w_out, h_out = int(w * echelle), int(h * echelle)
    fps_out = max(1.0, fps / max(1, subsample))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(chemin_sortie), fourcc, fps_out, (w_out, h_out))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Impossible d'ouvrir VideoWriter : {chemin_sortie}")

    coupures = set(int(c) for c in frames_coupure)
    fenetre_coupure = max(1, subsample) * 3  # bandeau visible ~3 frames analysees

    try:
        fi = 0
        while fi < total:
            ret, frame = cap.read()
            if not ret:
                break
            if fi % subsample == 0:
                if echelle < 1.0:
                    frame = cv2.resize(frame, (w_out, h_out))
                _dessiner_detections(frame, detections_trackees.get(fi), echelle)
                if any(0 <= fi - c < fenetre_coupure for c in coupures):
                    _dessiner_bandeau_coupure(frame)
                writer.write(frame)
            fi += 1
    finally:
        cap.release()
        writer.release()

    logger.info("Video annotee ecrite : %s (%dx%d @ %.1f fps)",
                chemin_sortie, w_out, h_out, fps_out)
    return chemin_sortie


def _dessiner_detections(frame: np.ndarray, dets: sv.Detections | None, echelle: float) -> None:
    """Dessine les boites + IDs sur la frame (in place)."""
    if dets is None or dets.tracker_id is None or len(dets) == 0:
        return
    for i in range(len(dets)):
        tid = int(dets.tracker_id[i])
        couleur = couleur_id(tid)
        x1, y1, x2, y2 = (float(v) * echelle for v in dets.xyxy[i])
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, couleur, 2)
        # etiquette ID lisible : fond plein + texte
        label = str(tid)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y_haut = max(0, p1[1] - th - 6)
        cv2.rectangle(frame, (p1[0], y_haut), (p1[0] + tw + 6, y_haut + th + 6), couleur, -1)
        cv2.putText(frame, label, (p1[0] + 3, y_haut + th + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)


def _dessiner_bandeau_coupure(frame: np.ndarray) -> None:
    """Bordure rouge + texte 'CHANGEMENT DE PLAN' (in place)."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 220), 6)
    cv2.putText(frame, "CHANGEMENT DE PLAN", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 220), 2, cv2.LINE_AA)
