"""Classification des equipes par KMeans sur la couleur du torse.

L'agent Claude Code devra ameliorer ce module :
- Passer la couleur de BGR vers HSV ou Lab pour plus de robustesse
- Permettre une selection manuelle du cluster MHB via ROI sur 1ere frame
- Gerer le cas du gardien qui a souvent un maillot different
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

import numpy as np
import supervision as sv
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


def extraire_couleur_torse(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Extrait la couleur mediane du torse d'un joueur dans une bbox.

    On prend une bande horizontale au milieu superieur de la bbox (15%-40% de la hauteur),
    sur 40% de la largeur centree.

    Args:
        frame: image BGR
        bbox: array [x1, y1, x2, y2]

    Returns:
        array [B, G, R] couleur mediane (ou zeros si bbox invalide)
    """
    x1, y1, x2, y2 = bbox.astype(int)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0:
        return np.zeros(3, dtype=np.float64)

    cx = (x1 + x2) // 2
    half_w = max(1, int(w * 0.20))
    y_h = y1 + int(h * 0.15)
    y_b = y1 + int(h * 0.40)
    crop = frame[y_h:y_b, cx - half_w : cx + half_w]

    if crop.size == 0:
        return np.zeros(3, dtype=np.float64)

    return np.median(crop.reshape(-1, 3), axis=0).astype(np.float64)


def classifier_equipes(
    detections_trackees: dict[int, sv.Detections],
    frames_par_index: dict[int, np.ndarray],
    classe_finale: dict[int, int],
    id_classe_joueur: int,
) -> tuple[dict[int, int], int]:
    """Classifie chaque tracker_id en equipe 0 ou 1 via KMeans sur couleur torse.

    Args:
        detections_trackees: dict frame_idx -> Detections (avec tracker_id)
        frames_par_index: dict frame_idx -> image BGR
        classe_finale: dict tracker_id -> classe stabilisee
        id_classe_joueur: id de la classe "joueur de champ"

    Returns:
        tuple (equipe_finale, cluster_mhb)
        - equipe_finale : dict tracker_id -> 0 ou 1
        - cluster_mhb : index du cluster considere comme MHB (le plus sombre par defaut)
    """
    couleurs_collectees: list[np.ndarray] = []
    tracker_par_couleur: list[int] = []

    for fi, dets in detections_trackees.items():
        if dets.tracker_id is None or fi not in frames_par_index:
            continue
        frame = frames_par_index[fi]
        for i in range(len(dets)):
            tid = int(dets.tracker_id[i])
            if classe_finale.get(tid) != id_classe_joueur:
                continue
            couleur = extraire_couleur_torse(frame, dets.xyxy[i])
            couleurs_collectees.append(couleur)
            tracker_par_couleur.append(tid)

    if len(couleurs_collectees) < 2:
        logger.warning("Trop peu d'echantillons couleur pour KMeans, fallback")
        return {}, 0

    X = np.array(couleurs_collectees, dtype=np.float64)
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    # Convention : MHB = cluster avec moyenne BGR la plus sombre
    moyennes = [km.cluster_centers_[i].mean() for i in range(2)]
    cluster_mhb = int(np.argmin(moyennes))

    # Vote majoritaire par tracker_id
    votes_par_tracker: dict[int, Counter] = defaultdict(Counter)
    for tid, label in zip(tracker_par_couleur, labels, strict=True):
        votes_par_tracker[tid][int(label)] += 1

    equipe_finale = {
        tid: int(compteur.most_common(1)[0][0])
        for tid, compteur in votes_par_tracker.items()
    }

    logger.info(
        "Classification equipes : %d joueurs classes, cluster_mhb=%d",
        len(equipe_finale),
        cluster_mhb,
    )
    return equipe_finale, cluster_mhb
