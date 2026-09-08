"""Detection des changements de plan (coupures camera) dans une video.

Un changement de plan invalide la continuite de tracking : perdre un ID a une
coupure n'est PAS un echec du tracker. Ce module repere les coupures pour que
le benchmark separe le tracking intra-plan (evaluable) de l'inter-plan (attendu
comme rupture).

Methode : correlation d'histogrammes HSV entre frames consecutives echantillonnees.
Une chute brutale de correlation = coupure franche. Pur OpenCV, pas de GPU.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _histogramme(frame: np.ndarray) -> np.ndarray:
    """Histogramme HSV normalise (teinte + saturation) d'une frame reduite."""
    petite = cv2.resize(frame, (64, 64))
    hsv = cv2.cvtColor(petite, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def detecter_changements_plan(
    chemin_video: str,
    subsample: int = 1,
    seuil_correlation: float = 0.5,
    max_frames: int | None = None,
) -> list[int]:
    """Detecte les frames de changement de plan (coupures camera).

    Args:
        chemin_video: chemin du MP4.
        subsample: 1 frame sur N analysee (doit matcher celui du tracking pour
            que les indices de coupure soient comparables aux tracks).
        seuil_correlation: en dessous de cette correlation d'histogramme entre
            deux frames analysees consecutives, on declare une coupure.
        max_frames: limite optionnelle (index frame source).

    Returns:
        liste triee des indices de frame ou un nouveau plan commence.

    Raises:
        RuntimeError: si la video ne s'ouvre pas.
    """
    cap = cv2.VideoCapture(str(chemin_video))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la video : {chemin_video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total = min(total, max_frames)

    coupures: list[int] = []
    hist_precedent: np.ndarray | None = None
    try:
        fi = 0
        while fi < total:
            ret, frame = cap.read()
            if not ret:
                break
            if fi % subsample == 0:
                hist = _histogramme(frame)
                if hist_precedent is not None:
                    corr = cv2.compareHist(hist_precedent, hist, cv2.HISTCMP_CORREL)
                    if corr < seuil_correlation:
                        coupures.append(fi)
                hist_precedent = hist
            fi += 1
    finally:
        cap.release()

    logger.info(
        "Changements de plan : %d coupure(s) detectee(s) sur %d frames (seuil=%.2f)",
        len(coupures),
        total,
        seuil_correlation,
    )
    return coupures


def segments_entre_coupures(
    frame_min: int, frame_max: int, coupures: list[int]
) -> list[tuple[int, int]]:
    """Decoupe l'intervalle [frame_min, frame_max] en segments intra-plan.

    Args:
        frame_min: premiere frame analysee.
        frame_max: derniere frame analysee.
        coupures: indices de changement de plan (debut de nouveau plan).

    Returns:
        liste de segments (debut, fin) inclusifs, un par plan continu.
    """
    bornes = sorted(c for c in coupures if frame_min < c <= frame_max)
    segments: list[tuple[int, int]] = []
    debut = frame_min
    for c in bornes:
        segments.append((debut, c - 1))
        debut = c
    segments.append((debut, frame_max))
    return segments
