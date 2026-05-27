"""Homographie : projection des coordonnees pixel vers le terrain en metres.

Une homographie est une transformation projective 3x3 qui mappe le plan image
au plan terrain. Necessite minimum 4 correspondances (pixel, terrain_m).
Avec 4 points : `getPerspectiveTransform`. Avec plus : `findHomography` + RANSAC.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TypedDict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Correspondance(TypedDict):
    """Une correspondance pixel <-> terrain pour calibrer l'homographie."""

    pixel: tuple[int, int]
    terrain_m: tuple[float, float]


@dataclass
class Homographie:
    """Homographie calibree, encapsule la matrice 3x3 et les methodes de projection."""

    matrice: np.ndarray
    nb_points: int
    methode: str  # "perspective_4pts", "ransac", "least_squares"

    def projeter_point(self, point_pixel: tuple[float, float]) -> tuple[float, float]:
        """Projette un point pixel vers le terrain (metres)."""
        pt = np.array([[point_pixel]], dtype=np.float32)
        proj = cv2.perspectiveTransform(pt, self.matrice)
        return float(proj[0][0][0]), float(proj[0][0][1])

    def projeter_bbox_pieds(self, bbox: np.ndarray) -> tuple[float, float]:
        """Projette le centre-bas d'une bbox (position des pieds) vers le terrain."""
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_bottom = float(bbox[3])
        return self.projeter_point((x_center, y_bottom))


def calibrer_homographie(correspondances: dict[str, Correspondance]) -> Homographie:
    """Calibre une homographie a partir de correspondances pixel <-> terrain.

    Args:
        correspondances: dict nom_point -> {"pixel": (x,y), "terrain_m": (X,Y)}

    Returns:
        Homographie calibree

    Raises:
        ValueError: si moins de 4 points ou si la calibration echoue
    """
    if len(correspondances) < 4:
        raise ValueError(
            f"Homographie : minimum 4 points requis, recu {len(correspondances)}"
        )

    pts_pix = np.array(
        [c["pixel"] for c in correspondances.values()], dtype=np.float32
    )
    pts_ter = np.array(
        [c["terrain_m"] for c in correspondances.values()], dtype=np.float32
    )

    if len(correspondances) == 4:
        H = cv2.getPerspectiveTransform(pts_pix, pts_ter)
        if H is None:
            raise ValueError("getPerspectiveTransform a echoue, points colineaires ?")
        return Homographie(matrice=H, nb_points=4, methode="perspective_4pts")

    # >4 points : RANSAC d'abord, fallback least-squares
    H, _ = cv2.findHomography(
        pts_pix, pts_ter, method=cv2.RANSAC, ransacReprojThreshold=2.0
    )
    if H is not None:
        return Homographie(
            matrice=H, nb_points=len(correspondances), methode="ransac"
        )

    logger.warning("RANSAC degenere, fallback least-squares")
    H, _ = cv2.findHomography(pts_pix, pts_ter, method=0)
    if H is None:
        raise ValueError("Calibration homographie KO meme en fallback")

    return Homographie(
        matrice=H, nb_points=len(correspondances), methode="least_squares"
    )
