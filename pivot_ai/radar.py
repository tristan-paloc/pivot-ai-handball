"""Rendu du radar 2D : terrain handball avec positions des joueurs."""

from __future__ import annotations

import cv2
import numpy as np

from pivot_ai.config import TerrainConfig


def dessiner_terrain_vide(config: TerrainConfig) -> np.ndarray:
    """Genere une image du terrain handball vide (lignes, surfaces de but, arcs 9m)."""
    radar = np.full((config.radar_h, config.radar_w, 3), 35, dtype=np.uint8)

    # Surface terrain (beige)
    cv2.rectangle(radar, config.m2px(0, 0), config.m2px(40, 20), (200, 160, 110), -1)
    # Bord
    cv2.rectangle(radar, config.m2px(0, 0), config.m2px(40, 20), (255, 255, 255), 2)
    # Ligne mediane
    cv2.line(radar, config.m2px(20, 0), config.m2px(20, 20), (255, 255, 255), 1)

    # Lignes 6m et 9m (demi-cercles)
    cd, cg = config.m2px(40, 10), config.m2px(0, 10)
    r6 = 6 * config.px_per_m
    r9 = 9 * config.px_per_m
    cv2.ellipse(radar, cd, (r6, r6), 0, 90, 270, (255, 255, 255), 2)
    cv2.ellipse(radar, cd, (r9, r9), 0, 90, 270, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.ellipse(radar, cg, (r6, r6), 0, -90, 90, (255, 255, 255), 2)
    cv2.ellipse(radar, cg, (r9, r9), 0, -90, 90, (255, 255, 255), 1, cv2.LINE_AA)

    # Lignes de but (rouge)
    cv2.line(radar, config.m2px(0, 8.5), config.m2px(0, 11.5), (0, 0, 220), 4)
    cv2.line(radar, config.m2px(40, 8.5), config.m2px(40, 11.5), (0, 0, 220), 4)

    return radar


def dessiner_legende(radar: np.ndarray) -> np.ndarray:
    """Ajoute la legende des couleurs en haut a gauche."""
    cv2.putText(radar, "MHB", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(radar, "ADV", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(radar, "GK", (80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 215, 255), 2, cv2.LINE_AA)
    cv2.putText(radar, "REF", (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 0, 255), 2, cv2.LINE_AA)
    return radar


def dessiner_radar(
    positions: list[tuple[float, float, tuple[int, int, int], str, int]],
    config: TerrainConfig | None = None,
) -> np.ndarray:
    """Genere un radar avec les positions de joueurs.

    Args:
        positions: liste de tuples (x_m, y_m, couleur_bgr, tag, tracker_id)
        config: configuration terrain (defaut : TerrainConfig())

    Returns:
        image BGR du radar
    """
    config = config or TerrainConfig()
    radar = dessiner_terrain_vide(config)
    radar = dessiner_legende(radar)

    for x_m, y_m, couleur, _tag, tid in positions:
        x_c = float(np.clip(x_m, 0, config.longueur_m))
        y_c = float(np.clip(y_m, 0, config.largeur_m))
        cx, cy = config.m2px(x_c, y_c)
        cv2.circle(radar, (cx, cy), 9, couleur, -1)
        cv2.circle(radar, (cx, cy), 9, (255, 255, 255), 2)
        cv2.putText(radar, str(tid), (cx + 11, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    return radar
