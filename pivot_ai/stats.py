"""Statistiques par joueur : distance parcourue, vitesse, heatmap, zone d'evolution.

ATTENTION : ne pas calculer les stats sur les positions interpolees, uniquement
sur les frames reellement detectees. Le flag `interpole` dans les positions
permet de filtrer.

Module a IMPLEMENTER par l'agent Claude Code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class PositionJoueur:
    """Une position de joueur a une frame donnee."""

    frame_idx: int
    x_m: float
    y_m: float
    interpole: bool = False


def calculer_stats_joueur(
    positions_par_tracker: dict[int, list[PositionJoueur]],
    fps: float,
    ignorer_interpolees: bool = True,
) -> pl.DataFrame:
    """Calcule les stats agregees par tracker_id.

    Args:
        positions_par_tracker: dict tracker_id -> liste de PositionJoueur
        fps: framerate du clip
        ignorer_interpolees: si True, exclut les positions interpolees du calcul

    Returns:
        DataFrame Polars avec colonnes :
        - tracker_id
        - distance_parcourue_m
        - vitesse_moyenne_m_s
        - vitesse_max_m_s
        - temps_total_s
        - x_min, x_max, y_min, y_max
        - centroide_x, centroide_y
        - nb_frames_detectees
    """
    raise NotImplementedError(
        "A implementer par Claude Code en suivant les specs du prompt initial. "
        "Indications : "
        "1) Filtrer positions interpolees si demande. "
        "2) Trier par frame_idx pour chaque tracker. "
        "3) Distance = somme des distances euclidiennes entre points consecutifs. "
        "4) Vitesse instantanee = distance / dt entre 2 frames consecutives "
        "   (attention au cas dt > 1/fps si frames sautees). "
        "5) Construire le DataFrame Polars avec toutes les colonnes attendues."
    )


def generer_heatmap_joueur(
    positions: list[PositionJoueur],
    longueur_terrain: float = 40.0,
    largeur_terrain: float = 20.0,
    resolution: float = 0.5,
) -> np.ndarray:
    """Genere une heatmap 2D de presence d'un joueur sur le terrain.

    Args:
        positions: liste de positions du joueur
        longueur_terrain: longueur en metres (defaut 40)
        largeur_terrain: largeur en metres (defaut 20)
        resolution: taille des cellules en metres (defaut 0.5m)

    Returns:
        array 2D shape (largeur/resolution, longueur/resolution) : densite normalisee
    """
    raise NotImplementedError(
        "A implementer par Claude Code. Indications : "
        "1) Initialiser une grille 2D zeros aux bonnes dimensions. "
        "2) Pour chaque position, incrementer la cellule correspondante. "
        "3) Optionnel : appliquer un filtre gaussien pour lissage (scipy.ndimage). "
        "4) Normaliser pour somme = 1."
    )


def calculer_largeur_bloc_defensif(
    positions_par_tracker: dict[int, list[PositionJoueur]],
    equipe_finale: dict[int, int],
    equipe_defendante: int,
    fps: float,
) -> pl.DataFrame:
    """Calcule la largeur du bloc defensif au cours du temps.

    Args:
        positions_par_tracker: positions par tracker_id
        equipe_finale: dict tracker_id -> equipe
        equipe_defendante: indice de l'equipe en defense
        fps: framerate

    Returns:
        DataFrame Polars avec colonnes (frame_idx, temps_s, largeur_y_m,
        x_moyen, nb_defenseurs_visibles)
    """
    raise NotImplementedError(
        "A implementer par Claude Code. Pour chaque frame : "
        "1) Recuperer les defenseurs presents. "
        "2) Calculer y_max - y_min des defenseurs (largeur laterale du bloc). "
        "3) x_moyen = position moyenne en longueur. "
        "4) Sortir un DataFrame temporel."
    )
