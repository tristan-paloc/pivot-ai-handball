"""Statistiques par joueur : distance parcourue, vitesse, heatmap, zone d'evolution.

ATTENTION : ne pas calculer les stats sur les positions interpolees, uniquement
sur les frames reellement detectees. Le flag `interpole` dans les positions
permet de filtrer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)


@dataclass
class PositionJoueur:
    """Une position de joueur a une frame donnee."""

    frame_idx: int
    x_m: float
    y_m: float
    interpole: bool = False


# Schema des colonnes attendues pour calculer_stats_joueur, ordre garanti.
_COLONNES_STATS: tuple[str, ...] = (
    "tracker_id",
    "distance_parcourue_m",
    "vitesse_moyenne_m_s",
    "vitesse_max_m_s",
    "temps_total_s",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "centroide_x",
    "centroide_y",
    "nb_frames_detectees",
)


# Vitesse max realiste pour un humain en sprint (~43 km/h, record monde 100m).
# Au-dela : artefact tracker (ID qui saute sur un autre joueur).
SEUIL_VITESSE_MAX_REALISTE_M_S: float = 12.0


def _stats_un_tracker(
    tracker_id: int,
    positions: list[PositionJoueur],
    fps: float,
    ignorer_interpolees: bool,
    seuil_vitesse_max_m_s: float | None,
) -> dict[str, float | int]:
    """Calcule les stats agregees pour un seul tracker."""
    if ignorer_interpolees:
        positions = [p for p in positions if not p.interpole]

    # Tri par frame_idx (les positions peuvent arriver dans le desordre).
    positions = sorted(positions, key=lambda p: p.frame_idx)
    nb = len(positions)

    if nb == 0:
        return {
            "tracker_id": tracker_id,
            "distance_parcourue_m": 0.0,
            "vitesse_moyenne_m_s": 0.0,
            "vitesse_max_m_s": 0.0,
            "temps_total_s": 0.0,
            "x_min": 0.0,
            "x_max": 0.0,
            "y_min": 0.0,
            "y_max": 0.0,
            "centroide_x": 0.0,
            "centroide_y": 0.0,
            "nb_frames_detectees": 0,
        }

    xs = np.array([p.x_m for p in positions], dtype=np.float64)
    ys = np.array([p.y_m for p in positions], dtype=np.float64)
    idxs = np.array([p.frame_idx for p in positions], dtype=np.float64)

    if nb == 1:
        # Pas de distance possible avec un seul point.
        return {
            "tracker_id": tracker_id,
            "distance_parcourue_m": 0.0,
            "vitesse_moyenne_m_s": 0.0,
            "vitesse_max_m_s": 0.0,
            "temps_total_s": 0.0,
            "x_min": float(xs[0]),
            "x_max": float(xs[0]),
            "y_min": float(ys[0]),
            "y_max": float(ys[0]),
            "centroide_x": float(xs[0]),
            "centroide_y": float(ys[0]),
            "nb_frames_detectees": 1,
        }

    # Distances euclidiennes entre points consecutifs.
    dx = np.diff(xs)
    dy = np.diff(ys)
    distances = np.sqrt(dx * dx + dy * dy)
    distance_totale = float(distances.sum())

    # Temps entre points consecutifs (gere les gaps de subsample).
    dt = np.diff(idxs) / fps
    # dt theoriquement > 0 puisque frames triees et uniques par tracker.
    dt_safe = np.where(dt > 0, dt, 1.0 / fps)
    vitesses = distances / dt_safe

    temps_total_s = float((idxs[-1] - idxs[0]) / fps)
    vitesse_moyenne = distance_totale / temps_total_s if temps_total_s > 0 else 0.0

    # Capper vitesse_max pour eviter les outliers tracker (ID qui saute sur
    # un autre joueur entre 2 frames echantillonnees). Le record humain 100m
    # est ~12.4 m/s ; tout au-dessus est un artefact.
    if seuil_vitesse_max_m_s is not None and len(vitesses) > 0:
        vitesses_filtrees = vitesses[vitesses <= seuil_vitesse_max_m_s]
        if len(vitesses_filtrees) > 0:
            vitesse_max = float(vitesses_filtrees.max())
        else:
            vitesse_max = float(seuil_vitesse_max_m_s)
    else:
        vitesse_max = float(vitesses.max())

    return {
        "tracker_id": tracker_id,
        "distance_parcourue_m": distance_totale,
        "vitesse_moyenne_m_s": vitesse_moyenne,
        "vitesse_max_m_s": vitesse_max,
        "temps_total_s": temps_total_s,
        "x_min": float(xs.min()),
        "x_max": float(xs.max()),
        "y_min": float(ys.min()),
        "y_max": float(ys.max()),
        "centroide_x": float(xs.mean()),
        "centroide_y": float(ys.mean()),
        "nb_frames_detectees": nb,
    }


def calculer_stats_joueur(
    positions_par_tracker: dict[int, list[PositionJoueur]],
    fps: float,
    ignorer_interpolees: bool = True,
    seuil_vitesse_max_m_s: float | None = SEUIL_VITESSE_MAX_REALISTE_M_S,
) -> pl.DataFrame:
    """Calcule les stats agregees par tracker_id.

    Args:
        positions_par_tracker: dict tracker_id -> liste de PositionJoueur
        fps: framerate du clip
        ignorer_interpolees: si True, exclut les positions interpolees du calcul
        seuil_vitesse_max_m_s: vitesse max realiste pour filtrer les outliers
            tracker dans vitesse_max_m_s. Mettre None pour desactiver le cap.

    Returns:
        DataFrame Polars avec colonnes :
        tracker_id, distance_parcourue_m, vitesse_moyenne_m_s, vitesse_max_m_s,
        temps_total_s, x_min, x_max, y_min, y_max, centroide_x, centroide_y,
        nb_frames_detectees.

    Raises:
        ValueError: si fps <= 0
    """
    if fps <= 0:
        raise ValueError(f"fps doit etre > 0, recu {fps}")

    if not positions_par_tracker:
        # DataFrame vide avec le schema attendu.
        return pl.DataFrame(
            schema={
                "tracker_id": pl.Int64,
                "distance_parcourue_m": pl.Float64,
                "vitesse_moyenne_m_s": pl.Float64,
                "vitesse_max_m_s": pl.Float64,
                "temps_total_s": pl.Float64,
                "x_min": pl.Float64,
                "x_max": pl.Float64,
                "y_min": pl.Float64,
                "y_max": pl.Float64,
                "centroide_x": pl.Float64,
                "centroide_y": pl.Float64,
                "nb_frames_detectees": pl.Int64,
            }
        )

    lignes = [
        _stats_un_tracker(
            tid, positions, fps, ignorer_interpolees, seuil_vitesse_max_m_s
        )
        for tid, positions in positions_par_tracker.items()
    ]
    df = pl.DataFrame(lignes).select(_COLONNES_STATS)
    logger.info("Stats calculees pour %d trackers", df.height)
    return df


def generer_heatmap_joueur(
    positions: list[PositionJoueur],
    longueur_terrain: float = 40.0,
    largeur_terrain: float = 20.0,
    resolution: float = 0.5,
    sigma_lissage: float | None = 1.0,
    ignorer_interpolees: bool = True,
) -> np.ndarray:
    """Genere une heatmap 2D de presence d'un joueur sur le terrain.

    Args:
        positions: liste de positions du joueur
        longueur_terrain: longueur en metres (defaut 40)
        largeur_terrain: largeur en metres (defaut 20)
        resolution: taille des cellules en metres (defaut 0.5m)
        sigma_lissage: sigma du filtre gaussien (None = pas de lissage)
        ignorer_interpolees: si True, exclut les positions interpolees

    Returns:
        array 2D shape (n_y, n_x) avec n_y = largeur/res, n_x = longueur/res.
        Densite normalisee (somme = 1) sauf si aucune position valide (zeros).

    Raises:
        ValueError: si dimensions ou resolution invalides
    """
    if resolution <= 0:
        raise ValueError(f"resolution doit etre > 0, recu {resolution}")
    if longueur_terrain <= 0 or largeur_terrain <= 0:
        raise ValueError("dimensions terrain doivent etre > 0")

    n_x = int(round(longueur_terrain / resolution))
    n_y = int(round(largeur_terrain / resolution))
    grille = np.zeros((n_y, n_x), dtype=np.float64)

    if ignorer_interpolees:
        positions = [p for p in positions if not p.interpole]

    if not positions:
        return grille

    # Indexation : axe 0 = y (largeur), axe 1 = x (longueur).
    for p in positions:
        # On clippe pour gerer les positions hors-terrain (bruit projection).
        ix = int(np.clip(p.x_m / resolution, 0, n_x - 1))
        iy = int(np.clip(p.y_m / resolution, 0, n_y - 1))
        grille[iy, ix] += 1.0

    if sigma_lissage is not None and sigma_lissage > 0:
        grille = gaussian_filter(grille, sigma=sigma_lissage, mode="constant")

    total = grille.sum()
    if total > 0:
        grille /= total

    return grille


def calculer_largeur_bloc_defensif(
    positions_par_tracker: dict[int, list[PositionJoueur]],
    equipe_finale: dict[int, int],
    equipe_defendante: int,
    fps: float,
    ignorer_interpolees: bool = True,
) -> pl.DataFrame:
    """Calcule la largeur du bloc defensif au cours du temps.

    Pour chaque frame ou au moins un defenseur est present, calcule la largeur
    laterale (y_max - y_min) du bloc et la position longitudinale moyenne.

    Args:
        positions_par_tracker: positions par tracker_id
        equipe_finale: dict tracker_id -> equipe
        equipe_defendante: indice de l'equipe en defense
        fps: framerate
        ignorer_interpolees: si True, exclut les positions interpolees

    Returns:
        DataFrame Polars trie par frame_idx avec colonnes :
        frame_idx, temps_s, largeur_y_m, x_moyen, nb_defenseurs_visibles.

    Raises:
        ValueError: si fps <= 0
    """
    if fps <= 0:
        raise ValueError(f"fps doit etre > 0, recu {fps}")

    # Agregation frame -> liste de (x_m, y_m) des defenseurs visibles.
    par_frame: dict[int, list[tuple[float, float]]] = {}
    for tid, positions in positions_par_tracker.items():
        if equipe_finale.get(tid) != equipe_defendante:
            continue
        for p in positions:
            if ignorer_interpolees and p.interpole:
                continue
            par_frame.setdefault(p.frame_idx, []).append((p.x_m, p.y_m))

    if not par_frame:
        return pl.DataFrame(
            schema={
                "frame_idx": pl.Int64,
                "temps_s": pl.Float64,
                "largeur_y_m": pl.Float64,
                "x_moyen": pl.Float64,
                "nb_defenseurs_visibles": pl.Int64,
            }
        )

    lignes = []
    for fi in sorted(par_frame.keys()):
        pts = par_frame[fi]
        xs = np.array([x for x, _ in pts], dtype=np.float64)
        ys = np.array([y for _, y in pts], dtype=np.float64)
        lignes.append(
            {
                "frame_idx": int(fi),
                "temps_s": float(fi / fps),
                "largeur_y_m": float(ys.max() - ys.min()) if len(ys) > 1 else 0.0,
                "x_moyen": float(xs.mean()),
                "nb_defenseurs_visibles": int(len(pts)),
            }
        )

    df = pl.DataFrame(lignes).select(
        ["frame_idx", "temps_s", "largeur_y_m", "x_moyen", "nb_defenseurs_visibles"]
    )
    logger.info("Largeur bloc defensif : %d frames calculees", df.height)
    return df
