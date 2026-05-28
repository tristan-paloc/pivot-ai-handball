"""Tests unitaires pour pivot_ai.stats."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from pivot_ai.stats import (
    PositionJoueur,
    calculer_largeur_bloc_defensif,
    calculer_stats_joueur,
    generer_heatmap_joueur,
)


def test_position_joueur_construction() -> None:
    """PositionJoueur se construit correctement."""
    p = PositionJoueur(frame_idx=10, x_m=15.5, y_m=8.2)
    assert p.frame_idx == 10
    assert p.x_m == 15.5
    assert p.y_m == 8.2
    assert not p.interpole


def test_distance_parcourue_trajectoire_droite() -> None:
    """Trajectoire droite : 26 positions, 0.4m/frame en x, distance totale = 10m."""
    fps = 25.0
    positions = [
        PositionJoueur(frame_idx=i, x_m=i * 0.4, y_m=10.0) for i in range(26)
    ]
    df = calculer_stats_joueur({42: positions}, fps=fps)

    assert df.height == 1
    ligne = df.row(0, named=True)
    assert ligne["tracker_id"] == 42
    assert math.isclose(ligne["distance_parcourue_m"], 10.0, abs_tol=1e-6)
    assert ligne["nb_frames_detectees"] == 26
    assert math.isclose(ligne["x_min"], 0.0, abs_tol=1e-6)
    assert math.isclose(ligne["x_max"], 10.0, abs_tol=1e-6)


def test_vitesse_moyenne_trajectoire_constante() -> None:
    """Vitesse constante 5 m/s : 0.2m/frame a 25fps, vitesse moyenne == 5.0."""
    fps = 25.0
    # 0.2 m par frame = 5 m/s exact.
    positions = [
        PositionJoueur(frame_idx=i, x_m=i * 0.2, y_m=5.0) for i in range(50)
    ]
    df = calculer_stats_joueur({1: positions}, fps=fps)

    ligne = df.row(0, named=True)
    assert math.isclose(ligne["vitesse_moyenne_m_s"], 5.0, abs_tol=1e-3)
    assert math.isclose(ligne["vitesse_max_m_s"], 5.0, abs_tol=1e-3)


def test_stats_ignore_interpolees() -> None:
    """Avec ignorer_interpolees=True, les positions interpole=True sont exclues."""
    fps = 25.0
    # 3 positions reelles : (0,0), (1,0), (2,0) -> distance = 2m
    # 1 position interpolee a (10,0) qui ne doit PAS compter.
    positions = [
        PositionJoueur(frame_idx=0, x_m=0.0, y_m=0.0, interpole=False),
        PositionJoueur(frame_idx=1, x_m=1.0, y_m=0.0, interpole=False),
        PositionJoueur(frame_idx=2, x_m=10.0, y_m=0.0, interpole=True),
        PositionJoueur(frame_idx=3, x_m=2.0, y_m=0.0, interpole=False),
    ]
    df = calculer_stats_joueur({7: positions}, fps=fps, ignorer_interpolees=True)
    ligne = df.row(0, named=True)
    assert math.isclose(ligne["distance_parcourue_m"], 2.0, abs_tol=1e-6)
    assert ligne["nb_frames_detectees"] == 3
    assert math.isclose(ligne["x_max"], 2.0, abs_tol=1e-6)

    # En incluant les interpolees, distance = 1 + 9 + 8 = 18m
    df2 = calculer_stats_joueur({7: positions}, fps=fps, ignorer_interpolees=False)
    ligne2 = df2.row(0, named=True)
    assert math.isclose(ligne2["distance_parcourue_m"], 18.0, abs_tol=1e-6)
    assert ligne2["nb_frames_detectees"] == 4


def test_stats_plusieurs_trackers() -> None:
    """Plusieurs trackers : 1 ligne par tracker, dans le bon ordre de colonnes."""
    positions = {
        1: [PositionJoueur(0, 0.0, 0.0), PositionJoueur(1, 3.0, 4.0)],  # 5m
        2: [PositionJoueur(0, 0.0, 0.0), PositionJoueur(1, 0.0, 1.0)],  # 1m
    }
    df = calculer_stats_joueur(positions, fps=25.0)
    assert df.height == 2
    assert df.columns[0] == "tracker_id"
    par_tid = {row["tracker_id"]: row for row in df.iter_rows(named=True)}
    assert math.isclose(par_tid[1]["distance_parcourue_m"], 5.0, abs_tol=1e-6)
    assert math.isclose(par_tid[2]["distance_parcourue_m"], 1.0, abs_tol=1e-6)


def test_stats_vide_renvoie_df_avec_schema() -> None:
    """Dict vide : DataFrame vide avec le bon schema."""
    df = calculer_stats_joueur({}, fps=25.0)
    assert df.height == 0
    assert "tracker_id" in df.columns
    assert "distance_parcourue_m" in df.columns


def test_stats_fps_invalide_leve_erreur() -> None:
    """fps <= 0 doit lever ValueError."""
    with pytest.raises(ValueError, match="fps"):
        calculer_stats_joueur({1: [PositionJoueur(0, 0, 0)]}, fps=0.0)


def test_heatmap_dimensions_et_normalisation() -> None:
    """Heatmap : bonnes dimensions et somme normalisee a 1."""
    positions = [
        PositionJoueur(frame_idx=i, x_m=float(i), y_m=10.0) for i in range(10)
    ]
    heatmap = generer_heatmap_joueur(positions, resolution=1.0, sigma_lissage=None)
    # 40m / 1m = 40 cells en x ; 20m / 1m = 20 cells en y
    assert heatmap.shape == (20, 40)
    assert math.isclose(heatmap.sum(), 1.0, abs_tol=1e-9)


def test_heatmap_concentration_position_unique() -> None:
    """Une seule position : densite concentree dans la cellule correspondante (sans lissage)."""
    positions = [PositionJoueur(frame_idx=0, x_m=20.0, y_m=10.0)]
    heatmap = generer_heatmap_joueur(positions, resolution=1.0, sigma_lissage=None)
    # Cellule (iy=10, ix=20) doit contenir toute la masse
    assert math.isclose(heatmap[10, 20], 1.0, abs_tol=1e-9)
    assert math.isclose(heatmap.sum(), 1.0, abs_tol=1e-9)


def test_heatmap_vide_retourne_zeros() -> None:
    """Aucune position : heatmap = zeros (pas de NaN)."""
    heatmap = generer_heatmap_joueur([], resolution=1.0)
    assert heatmap.shape == (20, 40)
    assert np.all(heatmap == 0.0)


def test_heatmap_ignore_interpolees() -> None:
    """Positions interpolees ne contribuent pas par defaut."""
    positions = [
        PositionJoueur(frame_idx=0, x_m=5.0, y_m=5.0, interpole=False),
        PositionJoueur(frame_idx=1, x_m=30.0, y_m=15.0, interpole=True),
    ]
    heatmap = generer_heatmap_joueur(positions, resolution=1.0, sigma_lissage=None)
    assert math.isclose(heatmap[5, 5], 1.0, abs_tol=1e-9)
    assert math.isclose(heatmap[15, 30], 0.0, abs_tol=1e-9)


def test_largeur_bloc_defensif_4_defenseurs_alignes() -> None:
    """4 defenseurs aux y=[5,8,11,14] sur la meme frame : largeur = 9m, nb = 4."""
    fps = 25.0
    positions_par_tracker = {
        10: [PositionJoueur(0, 10.0, 5.0)],
        11: [PositionJoueur(0, 12.0, 8.0)],
        12: [PositionJoueur(0, 14.0, 11.0)],
        13: [PositionJoueur(0, 11.0, 14.0)],
        20: [PositionJoueur(0, 30.0, 10.0)],  # attaquant adverse, ignore
    }
    equipe_finale = {10: 0, 11: 0, 12: 0, 13: 0, 20: 1}
    df = calculer_largeur_bloc_defensif(
        positions_par_tracker, equipe_finale, equipe_defendante=0, fps=fps
    )
    assert df.height == 1
    ligne = df.row(0, named=True)
    assert ligne["nb_defenseurs_visibles"] == 4
    assert math.isclose(ligne["largeur_y_m"], 9.0, abs_tol=1e-6)
    assert math.isclose(ligne["x_moyen"], (10 + 12 + 14 + 11) / 4, abs_tol=1e-6)
    assert math.isclose(ligne["temps_s"], 0.0, abs_tol=1e-9)


def test_largeur_bloc_defensif_temporel() -> None:
    """Plusieurs frames : tri par frame_idx et temps_s coherent."""
    fps = 25.0
    positions_par_tracker = {
        1: [
            PositionJoueur(0, 20.0, 5.0),
            PositionJoueur(25, 20.0, 6.0),
        ],
        2: [
            PositionJoueur(0, 20.0, 15.0),
            PositionJoueur(25, 20.0, 14.0),
        ],
    }
    equipe_finale = {1: 0, 2: 0}
    df = calculer_largeur_bloc_defensif(
        positions_par_tracker, equipe_finale, equipe_defendante=0, fps=fps
    )
    assert df.height == 2
    # Tri sur frame_idx
    assert df["frame_idx"].to_list() == [0, 25]
    assert math.isclose(df["temps_s"].to_list()[1], 1.0, abs_tol=1e-9)
    # frame 0 : largeur = 15-5 = 10
    # frame 25 : largeur = 14-6 = 8
    assert math.isclose(df["largeur_y_m"].to_list()[0], 10.0, abs_tol=1e-6)
    assert math.isclose(df["largeur_y_m"].to_list()[1], 8.0, abs_tol=1e-6)


def test_vitesse_max_outlier_cappee_par_defaut() -> None:
    """Un saut tracker (10m en 1 frame) donne >100 m/s : cappe a defaut 12 m/s max."""
    fps = 25.0
    positions = [
        PositionJoueur(frame_idx=0, x_m=0.0, y_m=0.0),
        PositionJoueur(frame_idx=1, x_m=0.5, y_m=0.0),  # ~12.5 m/s, juste au seuil
        PositionJoueur(frame_idx=2, x_m=20.0, y_m=0.0),  # saut tracker : 487.5 m/s
        PositionJoueur(frame_idx=3, x_m=20.4, y_m=0.0),  # 10 m/s
    ]
    df = calculer_stats_joueur({1: positions}, fps=fps)
    ligne = df.row(0, named=True)
    # Le saut de 487 m/s doit etre exclu, max conserve = 10 m/s
    assert ligne["vitesse_max_m_s"] <= 12.0
    assert ligne["vitesse_max_m_s"] == pytest.approx(10.0, abs=1e-3)


def test_vitesse_max_outlier_sans_cap() -> None:
    """Avec seuil_vitesse_max_m_s=None, les outliers passent."""
    fps = 25.0
    positions = [
        PositionJoueur(frame_idx=0, x_m=0.0, y_m=0.0),
        PositionJoueur(frame_idx=1, x_m=100.0, y_m=0.0),  # 2500 m/s
    ]
    df = calculer_stats_joueur({1: positions}, fps=fps, seuil_vitesse_max_m_s=None)
    ligne = df.row(0, named=True)
    assert ligne["vitesse_max_m_s"] > 1000.0


def test_largeur_bloc_defensif_vide() -> None:
    """Aucun defenseur de l'equipe demandee : DataFrame vide avec bon schema."""
    df = calculer_largeur_bloc_defensif(
        {1: [PositionJoueur(0, 0, 0)]}, {1: 0}, equipe_defendante=1, fps=25.0
    )
    assert df.height == 0
    assert isinstance(df, pl.DataFrame)
    assert "largeur_y_m" in df.columns
