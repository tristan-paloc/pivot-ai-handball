"""Tests des metriques de qualite du tracking (ID switches, nb joueurs estime)."""

from __future__ import annotations

from pivot_ai.metriques import (
    detecter_reprises_id,
    estimer_nb_joueurs,
    resumer_qualite_tracking,
)
from pivot_ai.stats import PositionJoueur


def _piste(tid_frames: range, x: float, y: float, dx: float = 0.0) -> list[PositionJoueur]:
    """Une piste : positions reelles sur des frames donnees, autour de (x, y)."""
    return [
        PositionJoueur(frame_idx=fi, x_m=x + i * dx, y_m=y)
        for i, fi in enumerate(tid_frames)
    ]


def test_reprise_simple_detectee() -> None:
    """Piste A finit (20,10) f=10 ; B demarre (20.5,10) f=14 -> 1 reprise."""
    positions = {
        1: _piste(range(0, 11, 2), 20.0, 10.0),   # finit frame 10 en (20,10)
        2: _piste(range(14, 30, 2), 20.5, 10.0),  # demarre frame 14 en (20.5,10)
    }
    reprises = detecter_reprises_id(positions, seuil_distance_m=3.0, seuil_frames=30)
    assert reprises == [(1, 2)]


def test_pistes_distantes_pas_de_reprise() -> None:
    """Deux pistes eloignees dans l'espace : pas de reprise."""
    positions = {
        1: _piste(range(0, 11, 2), 5.0, 5.0),
        2: _piste(range(14, 30, 2), 35.0, 15.0),  # trop loin
    }
    reprises = detecter_reprises_id(positions, seuil_distance_m=3.0, seuil_frames=30)
    assert reprises == []


def test_gap_temporel_trop_grand_pas_de_reprise() -> None:
    """Reprise proche en espace mais delai trop long : pas de reprise."""
    positions = {
        1: _piste(range(0, 11, 2), 20.0, 10.0),   # finit frame 10
        2: _piste(range(100, 120, 2), 20.5, 10.0),  # demarre frame 100 (gap 90)
    }
    reprises = detecter_reprises_id(positions, seuil_distance_m=3.0, seuil_frames=30)
    assert reprises == []


def test_chaine_de_trois_un_seul_joueur() -> None:
    """A -> B -> C recolles : 2 reprises, 1 joueur estime sur 3 pistes."""
    positions = {
        1: _piste(range(0, 11, 2), 20.0, 10.0),
        2: _piste(range(14, 25, 2), 20.5, 10.0),
        3: _piste(range(28, 40, 2), 21.0, 10.0),
    }
    reprises = detecter_reprises_id(positions, seuil_distance_m=3.0, seuil_frames=30)
    assert len(reprises) == 2
    assert estimer_nb_joueurs(positions, reprises=reprises) == 1


def test_deux_joueurs_distincts() -> None:
    """Deux joueurs qui ne se recollent jamais : 2 joueurs estimes."""
    positions = {
        1: _piste(range(0, 40, 2), 10.0, 5.0),
        2: _piste(range(0, 40, 2), 30.0, 15.0),
    }
    assert estimer_nb_joueurs(positions) == 2


def test_resume_qualite_complet() -> None:
    """Le resume agrege pistes, reprises, joueurs estimes et longueur moyenne."""
    positions = {
        1: _piste(range(0, 11, 2), 20.0, 10.0),    # 6 positions reelles
        2: _piste(range(14, 25, 2), 20.5, 10.0),   # 6 positions, reprise de 1
        3: _piste(range(0, 40, 2), 5.0, 5.0),      # joueur distinct
    }
    resume = resumer_qualite_tracking(positions)
    assert resume.nb_tracks == 3
    assert resume.nb_reprises_id == 1  # 1 -> 2
    assert resume.nb_joueurs_estimes == 2  # {1,2} recolles + {3}
    assert resume.longueur_track_moyenne > 0


def test_resume_ignore_positions_interpolees() -> None:
    """La longueur de piste ne compte que les positions reelles."""
    positions = {
        1: [
            PositionJoueur(0, 10.0, 5.0, interpole=False),
            PositionJoueur(1, 10.5, 5.0, interpole=True),
            PositionJoueur(2, 11.0, 5.0, interpole=False),
        ],
    }
    resume = resumer_qualite_tracking(positions)
    assert resume.longueur_track_moyenne == 2.0  # 2 reelles, l'interpolee ignoree


def test_dict_vide() -> None:
    """Aucune piste : resume a zero, pas de crash."""
    resume = resumer_qualite_tracking({})
    assert resume.nb_tracks == 0
    assert resume.nb_reprises_id == 0
    assert resume.nb_joueurs_estimes == 0
    assert resume.longueur_track_moyenne == 0.0
