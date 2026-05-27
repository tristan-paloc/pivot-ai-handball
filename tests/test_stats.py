"""Tests unitaires pour pivot_ai.stats.

Tests squelettes - a completer une fois les fonctions implementees par Claude Code.
"""

from __future__ import annotations

import pytest

from pivot_ai.stats import PositionJoueur


def test_position_joueur_construction() -> None:
    """PositionJoueur se construit correctement."""
    p = PositionJoueur(frame_idx=10, x_m=15.5, y_m=8.2)
    assert p.frame_idx == 10
    assert p.x_m == 15.5
    assert p.y_m == 8.2
    assert not p.interpole


@pytest.mark.skip(reason="A implementer par Claude Code")
def test_distance_parcourue_trajectoire_droite() -> None:
    """Sur une trajectoire en ligne droite de 10m, la distance calculee = 10m."""
    # positions = liste de 26 PositionJoueur, joueur qui avance de 0.4m/frame a 25fps
    # ...
    pass


@pytest.mark.skip(reason="A implementer par Claude Code")
def test_vitesse_moyenne_trajectoire_constante() -> None:
    """Avec une vitesse constante de 5 m/s, vitesse_moyenne == 5.0."""
    pass


@pytest.mark.skip(reason="A implementer par Claude Code")
def test_stats_ignore_interpolees() -> None:
    """Avec ignorer_interpolees=True, les positions interpole=True sont exclues."""
    pass
