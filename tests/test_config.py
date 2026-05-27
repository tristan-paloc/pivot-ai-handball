"""Tests unitaires pour pivot_ai.config."""

from __future__ import annotations

from pivot_ai.config import CLASSES_HANDBALL, CLASSES_INVERSE, POINTS_TERRAIN_M, TerrainConfig


def test_classes_inverse_coherente() -> None:
    """L'inversion des classes doit etre bijective."""
    for nom, idx in CLASSES_HANDBALL.items():
        assert CLASSES_INVERSE[idx] == nom


def test_terrain_dimensions_pixels() -> None:
    """Les dimensions pixel doivent etre coherentes avec px_per_m."""
    config = TerrainConfig()
    assert config.terrain_px_w == 40 * 22
    assert config.terrain_px_h == 20 * 22


def test_terrain_m2px_coins() -> None:
    """Les 4 coins doivent se mapper correctement."""
    config = TerrainConfig()
    x0, y0 = config.m2px(0, 0)
    x1, y1 = config.m2px(40, 20)
    assert x0 == config.origin_x
    assert y0 == config.origin_y + config.terrain_px_h
    assert x1 == config.origin_x + config.terrain_px_w
    assert y1 == config.origin_y


def test_terrain_est_dans_zone() -> None:
    """Le filtre de zone respecte les marges asymetriques."""
    config = TerrainConfig()
    assert config.est_dans_zone((20.0, 10.0))  # centre
    assert config.est_dans_zone((0.0, 5.0))    # bord X, dans la zone Y
    assert not config.est_dans_zone((20.0, 1.0))   # trop bas
    assert not config.est_dans_zone((20.0, 19.0))  # trop haut


def test_points_terrain_dimensions() -> None:
    """Les points caracteristiques sont dans le rectangle 40x20."""
    for nom, (x, y) in POINTS_TERRAIN_M.items():
        assert 0 <= x <= 40, f"Point {nom} hors terrain en X : {x}"
        assert 0 <= y <= 20, f"Point {nom} hors terrain en Y : {y}"
