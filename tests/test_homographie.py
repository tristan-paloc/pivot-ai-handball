"""Tests unitaires pour pivot_ai.homographie."""

from __future__ import annotations

import numpy as np
import pytest

from pivot_ai.homographie import calibrer_homographie


def test_calibrer_homographie_4_points_identite() -> None:
    """Avec 4 points alignes sur une grille reguliere, la projection est correcte."""
    correspondances = {
        "A": {"pixel": (100, 100), "terrain_m": (0.0, 0.0)},
        "B": {"pixel": (1000, 100), "terrain_m": (40.0, 0.0)},
        "C": {"pixel": (1000, 600), "terrain_m": (40.0, 20.0)},
        "D": {"pixel": (100, 600), "terrain_m": (0.0, 20.0)},
    }
    H = calibrer_homographie(correspondances)
    assert H.nb_points == 4
    assert H.methode == "perspective_4pts"

    # Projection d'un point connu (le centre du rectangle pixel = centre du terrain)
    x_m, y_m = H.projeter_point((550.0, 350.0))
    assert abs(x_m - 20.0) < 0.5
    assert abs(y_m - 10.0) < 0.5


def test_calibrer_homographie_moins_de_4_points_leve_erreur() -> None:
    """Moins de 4 points doit lever ValueError."""
    correspondances = {
        "A": {"pixel": (0, 0), "terrain_m": (0.0, 0.0)},
        "B": {"pixel": (100, 0), "terrain_m": (1.0, 0.0)},
        "C": {"pixel": (100, 100), "terrain_m": (1.0, 1.0)},
    }
    with pytest.raises(ValueError, match="minimum 4 points"):
        calibrer_homographie(correspondances)


def test_calibrer_homographie_7_points_ransac() -> None:
    """Avec >4 points, on utilise RANSAC."""
    correspondances = {
        f"P{i}": {"pixel": (100 + i * 50, 100 + i * 30),
                  "terrain_m": (float(i * 2), float(i))}
        for i in range(7)
    }
    H = calibrer_homographie(correspondances)
    assert H.nb_points == 7
    assert H.methode in ("ransac", "least_squares")


def test_projeter_bbox_pieds() -> None:
    """La projection bbox doit prendre le centre-bas."""
    correspondances = {
        "A": {"pixel": (0, 0), "terrain_m": (0.0, 0.0)},
        "B": {"pixel": (1000, 0), "terrain_m": (40.0, 0.0)},
        "C": {"pixel": (1000, 500), "terrain_m": (40.0, 20.0)},
        "D": {"pixel": (0, 500), "terrain_m": (0.0, 20.0)},
    }
    H = calibrer_homographie(correspondances)
    bbox = np.array([100.0, 100.0, 200.0, 400.0])
    x_m, y_m = H.projeter_bbox_pieds(bbox)
    # Le centre-bas est en (150, 400), qui projette sur terrain
    assert 5.0 < x_m < 7.0
    assert 15.0 < y_m < 17.0
