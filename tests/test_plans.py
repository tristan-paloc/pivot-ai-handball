"""Tests de la detection des changements de plan (coupures camera)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from pivot_ai.plans import detecter_changements_plan, segments_entre_coupures


def _video_deux_scenes(
    chemin: Path, n_par_scene: int = 15, fps: float = 25.0,
    couleur_a=(200, 60, 60), couleur_b=(60, 60, 200),
) -> int:
    """Ecrit une video : scene A (teinte 1) puis scene B (teinte differente).

    Retourne l'index de frame ou la scene B commence (la coupure attendue).
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(chemin), fourcc, fps, (160, 120))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter indisponible")
    for _ in range(n_par_scene):
        writer.write(np.full((120, 160, 3), couleur_a, dtype=np.uint8))
    for _ in range(n_par_scene):
        writer.write(np.full((120, 160, 3), couleur_b, dtype=np.uint8))
    writer.release()
    return n_par_scene


def test_detecte_une_coupure(tmp_path: Path) -> None:
    """Une video a deux scenes de teintes differentes -> une coupure au raccord."""
    source = tmp_path / "deux_scenes.mp4"
    debut_b = _video_deux_scenes(source, n_par_scene=15)
    coupures = detecter_changements_plan(str(source), seuil_correlation=0.5)
    assert len(coupures) == 1
    # tolerance d'1 frame sur la position du raccord
    assert abs(coupures[0] - debut_b) <= 1


def test_video_uniforme_pas_de_coupure(tmp_path: Path) -> None:
    """Une video d'une seule teinte -> aucune coupure."""
    source = tmp_path / "uniforme.mp4"
    _video_deux_scenes(source, n_par_scene=15, couleur_a=(120, 90, 60), couleur_b=(120, 90, 60))
    coupures = detecter_changements_plan(str(source), seuil_correlation=0.5)
    assert coupures == []


def test_segments_entre_coupures() -> None:
    """Le decoupage en segments intra-plan est correct."""
    assert segments_entre_coupures(0, 99, []) == [(0, 99)]
    assert segments_entre_coupures(0, 99, [40]) == [(0, 39), (40, 99)]
    assert segments_entre_coupures(0, 99, [30, 70]) == [(0, 29), (30, 69), (70, 99)]
    # coupures hors intervalle ignorees
    assert segments_entre_coupures(10, 50, [5, 60]) == [(10, 50)]
