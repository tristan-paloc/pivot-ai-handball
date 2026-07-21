"""Tests du backend BoT-SORT : config yaml + routage pipeline.

Le comportement reel de BoT-SORT+ReID (Ultralytics model.track) necessite
torch + GPU et se valide sur Colab. Ici on verifie ce qui est testable en
local : la config est valide, et le pipeline route bien vers le bon backend.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from pivot_ai.tracking import CHEMIN_CONFIG_BOTSORT


def test_config_botsort_existe_et_valide() -> None:
    """La config BoT-SORT est livree et active le ReID."""
    assert CHEMIN_CONFIG_BOTSORT.exists(), f"Config absente : {CHEMIN_CONFIG_BOTSORT}"
    cfg = yaml.safe_load(CHEMIN_CONFIG_BOTSORT.read_text(encoding="utf-8"))
    assert cfg["tracker_type"] == "botsort"
    assert cfg["with_reid"] is True
    # buffer augmente pour survivre aux occlusions
    assert cfg["track_buffer"] >= 30


def test_config_botsort_dans_le_package() -> None:
    """La config est bien sous pivot_ai/trackers/ (packagee)."""
    assert CHEMIN_CONFIG_BOTSORT.parent.name == "trackers"
    assert CHEMIN_CONFIG_BOTSORT.parent.parent.name == "pivot_ai"


def _homographie_identite() -> dict:
    return {
        "A": {"pixel": (0, 0), "terrain_m": (0.0, 0.0)},
        "B": {"pixel": (320, 0), "terrain_m": (40.0, 0.0)},
        "C": {"pixel": (320, 240), "terrain_m": (40.0, 20.0)},
        "D": {"pixel": (0, 240), "terrain_m": (0.0, 20.0)},
    }


def test_pipeline_route_vers_botsort(tmp_path: Path) -> None:
    """tracker='botsort' appelle tracker_video_botsort, pas la voie ByteTrack."""
    from tests.conftest import generer_video_factice

    source = tmp_path / "match.mp4"
    generer_video_factice(source, nb_frames=20, fps=25.0, largeur=320, hauteur=240)

    # Stub : le backend botsort est mocke pour renvoyer un tracking vide
    # (on verifie seulement le ROUTAGE, pas le tracker reel).
    from pivot_ai.pipeline import traiter_match_complet

    faux_retour = ({}, {})  # (detections_trackees, classe_finale) vides
    with (
        patch("pivot_ai.pipeline.tracker_video_botsort", return_value=faux_retour) as mock_bot,
        patch("pivot_ai.pipeline.detecter_video") as mock_detect,
        patch("pivot_ai.pipeline.tracker_detections") as mock_byte,
    ):
        traiter_match_complet(
            chemin_video=source,
            correspondances_homographie=_homographie_identite(),
            dossier_sortie=tmp_path / "out",
            subsample=5,
            generer_video_radar=False,
            decouper_actions=False,
            detecteur=object(),  # jamais utilise (backend mocke)
            tracker="botsort",
        )
    mock_bot.assert_called_once()
    mock_detect.assert_not_called()
    mock_byte.assert_not_called()


def test_pipeline_tracker_inconnu_leve_erreur(tmp_path: Path) -> None:
    """Un backend inconnu leve ValueError."""
    from pivot_ai.pipeline import traiter_match_complet
    from tests.conftest import generer_video_factice

    source = tmp_path / "match.mp4"
    generer_video_factice(source, nb_frames=10, fps=25.0, largeur=320, hauteur=240)
    with pytest.raises(ValueError, match="tracker inconnu"):
        traiter_match_complet(
            chemin_video=source,
            correspondances_homographie=_homographie_identite(),
            dossier_sortie=tmp_path / "out",
            detecteur=object(),
            tracker="deepsort",
        )
