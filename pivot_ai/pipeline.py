"""Pipeline orchestrateur : enchaine detection, tracking, equipes, stats, decoupage.

Module a IMPLEMENTER par l'agent Claude Code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ResultatPipeline:
    """Resultat complet d'un traitement de clip ou de match."""

    chemin_video_source: Path
    chemin_video_radar: Path | None
    stats_joueurs: Any  # polars.DataFrame
    actions_detectees: list[Any]  # list[Action]
    clips_decoupes: list[Path]
    metadonnees: dict[str, Any]


def traiter_match_complet(
    chemin_video: str | Path,
    correspondances_homographie: dict[str, Any],
    dossier_sortie: str | Path,
    subsample: int = 2,
    generer_video_radar: bool = True,
    decouper_actions: bool = True,
) -> ResultatPipeline:
    """Pipeline complet : video -> stats + clips d'actions.

    Args:
        chemin_video: video source
        correspondances_homographie: dict points pour calibrer homographie
            (voir pivot_ai.homographie.Correspondance)
        dossier_sortie: dossier ou ecrire stats, video radar, clips
        subsample: 1 frame sur N pour la detection
        generer_video_radar: si True, genere la video annotee + radar
        decouper_actions: si True, decoupe les actions detectees

    Returns:
        ResultatPipeline avec tous les artefacts
    """
    raise NotImplementedError(
        "A implementer par Claude Code. Etapes : "
        "1) Charger video, valider correspondances. "
        "2) Calibrer homographie via pivot_ai.homographie.calibrer_homographie. "
        "3) Detection : DetecteurLocal + detecter_video avec subsample. "
        "4) Tracking : tracker_detections. "
        "5) Equipes : classifier_equipes. "
        "6) Interpoler les positions sur frames non-echantillonnees. "
        "7) Stats : calculer_stats_joueur + sauvegarder en CSV/Parquet. "
        "8) Si generer_video_radar : generer la video SBS (broadcast + radar). "
        "9) Si decouper_actions : detecter_actions + decouper_clips_video. "
        "10) Construire et retourner ResultatPipeline."
    )
