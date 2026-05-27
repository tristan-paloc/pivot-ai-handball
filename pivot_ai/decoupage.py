"""Decoupage automatique d'un match en actions individuelles.

Approche heuristique simple, sans ML, basee sur des regles metier :
- Une action = sequence ou >= 6 joueurs sont presents sur le terrain
- Debut : transition de 0-2 joueurs a 6+ joueurs en moins de 2s
- Fin : retour a <4 joueurs OU pause >3s sans mouvement significatif

A AMELIORER plus tard avec un modele ML de classification d'evenements.

Module a IMPLEMENTER par l'agent Claude Code.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import supervision as sv

logger = logging.getLogger(__name__)


@dataclass
class Action:
    """Une action detectee dans le match."""

    frame_debut: int
    frame_fin: int
    duree_s: float
    equipe_en_attaque: int | None  # 0, 1 ou None si indetermine
    nb_joueurs_moyen: float


def detecter_actions(
    detections_trackees: dict[int, sv.Detections],
    classe_finale: dict[int, int],
    id_classe_joueur: int,
    fps: float,
    seuil_debut_joueurs: int = 6,
    seuil_fin_joueurs: int = 4,
    duree_min_action_s: float = 3.0,
    duree_max_pause_s: float = 3.0,
) -> list[Action]:
    """Detecte les actions dans un match par heuristique simple.

    Args:
        detections_trackees: dict frame_idx -> Detections
        classe_finale: dict tracker_id -> classe
        id_classe_joueur: id de la classe joueur de champ
        fps: framerate
        seuil_debut_joueurs: nb min de joueurs pour declencher une action
        seuil_fin_joueurs: nb sous lequel l'action est consideree finie
        duree_min_action_s: duree minimum pour qu'une action soit valide
        duree_max_pause_s: pause max autorisee dans une action

    Returns:
        liste d'Action triees par frame_debut
    """
    raise NotImplementedError(
        "A implementer par Claude Code. Algorithme : "
        "1) Pour chaque frame, compter le nombre de joueurs de champ presents. "
        "2) Parcourir les frames sequentiellement avec une machine a etats : "
        "   - etat 'attente' : si nb_joueurs >= seuil_debut, basculer en 'action' "
        "   - etat 'action' : si nb_joueurs < seuil_fin pendant > duree_max_pause_s, "
        "     cloturer l'action. "
        "3) Filtrer les actions de duree < duree_min_action_s. "
        "4) Pour chaque action, calculer l'equipe en attaque "
        "   (heuristique : equipe majoritaire dans la moitie offensive du terrain)."
    )


def decouper_clips_video(
    chemin_video_source: str | Path,
    actions: list[Action],
    dossier_sortie: str | Path,
    prefixe: str = "action",
) -> list[Path]:
    """Decoupe les actions detectees en fichiers MP4 separes via ffmpeg.

    Args:
        chemin_video_source: video du match complet
        actions: liste des actions a extraire
        dossier_sortie: dossier de sortie
        prefixe: prefixe des fichiers (defaut "action")

    Returns:
        liste des chemins des clips generes

    Raises:
        FileNotFoundError: si ffmpeg n'est pas installe
        RuntimeError: si un decoupage echoue
    """
    raise NotImplementedError(
        "A implementer par Claude Code. Indications : "
        "1) Verifier que ffmpeg est dispo (shutil.which). "
        "2) Pour chaque action, calculer t_debut et t_fin en secondes. "
        "3) Appeler : ffmpeg -ss {t_debut} -i {source} -t {duree} -c copy {sortie} "
        "   (utiliser subprocess.run avec check=True). "
        "4) Nommer chaque fichier : {prefixe}_{idx:03d}_{duree_s:.1f}s.mp4 "
        "5) Logger chaque clip genere."
    )


def _verifier_ffmpeg() -> None:
    """Verifie que ffmpeg est dispo, sinon raise."""
    import shutil
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg non trouve. Installe-le : "
            "apt-get install ffmpeg (Linux) ou brew install ffmpeg (Mac)."
        )
