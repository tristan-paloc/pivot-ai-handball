"""Decoupage automatique d'un match en actions individuelles.

Approche heuristique simple, sans ML, basee sur des regles metier :
- Une action = sequence ou >= 6 joueurs sont presents sur le terrain
- Debut : transition de 0-2 joueurs a 6+ joueurs en moins de 2s
- Fin : retour a <4 joueurs OU pause >3s sans mouvement significatif

A AMELIORER plus tard avec un modele ML de classification d'evenements.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
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


def _compter_joueurs_par_frame(
    detections_trackees: dict[int, sv.Detections],
    classe_finale: dict[int, int],
    id_classe_joueur: int,
) -> dict[int, int]:
    """Compte les joueurs de champ presents par frame, en utilisant la classe stabilisee.

    Args:
        detections_trackees: dict frame_idx -> Detections (avec tracker_id)
        classe_finale: dict tracker_id -> classe stabilisee
        id_classe_joueur: id de la classe joueur de champ

    Returns:
        dict frame_idx -> nb_joueurs_de_champ
    """
    compteur: dict[int, int] = {}
    for fi, dets in detections_trackees.items():
        if dets.tracker_id is None or len(dets) == 0:
            compteur[fi] = 0
            continue
        n = sum(
            1
            for tid in dets.tracker_id
            if classe_finale.get(int(tid)) == id_classe_joueur
        )
        compteur[fi] = int(n)
    return compteur


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

    Machine a etats a 2 etats (attente, action) parcourant les frames
    par ordre croissant. Transition attente -> action quand nb_joueurs
    >= seuil_debut_joueurs. Transition action -> attente quand la pause
    (nb_joueurs < seuil_fin_joueurs) dure plus de duree_max_pause_s.
    Les actions de duree < duree_min_action_s sont ecartees.

    Args:
        detections_trackees: dict frame_idx -> Detections
        classe_finale: dict tracker_id -> classe stabilisee
        id_classe_joueur: id de la classe joueur de champ
        fps: framerate
        seuil_debut_joueurs: nb min pour declencher une action
        seuil_fin_joueurs: nb sous lequel on considere la pause
        duree_min_action_s: duree minimum pour qu'une action soit valide
        duree_max_pause_s: pause max autorisee dans une action

    Returns:
        liste d'Action triees par frame_debut.

    Raises:
        ValueError: si fps <= 0 ou si seuils incoherents
    """
    if fps <= 0:
        raise ValueError(f"fps doit etre > 0, recu {fps}")
    if seuil_fin_joueurs > seuil_debut_joueurs:
        raise ValueError(
            "seuil_fin_joueurs ne peut pas etre superieur a seuil_debut_joueurs"
        )

    nb_par_frame = _compter_joueurs_par_frame(
        detections_trackees, classe_finale, id_classe_joueur
    )
    if not nb_par_frame:
        return []

    frames_triees = sorted(nb_par_frame.keys())

    etat = "attente"
    debut_action: int | None = None
    frame_derniere_presence: int | None = None
    actions: list[Action] = []

    def cloturer(fin_frame: int) -> None:
        """Cloture l'action courante et l'ajoute si valide.

        nb_joueurs_moyen est calcule sur la fenetre [debut_action, fin_frame]
        en s'appuyant uniquement sur les frames effectivement analysees.
        """
        nonlocal etat, debut_action, frame_derniere_presence
        if debut_action is None:
            etat = "attente"
            return
        duree_s = (fin_frame - debut_action) / fps
        if duree_s >= duree_min_action_s:
            frames_action = [
                nb_par_frame[f]
                for f in frames_triees
                if debut_action <= f <= fin_frame
            ]
            nb_moyen = (
                sum(frames_action) / len(frames_action) if frames_action else 0.0
            )
            actions.append(
                Action(
                    frame_debut=debut_action,
                    frame_fin=fin_frame,
                    duree_s=float(duree_s),
                    equipe_en_attaque=None,
                    nb_joueurs_moyen=float(nb_moyen),
                )
            )
        etat = "attente"
        debut_action = None
        frame_derniere_presence = None

    for fi in frames_triees:
        n = nb_par_frame[fi]
        if etat == "attente":
            if n >= seuil_debut_joueurs:
                etat = "action"
                debut_action = fi
                frame_derniere_presence = fi
        else:  # etat == "action"
            if n >= seuil_fin_joueurs:
                frame_derniere_presence = fi
            else:
                assert frame_derniere_presence is not None
                duree_pause_s = (fi - frame_derniere_presence) / fps
                if duree_pause_s > duree_max_pause_s:
                    cloturer(frame_derniere_presence)

    # Cloture l'action en cours a la fin du clip.
    if etat == "action" and frame_derniere_presence is not None:
        cloturer(frame_derniere_presence)

    logger.info(
        "Detection actions : %d actions retenues sur %d frames analysees",
        len(actions),
        len(frames_triees),
    )
    return actions


def _verifier_ffmpeg() -> None:
    """Verifie que ffmpeg est dispo, sinon raise."""
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg non trouve. Installe-le : "
            "apt-get install ffmpeg (Linux) ou brew install ffmpeg (Mac) "
            "ou winget install ffmpeg (Windows)."
        )


def _lire_fps(chemin_video: Path) -> float:
    """Lit le framerate d'une video via OpenCV."""
    cap = cv2.VideoCapture(str(chemin_video))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la video : {chemin_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"fps invalide ({fps}) pour la video : {chemin_video}")
    return fps


def decouper_clips_video(
    chemin_video_source: str | Path,
    actions: list[Action],
    dossier_sortie: str | Path,
    prefixe: str = "action",
) -> list[Path]:
    """Decoupe les actions detectees en fichiers MP4 separes via ffmpeg.

    Utilise -c copy pour eviter le re-encodage (rapide, qualite preservee).
    Les bornes ne tombent pas forcement sur des keyframes : ffmpeg peut
    arrondir au keyframe le plus proche.

    Args:
        chemin_video_source: video du match complet
        actions: liste des actions a extraire
        dossier_sortie: dossier de sortie (cree si absent)
        prefixe: prefixe des fichiers (defaut "action")

    Returns:
        liste des chemins des clips generes (peut etre plus courte que
        `actions` si certains decoupages echouent : warning + skip).

    Raises:
        FileNotFoundError: si ffmpeg n'est pas installe ou source absente
    """
    _verifier_ffmpeg()
    chemin_source = Path(chemin_video_source)
    if not chemin_source.exists():
        raise FileNotFoundError(f"Video source introuvable : {chemin_source}")

    sortie = Path(dossier_sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    fps = _lire_fps(chemin_source)
    clips_generes: list[Path] = []

    for idx, action in enumerate(actions):
        t_debut = action.frame_debut / fps
        nom = f"{prefixe}_{idx:03d}_{action.duree_s:.1f}s.mp4"
        chemin_sortie = sortie / nom
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{t_debut:.3f}",
            "-i",
            str(chemin_source),
            "-t",
            f"{action.duree_s:.3f}",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(chemin_sortie),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "Echec decoupage action %d (%.1fs) : %s",
                idx,
                action.duree_s,
                exc.stderr.decode("utf-8", errors="replace")[:200],
            )
            continue
        clips_generes.append(chemin_sortie)
        logger.info(
            "Clip genere : %s (debut=%.2fs, duree=%.2fs)",
            chemin_sortie.name,
            t_debut,
            action.duree_s,
        )

    logger.info(
        "Decoupage termine : %d/%d clips generes dans %s",
        len(clips_generes),
        len(actions),
        sortie,
    )
    return clips_generes
