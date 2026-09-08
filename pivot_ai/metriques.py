"""Metriques de qualite du tracking : reprises d'ID et nb de joueurs estime.

Permet de juger objectivement un tracker (ByteTrack vs BoT-SORT) sans regarder
la video. Idee : un joueur mal suivi apparait comme plusieurs pistes courtes.
Quand une piste s'eteint et qu'une nouvelle demarre juste apres, tout pres, il
s'agit tres probablement du meme joueur -> une "reprise d'ID" (ID switch). En
recollant ces chaines, on estime le nombre de joueurs reels.

Purement geometrique (positions terrain en metres), testable sans GPU.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from pivot_ai.stats import PositionJoueur

logger = logging.getLogger(__name__)


@dataclass
class ResumeQualiteTracking:
    """Synthese chiffree de la qualite d'un tracking."""

    nb_tracks: int
    nb_reprises_id: int
    nb_joueurs_estimes: int
    longueur_track_moyenne: float


@dataclass
class BornesTrack:
    """Premiere et derniere position reelle d'une piste."""

    frame_debut: int
    x_debut: float
    y_debut: float
    frame_fin: int
    x_fin: float
    y_fin: float
    nb_reel: int


def _bornes_track(positions: list[PositionJoueur]) -> BornesTrack | None:
    """Extrait les bornes temporelles/spatiales d'une piste (positions reelles)."""
    reels = sorted(
        (p for p in positions if not p.interpole), key=lambda p: p.frame_idx
    )
    if not reels:
        return None
    d, f = reels[0], reels[-1]
    return BornesTrack(
        frame_debut=d.frame_idx,
        x_debut=d.x_m,
        y_debut=d.y_m,
        frame_fin=f.frame_idx,
        x_fin=f.x_m,
        y_fin=f.y_m,
        nb_reel=len(reels),
    )


def _coupure_entre(frame_fin: int, frame_debut: int, frames_coupure: tuple[int, ...]) -> bool:
    """True si un changement de plan tombe dans l'intervalle ]frame_fin, frame_debut]."""
    return any(frame_fin < c <= frame_debut for c in frames_coupure)


def detecter_reprises_bornes(
    bornes: dict[int, BornesTrack],
    seuil_distance: float,
    seuil_frames: int,
    frames_coupure: tuple[int, ...] = (),
) -> list[tuple[int, int]]:
    """Coeur generique : detecte les reprises d'ID a partir de bornes de tracks.

    Reutilise en espace terrain (metres) comme en espace image (pixels) : seule
    l'unite de `seuil_distance` change. Une reprise (A, B) est comptee si B
    demarre apres la fin de A, dans un delai <= seuil_frames, a une distance
    <= seuil_distance, ET sans changement de plan entre les deux (une reprise
    qui enjambe une coupure camera n'est PAS un echec de tracking).

    Args:
        bornes: dict tracker_id -> BornesTrack (premiere/derniere position).
        seuil_distance: distance max entre fin de A et debut de B (m ou px).
        seuil_frames: delai max (frames) entre fin de A et debut de B.
        frames_coupure: indices de frames ou un changement de plan se produit.

    Returns:
        liste de couples (tracker_a, tracker_b) = reprises detectees.
    """
    fins = sorted(bornes.items(), key=lambda kv: kv[1].frame_fin)
    debut_pris: set[int] = set()
    reprises: list[tuple[int, int]] = []

    for tid_a, ba in fins:
        meilleur: int | None = None
        meilleure_dist: float | None = None
        for tid_b, bb in bornes.items():
            if tid_b == tid_a or tid_b in debut_pris:
                continue
            dt = bb.frame_debut - ba.frame_fin
            if dt <= 0 or dt > seuil_frames:
                continue
            if _coupure_entre(ba.frame_fin, bb.frame_debut, frames_coupure):
                continue
            dist = math.hypot(bb.x_debut - ba.x_fin, bb.y_debut - ba.y_fin)
            if dist > seuil_distance:
                continue
            if meilleure_dist is None or dist < meilleure_dist:
                meilleure_dist = dist
                meilleur = tid_b
        if meilleur is not None:
            reprises.append((tid_a, meilleur))
            debut_pris.add(meilleur)

    return reprises


def composantes_connexes(tids: list[int], reprises: list[tuple[int, int]]) -> int:
    """Nb de composantes connexes (union-find) : chaque reprise fusionne 2 tracks."""
    parent = {t: t for t in tids}

    def racine(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in reprises:
        if a in parent and b in parent:
            ra, rb = racine(a), racine(b)
            if ra != rb:
                parent[ra] = rb
    return len({racine(t) for t in tids})


def detecter_reprises_id(
    positions_par_tracker: dict[int, list[PositionJoueur]],
    seuil_distance_m: float = 3.0,
    seuil_frames: int = 30,
    frames_coupure: tuple[int, ...] = (),
) -> list[tuple[int, int]]:
    """Detecte les reprises d'ID en espace terrain (metres).

    Args:
        positions_par_tracker: positions reelles par tracker_id (avant filtre)
        seuil_distance_m: distance max (m) entre fin de A et debut de B
        seuil_frames: delai max (frames source) entre fin de A et debut de B
        frames_coupure: indices des changements de plan (reprises enjambant une
            coupure ignorees)

    Returns:
        liste de couples (tracker_a, tracker_b) = reprises detectees.
    """
    bornes = {
        tid: b
        for tid, pos in positions_par_tracker.items()
        if (b := _bornes_track(pos)) is not None
    }
    return detecter_reprises_bornes(
        bornes, seuil_distance_m, seuil_frames, frames_coupure
    )


def estimer_nb_joueurs(
    positions_par_tracker: dict[int, list[PositionJoueur]],
    reprises: list[tuple[int, int]] | None = None,
    seuil_distance_m: float = 3.0,
    seuil_frames: int = 30,
) -> int:
    """Estime le nb de joueurs reels en recollant les chaines de reprises.

    Union-find : chaque reprise (A, B) fusionne A et B dans le meme joueur.
    Le nb de composantes connexes = nb de joueurs logiques estimes.

    Args:
        positions_par_tracker: positions reelles par tracker_id
        reprises: reprises deja calculees (sinon recalculees)
        seuil_distance_m: passe a detecter_reprises_id si reprises=None
        seuil_frames: passe a detecter_reprises_id si reprises=None

    Returns:
        nb de joueurs estimes (composantes connexes).
    """
    if reprises is None:
        reprises = detecter_reprises_id(
            positions_par_tracker, seuil_distance_m, seuil_frames
        )
    return composantes_connexes(list(positions_par_tracker.keys()), reprises)


def resumer_qualite_tracking(
    positions_par_tracker: dict[int, list[PositionJoueur]],
    seuil_distance_m: float = 3.0,
    seuil_frames: int = 30,
) -> ResumeQualiteTracking:
    """Assemble les KPI de qualite du tracking.

    Args:
        positions_par_tracker: positions reelles par tracker_id (avant filtre)
        seuil_distance_m: distance max pour une reprise d'ID
        seuil_frames: delai max (frames source) pour une reprise d'ID

    Returns:
        ResumeQualiteTracking (nb_tracks, nb_reprises_id, nb_joueurs_estimes,
        longueur_track_moyenne).
    """
    nb_tracks = len(positions_par_tracker)
    reprises = detecter_reprises_id(
        positions_par_tracker, seuil_distance_m, seuil_frames
    )
    nb_joueurs = estimer_nb_joueurs(positions_par_tracker, reprises=reprises)

    longueurs = [
        sum(1 for p in pos if not p.interpole)
        for pos in positions_par_tracker.values()
    ]
    longueur_moy = sum(longueurs) / len(longueurs) if longueurs else 0.0

    resume = ResumeQualiteTracking(
        nb_tracks=nb_tracks,
        nb_reprises_id=len(reprises),
        nb_joueurs_estimes=nb_joueurs,
        longueur_track_moyenne=longueur_moy,
    )
    logger.info(
        "Qualite tracking : %d pistes, %d reprises d'ID, ~%d joueurs estimes "
        "(longueur moy %.1f frames)",
        resume.nb_tracks,
        resume.nb_reprises_id,
        resume.nb_joueurs_estimes,
        resume.longueur_track_moyenne,
    )
    return resume
