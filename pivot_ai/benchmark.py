"""Benchmark de la continuite d'identite du tracking (espace image).

Mesure la qualite d'un tracking directement sur les bboxes (pixels), SANS
homographie : la continuite d'ID est un probleme d'espace image. Reutilise le
coeur de detection de reprises de `metriques.py` (partage terrain/pixels).

Fournit, par run (clip x tracker) :
- nb de tracks
- longueur des tracks (moy / mediane / p90), en frames et en secondes
- nb de fragments courts
- reprises d'ID (switches suspectes), en tenant compte des changements de plan
- nb de joueurs estimes (recollage des reprises)
- taux de fragmentation

Purement geometrique -> testable sans GPU sur des detections synthetiques.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import asdict, dataclass, field

import supervision as sv

from pivot_ai.metriques import (
    BornesTrack,
    composantes_connexes,
    detecter_reprises_bornes,
)

logger = logging.getLogger(__name__)


def _centre_bbox(xyxy) -> tuple[float, float]:
    """Centre (x, y) en pixels d'une bbox [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def trajectoires_pixels(
    detections_trackees: dict[int, sv.Detections],
) -> dict[int, list[tuple[int, float, float]]]:
    """Extrait, par tracker_id, la liste (frame, cx, cy) des centres de bbox.

    Args:
        detections_trackees: dict frame_idx -> Detections (avec tracker_id).

    Returns:
        dict tracker_id -> liste triee de (frame_idx, cx_px, cy_px).
    """
    traj: dict[int, list[tuple[int, float, float]]] = {}
    for fi in sorted(detections_trackees.keys()):
        dets = detections_trackees[fi]
        if dets.tracker_id is None:
            continue
        for i in range(len(dets)):
            tid = int(dets.tracker_id[i])
            cx, cy = _centre_bbox(dets.xyxy[i])
            traj.setdefault(tid, []).append((int(fi), cx, cy))
    for tid in traj:
        traj[tid].sort(key=lambda p: p[0])
    return traj


def bornes_pixels(
    trajectoires: dict[int, list[tuple[int, float, float]]],
) -> dict[int, BornesTrack]:
    """Convertit des trajectoires pixel en BornesTrack (premiere/derniere position)."""
    bornes: dict[int, BornesTrack] = {}
    for tid, pts in trajectoires.items():
        if not pts:
            continue
        f0, x0, y0 = pts[0]
        f1, x1, y1 = pts[-1]
        bornes[tid] = BornesTrack(
            frame_debut=f0, x_debut=x0, y_debut=y0,
            frame_fin=f1, x_fin=x1, y_fin=y1, nb_reel=len(pts),
        )
    return bornes


@dataclass
class ResumeBenchmark:
    """Metriques de continuite d'ID pour un run (clip x tracker)."""

    tracker: str = ""
    clip: str = ""
    nb_tracks: int = 0
    nb_joueurs_estimes: int = 0
    nb_reprises_id: int = 0
    nb_fragments_courts: int = 0
    fragmentation: float = 0.0
    longueur_moyenne_frames: float = 0.0
    longueur_mediane_frames: float = 0.0
    longueur_p90_frames: float = 0.0
    longueur_moyenne_s: float = 0.0
    nb_coupures_plan: int = 0
    frames_coupure: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _percentile(valeurs: list[float], q: float) -> float:
    """Percentile q (0-100) simple, sans dependance numpy."""
    if not valeurs:
        return 0.0
    tri = sorted(valeurs)
    if len(tri) == 1:
        return float(tri[0])
    rang = (q / 100.0) * (len(tri) - 1)
    bas = int(rang)
    haut = min(bas + 1, len(tri) - 1)
    frac = rang - bas
    return float(tri[bas] + (tri[haut] - tri[bas]) * frac)


def resume_benchmark(
    detections_trackees: dict[int, sv.Detections],
    fps: float,
    subsample: int = 1,
    seuil_distance_px: float = 120.0,
    seuil_frames: int = 30,
    min_frames_fragment: int = 5,
    frames_coupure: tuple[int, ...] = (),
    tracker: str = "",
    clip: str = "",
) -> ResumeBenchmark:
    """Calcule les metriques de continuite d'ID pour un run.

    Args:
        detections_trackees: sortie du tracking (frame -> Detections + tracker_id).
        fps: framerate reel du clip.
        subsample: 1 frame sur N reellement trackee (pour convertir en secondes).
        seuil_distance_px: distance max (px) pour lier deux tracks (reprise d'ID).
        seuil_frames: delai max (frames source) pour une reprise.
        min_frames_fragment: en dessous, un track est un "fragment court".
        frames_coupure: indices des changements de plan (reprises enjambant une
            coupure ignorees).
        tracker, clip: etiquettes pour le rapport.

    Returns:
        ResumeBenchmark.
    """
    traj = trajectoires_pixels(detections_trackees)
    bornes = bornes_pixels(traj)
    nb_tracks = len(bornes)

    longueurs = [b.nb_reel for b in bornes.values()]
    reprises = detecter_reprises_bornes(
        bornes, seuil_distance_px, seuil_frames, tuple(frames_coupure)
    )
    nb_joueurs = composantes_connexes(list(bornes.keys()), reprises)

    # Duree reelle couverte par une frame trackee = subsample / fps.
    sec_par_frame_trackee = subsample / fps if fps > 0 else 0.0
    moy = statistics.fmean(longueurs) if longueurs else 0.0

    return ResumeBenchmark(
        tracker=tracker,
        clip=clip,
        nb_tracks=nb_tracks,
        nb_joueurs_estimes=nb_joueurs,
        nb_reprises_id=len(reprises),
        nb_fragments_courts=sum(1 for n in longueurs if n < min_frames_fragment),
        fragmentation=round(nb_tracks / max(1, nb_joueurs), 2),
        longueur_moyenne_frames=round(moy, 1),
        longueur_mediane_frames=round(statistics.median(longueurs), 1) if longueurs else 0.0,
        longueur_p90_frames=round(_percentile([float(x) for x in longueurs], 90), 1),
        longueur_moyenne_s=round(moy * sec_par_frame_trackee, 2),
        nb_coupures_plan=len(frames_coupure),
        frames_coupure=list(frames_coupure),
    )
