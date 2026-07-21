"""Tracking multi-objets via ByteTrack avec stabilisation des classes.

Le tracking attribue un ID stable aux objets entre frames. La classe predite
par le detecteur peut varier d'une frame a l'autre pour un meme tracker_id ;
on stabilise par vote majoritaire sur l'ensemble du clip.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict

import supervision as sv

logger = logging.getLogger(__name__)


def tracker_detections(
    detections_par_frame: dict[int, sv.Detections],
    fps: float,
    subsample: int = 1,
    track_activation_threshold: float = 0.25,
    lost_track_buffer: int = 30,
    minimum_matching_threshold: float = 0.8,
    minimum_consecutive_frames: int = 2,
) -> tuple[dict[int, sv.Detections], dict[int, int]]:
    """Applique ByteTrack et stabilise la classe par tracker_id.

    Le framerate transmis a ByteTrack est corrige du subsample : on ne feed
    qu'une frame sur `subsample`, donc le modele de mouvement doit raisonner
    sur le framerate effectif `fps / subsample` (sinon la prediction Kalman
    est fausse d'un facteur `subsample`, ce qui fragmente le tracking).

    `minimum_consecutive_frames > 1` supprime les tracks fantomes (detections
    isolees sur 1-2 frames qui recevaient un tracker_id ephemere).

    Args:
        detections_par_frame: sortie du detecteur, dict frame_idx -> Detections
        fps: framerate reel du clip
        subsample: 1 frame sur N reellement fournie au tracker
        track_activation_threshold: confiance min pour demarrer un track
        lost_track_buffer: nb de frames (effectives) ou un track perdu survit,
            pour re-matcher un joueur apres occlusion sans changer d'ID
        minimum_matching_threshold: seuil d'association IoU/embedding
        minimum_consecutive_frames: nb de frames consecutives requises avant
            qu'un track soit valide (>=2 elimine les fantomes 1-frame)

    Returns:
        tuple (detections_avec_ids, classe_finale)
        - detections_avec_ids : dict frame_idx -> Detections avec tracker_id
        - classe_finale : dict tracker_id -> classe stabilisee (vote majoritaire)
    """
    frame_rate_effectif = max(1, int(round(fps / max(1, subsample))))
    tracker = sv.ByteTrack(
        frame_rate=frame_rate_effectif,
        track_activation_threshold=track_activation_threshold,
        lost_track_buffer=lost_track_buffer,
        minimum_matching_threshold=minimum_matching_threshold,
        minimum_consecutive_frames=minimum_consecutive_frames,
    )
    detections_trackees: dict[int, sv.Detections] = {}
    classes_observees: dict[int, Counter] = defaultdict(Counter)

    for fi in sorted(detections_par_frame.keys()):
        dets = tracker.update_with_detections(detections_par_frame[fi])
        detections_trackees[fi] = dets

        if dets.tracker_id is None:
            continue

        for i in range(len(dets)):
            tid = int(dets.tracker_id[i])
            cid = int(dets.class_id[i])
            classes_observees[tid][cid] += 1

    classe_finale = {
        tid: int(compteur.most_common(1)[0][0])
        for tid, compteur in classes_observees.items()
    }

    logger.info(
        "Tracking termine : %d trackers stables sur %d frames",
        len(classe_finale),
        len(detections_trackees),
    )
    return detections_trackees, classe_finale
