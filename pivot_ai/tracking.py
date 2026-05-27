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
) -> tuple[dict[int, sv.Detections], dict[int, int]]:
    """Applique ByteTrack et stabilise la classe par tracker_id.

    Args:
        detections_par_frame: sortie du detecteur, dict frame_idx -> Detections
        fps: framerate du clip (utilise par ByteTrack pour la prediction)

    Returns:
        tuple (detections_avec_ids, classe_finale)
        - detections_avec_ids : dict frame_idx -> Detections avec tracker_id
        - classe_finale : dict tracker_id -> classe stabilisee (vote majoritaire)
    """
    tracker = sv.ByteTrack(frame_rate=max(1, int(round(fps))))
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
