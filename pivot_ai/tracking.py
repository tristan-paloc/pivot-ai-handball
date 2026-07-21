"""Tracking multi-objets via ByteTrack avec stabilisation des classes.

Le tracking attribue un ID stable aux objets entre frames. La classe predite
par le detecteur peut varier d'une frame a l'autre pour un meme tracker_id ;
on stabilise par vote majoritaire sur l'ensemble du clip.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import supervision as sv

if TYPE_CHECKING:
    from pivot_ai.detection import DetecteurLocal

logger = logging.getLogger(__name__)

# Config BoT-SORT+ReID livree avec le package.
CHEMIN_CONFIG_BOTSORT = Path(__file__).parent / "trackers" / "botsort_reid.yaml"


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


def tracker_video_botsort(
    chemin_video: str | Path,
    detecteur: DetecteurLocal,
    subsample: int = 1,
    chemin_config: str | Path | None = None,
    max_frames: int | None = None,
) -> tuple[dict[int, sv.Detections], dict[int, int]]:
    """Detecte ET tracke via BoT-SORT + ReID (Ultralytics `model.track`).

    Contrairement a ByteTrack (mouvement seul), BoT-SORT calcule une empreinte
    d'apparence par joueur et re-identifie apres occlusion, ce qui reduit
    fortement les sauts d'ID au contact. La detection et le tracking sont
    couples dans `model.track` (stateful, persist=True), donc cette fonction
    remplace a la fois detecter_video et tracker_detections.

    Le remap eventuel des classes (modele handball -> CLASSES_HANDBALL) est
    applique si le detecteur en a un.

    Args:
        chemin_video: chemin du MP4
        detecteur: DetecteurLocal (expose .model YOLO Ultralytics)
        subsample: 1 frame sur N fournie au tracker
        chemin_config: yaml BoT-SORT (defaut : config ReID livree)
        max_frames: limite optionnelle (index frame source)

    Returns:
        tuple (detections_trackees, classe_finale), meme format que
        tracker_detections pour compatibilite avec le reste du pipeline.

    Raises:
        RuntimeError: si la video ne s'ouvre pas
    """
    from pivot_ai.detection import _appliquer_remap

    config = str(chemin_config or CHEMIN_CONFIG_BOTSORT)
    cap = cv2.VideoCapture(str(chemin_video))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la video : {chemin_video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total = min(total, max_frames)

    remap = getattr(detecteur, "remap_classes", None)
    classes = detecteur._classes_predict
    detections_trackees: dict[int, sv.Detections] = {}
    classes_observees: dict[int, Counter] = defaultdict(Counter)

    try:
        fi = 0
        while fi < total:
            ret, frame = cap.read()
            if not ret:
                break
            if fi % subsample == 0:
                resultats = detecteur.model.track(
                    frame,
                    persist=True,
                    tracker=config,
                    conf=detecteur.config.confiance_min,
                    iou=detecteur.config.iou_min,
                    classes=classes,
                    verbose=False,
                    device=detecteur.config.device,
                )
                dets = sv.Detections.from_ultralytics(resultats[0])
                if remap:
                    dets = _appliquer_remap(dets, remap)
                detections_trackees[fi] = dets
                if dets.tracker_id is not None:
                    for i in range(len(dets)):
                        tid = int(dets.tracker_id[i])
                        classes_observees[tid][int(dets.class_id[i])] += 1
            fi += 1
    finally:
        cap.release()

    classe_finale = {
        tid: int(compteur.most_common(1)[0][0])
        for tid, compteur in classes_observees.items()
    }

    logger.info(
        "Tracking BoT-SORT termine : %d trackers sur %d frames analysees (ReID)",
        len(classe_finale),
        len(detections_trackees),
    )
    return detections_trackees, classe_finale
