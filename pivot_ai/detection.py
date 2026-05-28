"""Detection d'objets via YOLO local (GPU ou CPU).

Remplace la dependance Roboflow Cloud du pipeline d'origine. L'interface
retourne des `supervision.Detections` pour compatibilite avec le tracking.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from pivot_ai.config import ModeleConfig

logger = logging.getLogger(__name__)


class DetecteurLocal:
    """Wrapper autour d'Ultralytics YOLO pour inference locale.

    Usage:
        detecteur = DetecteurLocal(ModeleConfig())
        detections = detecteur.detecter(frame)
    """

    def __init__(self, config: ModeleConfig | None = None) -> None:
        from ultralytics import YOLO  # import lazy : evite de charger torch si inutile

        self.config = config or ModeleConfig()
        chemin = Path(self.config.chemin_modele)
        logger.info("Chargement modele YOLO depuis %s (device=%s)", chemin, self.config.device)
        self.model = YOLO(str(chemin))
        # Warmup : premiere inference est lente (compilation CUDA kernels)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, verbose=False, device=self.config.device)

    def detecter(self, frame: np.ndarray) -> sv.Detections:
        """Detecte les objets dans une frame BGR.

        Args:
            frame: image BGR shape (H, W, 3)

        Returns:
            sv.Detections avec xyxy, confidence, class_id
        """
        results = self.model.predict(
            frame,
            conf=self.config.confiance_min,
            iou=self.config.iou_min,
            classes=list(self.config.classes_a_garder),
            verbose=False,
            device=self.config.device,
        )
        if not results:
            return sv.Detections.empty()

        # supervision sait parser directement les resultats Ultralytics
        return sv.Detections.from_ultralytics(results[0])

    def detecter_batch(self, frames: list[np.ndarray]) -> list[sv.Detections]:
        """Detecte sur un batch de frames (plus rapide qu'en boucle)."""
        results = self.model.predict(
            frames,
            conf=self.config.confiance_min,
            iou=self.config.iou_min,
            classes=list(self.config.classes_a_garder),
            verbose=False,
            device=self.config.device,
        )
        return [sv.Detections.from_ultralytics(r) for r in results]


def detecter_video(
    chemin_video: str | Path,
    detecteur: DetecteurLocal,
    subsample: int = 1,
    max_frames: int | None = None,
    batch_size: int = 8,
) -> dict[int, sv.Detections]:
    """Detecte sur toutes les frames d'une video (echantillonnees).

    Iteration sequentielle (`cap.read()` en continu) avec skip des frames
    non echantillonnees. Bannit `cap.set(CAP_PROP_POS_FRAMES, ...)` qui
    re-decode depuis le keyframe precedent et explose les temps sur MP4.
    Batch les frames retenues avant `detecter_batch` pour saturer le GPU.

    Args:
        chemin_video: chemin vers le MP4
        detecteur: instance avec une methode `detecter_batch(list[ndarray])`
        subsample: 1 frame sur N
        max_frames: limite optionnelle (sur la frame index source)
        batch_size: taille de batch GPU (8 = bon defaut T4 / YOLOv8m / 640px)

    Returns:
        dict frame_index -> Detections
    """
    if subsample < 1:
        raise ValueError(f"subsample doit etre >= 1, recu {subsample}")
    if batch_size < 1:
        raise ValueError(f"batch_size doit etre >= 1, recu {batch_size}")

    cap = cv2.VideoCapture(str(chemin_video))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la video : {chemin_video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total = min(total, max_frames)

    detections_par_frame: dict[int, sv.Detections] = {}
    batch_frames: list[np.ndarray] = []
    batch_indices: list[int] = []

    def flush_batch() -> None:
        if not batch_frames:
            return
        resultats = detecteur.detecter_batch(batch_frames)
        for idx, dets in zip(batch_indices, resultats, strict=True):
            detections_par_frame[idx] = dets
        batch_frames.clear()
        batch_indices.clear()

    try:
        fi = 0
        while fi < total:
            ret, frame = cap.read()
            if not ret:
                break
            if fi % subsample == 0:
                batch_frames.append(frame)
                batch_indices.append(fi)
                if len(batch_frames) >= batch_size:
                    flush_batch()
            fi += 1
        flush_batch()
    finally:
        cap.release()

    logger.info(
        "Detection video terminee : %d frames analysees sur %d total "
        "(subsample=%d, batch_size=%d)",
        len(detections_par_frame),
        total,
        subsample,
        batch_size,
    )
    return detections_par_frame
