"""Fixtures et helpers partages pour la suite de tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import supervision as sv


def generer_video_factice(
    chemin: Path,
    nb_frames: int = 50,
    fps: float = 25.0,
    largeur: int = 320,
    hauteur: int = 240,
) -> None:
    """Genere un petit MP4 valide pour tester les pipelines bout-en-bout."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(chemin), fourcc, fps, (largeur, hauteur))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter ne peut pas creer la video de test")
    for i in range(nb_frames):
        frame = np.full((hauteur, largeur, 3), (i * 3) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def detections_fabriquees(
    bboxes: list[tuple[float, float, float, float]],
    class_ids: list[int],
    confiances: list[float] | None = None,
) -> sv.Detections:
    """Cree un sv.Detections sans tracker_id (pour entrer dans le tracker)."""
    n = len(bboxes)
    assert n == len(class_ids)
    if n == 0:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(bboxes, dtype=np.float32),
        confidence=np.array(confiances or [1.0] * n, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
    )
