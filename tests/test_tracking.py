"""Tests du tracking : suppression des fantomes, stabilite d'ID, framerate effectif."""

from __future__ import annotations

import numpy as np
import supervision as sv

from pivot_ai.tracking import tracker_detections


def _det(x: float, y: float, class_id: int = 0) -> sv.Detections:
    """Une detection carree 20px centree approximativement en (x, y)."""
    return sv.Detections(
        xyxy=np.array([[x, y, x + 20, y + 40]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([class_id], dtype=int),
    )


def _sequence_objet_mobile(nb_frames: int, x0: float, dx: float) -> dict[int, sv.Detections]:
    """Un objet qui avance regulierement, present sur toutes les frames."""
    return {fi: _det(x0 + fi * dx, 100.0) for fi in range(nb_frames)}


def test_objet_persistant_recoit_un_id_stable() -> None:
    """Un objet present sur de nombreuses frames obtient un unique tracker_id."""
    seq = _sequence_objet_mobile(nb_frames=20, x0=100.0, dx=2.0)
    _dets, classe_finale = tracker_detections(seq, fps=30.0, subsample=1)
    # Un seul objet reel -> idealement 1 seul tracker stable
    assert len(classe_finale) == 1
    assert next(iter(classe_finale.values())) == 0


def test_fantome_une_frame_supprime() -> None:
    """Une detection isolee sur 1 frame ne cree pas de tracker (minimum_consecutive_frames)."""
    seq: dict[int, sv.Detections] = {}
    # Objet stable sur 15 frames (autour de x=100, avance lentement)
    for fi in range(15):
        seq[fi] = _det(100.0 + fi * 2, 100.0)
    # Fantome : sur la frame 7, on ajoute une 2e detection isolee tres loin
    x_obj = 100.0 + 7 * 2
    seq[7] = sv.Detections(
        xyxy=np.array(
            [[x_obj, 100.0, x_obj + 20, 140.0], [500.0, 300.0, 520.0, 340.0]],
            dtype=np.float32,
        ),
        confidence=np.array([0.9, 0.9], dtype=np.float32),
        class_id=np.array([0, 0], dtype=int),
    )
    _dets, classe_finale = tracker_detections(
        seq, fps=30.0, subsample=1, minimum_consecutive_frames=3
    )
    # Le fantome ne doit pas generer de tracker durable : 1 seul tracker stable
    assert len(classe_finale) == 1


def test_framerate_effectif_corrige_du_subsample() -> None:
    """Le framerate transmis a ByteTrack tient compte du subsample.

    On verifie indirectement : avec subsample=3 et fps=30, le tracker doit
    fonctionner (framerate effectif=10) et tracker l'objet sans exploser.
    """
    seq = _sequence_objet_mobile(nb_frames=15, x0=100.0, dx=3.0)
    _dets, classe_finale = tracker_detections(seq, fps=30.0, subsample=3)
    # L'objet reste unique malgre le subsample
    assert len(classe_finale) == 1


def test_minimum_consecutive_frames_desactivable() -> None:
    """Avec minimum_consecutive_frames=1, comportement historique (pas de delai)."""
    seq = _sequence_objet_mobile(nb_frames=5, x0=100.0, dx=1.0)
    dets_trackees, _classe = tracker_detections(
        seq, fps=30.0, subsample=1, minimum_consecutive_frames=1
    )
    # Des la premiere frame un tracker_id est attribue
    premiere = dets_trackees[0]
    assert premiere.tracker_id is not None
    assert len(premiere) == 1


def test_deux_objets_deux_ids() -> None:
    """Deux objets distincts et persistants -> deux trackers."""
    seq: dict[int, sv.Detections] = {}
    for fi in range(15):
        seq[fi] = sv.Detections(
            xyxy=np.array(
                [[100 + fi * 2, 100, 120 + fi * 2, 140],
                 [400 - fi * 2, 100, 420 - fi * 2, 140]],
                dtype=np.float32,
            ),
            confidence=np.array([0.9, 0.9], dtype=np.float32),
            class_id=np.array([0, 0], dtype=int),
        )
    _dets, classe_finale = tracker_detections(seq, fps=30.0, subsample=1)
    assert len(classe_finale) == 2
