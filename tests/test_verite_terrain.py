"""Tests de la verite terrain : suivi de joueurs annotes + cause des pertes d'ID."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import supervision as sv

from pivot_ai.verite_terrain import (
    PointGT,
    _id_au_point,
    charger_verite_terrain,
    evaluer_joueur,
    evaluer_verite_terrain,
)


def _det_box(tid: int, cx: float, cy: float) -> tuple[list[float], int]:
    return [cx - 12, cy - 20, cx + 12, cy + 20], tid


def _frame_dets(boxes: list[tuple[list[float], int]]) -> sv.Detections:
    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array([b for b, _ in boxes], dtype=np.float32),
        confidence=np.ones(len(boxes), dtype=np.float32),
        class_id=np.zeros(len(boxes), dtype=int),
        tracker_id=np.array([t for _, t in boxes], dtype=int),
    )


def test_id_au_point() -> None:
    """Point dans une boite -> son ID ; sinon plus proche < seuil ; sinon None."""
    dets = _frame_dets([_det_box(7, 100, 100)])
    assert _id_au_point(dets, 100, 100, seuil_px=30) == 7   # dans la boite
    assert _id_au_point(dets, 100, 160, seuil_px=30) is None  # trop loin
    assert _id_au_point(dets, 100, 160, seuil_px=80) == 7   # proche < seuil
    assert _id_au_point(dets, 500, 500, seuil_px=30) is None


def test_joueur_parfaitement_suivi() -> None:
    """Meme ID sur tous les points -> taux 1.0, 0 switch, continuite OK."""
    dets = {f: _frame_dets([_det_box(1, 100, 100)]) for f in (0, 10, 20, 30)}
    points = [PointGT(f, 100, 100) for f in (0, 10, 20, 30)]
    e = evaluer_joueur(points, dets, seuil_px=40)
    assert e.id_dominant == 1
    assert e.taux_id_dominant == 1.0
    assert e.nb_switches == 0
    assert e.continuite_ok is True


def test_switch_cause_tracker() -> None:
    """ID change alors que la detection est continue -> cause tracker."""
    dets = {
        0: _frame_dets([_det_box(1, 100, 100)]),
        10: _frame_dets([_det_box(1, 100, 100)]),
        20: _frame_dets([_det_box(1, 100, 100)]),
        30: _frame_dets([_det_box(2, 100, 100)]),  # meme joueur, ID a saute
        40: _frame_dets([_det_box(2, 100, 100)]),
    }
    points = [PointGT(f, 100, 100) for f in (0, 10, 20, 30, 40)]
    e = evaluer_joueur(points, dets, seuil_px=40)
    assert e.nb_switches == 1
    assert e.nb_switches_cause_tracker == 1
    assert e.nb_switches_cause_detection == 0


def test_switch_cause_detection() -> None:
    """Un trou de detection precede le changement d'ID -> cause detection."""
    dets = {
        0: _frame_dets([_det_box(1, 100, 100)]),
        10: _frame_dets([_det_box(1, 100, 100)]),
        20: _frame_dets([]),                        # detection absente
        30: _frame_dets([_det_box(2, 100, 100)]),
    }
    points = [PointGT(f, 100, 100) for f in (0, 10, 20, 30)]
    e = evaluer_joueur(points, dets, seuil_px=40)
    assert e.nb_matches == 3
    assert e.duree_suivie_pct == 0.75
    assert e.nb_switches == 1
    assert e.nb_switches_cause_detection == 1
    assert e.nb_switches_cause_tracker == 0


def test_switch_a_une_coupure_non_compte() -> None:
    """Un changement d'ID a un changement de plan n'est pas un echec de tracking."""
    dets = {
        0: _frame_dets([_det_box(1, 100, 100)]),
        10: _frame_dets([_det_box(1, 100, 100)]),
        20: _frame_dets([_det_box(2, 100, 100)]),
        30: _frame_dets([_det_box(2, 100, 100)]),
    }
    points = [PointGT(f, 100, 100) for f in (0, 10, 20, 30)]
    e = evaluer_joueur(points, dets, seuil_px=40, frames_coupure=(15,))
    assert e.nb_switches == 0  # la coupure explique le changement


def test_charger_et_evaluer_verite_terrain(tmp_path: Path) -> None:
    """Chargement JSON + agregat (taux continuite moyen, part cause tracker)."""
    chemin = tmp_path / "vt.json"
    chemin.write_text(json.dumps({
        "clip": "evt_test.mp4",
        "joueurs": [
            {"nom": "a", "points": [{"frame": 0, "x": 100, "y": 100},
                                    {"frame": 10, "x": 100, "y": 100}]},
        ],
    }), encoding="utf-8")
    vt = charger_verite_terrain(chemin)
    assert vt.clip == "evt_test.mp4"
    assert vt.joueurs[0].nom == "a"

    dets = {0: _frame_dets([_det_box(1, 100, 100)]),
            10: _frame_dets([_det_box(1, 100, 100)])}
    agg = evaluer_verite_terrain(vt, dets, seuil_px=40)
    assert agg.clip == "evt_test.mp4"
    assert agg.taux_continuite_moyen == 1.0
    assert agg.part_cause_tracker == 0.0
