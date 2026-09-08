"""Tests des metriques de continuite d'ID en espace image (benchmark)."""

from __future__ import annotations

import numpy as np
import supervision as sv

from pivot_ai.benchmark import (
    bornes_pixels,
    detailler_reprises,
    resume_benchmark,
    trajectoires_pixels,
)
from pivot_ai.metriques import composantes_connexes, detecter_reprises_bornes


def _dets(items: list[tuple[int, float, float]]) -> sv.Detections:
    """Detections d'une frame a partir de (tracker_id, cx, cy) : boites 20x40."""
    if not items:
        return sv.Detections.empty()
    xyxy = np.array(
        [[cx - 10, cy - 20, cx + 10, cy + 20] for _, cx, cy in items], dtype=np.float32
    )
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.ones(len(items), dtype=np.float32),
        class_id=np.zeros(len(items), dtype=int),
        tracker_id=np.array([tid for tid, _, _ in items], dtype=int),
    )


def _scenario_fragmente() -> dict[int, sv.Detections]:
    """Tracker 1 (frames 0-8) puis 2 (frames 11-19) = meme joueur fragmente ;
    tracker 3 loin, present tout le temps = joueur distinct."""
    dets: dict[int, sv.Detections] = {}
    for fi in range(0, 9):
        dets[fi] = _dets([(1, 100 + fi, 100), (3, 600, 400)])
    for fi in range(9, 11):
        dets[fi] = _dets([(3, 600, 400)])  # tracker 1 disparu (occlusion)
    for fi in range(11, 20):
        dets[fi] = _dets([(2, 108 + (fi - 11), 100), (3, 600, 400)])
    return dets


def test_trajectoires_et_bornes() -> None:
    """Extraction des trajectoires pixel + bornes."""
    traj = trajectoires_pixels(_scenario_fragmente())
    assert set(traj.keys()) == {1, 2, 3}
    assert traj[1][0] == (0, 100.0, 100.0)
    bornes = bornes_pixels(traj)
    assert bornes[1].frame_debut == 0 and bornes[1].frame_fin == 8
    assert bornes[2].frame_debut == 11


def test_reprise_pixels_sans_coupure() -> None:
    """Sans coupure : tracker 1 -> tracker 2 forme une reprise (meme joueur)."""
    bornes = bornes_pixels(trajectoires_pixels(_scenario_fragmente()))
    reprises = detecter_reprises_bornes(bornes, seuil_distance=120.0, seuil_frames=30)
    assert (1, 2) in reprises
    # 3 tracks, 2 joueurs reels (1&2 recolles, 3 seul)
    assert composantes_connexes(list(bornes.keys()), reprises) == 2


def test_reprise_pixels_ignoree_si_coupure() -> None:
    """Avec une coupure entre la fin de 1 et le debut de 2, pas de reprise."""
    bornes = bornes_pixels(trajectoires_pixels(_scenario_fragmente()))
    reprises = detecter_reprises_bornes(
        bornes, seuil_distance=120.0, seuil_frames=30, frames_coupure=(10,)
    )
    assert (1, 2) not in reprises
    # 1, 2 et 3 comptes separement (la coupure explique la perte d'ID)
    assert composantes_connexes(list(bornes.keys()), reprises) == 3


def test_resume_benchmark() -> None:
    """Le resume agrege les metriques image-space."""
    r = resume_benchmark(
        _scenario_fragmente(), fps=25.0, subsample=1,
        seuil_distance_px=120.0, seuil_frames=30, min_frames_fragment=5,
        tracker="bytetrack", clip="synthetique",
    )
    assert r.nb_tracks == 3
    assert r.nb_reprises_id == 1
    assert r.nb_joueurs_estimes == 2
    assert r.fragmentation == round(3 / 2, 2)
    # tracker 3 present 20 frames -> p90 eleve ; moyenne > 0
    assert r.longueur_moyenne_frames > 0
    assert r.longueur_moyenne_s > 0


def test_resume_benchmark_avec_coupure() -> None:
    """Avec coupure, la reprise 1->2 n'est plus comptee : 3 joueurs estimes."""
    r = resume_benchmark(
        _scenario_fragmente(), fps=25.0, subsample=1,
        seuil_distance_px=120.0, seuil_frames=30, frames_coupure=(10,),
    )
    assert r.nb_reprises_id == 0
    assert r.nb_joueurs_estimes == 3
    assert r.nb_coupures_plan == 1


def test_cause_detection_quand_trou_vide() -> None:
    """Trou sans aucune detection pres de la derniere position -> cause detection."""
    dets = _scenario_fragmente()  # frames 9-10 : rien pres de (108, 100)
    bornes = bornes_pixels(trajectoires_pixels(dets))
    reprises = detecter_reprises_bornes(bornes, seuil_distance=120.0, seuil_frames=30)
    details = detailler_reprises(dets, bornes, reprises, fps=25.0, seuil_px=60.0)
    assert len(details) == 1
    assert details[0].cause == "detection"
    assert details[0].gap_frames == 3  # frame_debut 11 - frame_fin 8


def test_cause_tracker_quand_detection_presente_pendant_trou() -> None:
    """Detection presente pres de la derniere position pendant le trou -> cause tracker."""
    dets: dict[int, sv.Detections] = {}
    for fi in range(0, 9):
        dets[fi] = _dets([(1, 100 + fi, 100)])
    dets[9] = _dets([(3, 109, 100)])   # meme joueur toujours vu, ID different
    dets[10] = _dets([(3, 109, 100)])
    for fi in range(11, 20):
        dets[fi] = _dets([(2, 109, 100)])
    bornes = bornes_pixels(trajectoires_pixels(dets))
    reprises = detecter_reprises_bornes(bornes, seuil_distance=120.0, seuil_frames=30)
    details = detailler_reprises(dets, bornes, reprises, fps=25.0, seuil_px=60.0)
    causes = {(d.tracker_a, d.tracker_b): d.cause for d in details}
    assert causes.get((1, 3)) == "tracker" or causes.get((1, 2)) == "tracker"


def test_resume_compte_les_causes() -> None:
    """Le resume ventile les reprises par cause et expose le detail date."""
    r = resume_benchmark(
        _scenario_fragmente(), fps=25.0, subsample=1,
        seuil_distance_px=120.0, seuil_frames=30,
    )
    assert r.nb_reprises_cause_detection + r.nb_reprises_cause_tracker == r.nb_reprises_id
    assert r.nb_reprises_cause_detection == 1
    assert len(r.reprises_detail) == 1
    assert r.reprises_detail[0]["cause"] == "detection"
