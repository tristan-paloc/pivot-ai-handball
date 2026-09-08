"""Tests du runner de benchmark + video annotee (tracking injecte, sans GPU)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from pivot_ai.benchmark import (
    ecrire_comparatif,
    lancer_benchmark,
)
from pivot_ai.video_annotee import couleur_id, generer_video_annotee

from .conftest import generer_video_factice


def _dets(items: list[tuple[int, float, float]]) -> sv.Detections:
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


def _tracking_synthetique(nb_frames: int = 20) -> dict[int, sv.Detections]:
    """Deux joueurs distincts, l'un fragmente en 2 IDs (occlusion au milieu)."""
    dets: dict[int, sv.Detections] = {}
    for fi in range(nb_frames):
        items = [(3, 250.0, 180.0)]  # joueur stable
        if fi < 8:
            items.append((1, 60.0 + fi, 100.0))
        elif fi >= 11:
            items.append((2, 68.0 + (fi - 11), 100.0))
        dets[fi] = _dets(items)
    return dets


def test_couleur_id_stable_et_distincte() -> None:
    """La couleur d'un ID est stable et deux IDs voisins different."""
    assert couleur_id(5) == couleur_id(5)
    assert couleur_id(1) != couleur_id(2)


def test_generer_video_annotee(tmp_path: Path) -> None:
    """La video annotee est ecrite, lisible, a fps/subsample."""
    source = tmp_path / "src.mp4"
    generer_video_factice(source, nb_frames=20, fps=50.0, largeur=640, hauteur=360)
    sortie = tmp_path / "annotee.mp4"
    res = generer_video_annotee(
        source, _tracking_synthetique(20), sortie, subsample=2, frames_coupure=(10,)
    )
    assert res == sortie and sortie.exists()
    cap = cv2.VideoCapture(str(sortie))
    assert cap.isOpened()
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert nb >= 1


def test_lancer_benchmark_avec_tracking_injecte(tmp_path: Path) -> None:
    """Le runner compare 2 trackers sans GPU (tracking injecte) et ecrit les sorties."""
    source = tmp_path / "evt_test.mp4"
    generer_video_factice(source, nb_frames=20, fps=50.0, largeur=640, hauteur=360)
    sortie = tmp_path / "bench"

    appels: list[tuple[str, str]] = []

    def faux_tracking(chemin, tracker, subsample, fps):
        appels.append((Path(chemin).name, tracker))
        # botsort "recolle" (1 seul ID), bytetrack "fragmente" (2 IDs)
        if tracker == "botsort":
            d = {}
            for fi in range(20):
                items = [(3, 250.0, 180.0)]
                if fi < 8 or fi >= 11:
                    items.append((1, 60.0 + fi, 100.0))
                d[fi] = _dets(items)
            return d
        return _tracking_synthetique(20)

    runs = lancer_benchmark(
        clips=[source],
        trackers=["bytetrack", "botsort"],
        sortie=sortie,
        fonction_tracking=faux_tracking,
        subsample=2,
        seuil_distance_px=120.0,
        seuil_frames=30,
        generer_videos=True,
    )

    assert len(runs) == 2
    assert ("evt_test.mp4", "bytetrack") in appels
    assert (sortie / "comparatif.json").exists()
    assert (sortie / "comparatif.csv").exists()
    assert (sortie / "RAPPORT.md").exists()
    # chaque run a sa video annotee
    for r in runs:
        assert r.chemin_video_annotee is not None and r.chemin_video_annotee.exists()

    # bytetrack fragmente davantage que botsort sur ce scenario
    par_tracker = {r.resume.tracker: r.resume for r in runs}
    assert par_tracker["bytetrack"].nb_tracks >= par_tracker["botsort"].nb_tracks


def _tracking_cascade(nb_frames: int = 31) -> dict[int, sv.Detections]:
    """Un joueur fragmente en 3 IDs (avance regulierement) + un joueur distinct."""
    dets: dict[int, sv.Detections] = {}
    for fi in range(nb_frames):
        items = [(99, 600.0, 400.0)]
        x = 100.0 + fi * 3.0
        if fi <= 8:
            items.append((1, x, 100.0))
        elif 11 <= fi <= 19:
            items.append((2, x, 100.0))
        elif fi >= 22:
            items.append((3, x, 100.0))
        dets[fi] = _dets(items)
    return dets


def test_benchmark_variante_stitch(tmp_path: Path) -> None:
    """Le token bytetrack+stitch derive du bytetrack, reduit les tracks, porte ses merges."""
    source = tmp_path / "cascade.mp4"
    generer_video_factice(source, nb_frames=31, fps=25.0, largeur=640, hauteur=360)
    runs = lancer_benchmark(
        clips=[source], trackers=["bytetrack", "bytetrack+stitch"],
        sortie=tmp_path / "b",
        fonction_tracking=lambda c, t, s, f: _tracking_cascade(31),
        subsample=1, seuil_distance_px=120.0, seuil_frames=30,
        generer_videos=False, detecter_coupures=False,
    )
    par_tracker = {r.resume.tracker: r for r in runs}
    brut = par_tracker["bytetrack"].resume
    stitch = par_tracker["bytetrack+stitch"].resume
    # la cascade 1->2->3 est recollee : moins de tracks apres stitching
    assert stitch.nb_tracks < brut.nb_tracks
    assert stitch.nb_tracks == 2  # joueur recolle + joueur distinct
    # le run stitch porte son journal de merges + le fichier est ecrit
    assert par_tracker["bytetrack+stitch"].merges is not None
    assert (tmp_path / "b" / "merges_cascade__bytetrack+stitch.json").exists()
    assert (tmp_path / "b" / "merges_cascade__bytetrack+stitch.csv").exists()


def test_benchmark_sans_coupures(tmp_path: Path) -> None:
    """detecter_coupures=False : plan continu, aucune coupure n'est comptee."""
    source = tmp_path / "continu.mp4"
    generer_video_factice(source, nb_frames=20, fps=50.0, largeur=640, hauteur=360)
    runs = lancer_benchmark(
        clips=[source], trackers=["bytetrack"], sortie=tmp_path / "b",
        fonction_tracking=lambda c, t, s, f: _tracking_synthetique(20),
        subsample=2, seuil_distance_px=120.0, seuil_frames=30,
        generer_videos=False, detecter_coupures=False,
    )
    assert runs[0].resume.nb_coupures_plan == 0
    assert runs[0].resume.frames_coupure == []


def test_ecrire_comparatif_json_valide(tmp_path: Path) -> None:
    """comparatif.json est un JSON de la bonne forme."""
    source = tmp_path / "evt.mp4"
    generer_video_factice(source, nb_frames=12, fps=50.0, largeur=320, hauteur=240)
    runs = lancer_benchmark(
        clips=[source], trackers=["bytetrack"], sortie=tmp_path / "b",
        fonction_tracking=lambda c, t, s, f: _tracking_synthetique(12),
        generer_videos=False,
    )
    chemin_json, _ = ecrire_comparatif(runs, tmp_path / "b")
    data = json.loads(chemin_json.read_text(encoding="utf-8"))
    assert isinstance(data, list) and data and data[0]["tracker"] == "bytetrack"
