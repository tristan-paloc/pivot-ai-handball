"""Tests unitaires pour pivot_ai.pipeline."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import supervision as sv

from pivot_ai.config import CLASSES_HANDBALL, TerrainConfig
from pivot_ai.homographie import calibrer_homographie
from pivot_ai.pipeline import (
    COULEUR_ADV,
    COULEUR_GK,
    COULEUR_MHB,
    COULEUR_REF,
    ResultatPipeline,
    couleur_tracker,
    generer_video_sbs,
    interpoler_positions,
    projeter_detections_en_terrain,
    traiter_match_complet,
)
from pivot_ai.stats import PositionJoueur
from tests.conftest import generer_video_factice

# Homographie identite-like : pixels = metres (pratique pour tests)
_CORRESPONDANCES_TEST = {
    "A": {"pixel": (0, 0), "terrain_m": (0.0, 0.0)},
    "B": {"pixel": (40, 0), "terrain_m": (40.0, 0.0)},
    "C": {"pixel": (40, 20), "terrain_m": (40.0, 20.0)},
    "D": {"pixel": (0, 20), "terrain_m": (0.0, 20.0)},
}


# ---------------------------------------------------------------------------
# projeter_detections_en_terrain
# ---------------------------------------------------------------------------


def _detection_avec_ids(
    bboxes: list[tuple[float, float, float, float]],
    tracker_ids: list[int],
    class_ids: list[int],
) -> sv.Detections:
    n = len(bboxes)
    return sv.Detections(
        xyxy=np.array(bboxes, dtype=np.float32),
        confidence=np.ones(n, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
        tracker_id=np.array(tracker_ids, dtype=int),
    )


def test_projeter_filtre_classes_non_joueur() -> None:
    """Seuls les trackers classe 'players' sont conserves."""
    homographie = calibrer_homographie(_CORRESPONDANCES_TEST)
    # bbox centree en pixel (20, 10) -> centre-bas (20, 10) -> terrain (20, 10)
    dets = _detection_avec_ids(
        bboxes=[(15, 0, 25, 10), (15, 0, 25, 10)],
        tracker_ids=[1, 2],
        class_ids=[0, 1],
    )
    classe_finale = {1: 0, 2: 1}  # 1 joueur, 1 GK
    pos = projeter_detections_en_terrain(
        {0: dets}, classe_finale, homographie,
        id_classe_joueur=0, filtrer_zone=False,
    )
    assert set(pos.keys()) == {1}
    assert len(pos[1]) == 1
    assert math.isclose(pos[1][0].x_m, 20.0, abs_tol=1e-3)
    assert math.isclose(pos[1][0].y_m, 10.0, abs_tol=1e-3)
    assert pos[1][0].interpole is False


def test_projeter_filtre_zone_vire_les_bancs() -> None:
    """Avec filtrer_zone=True, les positions hors marges sont rejetees."""
    homographie = calibrer_homographie(_CORRESPONDANCES_TEST)
    # Le centre-bas du second tracker tombe a y=1.0m, sous marge_y_m=2.0
    dets = _detection_avec_ids(
        bboxes=[(15, 0, 25, 10), (15, 0, 25, 1)],
        tracker_ids=[1, 2],
        class_ids=[0, 0],
    )
    classe_finale = {1: 0, 2: 0}
    pos = projeter_detections_en_terrain(
        {0: dets}, classe_finale, homographie,
        id_classe_joueur=0, filtrer_zone=True,
    )
    assert set(pos.keys()) == {1}


# ---------------------------------------------------------------------------
# interpoler_positions
# ---------------------------------------------------------------------------


def test_interpoler_gap_petit_remplit_lineairement() -> None:
    """Tracker present a frames 0 et 4 : interpolation lineaire a 1, 2, 3."""
    positions = {
        7: [
            PositionJoueur(frame_idx=0, x_m=0.0, y_m=0.0),
            PositionJoueur(frame_idx=4, x_m=4.0, y_m=8.0),
        ]
    }
    res = interpoler_positions(positions, max_gap_frames=10)
    assert len(res[7]) == 5
    # frame 1 : alpha=0.25 -> x=1, y=2
    assert math.isclose(res[7][1].x_m, 1.0, abs_tol=1e-9)
    assert math.isclose(res[7][1].y_m, 2.0, abs_tol=1e-9)
    assert res[7][1].interpole is True
    # frame 4 : non interpolee
    assert res[7][4].frame_idx == 4
    assert res[7][4].interpole is False


def test_interpoler_gap_grand_ne_remplit_pas() -> None:
    """Gap superieur a max_gap_frames : pas d'interpolation, segments preserves."""
    positions = {
        1: [
            PositionJoueur(frame_idx=0, x_m=0.0, y_m=0.0),
            PositionJoueur(frame_idx=100, x_m=10.0, y_m=10.0),
        ]
    }
    res = interpoler_positions(positions, max_gap_frames=10)
    assert len(res[1]) == 2
    assert all(not p.interpole for p in res[1])


def test_interpoler_pas_d_extrapolation_aux_bords() -> None:
    """Pas d'interpolation avant la 1ere ni apres la derniere position."""
    positions = {
        1: [
            PositionJoueur(frame_idx=5, x_m=5.0, y_m=5.0),
            PositionJoueur(frame_idx=8, x_m=8.0, y_m=8.0),
        ]
    }
    res = interpoler_positions(positions, max_gap_frames=10)
    assert res[1][0].frame_idx == 5
    assert res[1][-1].frame_idx == 8


# ---------------------------------------------------------------------------
# couleur_tracker
# ---------------------------------------------------------------------------


def test_couleur_tracker_par_role() -> None:
    """Couleurs distinctes pour MHB, ADV, GK, REF."""
    classe = {
        1: CLASSES_HANDBALL["players"],
        2: CLASSES_HANDBALL["players"],
        3: CLASSES_HANDBALL["goalkeeper"],
        4: CLASSES_HANDBALL["referees"],
    }
    equipe = {1: 0, 2: 1}  # tracker 1 = cluster 0 (MHB), tracker 2 = cluster 1 (ADV)
    assert couleur_tracker(1, classe, equipe, cluster_mhb=0) == COULEUR_MHB
    assert couleur_tracker(2, classe, equipe, cluster_mhb=0) == COULEUR_ADV
    assert couleur_tracker(3, classe, equipe, cluster_mhb=0) == COULEUR_GK
    assert couleur_tracker(4, classe, equipe, cluster_mhb=0) == COULEUR_REF


# ---------------------------------------------------------------------------
# generer_video_sbs
# ---------------------------------------------------------------------------


def test_generer_video_sbs_cree_fichier_avec_dimensions(tmp_path: Path) -> None:
    """La video SBS est ecrite avec largeur = broadcast_w + radar_w."""
    source = tmp_path / "source.mp4"
    generer_video_factice(source, nb_frames=10, fps=25.0, largeur=320, hauteur=240)

    classe_finale = {1: CLASSES_HANDBALL["players"]}
    detections_trackees = {
        0: _detection_avec_ids([(100, 50, 200, 200)], [1], [0]),
    }
    positions_interpolees = {
        1: [PositionJoueur(frame_idx=fi, x_m=15.0, y_m=10.0) for fi in range(10)],
    }
    chemin_sortie = tmp_path / "sbs.mp4"
    config = TerrainConfig()
    res = generer_video_sbs(
        source,
        detections_trackees,
        positions_interpolees,
        classe_finale,
        equipe_finale={1: 0},
        cluster_mhb=0,
        chemin_sortie=chemin_sortie,
        terrain_config=config,
    )

    assert res == chemin_sortie
    assert chemin_sortie.exists()
    # Verifier dimensions du fichier produit
    cap = cv2.VideoCapture(str(chemin_sortie))
    assert cap.isOpened()
    largeur = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hauteur = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert largeur == config.broadcast_w + config.radar_w
    assert hauteur == config.broadcast_h
    assert nb >= 1


# ---------------------------------------------------------------------------
# traiter_match_complet (avec detecteur stub)
# ---------------------------------------------------------------------------


class DetecteurStub:
    """Detecteur stub : retourne une bbox de joueur fixe sur chaque frame.

    Sert a tester le pipeline complet sans dependance YOLO/GPU.
    """

    def __init__(self, bbox: tuple[float, float, float, float] = (140, 80, 180, 200)) -> None:
        self.bbox = bbox
        self.nb_appels_batch = 0  # pour les tests perf
        self.tailles_batch_observees: list[int] = []

    def detecter(self, frame: np.ndarray) -> sv.Detections:
        return sv.Detections(
            xyxy=np.array([self.bbox], dtype=np.float32),
            confidence=np.array([0.9], dtype=np.float32),
            class_id=np.array([CLASSES_HANDBALL["players"]], dtype=int),
        )

    def detecter_batch(self, frames: list[np.ndarray]) -> list[sv.Detections]:
        self.nb_appels_batch += 1
        self.tailles_batch_observees.append(len(frames))
        return [self.detecter(f) for f in frames]


def test_traiter_match_complet_bout_en_bout(tmp_path: Path) -> None:
    """Pipeline complet avec detecteur stub : sortie stats CSV/Parquet et resultat coherent."""
    source = tmp_path / "match.mp4"
    generer_video_factice(source, nb_frames=40, fps=25.0, largeur=320, hauteur=240)

    # Homographie : pixels (0..320, 0..240) -> terrain (0..40, 0..20)
    correspondances = {
        "A": {"pixel": (0, 0), "terrain_m": (0.0, 0.0)},
        "B": {"pixel": (320, 0), "terrain_m": (40.0, 0.0)},
        "C": {"pixel": (320, 240), "terrain_m": (40.0, 20.0)},
        "D": {"pixel": (0, 240), "terrain_m": (0.0, 20.0)},
    }
    sortie = tmp_path / "out"

    res = traiter_match_complet(
        chemin_video=source,
        correspondances_homographie=correspondances,
        dossier_sortie=sortie,
        subsample=2,
        generer_video_radar=False,  # accelere le test
        decouper_actions=False,
        detecteur=DetecteurStub(),
    )

    assert isinstance(res, ResultatPipeline)
    assert res.chemin_video_source == source
    assert res.chemin_video_radar is None
    assert (sortie / "stats_joueurs.csv").exists()
    assert (sortie / "stats_joueurs.parquet").exists()
    assert res.metadonnees["subsample"] == 2
    assert res.metadonnees["nb_frames_total"] == 40
    assert res.metadonnees["nb_trackers_total"] >= 1


def test_traiter_match_complet_avec_video_sbs(tmp_path: Path) -> None:
    """Pipeline avec generation video SBS : fichier mp4 ecrit."""
    source = tmp_path / "match.mp4"
    generer_video_factice(source, nb_frames=10, fps=25.0, largeur=320, hauteur=240)

    correspondances = {
        "A": {"pixel": (0, 0), "terrain_m": (0.0, 0.0)},
        "B": {"pixel": (320, 0), "terrain_m": (40.0, 0.0)},
        "C": {"pixel": (320, 240), "terrain_m": (40.0, 20.0)},
        "D": {"pixel": (0, 240), "terrain_m": (0.0, 20.0)},
    }
    sortie = tmp_path / "out"

    res = traiter_match_complet(
        chemin_video=source,
        correspondances_homographie=correspondances,
        dossier_sortie=sortie,
        subsample=1,
        generer_video_radar=True,
        decouper_actions=False,
        detecteur=DetecteurStub(),
    )
    assert res.chemin_video_radar is not None
    assert res.chemin_video_radar.exists()


def test_traiter_match_complet_video_inexistante(tmp_path: Path) -> None:
    """Video source absente : FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        traiter_match_complet(
            chemin_video=tmp_path / "absent.mp4",
            correspondances_homographie=_CORRESPONDANCES_TEST,
            dossier_sortie=tmp_path / "out",
            detecteur=DetecteurStub(),
        )


def test_traiter_match_complet_utilise_detecter_batch(tmp_path: Path) -> None:
    """Le pipeline appelle detecter_batch (et non detecter frame-par-frame)."""
    source = tmp_path / "match.mp4"
    generer_video_factice(source, nb_frames=40, fps=25.0, largeur=320, hauteur=240)
    correspondances = {
        "A": {"pixel": (0, 0), "terrain_m": (0.0, 0.0)},
        "B": {"pixel": (320, 0), "terrain_m": (40.0, 0.0)},
        "C": {"pixel": (320, 240), "terrain_m": (40.0, 20.0)},
        "D": {"pixel": (0, 240), "terrain_m": (0.0, 20.0)},
    }
    stub = DetecteurStub()
    traiter_match_complet(
        chemin_video=source,
        correspondances_homographie=correspondances,
        dossier_sortie=tmp_path / "out",
        subsample=2,
        generer_video_radar=False,
        decouper_actions=False,
        detecteur=stub,
        batch_size=8,
    )
    # 40 frames / subsample 2 = 20 frames echantillonnees.
    # Avec batch_size=8 : 8 + 8 + 4 = 3 appels.
    assert stub.nb_appels_batch == 3
    assert stub.tailles_batch_observees == [8, 8, 4]
