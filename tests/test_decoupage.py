"""Tests unitaires pour pivot_ai.decoupage."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest
import supervision as sv

from pivot_ai.decoupage import (
    Action,
    _compter_joueurs_par_frame,
    decouper_clips_video,
    detecter_actions,
)
from tests.conftest import generer_video_factice


def _detection_avec_tracker_ids(tracker_ids: list[int], class_ids: list[int]) -> sv.Detections:
    """Fabrique un sv.Detections synthetique avec tracker_id et class_id."""
    n = len(tracker_ids)
    assert n == len(class_ids)
    if n == 0:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.zeros((n, 4), dtype=np.float32),
        confidence=np.ones(n, dtype=np.float32),
        class_id=np.array(class_ids, dtype=int),
        tracker_id=np.array(tracker_ids, dtype=int),
    )


def _construire_sequence(profils: list[tuple[int, int]]) -> dict[int, sv.Detections]:
    """Construit une sequence frame -> Detections a partir de (frame_idx, nb_joueurs).

    On reutilise toujours les memes tracker_id 1..N (tous classe 0 = joueur).
    """
    detections: dict[int, sv.Detections] = {}
    for fi, n in profils:
        tids = list(range(1, n + 1))
        cids = [0] * n
        detections[fi] = _detection_avec_tracker_ids(tids, cids)
    return detections


# ---------------------------------------------------------------------------
# Compteur de joueurs par frame
# ---------------------------------------------------------------------------


def test_compter_joueurs_ignore_classes_non_joueur() -> None:
    """Seuls les trackers dont la classe stabilisee == id_classe_joueur sont comptes."""
    dets = _detection_avec_tracker_ids(
        tracker_ids=[1, 2, 3, 4],
        class_ids=[0, 0, 1, 2],  # mais classe stabilisee != cid forcement
    )
    classe_finale = {1: 0, 2: 0, 3: 1, 4: 2}  # 2 joueurs, 1 gk, 1 ref
    res = _compter_joueurs_par_frame({0: dets}, classe_finale, id_classe_joueur=0)
    assert res == {0: 2}


def test_compter_joueurs_frame_vide() -> None:
    """Frame sans detection = 0 joueurs."""
    res = _compter_joueurs_par_frame({5: sv.Detections.empty()}, {}, id_classe_joueur=0)
    assert res == {5: 0}


# ---------------------------------------------------------------------------
# Detection d'actions
# ---------------------------------------------------------------------------


def test_detecter_actions_sequence_propre() -> None:
    """Sequence : 50f vide, 100f a 7 joueurs, 50f vide -> 1 action."""
    fps = 25.0
    profils: list[tuple[int, int]] = []
    profils += [(i, 0) for i in range(50)]
    profils += [(i, 7) for i in range(50, 150)]  # action 4s
    profils += [(i, 0) for i in range(150, 200)]

    detections = _construire_sequence(profils)
    classe_finale = {tid: 0 for tid in range(1, 8)}

    actions = detecter_actions(
        detections,
        classe_finale,
        id_classe_joueur=0,
        fps=fps,
        seuil_debut_joueurs=6,
        seuil_fin_joueurs=4,
        duree_min_action_s=3.0,
        duree_max_pause_s=3.0,
    )
    assert len(actions) == 1
    a = actions[0]
    assert a.frame_debut == 50
    # Derniere frame avec >= seuil_fin = 149
    assert a.frame_fin == 149
    assert math.isclose(a.duree_s, (149 - 50) / fps, abs_tol=1e-6)
    assert a.duree_s >= 3.0
    assert math.isclose(a.nb_joueurs_moyen, 7.0, abs_tol=1e-6)
    assert a.equipe_en_attaque is None


def test_detecter_actions_filtre_duree_min() -> None:
    """Action courte (1.5s) ignoree si duree_min_action_s=3.0."""
    fps = 25.0
    profils: list[tuple[int, int]] = []
    profils += [(i, 0) for i in range(20)]
    profils += [(i, 7) for i in range(20, 40)]  # ~0.8s, trop court
    profils += [(i, 0) for i in range(40, 200)]

    detections = _construire_sequence(profils)
    classe_finale = {tid: 0 for tid in range(1, 8)}

    actions = detecter_actions(
        detections,
        classe_finale,
        id_classe_joueur=0,
        fps=fps,
        duree_min_action_s=3.0,
    )
    assert actions == []


def test_detecter_actions_pause_breve_dans_action() -> None:
    """Une pause < duree_max_pause_s ne casse pas l'action."""
    fps = 25.0
    profils: list[tuple[int, int]] = []
    profils += [(i, 0) for i in range(25)]
    profils += [(i, 7) for i in range(25, 100)]  # 3s d'action
    profils += [(i, 2) for i in range(100, 120)]  # pause ~0.8s (< 3s)
    profils += [(i, 7) for i in range(120, 200)]  # reprise 3.2s
    profils += [(i, 0) for i in range(200, 250)]

    detections = _construire_sequence(profils)
    classe_finale = {tid: 0 for tid in range(1, 8)}

    actions = detecter_actions(
        detections,
        classe_finale,
        id_classe_joueur=0,
        fps=fps,
        duree_min_action_s=3.0,
        duree_max_pause_s=3.0,
    )
    # Une seule action couvrant l'ensemble
    assert len(actions) == 1
    assert actions[0].frame_debut == 25
    assert actions[0].frame_fin == 199


def test_detecter_actions_deux_actions_distinctes() -> None:
    """Deux pics de 4s separes par une longue pause > 3s : 2 actions."""
    fps = 25.0
    profils: list[tuple[int, int]] = []
    profils += [(i, 0) for i in range(20)]
    profils += [(i, 7) for i in range(20, 120)]  # action 1 : 4s
    profils += [(i, 0) for i in range(120, 240)]  # pause longue 4.8s
    profils += [(i, 7) for i in range(240, 360)]  # action 2 : 4.8s
    profils += [(i, 0) for i in range(360, 400)]

    detections = _construire_sequence(profils)
    classe_finale = {tid: 0 for tid in range(1, 8)}

    actions = detecter_actions(
        detections,
        classe_finale,
        id_classe_joueur=0,
        fps=fps,
        duree_min_action_s=3.0,
        duree_max_pause_s=3.0,
    )
    assert len(actions) == 2
    assert actions[0].frame_debut == 20
    assert actions[1].frame_debut == 240
    # Triees par frame_debut
    assert actions[0].frame_debut < actions[1].frame_debut


def test_detecter_actions_action_jusqu_a_la_fin() -> None:
    """Une action qui ne se cloture pas avant la fin du clip est conservee."""
    fps = 25.0
    profils: list[tuple[int, int]] = []
    profils += [(i, 0) for i in range(25)]
    profils += [(i, 7) for i in range(25, 200)]  # 7s, jamais cloturee
    detections = _construire_sequence(profils)
    classe_finale = {tid: 0 for tid in range(1, 8)}

    actions = detecter_actions(
        detections, classe_finale, id_classe_joueur=0, fps=fps,
        duree_min_action_s=3.0,
    )
    assert len(actions) == 1
    assert actions[0].frame_debut == 25
    assert actions[0].frame_fin == 199


def test_detecter_actions_vide() -> None:
    """Aucune detection : aucune action."""
    actions = detecter_actions({}, {}, id_classe_joueur=0, fps=25.0)
    assert actions == []


def test_detecter_actions_fps_invalide() -> None:
    """fps <= 0 doit lever ValueError."""
    with pytest.raises(ValueError, match="fps"):
        detecter_actions({}, {}, id_classe_joueur=0, fps=0.0)


def test_detecter_actions_seuils_incoherents() -> None:
    """seuil_fin > seuil_debut doit lever ValueError."""
    with pytest.raises(ValueError, match="seuil"):
        detecter_actions(
            {}, {}, id_classe_joueur=0, fps=25.0,
            seuil_debut_joueurs=4, seuil_fin_joueurs=6,
        )


# ---------------------------------------------------------------------------
# Decoupage clips video (necessite ffmpeg + cv2 pour generer la source)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg non installe")
def test_decouper_clips_video_bout_en_bout(tmp_path: Path) -> None:
    """Decoupage reel via ffmpeg : verifie creation des fichiers et bornes nommage."""
    source = tmp_path / "source.mp4"
    generer_video_factice(source, nb_frames=100, fps=25.0, largeur=160, hauteur=120)  # 4s

    actions = [
        Action(frame_debut=0, frame_fin=25, duree_s=1.0,
               equipe_en_attaque=None, nb_joueurs_moyen=6.0),
        Action(frame_debut=50, frame_fin=75, duree_s=1.0,
               equipe_en_attaque=None, nb_joueurs_moyen=7.0),
    ]
    dossier = tmp_path / "clips"
    chemins = decouper_clips_video(source, actions, dossier, prefixe="test")

    assert len(chemins) == 2
    assert all(c.exists() for c in chemins)
    assert chemins[0].name == "test_000_1.0s.mp4"
    assert chemins[1].name == "test_001_1.0s.mp4"


def test_decouper_clips_video_source_inexistante(tmp_path: Path) -> None:
    """Source absente : FileNotFoundError."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg non installe")
    with pytest.raises(FileNotFoundError):
        decouper_clips_video(
            tmp_path / "absente.mp4",
            [Action(0, 25, 1.0, None, 6.0)],
            tmp_path / "out",
        )


def test_decouper_clips_video_liste_vide(tmp_path: Path) -> None:
    """Aucune action : aucun clip mais dossier cree."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg non installe")
    source = tmp_path / "src.mp4"
    generer_video_factice(source, nb_frames=50, fps=25.0, largeur=160, hauteur=120)
    dossier = tmp_path / "out"
    chemins = decouper_clips_video(source, [], dossier)
    assert chemins == []
    assert dossier.exists()
