"""Tests du remap de classes modele -> CLASSES_HANDBALL (sans YOLO/GPU)."""

from __future__ import annotations

import numpy as np
import supervision as sv

from pivot_ai.config import (
    CLASSES_HANDBALL,
    ModeleConfig,
    _normaliser_nom_classe,
    construire_remap_classes,
)
from pivot_ai.detection import _appliquer_remap


def test_normaliser_nom_classe() -> None:
    """Normalisation : casse, pluriel simple, separateurs."""
    assert _normaliser_nom_classe("Players") == "player"
    assert _normaliser_nom_classe("referees") == "referee"
    assert _normaliser_nom_classe("goalkeeper") == "goalkeeper"
    assert _normaliser_nom_classe("ball") == "ball"
    assert _normaliser_nom_classe("Goal_Keeper") == "goal keeper"


def test_remap_ordre_roboflow_alphabetique() -> None:
    """Dataset Roboflow alphabetique (ball=0, gk=1, players=2, ref=3) -> canonique."""
    noms_modele = {0: "ball", 1: "goalkeeper", 2: "players", 3: "referees"}
    remap = construire_remap_classes(noms_modele)
    # CLASSES_HANDBALL : players=0, goalkeeper=1, referees=2, ball=3
    assert remap[0] == CLASSES_HANDBALL["ball"]        # 3
    assert remap[1] == CLASSES_HANDBALL["goalkeeper"]  # 1
    assert remap[2] == CLASSES_HANDBALL["players"]     # 0
    assert remap[3] == CLASSES_HANDBALL["referees"]    # 2


def test_remap_noms_singuliers() -> None:
    """Noms singuliers (player/referee) matchent aussi."""
    noms_modele = {0: "player", 1: "goalkeeper", 2: "referee", 3: "ball"}
    remap = construire_remap_classes(noms_modele)
    assert remap[0] == CLASSES_HANDBALL["players"]
    assert remap[2] == CLASSES_HANDBALL["referees"]


def test_remap_ignore_classes_inconnues() -> None:
    """Une classe hors CLASSES_HANDBALL (ex : 'crowd') est absente du remap."""
    noms_modele = {0: "players", 1: "crowd", 2: "ball"}
    remap = construire_remap_classes(noms_modele)
    assert 1 not in remap
    assert remap[0] == CLASSES_HANDBALL["players"]
    assert remap[2] == CLASSES_HANDBALL["ball"]


def test_remap_coco_person_ne_matche_pas() -> None:
    """Un modele COCO (person, car...) ne matche aucune classe handball."""
    noms_modele = {0: "person", 1: "bicycle", 2: "car"}
    remap = construire_remap_classes(noms_modele)
    assert remap == {}


def test_appliquer_remap_sur_detections() -> None:
    """_appliquer_remap remappe les class_id et ecarte les classes inconnues."""
    # modele : 0=ball, 1=goalkeeper, 2=players, 3=referees ; + une classe 9 inconnue
    dets = sv.Detections(
        xyxy=np.array(
            [[0, 0, 1, 1], [1, 1, 2, 2], [2, 2, 3, 3], [3, 3, 4, 4]],
            dtype=np.float32,
        ),
        confidence=np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32),
        class_id=np.array([2, 1, 0, 9], dtype=int),  # players, gk, ball, inconnu
    )
    remap = {0: 3, 1: 1, 2: 0, 3: 2}  # canonique
    out = _appliquer_remap(dets, remap)
    # La classe 9 (inconnue) est ecartee -> 3 detections restantes
    assert len(out) == 3
    # class_id remappes : 2->0 (players), 1->1 (gk), 0->3 (ball)
    assert sorted(out.class_id.tolist()) == [0, 1, 3]


def test_appliquer_remap_detections_vides() -> None:
    """Remap sur Detections vide ne casse pas."""
    out = _appliquer_remap(sv.Detections.empty(), {0: 0})
    assert len(out) == 0


def test_modele_config_pour_handball() -> None:
    """La factory pour_handball active le remap et garde toutes les classes."""
    cfg = ModeleConfig.pour_handball("handball.pt", device="cpu")
    assert cfg.chemin_modele == "handball.pt"
    assert cfg.classes_a_garder is None
    assert cfg.remapper_vers_classes_handball is True
    assert cfg.device == "cpu"


def test_modele_config_defaut_coco_inchange() -> None:
    """La config par defaut reste COCO person, sans remap (retrocompat)."""
    cfg = ModeleConfig()
    assert cfg.classes_a_garder == (0,)
    assert cfg.remapper_vers_classes_handball is False
