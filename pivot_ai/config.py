"""Configuration centrale : constantes, schemas, parametres terrain."""

from __future__ import annotations

import logging
from dataclasses import dataclass

# ----------------------------------------------------------------------------
# Classes du modele de detection handball
# ----------------------------------------------------------------------------

CLASSES_HANDBALL: dict[str, int] = {
    "players": 0,
    "goalkeeper": 1,
    "referees": 2,
    "ball": 3,
}

CLASSES_INVERSE: dict[int, str] = {v: k for k, v in CLASSES_HANDBALL.items()}


# ----------------------------------------------------------------------------
# Terrain handball : 40m x 20m
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class TerrainConfig:
    """Configuration geometrique du terrain et du rendu radar."""

    # Dimensions reelles (metres)
    longueur_m: float = 40.0
    largeur_m: float = 20.0

    # Rendu radar (pixels)
    radar_w: int = 960
    radar_h: int = 540
    broadcast_w: int = 960
    broadcast_h: int = 540
    px_per_m: int = 22

    # Filtre asymetrique (vire bancs/staff sur les bords)
    marge_x_m: float = 0.0
    marge_y_m: float = 2.0

    @property
    def terrain_px_w(self) -> int:
        return int(self.longueur_m * self.px_per_m)

    @property
    def terrain_px_h(self) -> int:
        return int(self.largeur_m * self.px_per_m)

    @property
    def origin_x(self) -> int:
        return (self.radar_w - self.terrain_px_w) // 2

    @property
    def origin_y(self) -> int:
        return (self.radar_h - self.terrain_px_h) // 2

    def m2px(self, x_m: float, y_m: float) -> tuple[int, int]:
        """Convertit des coordonnees terrain (metres) en pixels radar."""
        x_px = self.origin_x + int(round(x_m * self.px_per_m))
        y_px = self.origin_y + int(round((self.largeur_m - y_m) * self.px_per_m))
        return x_px, y_px

    def est_dans_zone(self, point_m: tuple[float, float]) -> bool:
        """Verifie qu'un point est dans la zone exploitable du terrain."""
        x, y = point_m
        return (
            self.marge_x_m <= x <= self.longueur_m - self.marge_x_m
            and self.marge_y_m <= y <= self.largeur_m - self.marge_y_m
        )


# ----------------------------------------------------------------------------
# Points caracteristiques du terrain (pour homographie)
# ----------------------------------------------------------------------------
# Voir docs/terrain_handball.png pour le schema visuel

POINTS_TERRAIN_M: dict[str, tuple[float, float]] = {
    # Coins
    "A": (0.0, 0.0),
    "B": (40.0, 0.0),
    "C": (40.0, 20.0),
    "D": (0.0, 20.0),
    # Ligne mediane
    "E": (20.0, 0.0),
    "F": (20.0, 20.0),
    # Intersections lignes 6m avec lignes de touche
    "G": (0.0, 4.0),
    "H": (0.0, 16.0),
    "I": (40.0, 4.0),
    "J": (40.0, 16.0),
    # Sommets des demi-cercles 6m (sur la ligne mediane Y=10)
    "K": (6.0, 10.0),
    "L": (34.0, 10.0),
    # Poteaux de but
    "M_prime": (0.0, 8.5),
    "N_prime": (0.0, 11.5),
    "M": (40.0, 8.5),
    "N": (40.0, 11.5),
    # Sommets des arcs 9m
    "O_prime": (9.0, 10.0),
    "O": (31.0, 10.0),
}


# ----------------------------------------------------------------------------
# Modele YOLO local
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class ModeleConfig:
    """Configuration du modele de detection.

    Deux usages :
    - COCO generique (defaut) : `classes_a_garder=(0,)` garde la classe person,
      pas de remap (les noms COCO ne matchent pas CLASSES_HANDBALL).
    - Modele fine-tune handball : utiliser `ModeleConfig.pour_handball(chemin)`.
      Les class_id du modele sont remappes vers CLASSES_HANDBALL par nom, ce qui
      rend le pipeline robuste a l'ordre des classes du dataset d'entrainement.
    """

    chemin_modele: str = "yolov8m.pt"
    confiance_min: float = 0.35
    iou_min: float = 0.5
    # None = garder toutes les classes detectees par le modele.
    classes_a_garder: tuple[int, ...] | None = (0,)  # 0 = person en COCO
    device: str = "cuda"  # "cuda", "cpu", ou "mps" sur Mac
    # Si True, DetecteurLocal remappe les class_id du modele vers CLASSES_HANDBALL
    # via les noms de classes (model.names). Les classes inconnues sont ecartees.
    remapper_vers_classes_handball: bool = False

    @classmethod
    def pour_handball(
        cls,
        chemin_modele: str,
        confiance_min: float = 0.35,
        iou_min: float = 0.5,
        device: str = "cuda",
    ) -> ModeleConfig:
        """Config pour un modele fine-tune handball (4 classes players/gk/ref/ball).

        Garde toutes les classes du modele et active le remap par nom vers
        CLASSES_HANDBALL.

        Args:
            chemin_modele: chemin vers le .pt fine-tune handball
            confiance_min: seuil de confiance
            iou_min: seuil IoU NMS
            device: "cuda", "cpu" ou "mps"

        Returns:
            ModeleConfig prete pour l'inference handball.
        """
        return cls(
            chemin_modele=chemin_modele,
            confiance_min=confiance_min,
            iou_min=iou_min,
            classes_a_garder=None,
            device=device,
            remapper_vers_classes_handball=True,
        )


def _normaliser_nom_classe(nom: str) -> str:
    """Normalise un nom de classe pour matching robuste (casse, pluriel, espaces)."""
    n = nom.strip().lower().replace("_", " ").replace("-", " ").strip()
    # Singularise un eventuel pluriel simple (players -> player, referees -> referee).
    if n.endswith("s") and len(n) > 1:
        n = n[:-1]
    return n


def construire_remap_classes(
    noms_modele: dict[int, str],
    classes_cibles: dict[str, int] | None = None,
) -> dict[int, int]:
    """Construit le mapping class_id modele -> class_id canonique par nom.

    Permet d'utiliser un modele dont l'ordre des classes differe de
    CLASSES_HANDBALL (ex : export Roboflow alphabetique ball/goalkeeper/
    players/referees) sans casser le code aval qui raisonne en ids canoniques.

    Args:
        noms_modele: dict class_id -> nom, typiquement `model.names` d'Ultralytics
        classes_cibles: mapping nom -> id canonique (defaut : CLASSES_HANDBALL)

    Returns:
        dict class_id_modele -> class_id_canonique, uniquement pour les classes
        dont le nom matche une cible. Les classes non reconnues sont absentes.
    """
    cibles = classes_cibles if classes_cibles is not None else CLASSES_HANDBALL
    cibles_normalisees = {_normaliser_nom_classe(nom): idx for nom, idx in cibles.items()}

    remap: dict[int, int] = {}
    for cid_modele, nom in noms_modele.items():
        cle = _normaliser_nom_classe(nom)
        if cle in cibles_normalisees:
            remap[int(cid_modele)] = cibles_normalisees[cle]
    return remap


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------

def configurer_logging(niveau: int = logging.INFO) -> None:
    """Configure le logging global du projet."""
    logging.basicConfig(
        level=niveau,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
