"""Verite terrain legere : suivre 2-3 joueurs annotes pour des metriques interpretables.

On annote la position (point) de quelques joueurs de reference sur des frames
echantillonnees. On confronte ensuite chaque point a la piste (tracker_id) qui
le recouvre pour mesurer, PAR joueur reel :
- la duree correctement suivie,
- le taux de temps sous une seule identite (continuite -> critere 70%),
- le nombre de pertes d'identite (switches), hors changements de plan,
- et surtout la CAUSE de chaque perte : detection absente (trou de detection)
  vs tracker qui change d'ID alors que la detection etait la.

Format JSON attendu :
{
  "clip": "evt_054_...mp4",
  "joueurs": [
    {"nom": "hugo-monte-dos-santos",
     "points": [{"frame": 0, "x": 812, "y": 430}, {"frame": 25, "x": 790, "y": 415}, ...]}
  ]
}
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import supervision as sv

from pivot_ai.metriques import _coupure_entre

logger = logging.getLogger(__name__)


@dataclass
class PointGT:
    """Position annotee d'un joueur a une frame."""

    frame: int
    x: float
    y: float


@dataclass
class JoueurGT:
    """Un joueur de reference annote."""

    nom: str
    points: list[PointGT]


@dataclass
class VeriteTerrain:
    """Verite terrain d'un clip : quelques joueurs annotes."""

    clip: str
    joueurs: list[JoueurGT]


def charger_verite_terrain(chemin_json: str | Path) -> VeriteTerrain:
    """Charge une verite terrain depuis un JSON."""
    data = json.loads(Path(chemin_json).read_text(encoding="utf-8"))
    joueurs = [
        JoueurGT(
            nom=j["nom"],
            points=[PointGT(int(p["frame"]), float(p["x"]), float(p["y"])) for p in j["points"]],
        )
        for j in data.get("joueurs", [])
    ]
    return VeriteTerrain(clip=data.get("clip", ""), joueurs=joueurs)


@dataclass
class EvalJoueur:
    """Metriques de continuite d'ID pour un joueur annote."""

    nom: str
    nb_points: int
    nb_matches: int
    duree_suivie_pct: float
    id_dominant: int | None
    taux_id_dominant: float
    nb_switches: int
    nb_switches_cause_detection: int
    nb_switches_cause_tracker: int
    continuite_ok: bool  # taux_id_dominant >= seuil

    def as_dict(self) -> dict:
        return asdict(self)


def _id_au_point(dets: sv.Detections | None, x: float, y: float, seuil_px: float) -> int | None:
    """tracker_id de la piste qui recouvre (x, y) ; sinon la plus proche < seuil_px."""
    if dets is None or dets.tracker_id is None or len(dets) == 0:
        return None
    contenant: list[tuple[float, int]] = []
    proche: list[tuple[float, int]] = []
    for i in range(len(dets)):
        x1, y1, x2, y2 = (float(v) for v in dets.xyxy[i])
        tid = int(dets.tracker_id[i])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        d = math.hypot(cx - x, cy - y)
        if x1 <= x <= x2 and y1 <= y <= y2:
            contenant.append((d, tid))
        elif d <= seuil_px:
            proche.append((d, tid))
    if contenant:
        return min(contenant)[1]
    if proche:
        return min(proche)[1]
    return None


def evaluer_joueur(
    points: list[PointGT],
    detections_trackees: dict[int, sv.Detections],
    seuil_px: float,
    frames_coupure: tuple[int, ...] = (),
    seuil_continuite: float = 0.70,
    nom: str = "",
) -> EvalJoueur:
    """Confronte les points annotes d'un joueur au tracking et calcule ses metriques.

    Args:
        points: points annotes du joueur (tries par frame).
        detections_trackees: dict frame -> Detections (avec tracker_id).
        seuil_px: rayon max pour associer un point a une piste.
        frames_coupure: changements de plan (un switch a une coupure n'est pas
            compte comme echec).
        seuil_continuite: seuil du taux d'ID dominant pour valider la continuite.
        nom: etiquette.

    Returns:
        EvalJoueur.
    """
    pts = sorted(points, key=lambda p: p.frame)
    seq: list[int] = []
    switches = sw_det = sw_trk = 0
    last_id: int | None = None
    last_frame: int | None = None
    gap_depuis_match = False

    for p in pts:
        idp = _id_au_point(detections_trackees.get(p.frame), p.x, p.y, seuil_px)
        if idp is None:
            gap_depuis_match = True  # detection absente sur ce point
            continue
        seq.append(idp)
        est_switch = (
            last_id is not None
            and idp != last_id
            and not _coupure_entre(last_frame, p.frame, frames_coupure)
        )
        if est_switch:
            switches += 1
            if gap_depuis_match:
                sw_det += 1  # un trou de detection explique le changement
            else:
                sw_trk += 1  # detection continue mais l'ID a change = faute tracker
        last_id, last_frame = idp, p.frame
        gap_depuis_match = False

    nb_matches = len(seq)
    id_dominant: int | None = None
    taux = 0.0
    if seq:
        id_dominant, occ = Counter(seq).most_common(1)[0]
        taux = occ / nb_matches

    return EvalJoueur(
        nom=nom,
        nb_points=len(pts),
        nb_matches=nb_matches,
        duree_suivie_pct=round(nb_matches / len(pts), 3) if pts else 0.0,
        id_dominant=id_dominant,
        taux_id_dominant=round(taux, 3),
        nb_switches=switches,
        nb_switches_cause_detection=sw_det,
        nb_switches_cause_tracker=sw_trk,
        continuite_ok=taux >= seuil_continuite,
    )


@dataclass
class EvalVeriteTerrain:
    """Agregat de la verite terrain sur tous les joueurs annotes d'un clip."""

    clip: str
    joueurs: list[EvalJoueur] = field(default_factory=list)

    @property
    def taux_continuite_moyen(self) -> float:
        if not self.joueurs:
            return 0.0
        return round(sum(j.taux_id_dominant for j in self.joueurs) / len(self.joueurs), 3)

    @property
    def part_cause_tracker(self) -> float:
        """Part des switches imputables au tracker (vs detection)."""
        det = sum(j.nb_switches_cause_detection for j in self.joueurs)
        trk = sum(j.nb_switches_cause_tracker for j in self.joueurs)
        return round(trk / (det + trk), 3) if (det + trk) else 0.0

    def as_dict(self) -> dict:
        return {
            "clip": self.clip,
            "taux_continuite_moyen": self.taux_continuite_moyen,
            "part_cause_tracker": self.part_cause_tracker,
            "joueurs": [j.as_dict() for j in self.joueurs],
        }


def evaluer_verite_terrain(
    verite: VeriteTerrain,
    detections_trackees: dict[int, sv.Detections],
    seuil_px: float,
    frames_coupure: tuple[int, ...] = (),
    seuil_continuite: float = 0.70,
) -> EvalVeriteTerrain:
    """Evalue tous les joueurs annotes d'un clip."""
    evals = [
        evaluer_joueur(
            j.points, detections_trackees, seuil_px, frames_coupure,
            seuil_continuite, nom=j.nom,
        )
        for j in verite.joueurs
    ]
    return EvalVeriteTerrain(clip=verite.clip, joueurs=evals)


def extraire_frames_pour_annotation(
    chemin_video: str | Path,
    dossier_sortie: str | Path,
    intervalle_s: float = 1.0,
    subsample: int = 2,
) -> Path:
    """Extrait des frames a intervalle regulier + un gabarit JSON a completer.

    Ouvre chaque JPG, note (x, y) des 2-3 joueurs de reference, et reporte ces
    points dans le gabarit `verite_terrain.json`.

    Args:
        chemin_video: clip a annoter.
        dossier_sortie: dossier ou ecrire les frames + le gabarit.
        intervalle_s: espacement entre frames extraites (secondes).
        subsample: aligne les indices de frame sur ceux du tracking.

    Returns:
        Path du gabarit JSON.
    """
    chemin_video = str(chemin_video)
    sortie = Path(dossier_sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(chemin_video)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la video : {chemin_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pas = max(subsample, int(round(intervalle_s * fps)))
    pas -= pas % max(1, subsample)  # multiple du subsample -> frame reellement trackee

    frames_annotables: list[int] = []
    fi = 0
    while fi < total:
        ret, frame = cap.read()
        if not ret:
            break
        if fi % pas == 0:
            cv2.imwrite(str(sortie / f"frame_{fi:05d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            frames_annotables.append(fi)
        fi += 1
    cap.release()

    gabarit = {
        "clip": Path(chemin_video).name,
        "_aide": "Pour chaque joueur, ajoute un point {frame, x, y} par frame extraite.",
        "frames_extraites": frames_annotables,
        "joueurs": [{"nom": "joueur_1", "points": []}],
    }
    chemin_gabarit = sortie / "verite_terrain.json"
    chemin_gabarit.write_text(json.dumps(gabarit, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Annotation : %d frames extraites dans %s (gabarit : %s)",
        len(frames_annotables), sortie, chemin_gabarit,
    )
    return chemin_gabarit
