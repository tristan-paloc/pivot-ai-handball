"""Recollage de fragments de tracks apres ByteTrack (offline track stitching).

ByteTrack (mouvement seul) fragmente l'identite d'un joueur en plusieurs pistes
courtes des qu'il y a occlusion breve, contact ou croisement. Ce module relie
APRES COUP les fragments qui correspondent manifestement au meme joueur reel,
sans toucher a ByteTrack et de facon entierement reversible.

Principe : purement geometrique et conservateur. Pour chaque piste A qui se
termine, on cherche la piste B qui reprend juste apres, au bon endroit (position
extrapolee depuis la vitesse de sortie de A), de meme classe/role et de taille
compatible. Chaque recollage recoit un score et une categorie :
- "sur"     : score eleve, candidat unique -> recolle
- "probable": score moyen, candidat unique -> recolle
- "ambigu"  : plusieurs candidats plausibles OU score trop bas -> REFUSE

On prefere laisser des fragments non recolles plutot que fusionner deux joueurs
differents. Un union-find chaine les recollages (24->26->29->32->33 devient un
seul ID). Testable sans GPU sur des detections synthetiques.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)

# Seuils par defaut : point de depart CONSERVATEUR, a affiner apres les premiers
# resultats reels (ne pas sur-optimiser a l'aveugle).
GAP_MAX_S = 1.0            # ecart temporel max entre fin de A et debut de B
TOL_BASE = 0.6            # tolerance de position a gap nul, en hauteurs de bbox
TOL_PENTE = 0.02          # croissance de la tolerance par frame de gap
RATIO_TAILLE_MAX = 1.8    # ratio max des tailles de bbox (A vs B)
SEUIL_SUR = 0.70          # score >= -> merge "sur"
SEUIL_PROBABLE = 0.50     # score >= -> merge "probable"
MARGE_UNICITE = 0.10      # ecart de score min entre le 1er et le 2e candidat


@dataclass
class FragmentTrack:
    """Resume geometrique d'une piste (fragment) pour le recollage."""

    tid: int
    frame_debut: int
    frame_fin: int
    x_debut: float
    y_debut: float
    x_fin: float
    y_fin: float
    vx: float  # vitesse de sortie (px par frame source)
    vy: float
    w_moy: float
    h_moy: float
    classe: int
    nb_points: int


@dataclass
class CandidatMerge:
    """Un recollage evalue A->B, date, score et categorise."""

    tracker_a: int
    tracker_b: int
    seconde: float
    gap_frames: int
    distance_px: float
    distance_predite_px: float
    ratio_taille: float
    score: float
    categorie: str  # "sur" | "probable" | "ambigu"
    retenu: bool
    criteres: list[str]

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResultatStitching:
    """Sortie du recollage : detections relabelisees + journal des merges."""

    detections: dict[int, sv.Detections]
    merges: list[CandidatMerge]
    relabel: dict[int, int]


def _vitesse_sortie(pts: list[tuple[int, float, float, float, float, int]]) -> tuple[float, float]:
    """Vitesse (px/frame source) estimee sur les dernieres positions de la piste."""
    if len(pts) < 2:
        return 0.0, 0.0
    j = min(len(pts) - 1, 4)
    f0, x0, y0 = pts[-1 - j][0], pts[-1 - j][1], pts[-1 - j][2]
    f1, x1, y1 = pts[-1][0], pts[-1][1], pts[-1][2]
    df = f1 - f0
    if df <= 0:
        return 0.0, 0.0
    return (x1 - x0) / df, (y1 - y0) / df


def resumer_fragments(
    detections_trackees: dict[int, sv.Detections],
) -> dict[int, FragmentTrack]:
    """Extrait, par tracker_id, le resume geometrique utile au recollage.

    Args:
        detections_trackees: dict frame -> Detections (avec tracker_id, class_id).

    Returns:
        dict tracker_id -> FragmentTrack.
    """
    brut: dict[int, list[tuple[int, float, float, float, float, int]]] = {}
    for fi in sorted(detections_trackees.keys()):
        dets = detections_trackees[fi]
        if dets.tracker_id is None:
            continue
        for i in range(len(dets)):
            tid = int(dets.tracker_id[i])
            x1, y1, x2, y2 = (float(v) for v in dets.xyxy[i])
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            w, h = x2 - x1, y2 - y1
            cls = int(dets.class_id[i]) if dets.class_id is not None else -1
            brut.setdefault(tid, []).append((int(fi), cx, cy, w, h, cls))

    fragments: dict[int, FragmentTrack] = {}
    for tid, pts in brut.items():
        pts.sort(key=lambda p: p[0])
        vx, vy = _vitesse_sortie(pts)
        classe = Counter(p[5] for p in pts).most_common(1)[0][0]
        fragments[tid] = FragmentTrack(
            tid=tid,
            frame_debut=pts[0][0],
            frame_fin=pts[-1][0],
            x_debut=pts[0][1],
            y_debut=pts[0][2],
            x_fin=pts[-1][1],
            y_fin=pts[-1][2],
            vx=vx,
            vy=vy,
            w_moy=sum(p[3] for p in pts) / len(pts),
            h_moy=sum(p[4] for p in pts) / len(pts),
            classe=int(classe),
            nb_points=len(pts),
        )
    return fragments


def _ratio_taille(a: FragmentTrack, b: FragmentTrack) -> float:
    """Ratio max des tailles de bbox (>= 1). Grand = tailles tres differentes."""
    if min(a.w_moy, b.w_moy, a.h_moy, b.h_moy) <= 0:
        return math.inf
    rw = max(a.w_moy / b.w_moy, b.w_moy / a.w_moy)
    rh = max(a.h_moy / b.h_moy, b.h_moy / a.h_moy)
    return max(rw, rh)


def evaluer_candidats(
    fragments: dict[int, FragmentTrack],
    fps: float,
    gap_max_s: float = GAP_MAX_S,
    tol_base: float = TOL_BASE,
    tol_pente: float = TOL_PENTE,
    ratio_taille_max: float = RATIO_TAILLE_MAX,
    seuil_sur: float = SEUIL_SUR,
    seuil_probable: float = SEUIL_PROBABLE,
    marge_unicite: float = MARGE_UNICITE,
) -> list[CandidatMerge]:
    """Evalue les recollages possibles et les categorise (sans les appliquer).

    Pour chaque piste A (par ordre de fin), on liste les pistes B geometriquement
    plausibles (fenetre temporelle, meme classe, position extrapolee dans la
    tolerance, taille compatible), on les score et on decide :
    - candidat unique et score haut/moyen -> "sur"/"probable" (retenu) ;
    - plusieurs candidats proches, ou score trop bas -> "ambigu" (refuse).

    Un B deja retenu comme reprise d'un A ne peut pas etre reclame par un autre A
    (evite de fusionner deux joueurs sous un meme ID).

    Args:
        fragments: resume des pistes (resumer_fragments).
        fps: framerate reel (pour l'horodatage et la fenetre temporelle).
        gap_max_s: ecart temporel max (s) entre fin de A et debut de B.
        tol_base, tol_pente: tolerance de position = (tol_base + tol_pente*gap)
            * hauteur_bbox_moyenne.
        ratio_taille_max: ratio de taille max tolere.
        seuil_sur, seuil_probable: seuils de score des categories.
        marge_unicite: ecart de score min entre 1er et 2e candidat pour ne PAS
            juger le recollage ambigu.

    Returns:
        liste de CandidatMerge (retenus ET refuses), triee par seconde.
    """
    gap_max = max(1, int(round(gap_max_s * fps)))
    ordre = sorted(fragments.values(), key=lambda f: (f.frame_fin, f.tid))
    debut_pris: set[int] = set()
    merges: list[CandidatMerge] = []

    for a in ordre:
        cands: list[tuple[float, FragmentTrack, int, float, float, float, float]] = []
        for b in fragments.values():
            if b.tid == a.tid or b.tid in debut_pris:
                continue
            gap = b.frame_debut - a.frame_fin
            if gap <= 0 or gap > gap_max:  # B doit demarrer strictement apres A
                continue
            if b.classe != a.classe:  # role/classe incompatible
                continue
            pred_x = a.x_fin + a.vx * gap
            pred_y = a.y_fin + a.vy * gap
            dist_pred = math.hypot(b.x_debut - pred_x, b.y_debut - pred_y)
            taille_ref = max(1.0, (a.h_moy + b.h_moy) / 2.0)
            tol = (tol_base + tol_pente * gap) * taille_ref
            if dist_pred > tol:
                continue
            ratio = _ratio_taille(a, b)
            if ratio > ratio_taille_max:
                continue
            dist = math.hypot(b.x_debut - a.x_fin, b.y_debut - a.y_fin)
            term_dist = 1.0 - min(1.0, dist_pred / tol)
            term_gap = 1.0 - min(1.0, gap / gap_max)
            term_size = (
                1.0 - min(1.0, (ratio - 1.0) / (ratio_taille_max - 1.0))
                if ratio_taille_max > 1.0 else 1.0
            )
            score = 0.5 * term_dist + 0.25 * term_gap + 0.25 * term_size
            cands.append((score, b, gap, dist, dist_pred, ratio, tol))

        if not cands:
            continue
        cands.sort(key=lambda c: c[0], reverse=True)
        score, b, gap, dist, dist_pred, ratio, tol = cands[0]
        unique = len(cands) == 1 or (cands[0][0] - cands[1][0]) >= marge_unicite

        if not unique:
            categorie, retenu = "ambigu", False
        elif score >= seuil_sur:
            categorie, retenu = "sur", True
        elif score >= seuil_probable:
            categorie, retenu = "probable", True
        else:
            categorie, retenu = "ambigu", False

        criteres = [
            f"gap {gap}f",
            f"dist_pred {round(dist_pred)}px<=tol {round(tol)}px",
            f"classe {a.classe}",
            f"ratio_taille {round(ratio, 2)}",
        ]
        if len(cands) == 1:
            criteres.append("candidat unique")
        else:
            criteres.append(
                f"{len(cands)} candidats (marge {round(cands[0][0] - cands[1][0], 2)})"
            )
        if not retenu:
            criteres.append(
                "refuse: plusieurs candidats proches" if not unique
                else f"refuse: score {round(score, 2)} < {seuil_probable}"
            )

        if retenu:
            debut_pris.add(b.tid)
        merges.append(
            CandidatMerge(
                tracker_a=a.tid,
                tracker_b=b.tid,
                seconde=round(a.frame_fin / fps, 2) if fps > 0 else 0.0,
                gap_frames=gap,
                distance_px=round(dist, 1),
                distance_predite_px=round(dist_pred, 1),
                ratio_taille=round(ratio, 2),
                score=round(score, 3),
                categorie=categorie,
                retenu=retenu,
                criteres=criteres,
            )
        )

    merges.sort(key=lambda m: m.seconde)
    return merges


def construire_relabel(
    fragments: dict[int, FragmentTrack],
    merges: list[CandidatMerge],
) -> dict[int, int]:
    """Construit le mapping ancien_id -> id_canonique a partir des merges retenus.

    Union-find sur les recollages retenus ; l'id canonique d'un groupe est celui
    de la piste qui demarre le plus tot (chaine 24->26->29->32->33 -> 24).
    """
    parent = {t: t for t in fragments}

    def racine(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for m in merges:
        if m.retenu and m.tracker_a in parent and m.tracker_b in parent:
            ra, rb = racine(m.tracker_a), racine(m.tracker_b)
            if ra != rb:
                parent[ra] = rb

    groupes: dict[int, list[int]] = {}
    for t in fragments:
        groupes.setdefault(racine(t), []).append(t)

    relabel: dict[int, int] = {}
    for membres in groupes.values():
        canon = min(membres, key=lambda t: (fragments[t].frame_debut, t))
        for t in membres:
            relabel[t] = canon
    return relabel


def _appliquer_relabel(
    detections_trackees: dict[int, sv.Detections],
    relabel: dict[int, int],
) -> dict[int, sv.Detections]:
    """Recree les detections avec les tracker_id remappes (sans muter l'entree)."""
    out: dict[int, sv.Detections] = {}
    for fi, dets in detections_trackees.items():
        if dets.tracker_id is None:
            out[fi] = dets
            continue
        nouveaux = np.array(
            [relabel.get(int(t), int(t)) for t in dets.tracker_id],
            dtype=dets.tracker_id.dtype,
        )
        out[fi] = dataclasses.replace(dets, tracker_id=nouveaux)
    return out


def recoller_tracks(
    detections_trackees: dict[int, sv.Detections],
    fps: float,
    gap_max_s: float = GAP_MAX_S,
    tol_base: float = TOL_BASE,
    tol_pente: float = TOL_PENTE,
    ratio_taille_max: float = RATIO_TAILLE_MAX,
    seuil_sur: float = SEUIL_SUR,
    seuil_probable: float = SEUIL_PROBABLE,
    marge_unicite: float = MARGE_UNICITE,
) -> ResultatStitching:
    """Recolle les fragments de tracks apres ByteTrack (post-traitement offline).

    Args:
        detections_trackees: sortie de ByteTrack (frame -> Detections + tracker_id).
        fps: framerate reel du clip.
        gap_max_s, tol_base, tol_pente, ratio_taille_max, seuil_sur,
            seuil_probable, marge_unicite: parametres du recollage (defauts
            conservateurs).

    Returns:
        ResultatStitching (detections relabelisees, journal des merges, relabel).
    """
    fragments = resumer_fragments(detections_trackees)
    merges = evaluer_candidats(
        fragments, fps,
        gap_max_s=gap_max_s, tol_base=tol_base, tol_pente=tol_pente,
        ratio_taille_max=ratio_taille_max, seuil_sur=seuil_sur,
        seuil_probable=seuil_probable, marge_unicite=marge_unicite,
    )
    relabel = construire_relabel(fragments, merges)
    detections = _appliquer_relabel(detections_trackees, relabel)

    retenus = [m for m in merges if m.retenu]
    logger.info(
        "Stitching : %d fragments -> %d pistes (%d merges retenus, %d ambigus refuses)",
        len(fragments),
        len({relabel[t] for t in fragments}) if fragments else 0,
        len(retenus),
        len(merges) - len(retenus),
    )
    return ResultatStitching(detections=detections, merges=merges, relabel=relabel)
