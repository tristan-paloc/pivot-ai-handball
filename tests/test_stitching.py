"""Tests du recollage de tracks (offline stitching) : conservateur, sans GPU.

Verifie que le stitching recolle les fragments manifestement identiques (cascade)
et REFUSE les cas ambigus, incompatibles en classe, ou geometriquement invraisemblables.
"""

from __future__ import annotations

import numpy as np
import supervision as sv

from pivot_ai.stitching import (
    evaluer_candidats,
    recoller_tracks,
    resumer_fragments,
)


def _dets(items: list[tuple[int, float, float, int]], taille: float = 40.0) -> sv.Detections:
    """Detections d'une frame : (tracker_id, cx, cy, class_id), bbox largeur/2 x hauteur."""
    if not items:
        return sv.Detections.empty()
    demi_l, demi_h = taille / 4.0, taille / 2.0
    xyxy = np.array(
        [[cx - demi_l, cy - demi_h, cx + demi_l, cy + demi_h] for _, cx, cy, _ in items],
        dtype=np.float32,
    )
    return sv.Detections(
        xyxy=xyxy,
        confidence=np.ones(len(items), dtype=np.float32),
        class_id=np.array([c for _, _, _, c in items], dtype=int),
        tracker_id=np.array([t for t, _, _, _ in items], dtype=int),
    )


def _cascade() -> dict[int, sv.Detections]:
    """Un joueur (classe 0) fragmente en 3 IDs (24, 26, 29) avancant a vitesse
    constante vers la droite, avec 2 trous courts ; + un joueur distinct (99) au loin."""
    dets: dict[int, sv.Detections] = {}
    x = 100.0
    for fi in range(0, 31):
        items = [(99, 600.0, 400.0, 0)]  # joueur distinct, present tout le temps
        x = 100.0 + fi * 3.0  # deplacement regulier 3 px/frame
        if fi <= 8:
            items.append((24, x, 100.0, 0))
        elif 11 <= fi <= 19:
            items.append((26, x, 100.0, 0))
        elif fi >= 22:
            items.append((29, x, 100.0, 0))
        dets[fi] = _dets(items)
    return dets


def test_resumer_fragments() -> None:
    """Chaque piste devient un FragmentTrack avec bornes, vitesse, taille, classe."""
    frags = resumer_fragments(_cascade())
    assert set(frags) == {24, 26, 29, 99}
    assert frags[24].frame_debut == 0 and frags[24].frame_fin == 8
    assert frags[24].vx > 0  # avance vers la droite
    assert frags[24].classe == 0
    assert frags[99].vx == 0.0  # immobile


def test_cascade_recollee_en_un_id() -> None:
    """La cascade 24->26->29 est recollee en un seul ID ; le joueur distinct reste seul."""
    res = recoller_tracks(_cascade(), fps=25.0)
    # 24 est le plus precoce -> id canonique de la chaine
    assert res.relabel[26] == 24
    assert res.relabel[29] == 24
    assert res.relabel[99] == 99  # jamais fusionne
    # apres recollage : 2 pistes distinctes seulement
    ids_finaux = {int(t) for d in res.detections.values() if d.tracker_id is not None
                  for t in d.tracker_id}
    assert ids_finaux == {24, 99}
    # tous les merges de la chaine sont retenus
    retenus = {(m.tracker_a, m.tracker_b) for m in res.merges if m.retenu}
    assert (24, 26) in retenus and (26, 29) in retenus


def test_merge_ambigu_refuse() -> None:
    """Deux candidats aussi plausibles pour un meme A -> aucun recollage (ambigu)."""
    dets: dict[int, sv.Detections] = {}
    for fi in range(0, 6):
        dets[fi] = _dets([(1, 100.0, 100.0, 0)])  # A immobile, s'arrete a fi=5
    # deux reprises equidistantes de la derniere position de A, meme classe/taille
    for fi in range(8, 14):
        dets[fi] = _dets([(2, 112.0, 100.0, 0), (3, 100.0, 112.0, 0)])
    res = recoller_tracks(dets, fps=25.0)
    assert res.relabel[1] == 1  # A non recolle
    merge_a = next(m for m in res.merges if m.tracker_a == 1)
    assert merge_a.categorie == "ambigu" and merge_a.retenu is False


def test_classe_incompatible_pas_de_merge() -> None:
    """Un joueur (classe 0) et un arbitre (classe 2) ne sont jamais recolles."""
    dets: dict[int, sv.Detections] = {}
    for fi in range(0, 6):
        dets[fi] = _dets([(1, 100.0, 100.0, 0)])       # joueur
    for fi in range(8, 14):
        dets[fi] = _dets([(2, 103.0, 100.0, 2)])       # arbitre, meme endroit
    res = recoller_tracks(dets, fps=25.0)
    assert res.relabel[1] == 1 and res.relabel[2] == 2
    assert all(not m.retenu for m in res.merges)


def test_trop_loin_pas_de_merge() -> None:
    """Reprise trop eloignee de la position extrapolee -> pas candidat, pas de merge."""
    dets: dict[int, sv.Detections] = {}
    for fi in range(0, 6):
        dets[fi] = _dets([(1, 100.0, 100.0, 0)])       # A quasi immobile
    for fi in range(8, 14):
        dets[fi] = _dets([(2, 500.0, 100.0, 0)])       # B tres loin
    res = recoller_tracks(dets, fps=25.0)
    assert res.relabel[1] == 1 and res.relabel[2] == 2
    assert not any(m.retenu for m in res.merges)


def test_gap_trop_long_pas_de_merge() -> None:
    """Un trou plus long que gap_max n'est pas recolle."""
    dets: dict[int, sv.Detections] = {}
    for fi in range(0, 6):
        dets[fi] = _dets([(1, 100.0, 100.0, 0)])
    for fi in range(70, 76):  # ~2.6 s plus tard a 25 fps
        dets[fi] = _dets([(2, 100.0, 100.0, 0)])
    res = recoller_tracks(dets, fps=25.0, gap_max_s=1.0)
    assert res.relabel[2] == 2
    assert not any(m.retenu for m in res.merges)


def test_un_seul_a_par_b() -> None:
    """Un meme B ne peut pas etre reclame par deux A (pas de fusion de 2 joueurs)."""
    dets: dict[int, sv.Detections] = {}
    # deux A distincts se terminant au meme endroit a la meme frame
    for fi in range(0, 6):
        dets[fi] = _dets([(1, 100.0, 100.0, 0), (2, 100.5, 100.0, 0)])
    for fi in range(8, 14):
        dets[fi] = _dets([(3, 101.0, 100.0, 0)])  # un seul B
    merges = evaluer_candidats(resumer_fragments(dets), fps=25.0)
    retenus_vers_3 = [m for m in merges if m.tracker_b == 3 and m.retenu]
    assert len(retenus_vers_3) <= 1  # B=3 reclame au plus une fois
