"""Benchmark de la continuite d'identite du tracking (espace image).

Mesure la qualite d'un tracking directement sur les bboxes (pixels), SANS
homographie : la continuite d'ID est un probleme d'espace image. Reutilise le
coeur de detection de reprises de `metriques.py` (partage terrain/pixels).

Fournit, par run (clip x tracker) :
- nb de tracks
- longueur des tracks (moy / mediane / p90), en frames et en secondes
- nb de fragments courts
- reprises d'ID (switches suspectes), en tenant compte des changements de plan
- nb de joueurs estimes (recollage des reprises)
- taux de fragmentation

Purement geometrique -> testable sans GPU sur des detections synthetiques.
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import supervision as sv

from pivot_ai.metriques import (
    BornesTrack,
    composantes_connexes,
    detecter_reprises_bornes,
)
from pivot_ai.plans import detecter_changements_plan

logger = logging.getLogger(__name__)

# Signature d'une fonction de tracking injectable (permet de tester le runner
# sans GPU en fournissant des detections synthetiques).
FonctionTracking = Callable[[str, str, int, float], dict[int, sv.Detections]]


def _centre_bbox(xyxy) -> tuple[float, float]:
    """Centre (x, y) en pixels d'une bbox [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def trajectoires_pixels(
    detections_trackees: dict[int, sv.Detections],
) -> dict[int, list[tuple[int, float, float]]]:
    """Extrait, par tracker_id, la liste (frame, cx, cy) des centres de bbox.

    Args:
        detections_trackees: dict frame_idx -> Detections (avec tracker_id).

    Returns:
        dict tracker_id -> liste triee de (frame_idx, cx_px, cy_px).
    """
    traj: dict[int, list[tuple[int, float, float]]] = {}
    for fi in sorted(detections_trackees.keys()):
        dets = detections_trackees[fi]
        if dets.tracker_id is None:
            continue
        for i in range(len(dets)):
            tid = int(dets.tracker_id[i])
            cx, cy = _centre_bbox(dets.xyxy[i])
            traj.setdefault(tid, []).append((int(fi), cx, cy))
    for tid in traj:
        traj[tid].sort(key=lambda p: p[0])
    return traj


def bornes_pixels(
    trajectoires: dict[int, list[tuple[int, float, float]]],
) -> dict[int, BornesTrack]:
    """Convertit des trajectoires pixel en BornesTrack (premiere/derniere position)."""
    bornes: dict[int, BornesTrack] = {}
    for tid, pts in trajectoires.items():
        if not pts:
            continue
        f0, x0, y0 = pts[0]
        f1, x1, y1 = pts[-1]
        bornes[tid] = BornesTrack(
            frame_debut=f0, x_debut=x0, y_debut=y0,
            frame_fin=f1, x_fin=x1, y_fin=y1, nb_reel=len(pts),
        )
    return bornes


@dataclass
class ResumeBenchmark:
    """Metriques de continuite d'ID pour un run (clip x tracker)."""

    tracker: str = ""
    clip: str = ""
    nb_tracks: int = 0
    nb_joueurs_estimes: int = 0
    nb_reprises_id: int = 0
    nb_fragments_courts: int = 0
    fragmentation: float = 0.0
    longueur_moyenne_frames: float = 0.0
    longueur_mediane_frames: float = 0.0
    longueur_p90_frames: float = 0.0
    longueur_moyenne_s: float = 0.0
    nb_coupures_plan: int = 0
    frames_coupure: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _percentile(valeurs: list[float], q: float) -> float:
    """Percentile q (0-100) simple, sans dependance numpy."""
    if not valeurs:
        return 0.0
    tri = sorted(valeurs)
    if len(tri) == 1:
        return float(tri[0])
    rang = (q / 100.0) * (len(tri) - 1)
    bas = int(rang)
    haut = min(bas + 1, len(tri) - 1)
    frac = rang - bas
    return float(tri[bas] + (tri[haut] - tri[bas]) * frac)


def resume_benchmark(
    detections_trackees: dict[int, sv.Detections],
    fps: float,
    subsample: int = 1,
    seuil_distance_px: float = 120.0,
    seuil_frames: int = 30,
    min_frames_fragment: int = 5,
    frames_coupure: tuple[int, ...] = (),
    tracker: str = "",
    clip: str = "",
) -> ResumeBenchmark:
    """Calcule les metriques de continuite d'ID pour un run.

    Args:
        detections_trackees: sortie du tracking (frame -> Detections + tracker_id).
        fps: framerate reel du clip.
        subsample: 1 frame sur N reellement trackee (pour convertir en secondes).
        seuil_distance_px: distance max (px) pour lier deux tracks (reprise d'ID).
        seuil_frames: delai max (frames source) pour une reprise.
        min_frames_fragment: en dessous, un track est un "fragment court".
        frames_coupure: indices des changements de plan (reprises enjambant une
            coupure ignorees).
        tracker, clip: etiquettes pour le rapport.

    Returns:
        ResumeBenchmark.
    """
    traj = trajectoires_pixels(detections_trackees)
    bornes = bornes_pixels(traj)
    nb_tracks = len(bornes)

    longueurs = [b.nb_reel for b in bornes.values()]
    reprises = detecter_reprises_bornes(
        bornes, seuil_distance_px, seuil_frames, tuple(frames_coupure)
    )
    nb_joueurs = composantes_connexes(list(bornes.keys()), reprises)

    # Duree reelle couverte par une frame trackee = subsample / fps.
    sec_par_frame_trackee = subsample / fps if fps > 0 else 0.0
    moy = statistics.fmean(longueurs) if longueurs else 0.0

    return ResumeBenchmark(
        tracker=tracker,
        clip=clip,
        nb_tracks=nb_tracks,
        nb_joueurs_estimes=nb_joueurs,
        nb_reprises_id=len(reprises),
        nb_fragments_courts=sum(1 for n in longueurs if n < min_frames_fragment),
        fragmentation=round(nb_tracks / max(1, nb_joueurs), 2),
        longueur_moyenne_frames=round(moy, 1),
        longueur_mediane_frames=round(statistics.median(longueurs), 1) if longueurs else 0.0,
        longueur_p90_frames=round(_percentile([float(x) for x in longueurs], 90), 1),
        longueur_moyenne_s=round(moy * sec_par_frame_trackee, 2),
        nb_coupures_plan=len(frames_coupure),
        frames_coupure=list(frames_coupure),
    )


# ---------------------------------------------------------------------------
# Runner : lance le benchmark sur des clips x trackers
# ---------------------------------------------------------------------------


@dataclass
class RunBenchmark:
    """Un run (clip x tracker) : metriques + chemin de la video annotee."""

    resume: ResumeBenchmark
    chemin_video_annotee: Path | None = None


def _metadonnees_video(chemin_video: str) -> tuple[float, int, int]:
    """Lit (fps, largeur, hauteur) d'une video."""
    cap = cv2.VideoCapture(chemin_video)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la video : {chemin_video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, w, h


def _tracking_reel(
    chemin_video: str, tracker: str, subsample: int, fps: float,
    modele_config=None,
) -> dict[int, sv.Detections]:
    """Fonction de tracking par defaut (necessite le modele + torch/GPU)."""
    from pivot_ai.detection import DetecteurLocal, detecter_video
    from pivot_ai.tracking import tracker_detections, tracker_video_botsort

    detecteur = DetecteurLocal(modele_config)
    if tracker == "bytetrack":
        dets_par_frame = detecter_video(chemin_video, detecteur, subsample=subsample)
        trackees, _ = tracker_detections(dets_par_frame, fps=fps, subsample=subsample)
        return trackees
    if tracker == "botsort":
        trackees, _ = tracker_video_botsort(chemin_video, detecteur, subsample=subsample)
        return trackees
    raise ValueError(f"tracker inconnu : {tracker!r} (attendu 'bytetrack' ou 'botsort')")


def lancer_benchmark(
    clips: list[str | Path],
    trackers: list[str],
    sortie: str | Path,
    modele_config=None,
    subsample: int = 2,
    fonction_tracking: FonctionTracking | None = None,
    seuil_distance_px: float | None = None,
    seuil_frames: int | None = None,
    min_frames_fragment: int = 5,
    generer_videos: bool = True,
) -> list[RunBenchmark]:
    """Lance le benchmark de continuite d'ID sur chaque (clip x tracker).

    Args:
        clips: liste de chemins video.
        trackers: sous-ensemble de {"bytetrack", "botsort"}.
        sortie: dossier de sortie (metriques + videos annotees + rapport).
        modele_config: ModeleConfig (idealement le modele handball fine-tune).
        subsample: 1 frame sur N pour detection/tracking et coupures.
        fonction_tracking: injectable (tests sans GPU) ; defaut = _tracking_reel.
        seuil_distance_px: distance max (px) pour une reprise ; defaut 8% largeur.
        seuil_frames: delai max (frames source) ; defaut ~1s (fps).
        min_frames_fragment: seuil "fragment court".
        generer_videos: si True, produit une video annotee par run.

    Returns:
        liste de RunBenchmark. Ecrit aussi comparatif.json/.csv et RAPPORT.md.
    """
    from pivot_ai.video_annotee import generer_video_annotee

    sortie = Path(sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    clips = [str(c) for c in clips]

    runs: list[RunBenchmark] = []
    for clip in clips:
        nom_clip = Path(clip).name
        fps, largeur, _hauteur = _metadonnees_video(clip)
        s_px = seuil_distance_px if seuil_distance_px is not None else 0.08 * largeur
        s_fr = seuil_frames if seuil_frames is not None else int(round(fps))  # ~1 s
        coupures = tuple(detecter_changements_plan(clip, subsample=subsample))
        logger.info("Clip %s : %.1f fps, %d coupure(s)", nom_clip, fps, len(coupures))

        for tracker in trackers:
            fn = fonction_tracking or (
                lambda c, t, sub, f: _tracking_reel(c, t, sub, f, modele_config)
            )
            trackees = fn(clip, tracker, subsample, fps)
            resume = resume_benchmark(
                trackees, fps=fps, subsample=subsample,
                seuil_distance_px=s_px, seuil_frames=s_fr,
                min_frames_fragment=min_frames_fragment,
                frames_coupure=coupures, tracker=tracker, clip=nom_clip,
            )
            chemin_annote: Path | None = None
            if generer_videos:
                chemin_annote = generer_video_annotee(
                    clip, trackees, sortie / f"{Path(clip).stem}__{tracker}.mp4",
                    subsample=subsample, frames_coupure=coupures,
                )
            runs.append(RunBenchmark(resume=resume, chemin_video_annotee=chemin_annote))

    ecrire_comparatif(runs, sortie)
    ecrire_rapport(runs, sortie)
    return runs


def ecrire_comparatif(runs: list[RunBenchmark], sortie: str | Path) -> tuple[Path, Path]:
    """Ecrit comparatif.json et comparatif.csv des runs. Retourne (json, csv)."""
    sortie = Path(sortie)
    lignes = [r.resume.as_dict() for r in runs]

    chemin_json = sortie / "comparatif.json"
    chemin_json.write_text(json.dumps(lignes, indent=2, ensure_ascii=False), encoding="utf-8")

    chemin_csv = sortie / "comparatif.csv"
    if lignes:
        colonnes = [c for c in lignes[0] if c != "frames_coupure"]
        with chemin_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=colonnes, extrasaction="ignore")
            w.writeheader()
            w.writerows(lignes)
    return chemin_json, chemin_csv


def ecrire_rapport(runs: list[RunBenchmark], sortie: str | Path) -> Path:
    """Ecrit un RAPPORT.md comparatif lisible (table + lecture data-driven)."""
    sortie = Path(sortie)
    lignes: list[str] = ["# Benchmark continuite d'identite\n"]

    entete = (
        "| clip | tracker | tracks | joueurs est. | reprises ID | "
        "fragments courts | fragmentation | long. moy (s) | coupures |"
    )
    sep = "|" + "---|" * 9
    lignes += [entete, sep]
    for r in runs:
        m = r.resume
        lignes.append(
            f"| {m.clip} | {m.tracker} | {m.nb_tracks} | {m.nb_joueurs_estimes} | "
            f"{m.nb_reprises_id} | {m.nb_fragments_courts} | {m.fragmentation} | "
            f"{m.longueur_moyenne_s} | {m.nb_coupures_plan} |"
        )

    # Lecture : meilleur tracker = moins de reprises + fragmentation la plus basse.
    par_tracker: dict[str, list[ResumeBenchmark]] = {}
    for r in runs:
        par_tracker.setdefault(r.resume.tracker, []).append(r.resume)
    lignes.append("\n## Lecture\n")
    if par_tracker:
        def score(resumes: list[ResumeBenchmark]) -> float:
            return sum(x.nb_reprises_id for x in resumes) + sum(x.fragmentation for x in resumes)
        classement = sorted(par_tracker.items(), key=lambda kv: score(kv[1]))
        meilleur = classement[0][0]
        lignes.append(
            f"- **Tracker le plus continu sur ces clips : `{meilleur}`** "
            "(total reprises d'ID + fragmentation le plus bas).\n"
        )
        for tr, resumes in classement:
            tot_rep = sum(x.nb_reprises_id for x in resumes)
            lignes.append(f"  - `{tr}` : {tot_rep} reprises d'ID cumulees sur {len(resumes)} clip(s).")

    lignes.append(
        "\n> Part detection vs tracker et taux de recuperation d'ID sur occlusions "
        "brefs : voir les metriques de verite terrain (etape 3) — non calculables "
        "sans annotation des joueurs de reference.\n"
    )

    chemin = sortie / "RAPPORT.md"
    chemin.write_text("\n".join(lignes) + "\n", encoding="utf-8")
    return chemin
