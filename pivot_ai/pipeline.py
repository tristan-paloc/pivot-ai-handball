"""Pipeline orchestrateur : enchaine detection, tracking, equipes, stats, decoupage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import polars as pl
import supervision as sv

from pivot_ai.config import CLASSES_HANDBALL, ModeleConfig, TerrainConfig
from pivot_ai.decoupage import Action, decouper_clips_video, detecter_actions
from pivot_ai.detection import detecter_video
from pivot_ai.equipes import classifier_equipes
from pivot_ai.homographie import Correspondance, Homographie, calibrer_homographie
from pivot_ai.radar import dessiner_radar
from pivot_ai.stats import (
    PositionJoueur,
    calculer_largeur_bloc_defensif,
    calculer_stats_joueur,
)
from pivot_ai.tracking import tracker_detections

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Couleurs d'affichage par role (BGR, coherent avec radar.dessiner_legende)
# ---------------------------------------------------------------------------

COULEUR_MHB: tuple[int, int, int] = (0, 255, 0)        # vert
COULEUR_ADV: tuple[int, int, int] = (0, 0, 255)        # rouge
COULEUR_GK: tuple[int, int, int] = (0, 215, 255)       # jaune-orange
COULEUR_REF: tuple[int, int, int] = (255, 0, 255)      # magenta
COULEUR_INCONNU: tuple[int, int, int] = (200, 200, 200)  # gris


class _ProtocoleDetecteur(Protocol):
    """Interface attendue pour un detecteur injectable (DetecteurLocal ou stub)."""

    def detecter(self, frame: np.ndarray) -> sv.Detections: ...

    def detecter_batch(self, frames: list[np.ndarray]) -> list[sv.Detections]: ...


@dataclass
class ResultatPipeline:
    """Resultat complet d'un traitement de clip ou de match.

    Attributes:
        chemin_video_source: video d'entree
        chemin_video_radar: video SBS broadcast+radar si generee, sinon None
        stats_joueurs: DataFrame Polars agrege par tracker
        positions_par_tracker: positions terrain (m) brutes, par tracker_id.
            Utile pour heatmaps et analyses spatiales detaillees.
        actions_detectees: actions issues de detecter_actions
        clips_decoupes: chemins des clips d'actions ffmpeg
        metadonnees: dict d'infos (fps, nb_trackers, etc.)
    """

    chemin_video_source: Path
    chemin_video_radar: Path | None
    stats_joueurs: pl.DataFrame
    positions_par_tracker: dict[int, list[PositionJoueur]]
    actions_detectees: list[Action]
    clips_decoupes: list[Path]
    metadonnees: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Projection des detections en coordonnees terrain
# ---------------------------------------------------------------------------


def projeter_detections_en_terrain(
    detections_trackees: dict[int, sv.Detections],
    classe_finale: dict[int, int],
    homographie: Homographie,
    terrain_config: TerrainConfig | None = None,
    id_classe_joueur: int = CLASSES_HANDBALL["players"],
    filtrer_zone: bool = True,
) -> dict[int, list[PositionJoueur]]:
    """Projette les bboxes (centre-bas) en coordonnees terrain (metres).

    Seuls les trackers dont la classe stabilisee == id_classe_joueur sont retenus.
    Les positions hors `est_dans_zone` sont ecartees si `filtrer_zone=True`
    (utile pour virer les bancs et le staff).

    Args:
        detections_trackees: dict frame_idx -> Detections (avec tracker_id)
        classe_finale: dict tracker_id -> classe stabilisee
        homographie: matrice de projection pixel -> metres
        terrain_config: configuration terrain (defaut : TerrainConfig())
        id_classe_joueur: id de la classe joueur de champ
        filtrer_zone: applique le filtre `est_dans_zone` si True

    Returns:
        dict tracker_id -> liste de PositionJoueur (non interpolees), triee par frame.
    """
    config = terrain_config or TerrainConfig()
    positions_par_tracker: dict[int, list[PositionJoueur]] = {}

    for fi in sorted(detections_trackees.keys()):
        dets = detections_trackees[fi]
        if dets.tracker_id is None:
            continue
        for i in range(len(dets)):
            tid = int(dets.tracker_id[i])
            if classe_finale.get(tid) != id_classe_joueur:
                continue
            x_m, y_m = homographie.projeter_bbox_pieds(dets.xyxy[i])
            if filtrer_zone and not config.est_dans_zone((x_m, y_m)):
                continue
            positions_par_tracker.setdefault(tid, []).append(
                PositionJoueur(
                    frame_idx=int(fi),
                    x_m=float(x_m),
                    y_m=float(y_m),
                    interpole=False,
                )
            )

    logger.info(
        "Projection terrain : %d trackers conserves",
        len(positions_par_tracker),
    )
    return positions_par_tracker


# ---------------------------------------------------------------------------
# Interpolation des positions (rendu visuel uniquement)
# ---------------------------------------------------------------------------


def filtrer_traceurs_courts(
    positions_par_tracker: dict[int, list[PositionJoueur]],
    min_frames: int,
) -> dict[int, list[PositionJoueur]]:
    """Ecarte les trackers vus sur trop peu de frames reelles (fragments).

    ByteTrack (sans re-identification par apparence) fragmente les joueurs
    occultes en multiples IDs courts. Ce filtre supprime la longue traine de
    fragments pour ne garder que les trajectoires substantielles, exploitables
    pour les stats.

    Args:
        positions_par_tracker: positions reelles par tracker_id
        min_frames: nb minimum de positions reelles pour conserver un tracker
            (<= 1 desactive le filtre)

    Returns:
        dict filtre tracker_id -> positions.
    """
    if min_frames <= 1:
        return positions_par_tracker
    filtre = {
        tid: positions
        for tid, positions in positions_par_tracker.items()
        if sum(1 for p in positions if not p.interpole) >= min_frames
    }
    logger.info(
        "Filtre fragments : %d trackers conserves sur %d (min_frames=%d)",
        len(filtre),
        len(positions_par_tracker),
        min_frames,
    )
    return filtre


def interpoler_positions(
    positions_par_tracker: dict[int, list[PositionJoueur]],
    max_gap_frames: int = 10,
) -> dict[int, list[PositionJoueur]]:
    """Interpole lineairement les frames manquantes entre apparitions consecutives.

    On interpole UNIQUEMENT si le gap entre deux apparitions <= max_gap_frames.
    Cela couvre les frames sautees par subsample sans extrapoler en cas de
    disparition prolongee (occlusion ou sortie de champ).

    Les positions generees portent `interpole=True` pour pouvoir etre filtrees
    dans le calcul des stats.

    Args:
        positions_par_tracker: positions reelles (non-interpolees)
        max_gap_frames: gap maximum tolere pour interpolation (inclus)

    Returns:
        dict tracker_id -> liste etendue (reelles + interpolees), triee par frame.
    """
    resultat: dict[int, list[PositionJoueur]] = {}
    nb_interpolees = 0

    for tid, positions in positions_par_tracker.items():
        positions = sorted(positions, key=lambda p: p.frame_idx)
        if not positions:
            resultat[tid] = []
            continue

        nouvelle: list[PositionJoueur] = [positions[0]]
        for prec, suiv in zip(positions, positions[1:], strict=False):
            gap = suiv.frame_idx - prec.frame_idx
            if gap > 1 and gap <= max_gap_frames:
                for k in range(1, gap):
                    alpha = k / gap
                    nouvelle.append(
                        PositionJoueur(
                            frame_idx=prec.frame_idx + k,
                            x_m=prec.x_m + alpha * (suiv.x_m - prec.x_m),
                            y_m=prec.y_m + alpha * (suiv.y_m - prec.y_m),
                            interpole=True,
                        )
                    )
                    nb_interpolees += 1
            nouvelle.append(suiv)
        resultat[tid] = nouvelle

    logger.info(
        "Interpolation : %d positions interpolees pour %d trackers",
        nb_interpolees,
        len(resultat),
    )
    return resultat


# ---------------------------------------------------------------------------
# Couleur d'un tracker selon son role et son equipe
# ---------------------------------------------------------------------------


def couleur_tracker(
    tracker_id: int,
    classe_finale: dict[int, int],
    equipe_finale: dict[int, int],
    cluster_mhb: int,
) -> tuple[int, int, int]:
    """Retourne la couleur BGR a utiliser pour un tracker donne."""
    classe = classe_finale.get(tracker_id)
    if classe == CLASSES_HANDBALL["goalkeeper"]:
        return COULEUR_GK
    if classe == CLASSES_HANDBALL["referees"]:
        return COULEUR_REF
    if classe == CLASSES_HANDBALL["players"]:
        eq = equipe_finale.get(tracker_id)
        if eq is None:
            return COULEUR_ADV
        return COULEUR_MHB if eq == cluster_mhb else COULEUR_ADV
    return COULEUR_INCONNU


# ---------------------------------------------------------------------------
# Generation de la video SBS (broadcast + radar)
# ---------------------------------------------------------------------------


def generer_video_sbs(
    chemin_video_source: str | Path,
    detections_trackees: dict[int, sv.Detections],
    positions_par_tracker_interpolees: dict[int, list[PositionJoueur]],
    classe_finale: dict[int, int],
    equipe_finale: dict[int, int],
    cluster_mhb: int,
    chemin_sortie: str | Path,
    terrain_config: TerrainConfig | None = None,
) -> Path:
    """Genere une video side-by-side : broadcast annote + radar 2D.

    Resolution finale : broadcast_w + radar_w (largeur) x broadcast_h (hauteur).
    Encodage mp4v. Les bboxes du broadcast sont annotees avec la couleur d'equipe
    et le tracker_id. Le radar utilise les positions interpolees (rendu fluide).

    Args:
        chemin_video_source: video originale
        detections_trackees: detections + tracker_id par frame echantillonnee
        positions_par_tracker_interpolees: positions terrain (avec interpolation)
        classe_finale: dict tracker_id -> classe stabilisee
        equipe_finale: dict tracker_id -> equipe 0 ou 1
        cluster_mhb: index du cluster MHB
        chemin_sortie: chemin du MP4 de sortie
        terrain_config: config terrain (defaut : TerrainConfig())

    Returns:
        Path du fichier genere.

    Raises:
        RuntimeError: si la source ou le writer ne s'ouvrent pas
    """
    config = terrain_config or TerrainConfig()
    chemin_source = Path(chemin_video_source)
    chemin_sortie = Path(chemin_sortie)
    chemin_sortie.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(chemin_source))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la video : {chemin_source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or nb_frames <= 0:
        cap.release()
        raise RuntimeError(f"Metadonnees video invalides : fps={fps}, n={nb_frames}")

    # Hauteur commune broadcast + radar (les deux sont a config.broadcast_h/radar_h)
    if config.broadcast_h != config.radar_h:
        cap.release()
        raise RuntimeError(
            "broadcast_h doit egaler radar_h pour le hstack SBS "
            f"(actuel : {config.broadcast_h} vs {config.radar_h})"
        )
    largeur_sbs = config.broadcast_w + config.radar_w
    hauteur_sbs = config.broadcast_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(chemin_sortie), fourcc, fps, (largeur_sbs, hauteur_sbs))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Impossible d'ouvrir VideoWriter : {chemin_sortie}")

    # Index inverse : frame -> liste (tid, x_m, y_m) pour le radar
    positions_par_frame: dict[int, list[tuple[int, float, float]]] = {}
    for tid, positions in positions_par_tracker_interpolees.items():
        for p in positions:
            positions_par_frame.setdefault(p.frame_idx, []).append((tid, p.x_m, p.y_m))

    try:
        for fi in range(nb_frames):
            ret, frame = cap.read()
            if not ret:
                break
            broadcast = cv2.resize(frame, (config.broadcast_w, config.broadcast_h))

            # Annoter les boxes (seulement sur les frames echantillonnees)
            if fi in detections_trackees:
                dets = detections_trackees[fi]
                if dets.tracker_id is not None and len(dets) > 0:
                    sx = config.broadcast_w / float(frame.shape[1])
                    sy = config.broadcast_h / float(frame.shape[0])
                    for i in range(len(dets)):
                        tid = int(dets.tracker_id[i])
                        couleur = couleur_tracker(
                            tid, classe_finale, equipe_finale, cluster_mhb
                        )
                        x1, y1, x2, y2 = dets.xyxy[i]
                        p1 = (int(x1 * sx), int(y1 * sy))
                        p2 = (int(x2 * sx), int(y2 * sy))
                        cv2.rectangle(broadcast, p1, p2, couleur, 2)
                        cv2.putText(
                            broadcast,
                            str(tid),
                            (p1[0], max(0, p1[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            couleur,
                            1,
                            cv2.LINE_AA,
                        )

            # Radar : positions de la frame courante (interpolees incluses)
            positions_radar: list[tuple[float, float, tuple[int, int, int], str, int]] = []
            for tid, x_m, y_m in positions_par_frame.get(fi, []):
                couleur = couleur_tracker(tid, classe_finale, equipe_finale, cluster_mhb)
                positions_radar.append((x_m, y_m, couleur, "", tid))
            radar = dessiner_radar(positions_radar, config)

            sbs = np.hstack([broadcast, radar])
            writer.write(sbs)
    finally:
        cap.release()
        writer.release()

    logger.info("Video SBS generee : %s (%dx%d @ %.1f fps)",
                chemin_sortie, largeur_sbs, hauteur_sbs, fps)
    return chemin_sortie


# ---------------------------------------------------------------------------
# Orchestrateur principal
# ---------------------------------------------------------------------------


def _lire_metadonnees_video(chemin: Path) -> tuple[float, int]:
    """Lit fps et nb_frames d'une video."""
    cap = cv2.VideoCapture(str(chemin))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la video : {chemin}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps <= 0 or nb <= 0:
        raise RuntimeError(f"Metadonnees video invalides : fps={fps}, n={nb}")
    return fps, nb


def _extraire_frames_indexees(
    chemin: Path, indices: list[int]
) -> dict[int, np.ndarray]:
    """Extrait les frames aux indices donnes en une passe sequentielle.

    Iteration unique avec cap.read() et collecte des frames dont l'index
    est dans `indices`. Evite les `cap.set(CAP_PROP_POS_FRAMES, ...)`
    qui forcent un re-decode depuis le keyframe precedent (tres lent
    sur MP4 H.264).
    """
    cap = cv2.VideoCapture(str(chemin))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la video : {chemin}")
    indices_set = set(int(i) for i in indices)
    if not indices_set:
        cap.release()
        return {}
    max_idx = max(indices_set)
    frames: dict[int, np.ndarray] = {}
    try:
        fi = 0
        while fi <= max_idx:
            ret, frame = cap.read()
            if not ret:
                break
            if fi in indices_set:
                frames[fi] = frame
            fi += 1
    finally:
        cap.release()
    return frames


def traiter_match_complet(
    chemin_video: str | Path,
    correspondances_homographie: dict[str, Correspondance],
    dossier_sortie: str | Path,
    subsample: int = 2,
    generer_video_radar: bool = True,
    decouper_actions: bool = True,
    detecteur: _ProtocoleDetecteur | None = None,
    modele_config: ModeleConfig | None = None,
    terrain_config: TerrainConfig | None = None,
    batch_size: int = 8,
    min_frames_track: int = 0,
) -> ResultatPipeline:
    """Pipeline complet : video -> stats + clips d'actions + video radar SBS.

    Args:
        chemin_video: video source
        correspondances_homographie: points pour calibrer l'homographie
        dossier_sortie: dossier ou ecrire stats, video radar, clips
        subsample: 1 frame sur N pour la detection (2 par defaut)
        generer_video_radar: si True, genere la video annotee + radar SBS
        decouper_actions: si True, decoupe les actions detectees via ffmpeg
        detecteur: detecteur injecte (defaut : DetecteurLocal(modele_config)).
            Permet d'injecter un stub pour les tests.
        modele_config: config du detecteur si on instancie par defaut
        terrain_config: config terrain (defaut : TerrainConfig())
        batch_size: taille de batch GPU pour la detection
        min_frames_track: filtre les trackers vus sur moins de N frames reelles
            (fragments). 0 desactive le filtre. Recommande ~8-12 en production
            pour ne garder que les trajectoires exploitables.

    Returns:
        ResultatPipeline avec tous les artefacts produits.

    Raises:
        FileNotFoundError: si la video source n'existe pas
    """
    chemin_source = Path(chemin_video)
    if not chemin_source.exists():
        raise FileNotFoundError(f"Video introuvable : {chemin_source}")

    sortie = Path(dossier_sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    config = terrain_config or TerrainConfig()

    # 1. Homographie
    homographie = calibrer_homographie(correspondances_homographie)
    logger.info(
        "Homographie calibree : %d points, methode=%s",
        homographie.nb_points,
        homographie.methode,
    )

    # 2. Metadonnees video
    fps, nb_frames_total = _lire_metadonnees_video(chemin_source)
    estimation_s = nb_frames_total / max(subsample, 1) / 25.0  # ~25 inf/s sur T4
    logger.info(
        "Video : %d frames @ %.1f fps. Estimation detection (T4) : ~%.1fs",
        nb_frames_total,
        fps,
        estimation_s,
    )

    # 3. Detection (injectable pour tests)
    if detecteur is None:
        from pivot_ai.detection import DetecteurLocal
        detecteur = DetecteurLocal(modele_config)
    detections_par_frame = detecter_video(
        chemin_source, detecteur, subsample=subsample, batch_size=batch_size
    )

    # 4. Tracking (framerate corrige du subsample pour le modele de mouvement)
    detections_trackees, classe_finale = tracker_detections(
        detections_par_frame, fps=fps, subsample=subsample
    )

    # 5. Equipes (necessite les frames echantillonnees pour extraire les couleurs)
    frames_echantillonnees = _extraire_frames_indexees(
        chemin_source, list(detections_trackees.keys())
    )
    equipe_finale, cluster_mhb = classifier_equipes(
        detections_trackees,
        frames_echantillonnees,
        classe_finale,
        id_classe_joueur=CLASSES_HANDBALL["players"],
    )

    # 6. Projection terrain
    positions_par_tracker = projeter_detections_en_terrain(
        detections_trackees,
        classe_finale,
        homographie,
        terrain_config=config,
    )

    # 6bis. Filtre des fragments courts (traine de tracks ephemeres)
    nb_trackers_avant_filtre = len(positions_par_tracker)
    positions_par_tracker = filtrer_traceurs_courts(
        positions_par_tracker, min_frames=min_frames_track
    )

    # 7. Interpolation (pour rendu radar uniquement)
    max_gap = max(2, subsample * 5)
    positions_interpolees = interpoler_positions(
        positions_par_tracker, max_gap_frames=max_gap
    )

    # 8. Stats joueurs (positions reelles uniquement)
    stats_df = calculer_stats_joueur(positions_par_tracker, fps=fps)
    chemin_csv = sortie / "stats_joueurs.csv"
    chemin_parquet = sortie / "stats_joueurs.parquet"
    stats_df.write_csv(chemin_csv)
    stats_df.write_parquet(chemin_parquet)
    logger.info("Stats sauvegardees : %s / %s", chemin_csv, chemin_parquet)

    # Largeur bloc defensif pour les 2 equipes (si on a une classification)
    if equipe_finale:
        for eq_idx in (0, 1):
            df_bloc = calculer_largeur_bloc_defensif(
                positions_par_tracker,
                equipe_finale,
                equipe_defendante=eq_idx,
                fps=fps,
            )
            suffixe = "mhb" if eq_idx == cluster_mhb else "adv"
            chemin_bloc = sortie / f"largeur_bloc_defensif_{suffixe}.csv"
            df_bloc.write_csv(chemin_bloc)

    # 9. Video radar SBS
    chemin_video_radar: Path | None = None
    if generer_video_radar:
        chemin_video_radar = sortie / "match_radar_sbs.mp4"
        generer_video_sbs(
            chemin_source,
            detections_trackees,
            positions_interpolees,
            classe_finale,
            equipe_finale,
            cluster_mhb,
            chemin_video_radar,
            terrain_config=config,
        )

    # 10. Decoupage actions (sur les detections brutes, pas les interpolees)
    actions: list[Action] = []
    clips: list[Path] = []
    if decouper_actions:
        actions = detecter_actions(
            detections_trackees,
            classe_finale,
            id_classe_joueur=CLASSES_HANDBALL["players"],
            fps=fps,
        )
        if actions:
            try:
                clips = decouper_clips_video(
                    chemin_source, actions, sortie / "actions"
                )
            except FileNotFoundError as exc:
                logger.warning(
                    "Decoupage clips ignore (ffmpeg manquant) : %s", exc
                )

    metadonnees = {
        "fps": fps,
        "nb_frames_total": nb_frames_total,
        "subsample": subsample,
        "max_gap_interpolation": max_gap,
        "homographie_methode": homographie.methode,
        "homographie_nb_points": homographie.nb_points,
        "cluster_mhb": cluster_mhb,
        "nb_trackers_total": len(classe_finale),
        "nb_joueurs_classes": len(equipe_finale),
        "min_frames_track": min_frames_track,
        "nb_joueurs_avant_filtre": nb_trackers_avant_filtre,
        "nb_joueurs_stats": len(positions_par_tracker),
        "nb_actions_detectees": len(actions),
    }
    logger.info("Pipeline termine. Metadonnees : %s", metadonnees)

    return ResultatPipeline(
        chemin_video_source=chemin_source,
        chemin_video_radar=chemin_video_radar,
        stats_joueurs=stats_df,
        positions_par_tracker=positions_par_tracker,
        actions_detectees=actions,
        clips_decoupes=clips,
        metadonnees=metadonnees,
    )
