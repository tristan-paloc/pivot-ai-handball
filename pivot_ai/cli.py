"""CLI pour pivot-ai. Usage : `pivot-ai traiter --video X.mp4 --output dossier/`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from pivot_ai.config import configurer_logging
from pivot_ai.homographie import Correspondance
from pivot_ai.pipeline import ResultatPipeline, traiter_match_complet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chargement des correspondances homographie depuis JSON
# ---------------------------------------------------------------------------


def _charger_correspondances(chemin: Path) -> dict[str, Correspondance]:
    """Charge des correspondances pixel <-> terrain depuis un fichier JSON.

    Format attendu :
        {
          "A": {"pixel": [x, y], "terrain_m": [X, Y]},
          "B": {"pixel": [...], "terrain_m": [...]},
          ...
        }
    Minimum 4 points pour calibrer une homographie.

    Args:
        chemin: chemin vers le fichier JSON

    Returns:
        dict nom_point -> Correspondance (avec tuples pour pixel et terrain_m)

    Raises:
        FileNotFoundError: si le fichier n'existe pas
        ValueError: si le format est invalide (< 4 points ou structure KO)
        json.JSONDecodeError: si le JSON est mal forme
    """
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier homographie introuvable : {chemin}")

    with chemin.open("r", encoding="utf-8") as f:
        brut: Any = json.load(f)

    if not isinstance(brut, dict):
        raise ValueError(
            f"Fichier homographie : racine doit etre un objet JSON, recu {type(brut).__name__}"
        )

    # Les cles commencant par "_" sont des commentaires/metadonnees, ignorees.
    points_brut = {k: v for k, v in brut.items() if not k.startswith("_")}
    if len(points_brut) < 4:
        raise ValueError(
            f"Fichier homographie : minimum 4 points requis, recu {len(points_brut)}"
        )

    correspondances: dict[str, Correspondance] = {}
    for nom, donnees in points_brut.items():
        if not isinstance(donnees, dict):
            raise ValueError(f"Point '{nom}' : doit etre un objet avec 'pixel' et 'terrain_m'")
        if "pixel" not in donnees or "terrain_m" not in donnees:
            raise ValueError(
                f"Point '{nom}' : cles 'pixel' et 'terrain_m' obligatoires"
            )
        pixel = donnees["pixel"]
        terrain = donnees["terrain_m"]
        if not (isinstance(pixel, list) and len(pixel) == 2):
            raise ValueError(f"Point '{nom}' : 'pixel' doit etre une liste de 2 nombres")
        if not (isinstance(terrain, list) and len(terrain) == 2):
            raise ValueError(f"Point '{nom}' : 'terrain_m' doit etre une liste de 2 nombres")
        correspondances[nom] = {
            "pixel": (int(pixel[0]), int(pixel[1])),
            "terrain_m": (float(terrain[0]), float(terrain[1])),
        }

    return correspondances


# ---------------------------------------------------------------------------
# Affichage du recap final
# ---------------------------------------------------------------------------


def _logger_recap(resultat: ResultatPipeline, dossier_sortie: Path) -> None:
    """Logge un recap des artefacts produits."""
    logger.info("=" * 60)
    logger.info("Pipeline termine. Artefacts dans : %s", dossier_sortie)
    logger.info("-" * 60)
    logger.info("Video source : %s", resultat.chemin_video_source)
    logger.info("Stats joueurs : %s", dossier_sortie / "stats_joueurs.csv")
    logger.info("                %s", dossier_sortie / "stats_joueurs.parquet")
    logger.info("Nb trackers totaux : %d",
                resultat.metadonnees.get("nb_trackers_total", 0))
    logger.info("Nb joueurs classes en equipe : %d",
                resultat.metadonnees.get("nb_joueurs_classes", 0))
    if resultat.chemin_video_radar is not None:
        logger.info("Video radar SBS : %s", resultat.chemin_video_radar)
    else:
        logger.info("Video radar SBS : (non generee)")
    logger.info("Nb actions detectees : %d", len(resultat.actions_detectees))
    logger.info("Nb clips decoupes : %d", len(resultat.clips_decoupes))
    if resultat.clips_decoupes:
        for clip in resultat.clips_decoupes:
            logger.info("  - %s", clip)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Sous-commande `pivot-ai traiter`
# ---------------------------------------------------------------------------


def commande_traiter(args: argparse.Namespace) -> int:
    """Commande `pivot-ai traiter`."""
    chemin_video = Path(args.video)
    if not chemin_video.exists():
        logger.error("Video introuvable : %s", chemin_video)
        return 1

    if not args.homographie:
        logger.error(
            "Argument --homographie obligatoire : "
            "fournir un fichier JSON avec les correspondances pixel <-> terrain."
        )
        return 2

    chemin_homographie = Path(args.homographie)
    try:
        correspondances = _charger_correspondances(chemin_homographie)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        logger.error("Erreur chargement homographie : %s", exc)
        return 1

    dossier_sortie = Path(args.output)
    dossier_sortie.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Lancement pipeline : video=%s, homographie=%d points, subsample=%d",
        chemin_video,
        len(correspondances),
        args.subsample,
    )

    try:
        resultat = traiter_match_complet(
            chemin_video=chemin_video,
            correspondances_homographie=correspondances,
            dossier_sortie=dossier_sortie,
            subsample=args.subsample,
            generer_video_radar=not args.no_video_radar,
            decouper_actions=not args.no_decoupage,
        )
    except Exception:
        logger.exception("Echec du pipeline")
        return 1

    _logger_recap(resultat, dossier_sortie)
    return 0


# ---------------------------------------------------------------------------
# Parser et point d'entree
# ---------------------------------------------------------------------------


def construire_parser() -> argparse.ArgumentParser:
    """Construit le parser CLI (separe du main pour faciliter les tests)."""
    parser = argparse.ArgumentParser(
        prog="pivot-ai",
        description="Pipeline scouting handball : detection, tracking, stats, decoupage.",
    )
    sous = parser.add_subparsers(dest="commande", required=True)

    p_traiter = sous.add_parser(
        "traiter", help="Traite une video et genere stats + clips."
    )
    p_traiter.add_argument(
        "--video", required=True, help="Chemin vers le MP4 a traiter"
    )
    p_traiter.add_argument(
        "--output", required=True, help="Dossier de sortie"
    )
    p_traiter.add_argument(
        "--homographie",
        required=True,
        help="Fichier JSON avec les correspondances pixel <-> terrain (>= 4 points)",
    )
    p_traiter.add_argument(
        "--subsample", type=int, default=2, help="1 frame sur N (defaut 2)"
    )
    p_traiter.add_argument(
        "--no-video-radar",
        action="store_true",
        help="Ne pas generer la video SBS radar (par defaut elle est generee)",
    )
    p_traiter.add_argument(
        "--no-decoupage",
        action="store_true",
        help="Ne pas decouper les actions detectees (par defaut elles le sont)",
    )
    p_traiter.set_defaults(func=commande_traiter)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Point d'entree CLI."""
    configurer_logging()
    parser = construire_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
