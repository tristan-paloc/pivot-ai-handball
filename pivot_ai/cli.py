"""CLI pour pivot-ai. Usage : `pivot-ai traiter --video X.mp4 --output dossier/`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pivot_ai.config import configurer_logging

logger = logging.getLogger(__name__)


def commande_traiter(args: argparse.Namespace) -> int:
    """Commande `pivot-ai traiter`."""
    chemin_video = Path(args.video)
    if not chemin_video.exists():
        logger.error("Video introuvable : %s", chemin_video)
        return 1

    dossier_sortie = Path(args.output)
    dossier_sortie.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        "CLI a finaliser par Claude Code. Doit appeler "
        "pivot_ai.pipeline.traiter_match_complet avec les correspondances "
        "d'homographie chargees depuis un fichier JSON/TOML passe en argument."
    )


def main() -> int:
    """Point d'entree CLI."""
    configurer_logging()

    parser = argparse.ArgumentParser(
        prog="pivot-ai",
        description="Pipeline scouting handball : detection, tracking, stats, decoupage.",
    )
    sous = parser.add_subparsers(dest="commande", required=True)

    p_traiter = sous.add_parser("traiter", help="Traite une video et genere stats + clips.")
    p_traiter.add_argument("--video", required=True, help="Chemin vers le MP4 a traiter")
    p_traiter.add_argument("--output", required=True, help="Dossier de sortie")
    p_traiter.add_argument("--homographie", help="Fichier JSON avec les correspondances homographie")
    p_traiter.add_argument("--subsample", type=int, default=2, help="1 frame sur N (defaut 2)")
    p_traiter.set_defaults(func=commande_traiter)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
