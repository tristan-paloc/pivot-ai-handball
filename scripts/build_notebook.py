"""Genere le notebook Colab pivot-ai-handball a partir de cellules definies en Python.

Permet de tenir le contenu en code Python lisible plutot que de hand-editer le JSON
.ipynb. Idempotent : peut etre relance.

Usage : python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "colab_pivot_ai.ipynb"


def md(source: str) -> dict:
    """Cree une cellule markdown."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]],
    }


def code(source: str) -> dict:
    """Cree une cellule code."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]],
    }


CELLULES = [
    md(
        "# pivot-ai-handball — Notebook Colab\n"
        "\n"
        "Notebook \"thin\" : clone le repo, installe les deps, monte Drive et appelle le module Python.\n"
        "Aucune logique metier ici. Tout est dans `pivot_ai/`.\n"
        "\n"
        "**Prerequis :** runtime GPU (Edition > Parametres du notebook > Accelerateur materiel = T4 GPU)."
    ),
    md(
        "## 1. Authentification GitHub (repo prive)\n"
        "\n"
        "Le repo `tristan-paloc/pivot-ai-handball` est prive : il faut un **Personal Access Token (PAT)** GitHub pour le cloner depuis Colab.\n"
        "\n"
        "### Setup (a faire une fois)\n"
        "\n"
        "1. Generer un PAT sur GitHub :\n"
        "   - Va sur https://github.com/settings/tokens?type=beta (fine-grained, recommande)\n"
        "   - **Resource owner** : `tristan-paloc`\n"
        "   - **Repository access** : *Only select repositories* → `pivot-ai-handball`\n"
        "   - **Permissions** : Repository permissions → *Contents : Read-only*\n"
        "   - Genere et copie le token (ne sera plus jamais affiche).\n"
        "2. Dans Colab : icone **clef** (Secrets) dans la barre laterale gauche.\n"
        "   - **Name** : `GITHUB_TOKEN`\n"
        "   - **Value** : colle ton PAT\n"
        "   - Coche **Notebook access** pour autoriser ce notebook a le lire.\n"
        "\n"
        "Le secret est stocke chiffre cote Google et n'apparait jamais dans le notebook."
    ),
    md("## 2. Clone repo + install"),
    code(
        "import os, sys, subprocess\n"
        "from google.colab import userdata\n"
        "\n"
        "REPO_OWNER = \"tristan-paloc\"\n"
        "REPO_NAME  = \"pivot-ai-handball\"\n"
        "BRANCH     = \"main\"\n"
        "REPO_DIR   = f\"/content/{REPO_NAME}\"\n"
        "\n"
        "try:\n"
        "    GITHUB_TOKEN = userdata.get(\"GITHUB_TOKEN\")\n"
        "except userdata.SecretNotFoundError as exc:\n"
        "    raise RuntimeError(\n"
        "        \"GITHUB_TOKEN absent des Secrets Colab. \"\n"
        "        \"Cree-le en suivant les instructions de la cellule precedente.\"\n"
        "    ) from exc\n"
        "\n"
        "REPO_URL = f\"https://{GITHUB_TOKEN}@github.com/{REPO_OWNER}/{REPO_NAME}.git\"\n"
        "\n"
        "if not os.path.exists(REPO_DIR):\n"
        "    res = subprocess.run(\n"
        "        [\"git\", \"clone\", \"--branch\", BRANCH, \"--depth\", \"1\", REPO_URL, REPO_DIR],\n"
        "        capture_output=True, text=True,\n"
        "    )\n"
        "    if res.returncode != 0:\n"
        "        # Ne jamais logger le token : on masque l'URL dans l'erreur.\n"
        "        msg = res.stderr.replace(GITHUB_TOKEN, \"***\")\n"
        "        raise RuntimeError(f\"Echec git clone :\\n{msg}\")\n"
        "    print(f\"Repo clone dans {REPO_DIR}\")\n"
        "else:\n"
        "    res = subprocess.run(\n"
        "        [\"git\", \"-C\", REPO_DIR, \"pull\"], capture_output=True, text=True\n"
        "    )\n"
        "    if res.returncode != 0:\n"
        "        msg = res.stderr.replace(GITHUB_TOKEN, \"***\")\n"
        "        raise RuntimeError(f\"Echec git pull :\\n{msg}\")\n"
        "    print(f\"Repo a jour dans {REPO_DIR}\")\n"
        "\n"
        "%cd {REPO_DIR}\n"
        "!pip install -q -e \".[dev]\"\n"
        "\n"
        "if REPO_DIR not in sys.path:\n"
        "    sys.path.insert(0, REPO_DIR)\n"
        "\n"
        "import pivot_ai\n"
        "print(f\"pivot_ai version : {pivot_ai.__version__}\")"
    ),
    md("## 3. Verifier GPU"),
    code(
        "import torch\n"
        "print(f\"CUDA dispo : {torch.cuda.is_available()}\")\n"
        "if torch.cuda.is_available():\n"
        "    print(f\"GPU : {torch.cuda.get_device_name(0)}\")\n"
        "    print(f\"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} Go\")"
    ),
    md("## 4. Monter Google Drive"),
    code(
        "from google.colab import drive\n"
        "drive.mount('/content/drive', force_remount=False)\n"
        "\n"
        "DRIVE_ROOT  = \"/content/drive/MyDrive/PIVOT_AI\"\n"
        "CLIPS_DIR   = f\"{DRIVE_ROOT}/raw_clips\"\n"
        "OUTPUTS_DIR = f\"{DRIVE_ROOT}/outputs\"\n"
        "os.makedirs(OUTPUTS_DIR, exist_ok=True)\n"
        "\n"
        "!ls {CLIPS_DIR}"
    ),
    md(
        "## 5. Choisir un clip et relever les points d'homographie\n"
        "\n"
        "Affiche une frame du clip. **Survole avec la souris** pour relever les pixels des points caracteristiques du terrain.\n"
        "\n"
        "**Points a relever (au minimum 4, idealement 6-8 pour activer RANSAC) :**\n"
        "\n"
        "| Nom | Description | Coords terrain (m) |\n"
        "|-----|-------------|---------------------|\n"
        "| A | Coin terrain haut-gauche (proche bas image) | (0, 0) |\n"
        "| B | Coin terrain haut-droite | (40, 0) |\n"
        "| C | Coin terrain bas-droite (proche haut image) | (40, 20) |\n"
        "| D | Coin terrain bas-gauche | (0, 20) |\n"
        "| E | Ligne mediane cote bas | (20, 0) |\n"
        "| F | Ligne mediane cote haut | (20, 20) |\n"
        "| M_prime | Poteau de but gauche, cote bas | (0, 8.5) |\n"
        "| N_prime | Poteau de but gauche, cote haut | (0, 11.5) |\n"
        "| M | Poteau de but droit, cote bas | (40, 8.5) |\n"
        "| N | Poteau de but droit, cote haut | (40, 11.5) |\n"
        "| K | Sommet demi-cercle 6m gauche | (6, 10) |\n"
        "| L | Sommet demi-cercle 6m droite | (34, 10) |\n"
        "| O_prime | Sommet arc 9m gauche | (9, 10) |\n"
        "| O | Sommet arc 9m droite | (31, 10) |\n"
        "\n"
        "Voir `pivot_ai/config.py` (POINTS_TERRAIN_M) pour la liste complete."
    ),
    code(
        "import cv2\n"
        "import plotly.express as px\n"
        "\n"
        "CLIP = f\"{CLIPS_DIR}/attaque_mhb_stable_58s.mp4\"\n"
        "\n"
        "cap = cv2.VideoCapture(CLIP)\n"
        "total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n"
        "fps = cap.get(cv2.CAP_PROP_FPS)\n"
        "FRAME_REF = total // 2\n"
        "cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_REF)\n"
        "ret, frame_ref = cap.read()\n"
        "cap.release()\n"
        "\n"
        "print(f\"Clip : {total} frames @ {fps:.1f} fps ({total/fps:.1f}s)\")\n"
        "\n"
        "fig = px.imshow(cv2.cvtColor(frame_ref, cv2.COLOR_BGR2RGB))\n"
        "fig.update_layout(width=1300, height=730, dragmode='pan',\n"
        "                  title=f\"Frame {FRAME_REF} — survole pour relever les pixels\")\n"
        "fig.show()"
    ),
    code(
        "# A remplir avec les pixels releves sur la frame ci-dessus.\n"
        "# Plus tu mets de points (>= 5), plus l'estimation RANSAC est robuste.\n"
        "correspondances = {\n"
        "    \"C\":       {\"pixel\": (879, 319),  \"terrain_m\": (40, 20)},\n"
        "    \"J\":       {\"pixel\": (974, 355),  \"terrain_m\": (40, 16)},\n"
        "    \"N\":       {\"pixel\": (1250, 454), \"terrain_m\": (40, 11.5)},\n"
        "    \"M\":       {\"pixel\": (1426, 524), \"terrain_m\": (40, 8.5)},\n"
        "    \"I\":       {\"pixel\": (1889, 694), \"terrain_m\": (40, 4)},\n"
        "    \"L\":       {\"pixel\": (837, 571),  \"terrain_m\": (34, 10)},\n"
        "    \"O\":       {\"pixel\": (584, 641),  \"terrain_m\": (31, 10)},\n"
        "}\n"
        "print(f\"{len(correspondances)} correspondances renseignees\")"
    ),
    md(
        "## 6. Test rapide sur clip court\n"
        "\n"
        "Avant le pipeline complet, on lance une passe rapide pour valider l'enchainement :\n"
        "- subsample agressif (1 frame sur 5)\n"
        "- pas de video radar ni decoupage : on veut juste verifier que la chaine tient debout\n"
        "\n"
        "Sur un clip de 60s a 25fps : ~300 inferences au lieu de 1500, soit ~10-15s sur T4."
    ),
    code(
        "from pivot_ai.pipeline import traiter_match_complet\n"
        "import time\n"
        "\n"
        "DOSSIER_TEST = f\"{OUTPUTS_DIR}/_test_rapide\"\n"
        "\n"
        "t0 = time.time()\n"
        "resultat_test = traiter_match_complet(\n"
        "    chemin_video=CLIP,\n"
        "    correspondances_homographie=correspondances,\n"
        "    dossier_sortie=DOSSIER_TEST,\n"
        "    subsample=5,\n"
        "    generer_video_radar=False,\n"
        "    decouper_actions=False,\n"
        ")\n"
        "duree = time.time() - t0\n"
        "\n"
        "print(f\"\\nTest rapide OK en {duree:.1f}s\")\n"
        "print(f\"Trackers detectes : {resultat_test.metadonnees['nb_trackers_total']}\")\n"
        "print(f\"Joueurs classes en equipe : {resultat_test.metadonnees['nb_joueurs_classes']}\")\n"
        "print(f\"Methode homographie : {resultat_test.metadonnees['homographie_methode']}\")\n"
        "print(f\"Stats (10 premieres lignes) :\")\n"
        "print(resultat_test.stats_joueurs.head(10))"
    ),
    md(
        "## 7. Pipeline complet\n"
        "\n"
        "Avec `subsample=2` (1 frame sur 2), on a une bonne qualite de tracking pour ~1500 inferences/min sur T4.\n"
        "\n"
        "**Estimation temps** (vidant les caches GPU avant) :"
    ),
    code(
        "nb_inferences = total // 2  # subsample=2\n"
        "vitesse_t4 = 25  # inferences/s en moyenne sur T4 avec YOLOv8m\n"
        "duree_estimee_min = nb_inferences / vitesse_t4 / 60\n"
        "print(f\"Estimation : ~{duree_estimee_min:.1f} min de detection sur T4\")\n"
        "print(f\"+ tracking + classification + video SBS + decoupage ≈ +20-30%\")\n"
        "print(f\"Total estime : {duree_estimee_min * 1.3:.1f} min\")"
    ),
    code(
        "import torch\n"
        "if torch.cuda.is_available():\n"
        "    torch.cuda.empty_cache()\n"
        "\n"
        "t0 = time.time()\n"
        "resultat = traiter_match_complet(\n"
        "    chemin_video=CLIP,\n"
        "    correspondances_homographie=correspondances,\n"
        "    dossier_sortie=OUTPUTS_DIR,\n"
        "    subsample=2,\n"
        "    generer_video_radar=True,\n"
        "    decouper_actions=True,\n"
        ")\n"
        "duree_min = (time.time() - t0) / 60\n"
        "\n"
        "print(f\"\\nPipeline complet termine en {duree_min:.1f} min\")\n"
        "\n"
        "print(\"\\n=== STATS PAR JOUEUR ===\")\n"
        "print(resultat.stats_joueurs)\n"
        "\n"
        "print(f\"\\n=== ACTIONS DETECTEES : {len(resultat.actions_detectees)} ===\")\n"
        "for i, action in enumerate(resultat.actions_detectees):\n"
        "    print(f\"  Action {i+1}: frames {action.frame_debut}-{action.frame_fin} \"\n"
        "          f\"({action.duree_s:.1f}s, {action.nb_joueurs_moyen:.1f} joueurs moyens)\")\n"
        "\n"
        "print(f\"\\n=== CLIPS DECOUPES : {len(resultat.clips_decoupes)} ===\")\n"
        "for clip in resultat.clips_decoupes:\n"
        "    print(f\"  {clip}\")\n"
        "\n"
        "print(f\"\\n=== ARTEFACTS ===\")\n"
        "print(f\"  CSV  : {OUTPUTS_DIR}/stats_joueurs.csv\")\n"
        "print(f\"  Parquet : {OUTPUTS_DIR}/stats_joueurs.parquet\")\n"
        "print(f\"  Video radar SBS : {resultat.chemin_video_radar}\")"
    ),
    md("## 8. Visualiser quelques frames du rendu final"),
    code(
        "import matplotlib.pyplot as plt\n"
        "\n"
        "video_radar = resultat.chemin_video_radar\n"
        "cap = cv2.VideoCapture(str(video_radar))\n"
        "total_radar = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n"
        "\n"
        "fig, axes = plt.subplots(5, 1, figsize=(20, 22))\n"
        "for ax, idx in zip(axes, [0, total_radar//4, total_radar//2, 3*total_radar//4, total_radar-1]):\n"
        "    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)\n"
        "    ret, frame = cap.read()\n"
        "    if ret:\n"
        "        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))\n"
        "        ax.set_title(f\"Frame {idx}/{total_radar}\")\n"
        "        ax.axis('off')\n"
        "cap.release()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 9. Heatmap d'un joueur\n"
        "\n"
        "Visualise la densite de presence d'un joueur sur le terrain. On prend par defaut le tracker le plus present."
    ),
    code(
        "import polars as pl\n"
        "import plotly.express as px\n"
        "from pivot_ai.stats import generer_heatmap_joueur\n"
        "\n"
        "# Tracker le plus present dans le clip\n"
        "stats_tries = resultat.stats_joueurs.sort(\"nb_frames_detectees\", descending=True)\n"
        "tracker_choisi = int(stats_tries[\"tracker_id\"][0])\n"
        "print(f\"Tracker selectionne : {tracker_choisi} \"\n"
        "      f\"({stats_tries['nb_frames_detectees'][0]} frames detectees)\")\n"
        "print(stats_tries.filter(pl.col(\"tracker_id\") == tracker_choisi))\n"
        "\n"
        "# Heatmap reelle a partir des positions terrain stockees dans ResultatPipeline\n"
        "positions_joueur = resultat.positions_par_tracker.get(tracker_choisi, [])\n"
        "heatmap = generer_heatmap_joueur(positions_joueur, resolution=0.5, sigma_lissage=1.5)\n"
        "\n"
        "fig = px.imshow(\n"
        "    heatmap,\n"
        "    origin=\"lower\",\n"
        "    aspect=\"equal\",\n"
        "    color_continuous_scale=\"Hot\",\n"
        "    title=f\"Heatmap presence joueur tracker_id={tracker_choisi}\",\n"
        "    labels={\"x\": \"x_terrain (cellules de 0.5m)\", \"y\": \"y_terrain (cellules de 0.5m)\"},\n"
        ")\n"
        "fig.update_layout(width=900, height=500)\n"
        "fig.show()"
    ),
    md(
        "## 10. Largeur du bloc defensif au cours du temps\n"
        "\n"
        "Charge le DataFrame de la largeur du bloc defensif (MHB ou ADV) et le trace via plotly."
    ),
    code(
        "import polars as pl\n"
        "import plotly.express as px\n"
        "\n"
        "chemin_bloc_mhb = f\"{OUTPUTS_DIR}/largeur_bloc_defensif_mhb.csv\"\n"
        "chemin_bloc_adv = f\"{OUTPUTS_DIR}/largeur_bloc_defensif_adv.csv\"\n"
        "\n"
        "if os.path.exists(chemin_bloc_mhb):\n"
        "    df_mhb = pl.read_csv(chemin_bloc_mhb).with_columns(equipe=pl.lit(\"MHB\"))\n"
        "    df_adv = pl.read_csv(chemin_bloc_adv).with_columns(equipe=pl.lit(\"ADV\"))\n"
        "    df = pl.concat([df_mhb, df_adv])\n"
        "    fig = px.line(\n"
        "        df.to_pandas(),\n"
        "        x=\"temps_s\", y=\"largeur_y_m\", color=\"equipe\",\n"
        "        title=\"Largeur du bloc defensif au cours du temps\",\n"
        "        labels={\"temps_s\": \"Temps (s)\", \"largeur_y_m\": \"Largeur (m)\"},\n"
        "    )\n"
        "    fig.show()\n"
        "else:\n"
        "    print(\"Pas de fichier largeur_bloc_defensif_*.csv : classification d'equipes a peut-etre echoue (trop peu de joueurs ?)\")"
    ),
]


def construire_notebook() -> dict:
    return {
        "cells": CELLULES,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    notebook = construire_notebook()
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Notebook ecrit : {NOTEBOOK_PATH} ({len(CELLULES)} cellules)")


if __name__ == "__main__":
    main()
