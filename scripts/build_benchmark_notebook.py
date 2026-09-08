"""Genere le notebook Colab de benchmark de continuite d'ID.

Clone la branche feat/benchmark-tracking, installe, monte Drive, lance
`pivot-ai benchmark` sur les 3 clips, affiche le comparatif + des frames annotees.

Usage : python scripts/build_benchmark_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "benchmark_colab.ipynb"
BRANCHE = "feat/benchmark-tracking"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]],
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")[:-1]] + [source.split("\n")[-1]],
    }


CELLULES = [
    md(
        "# Benchmark continuite d'identite (tracking)\n"
        "\n"
        "Compare **ByteTrack** vs **BoT-SORT + ReID** sur des clips handball, avec le\n"
        "modele fine-tune. Produit : videos annotees (IDs colores + coupures de plan),\n"
        "`comparatif.csv/json` et `RAPPORT.md`.\n"
        "\n"
        "**Prerequis** : runtime **T4 GPU**, et les 3 clips deposes sur Drive dans\n"
        "`/MyDrive/PIVOT_AI/benchmark/`."
    ),
    md("## 1. Clone (branche benchmark) + install"),
    code(
        "import os, subprocess, sys\n"
        "\n"
        "REPO_OWNER = \"tristan-paloc\"\n"
        "REPO_NAME  = \"pivot-ai-handball\"\n"
        f"BRANCHE    = \"{BRANCHE}\"\n"
        "REPO_DIR   = f\"/content/{REPO_NAME}\"\n"
        "REPO_URL   = f\"https://github.com/{REPO_OWNER}/{REPO_NAME}.git\"\n"
        "\n"
        "if not os.path.exists(REPO_DIR):\n"
        "    res = subprocess.run([\"git\", \"clone\", \"--branch\", BRANCHE, \"--depth\", \"1\",\n"
        "                          REPO_URL, REPO_DIR], capture_output=True, text=True)\n"
        "    if res.returncode != 0:\n"
        "        raise RuntimeError(res.stderr)\n"
        "else:\n"
        "    subprocess.run([\"git\", \"-C\", REPO_DIR, \"fetch\", \"origin\", BRANCHE],\n"
        "                   capture_output=True, text=True)\n"
        "    subprocess.run([\"git\", \"-C\", REPO_DIR, \"checkout\", BRANCHE],\n"
        "                   capture_output=True, text=True)\n"
        "    subprocess.run([\"git\", \"-C\", REPO_DIR, \"pull\"], capture_output=True, text=True)\n"
        "\n"
        "%cd {REPO_DIR}\n"
        "!pip install -q -e .\n"
        "if REPO_DIR not in sys.path:\n"
        "    sys.path.insert(0, REPO_DIR)\n"
        "import pivot_ai; print(f\"pivot_ai version : {pivot_ai.__version__}\")"
    ),
    md("## 2. Verifier le GPU"),
    code(
        "import torch\n"
        "print(f\"CUDA dispo : {torch.cuda.is_available()}\")\n"
        "if torch.cuda.is_available():\n"
        "    print(f\"GPU : {torch.cuda.get_device_name(0)}\")\n"
        "else:\n"
        "    raise RuntimeError(\"Active le GPU : Execution > Modifier le type d'execution > T4 GPU\")"
    ),
    md("## 3. Monter Google Drive"),
    code(
        "from google.colab import drive\n"
        "drive.mount('/content/drive', force_remount=False)\n"
        "\n"
        "DOSSIER_CLIPS  = \"/content/drive/MyDrive/PIVOT_AI/benchmark\"\n"
        "DOSSIER_SORTIE = \"/content/drive/MyDrive/PIVOT_AI/benchmark_out\"\n"
        "MODELE         = \"/content/drive/MyDrive/PIVOT_AI/models/handball_yolov8m.pt\"\n"
        "\n"
        "assert os.path.isdir(DOSSIER_CLIPS), f\"Depose les clips dans {DOSSIER_CLIPS}\"\n"
        "assert os.path.exists(MODELE), f\"Modele introuvable : {MODELE}\"\n"
        "!ls -la {DOSSIER_CLIPS}"
    ),
    md(
        "## 4. Lancer le benchmark\n"
        "\n"
        "ByteTrack vs BoT-SORT sur les 3 clips. Compter quelques minutes sur T4\n"
        "(BoT-SORT/ReID est le plus lent)."
    ),
    code(
        "!python -m pivot_ai.cli benchmark --clips {DOSSIER_CLIPS} "
        "--sortie {DOSSIER_SORTIE} --modele {MODELE} "
        "--trackers bytetrack,botsort --subsample 2"
    ),
    md("## 5. Comparatif chiffre"),
    code(
        "import polars as pl\n"
        "df = pl.read_csv(f\"{DOSSIER_SORTIE}/comparatif.csv\")\n"
        "print(df)\n"
        "print(\"\\n=== RAPPORT ===\\n\")\n"
        "print(open(f\"{DOSSIER_SORTIE}/RAPPORT.md\", encoding=\"utf-8\").read())"
    ),
    md(
        "## 6. Verification visuelle\n"
        "\n"
        "Quelques frames d'une video annotee (couleur = ID ; si la couleur d'un joueur\n"
        "change sans coupure de plan, c'est une casse d'ID)."
    ),
    code(
        "import glob\n"
        "import cv2\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "videos = sorted(glob.glob(f\"{DOSSIER_SORTIE}/*.mp4\"))\n"
        "print(\"Videos annotees :\")\n"
        "for v in videos:\n"
        "    print(\"  \", v)\n"
        "\n"
        "if videos:\n"
        "    cap = cv2.VideoCapture(videos[0])\n"
        "    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n"
        "    fig, axes = plt.subplots(2, 2, figsize=(18, 10))\n"
        "    for ax, frac in zip(axes.flat, [0.2, 0.45, 0.7, 0.9]):\n"
        "        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))\n"
        "        ok, frame = cap.read()\n"
        "        if ok:\n"
        "            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))\n"
        "            ax.set_title(os.path.basename(videos[0]))\n"
        "            ax.axis('off')\n"
        "    cap.release()\n"
        "    plt.tight_layout(); plt.show()"
    ),
]


def main() -> None:
    notebook = {
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
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Notebook ecrit : {NOTEBOOK_PATH} ({len(CELLULES)} cellules)")


if __name__ == "__main__":
    main()
