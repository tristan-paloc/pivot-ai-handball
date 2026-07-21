"""Genere le notebook Colab d'entrainement du modele YOLO handball.

Fine-tune un YOLOv8 sur le dataset Roboflow handball-detection-fj8rc
(4 classes players/goalkeeper/referees/ball) sur GPU T4, puis exporte
les poids vers Drive pour usage dans le pipeline d'inference.

Usage : python scripts/build_train_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "train_handball_yolo.ipynb"


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
        "# Entrainement modele YOLO handball\n"
        "\n"
        "Fine-tune un YOLOv8 sur le dataset **handball-detection-fj8rc** (Roboflow Universe) :\n"
        "4 classes `players` / `goalkeeper` / `referees` / `ball`.\n"
        "\n"
        "Objectif : remplacer le YOLOv8m COCO generique (qui detecte des 'person' et fragmente\n"
        "le tracking) par un detecteur specialise handball qui distingue joueurs, gardiens,\n"
        "arbitres et ballon.\n"
        "\n"
        "**Sortie** : `best.pt` sauvegarde sur Drive, a passer ensuite a\n"
        "`ModeleConfig.pour_handball(...)` dans le notebook d'inference.\n"
        "\n"
        "**Prerequis** : runtime **T4 GPU** (Execution > Modifier le type d'execution)."
    ),
    md(
        "## 1. Cle API Roboflow\n"
        "\n"
        "Le dataset est sur Roboflow Universe (gratuit). Il faut une cle API :\n"
        "\n"
        "1. Cree un compte gratuit sur https://roboflow.com\n"
        "2. Recupere ta cle : https://app.roboflow.com/settings/api\n"
        "3. Dans Colab : icone **clef** (Secrets) dans la barre laterale gauche\n"
        "   - **Name** : `ROBOFLOW_API_KEY`\n"
        "   - **Value** : colle ta cle\n"
        "   - Active **Notebook access**\n"
        "\n"
        "Le dataset public : https://universe.roboflow.com/handballdetectionvictorcollado/handball-detection-fj8rc"
    ),
    md("## 2. Install"),
    code(
        "!pip install -q ultralytics roboflow\n"
        "\n"
        "import torch\n"
        "print(f\"CUDA dispo : {torch.cuda.is_available()}\")\n"
        "if torch.cuda.is_available():\n"
        "    print(f\"GPU : {torch.cuda.get_device_name(0)}\")\n"
        "    print(f\"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} Go\")\n"
        "else:\n"
        "    raise RuntimeError(\"Pas de GPU : Execution > Modifier le type d'execution > T4 GPU\")"
    ),
    md(
        "## 3. Telecharger le dataset\n"
        "\n"
        "`VERSION` = numero de version du dataset Roboflow. La v2 est celle du pipeline d'origine.\n"
        "Verifie sur la page du dataset si une version plus recente (plus d'images) existe et ajuste."
    ),
    code(
        "from google.colab import userdata\n"
        "from roboflow import Roboflow\n"
        "\n"
        "WORKSPACE = \"handballdetectionvictorcollado\"\n"
        "PROJECT   = \"handball-detection-fj8rc\"\n"
        "VERSION   = 2  # ajuste si une version plus recente existe\n"
        "\n"
        "try:\n"
        "    ROBOFLOW_API_KEY = userdata.get(\"ROBOFLOW_API_KEY\")\n"
        "except userdata.SecretNotFoundError as exc:\n"
        "    raise RuntimeError(\n"
        "        \"ROBOFLOW_API_KEY absent des Secrets Colab. Voir cellule 1.\"\n"
        "    ) from exc\n"
        "\n"
        "rf = Roboflow(api_key=ROBOFLOW_API_KEY)\n"
        "project = rf.workspace(WORKSPACE).project(PROJECT)\n"
        "dataset = project.version(VERSION).download(\"yolov8\")\n"
        "print(f\"Dataset telecharge dans : {dataset.location}\")"
    ),
    md("## 4. Inspecter le dataset (classes, nb d'images)"),
    code(
        "import yaml, os, glob\n"
        "\n"
        "data_yaml = os.path.join(dataset.location, \"data.yaml\")\n"
        "with open(data_yaml) as f:\n"
        "    cfg = yaml.safe_load(f)\n"
        "\n"
        "print(\"Classes (ordre du dataset) :\", cfg.get(\"names\"))\n"
        "print(\"nc :\", cfg.get(\"nc\"))\n"
        "for split in (\"train\", \"valid\", \"test\"):\n"
        "    d = os.path.join(dataset.location, split, \"images\")\n"
        "    if os.path.isdir(d):\n"
        "        n = len(glob.glob(os.path.join(d, \"*\")))\n"
        "        print(f\"{split:6s} : {n} images\")\n"
        "\n"
        "print(\"\\nNote : l'ordre des classes ici peut differer de CLASSES_HANDBALL.\")\n"
        "print(\"Le pipeline remappe par NOM via ModeleConfig.pour_handball, donc peu importe l'ordre.\")"
    ),
    md(
        "## 5. Entrainement\n"
        "\n"
        "- `MODELE_BASE` : `yolov8m.pt` (bon compromis). Passe a `yolov8s.pt` pour aller plus vite,\n"
        "  ou `yolov8l.pt` pour plus de precision si le quota GPU le permet.\n"
        "- `EPOCHS` : 50 est un bon point de depart. `patience` arrete si plus d'amelioration.\n"
        "- `BATCH` : 16 tient sur T4 15 Go en 640px pour yolov8m ; baisse a 8 si OOM.\n"
        "\n"
        "Duree indicative T4 : ~30 s a 2 min / epoch selon la taille du dataset."
    ),
    code(
        "from ultralytics import YOLO\n"
        "\n"
        "MODELE_BASE = \"yolov8m.pt\"\n"
        "EPOCHS      = 50\n"
        "IMGSZ       = 640\n"
        "BATCH       = 16\n"
        "\n"
        "model = YOLO(MODELE_BASE)\n"
        "resultats = model.train(\n"
        "    data=data_yaml,\n"
        "    epochs=EPOCHS,\n"
        "    imgsz=IMGSZ,\n"
        "    batch=BATCH,\n"
        "    patience=15,\n"
        "    name=\"handball_yolo\",\n"
        "    project=\"/content/runs\",\n"
        "    verbose=True,\n"
        ")\n"
        "print(\"\\nEntrainement termine.\")\n"
        "print(f\"Poids best : {model.trainer.best}\")"
    ),
    md("## 6. Metriques de validation"),
    code(
        "metrics = model.val()\n"
        "print(f\"mAP50-95 : {metrics.box.map:.4f}\")\n"
        "print(f\"mAP50    : {metrics.box.map50:.4f}\")\n"
        "print(f\"mAP75    : {metrics.box.map75:.4f}\")\n"
        "print(\"\\nmAP50 par classe :\")\n"
        "for i, nom in enumerate(cfg.get(\"names\", [])):\n"
        "    try:\n"
        "        print(f\"  {nom:12s} : {metrics.box.ap50[i]:.4f}\")\n"
        "    except (IndexError, TypeError):\n"
        "        pass"
    ),
    md(
        "## 7. Exporter les poids vers Drive\n"
        "\n"
        "Copie `best.pt` dans ton Drive pour le reutiliser dans le pipeline d'inference."
    ),
    code(
        "import shutil, os\n"
        "from google.colab import drive\n"
        "\n"
        "drive.mount('/content/drive', force_remount=False)\n"
        "DOSSIER_MODELES = \"/content/drive/MyDrive/PIVOT_AI/models\"\n"
        "os.makedirs(DOSSIER_MODELES, exist_ok=True)\n"
        "\n"
        "chemin_best = str(model.trainer.best)\n"
        "chemin_cible = os.path.join(DOSSIER_MODELES, \"handball_yolov8m.pt\")\n"
        "shutil.copy(chemin_best, chemin_cible)\n"
        "print(f\"Modele exporte : {chemin_cible}\")\n"
        "print(\"\\nA utiliser dans le notebook d'inference :\")\n"
        "print(\"  from pivot_ai.config import ModeleConfig\")\n"
        "print(f\"  cfg = ModeleConfig.pour_handball('{chemin_cible}')\")"
    ),
    md("## 8. Verification visuelle sur quelques images de validation"),
    code(
        "import glob, random\n"
        "import matplotlib.pyplot as plt\n"
        "import cv2\n"
        "\n"
        "images_val = glob.glob(os.path.join(dataset.location, \"valid\", \"images\", \"*\"))\n"
        "echantillon = images_val[:6] if len(images_val) >= 6 else images_val\n"
        "\n"
        "if echantillon:\n"
        "    preds = model.predict(echantillon, conf=0.35, verbose=False)\n"
        "    fig, axes = plt.subplots(2, 3, figsize=(20, 10))\n"
        "    for ax, pred in zip(axes.flat, preds):\n"
        "        annotee = pred.plot()  # BGR annote\n"
        "        ax.imshow(cv2.cvtColor(annotee, cv2.COLOR_BGR2RGB))\n"
        "        ax.axis('off')\n"
        "    plt.tight_layout()\n"
        "    plt.show()\n"
        "else:\n"
        "    print(\"Pas d'images de validation trouvees.\")"
    ),
    md(
        "## Suite : utiliser le modele dans le pipeline\n"
        "\n"
        "Dans `notebooks/colab_pivot_ai.ipynb`, remplace l'appel au pipeline pour pointer\n"
        "vers le modele handball :\n"
        "\n"
        "```python\n"
        "from pivot_ai.config import ModeleConfig\n"
        "\n"
        "config_handball = ModeleConfig.pour_handball(\n"
        "    \"/content/drive/MyDrive/PIVOT_AI/models/handball_yolov8m.pt\"\n"
        ")\n"
        "resultat = traiter_match_complet(\n"
        "    chemin_video=CLIP,\n"
        "    correspondances_homographie=correspondances,\n"
        "    dossier_sortie=OUTPUTS_DIR,\n"
        "    subsample=2,\n"
        "    modele_config=config_handball,\n"
        ")\n"
        "```\n"
        "\n"
        "Le pipeline detectera alors joueurs / gardiens / arbitres / ballon distinctement,\n"
        "ce qui reduit fortement la fragmentation du tracking."
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
