"""Tests de validation des notebooks Colab et de leurs builders."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"

NOTEBOOK_INFERENCE = NOTEBOOKS_DIR / "colab_pivot_ai.ipynb"
NOTEBOOK_TRAIN = NOTEBOOKS_DIR / "train_handball_yolo.ipynb"

TOUS_NOTEBOOKS = [NOTEBOOK_INFERENCE, NOTEBOOK_TRAIN]


# ---------------------------------------------------------------------------
# Checks generiques appliques aux deux notebooks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("notebook", TOUS_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_existe(notebook: Path) -> None:
    """Le fichier notebook est present."""
    assert notebook.exists(), f"Notebook absent : {notebook}"


@pytest.mark.parametrize("notebook", TOUS_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_json_valide(notebook: Path) -> None:
    """Le notebook est un JSON valide avec la structure attendue."""
    contenu = json.loads(notebook.read_text(encoding="utf-8"))
    assert isinstance(contenu, dict)
    assert "cells" in contenu
    assert "metadata" in contenu
    assert "nbformat" in contenu


@pytest.mark.parametrize("notebook", TOUS_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_metadata_colab_gpu(notebook: Path) -> None:
    """Les metadonnees Colab declarent GPU T4."""
    contenu = json.loads(notebook.read_text(encoding="utf-8"))
    metadata = contenu["metadata"]
    assert metadata.get("accelerator") == "GPU"
    assert metadata.get("colab", {}).get("gpuType") == "T4"


@pytest.mark.parametrize("notebook", TOUS_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_cellules_bien_typees(notebook: Path) -> None:
    """Chaque cellule a un cell_type valide et une source."""
    contenu = json.loads(notebook.read_text(encoding="utf-8"))
    for i, cell in enumerate(contenu["cells"]):
        assert cell["cell_type"] in {"markdown", "code"}, f"Cellule {i} : type inconnu"
        assert "source" in cell, f"Cellule {i} sans source"


@pytest.mark.parametrize("notebook", TOUS_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_cellules_code_parse_python(notebook: Path) -> None:
    """Chaque cellule code parse en Python valide (magics !/% remplaces par pass)."""
    contenu = json.loads(notebook.read_text(encoding="utf-8"))
    for i, cell in enumerate(contenu["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        lignes = []
        for ligne in src.split("\n"):
            stripped = ligne.lstrip()
            if stripped.startswith(("!", "%")):
                indent = ligne[: len(ligne) - len(stripped)]
                lignes.append(indent + "pass")
            else:
                lignes.append(ligne)
        try:
            ast.parse("\n".join(lignes))
        except SyntaxError as exc:
            raise AssertionError(f"{notebook.name} cellule {i} : SyntaxError {exc}") from exc


@pytest.mark.parametrize("notebook", TOUS_NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_pas_de_token_en_dur(notebook: Path) -> None:
    """Aucun token/cle en dur dans les notebooks."""
    contenu = notebook.read_text(encoding="utf-8")
    assert "ghp_" not in contenu
    assert "github_pat_" not in contenu


# ---------------------------------------------------------------------------
# Checks specifiques au notebook d'inference
# ---------------------------------------------------------------------------


def test_inference_url_repo_remplacee() -> None:
    """L'URL placeholder est substituee."""
    contenu = NOTEBOOK_INFERENCE.read_text(encoding="utf-8")
    assert "REMPLACE_PAR_TON_USERNAME" not in contenu
    assert "tristan-paloc" in contenu
    assert "pivot-ai-handball" in contenu


def test_inference_appelle_pipeline() -> None:
    """Le notebook d'inference utilise traiter_match_complet."""
    contenu = NOTEBOOK_INFERENCE.read_text(encoding="utf-8")
    assert "traiter_match_complet" in contenu
    assert "pivot_ai.pipeline" in contenu


def test_inference_clone_anonyme_repo_public() -> None:
    """Repo public : clone HTTPS anonyme, pas de PAT."""
    contenu = NOTEBOOK_INFERENCE.read_text(encoding="utf-8")
    assert "https://github.com/" in contenu
    assert "GITHUB_TOKEN" not in contenu
    assert "userdata.get" not in contenu


def test_inference_propose_modele_handball() -> None:
    """Le notebook d'inference propose de charger le modele handball fine-tune."""
    contenu = NOTEBOOK_INFERENCE.read_text(encoding="utf-8")
    assert "ModeleConfig.pour_handball" in contenu
    assert "modele_config=modele_config" in contenu


# ---------------------------------------------------------------------------
# Checks specifiques au notebook d'entrainement
# ---------------------------------------------------------------------------


def test_train_utilise_roboflow_et_dataset() -> None:
    """Le notebook d'entrainement telecharge le dataset handball via Roboflow."""
    contenu = NOTEBOOK_TRAIN.read_text(encoding="utf-8")
    assert "roboflow" in contenu.lower()
    assert "handball-detection-fj8rc" in contenu
    assert "handballdetectionvictorcollado" in contenu
    assert "ROBOFLOW_API_KEY" in contenu


def test_train_entraine_et_exporte() -> None:
    """Le notebook entraine un YOLO et exporte best.pt vers Drive."""
    contenu = NOTEBOOK_TRAIN.read_text(encoding="utf-8")
    assert "model.train" in contenu
    assert "handball_yolov8m.pt" in contenu
    assert "ModeleConfig.pour_handball" in contenu


# ---------------------------------------------------------------------------
# Idempotence des builders
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("builder", "notebook"),
    [
        ("build_notebook.py", NOTEBOOK_INFERENCE),
        ("build_train_notebook.py", NOTEBOOK_TRAIN),
    ],
    ids=["inference", "train"],
)
def test_builder_notebook_idempotent(builder: str, notebook: Path) -> None:
    """Chaque builder reecrit le meme JSON octet pour octet."""
    script = SCRIPTS_DIR / builder
    assert script.exists()
    avant = notebook.read_bytes()
    subprocess.run([sys.executable, str(script)], check=True, capture_output=True)
    apres = notebook.read_bytes()
    assert avant == apres, f"{builder} n'est pas idempotent"
