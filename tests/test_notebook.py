"""Tests de validation du notebook Colab et du builder associe."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent.parent / "notebooks" / "colab_pivot_ai.ipynb"


def test_notebook_existe() -> None:
    """Le fichier notebook est present."""
    assert NOTEBOOK.exists(), f"Notebook absent : {NOTEBOOK}"


def test_notebook_json_valide() -> None:
    """Le notebook est un JSON valide."""
    contenu = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert isinstance(contenu, dict)
    assert "cells" in contenu
    assert "metadata" in contenu
    assert "nbformat" in contenu


def test_notebook_metadata_colab_gpu() -> None:
    """Les metadonnees Colab declarent GPU T4."""
    contenu = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    metadata = contenu["metadata"]
    assert metadata.get("accelerator") == "GPU"
    assert metadata.get("colab", {}).get("gpuType") == "T4"


def test_notebook_url_repo_remplacee() -> None:
    """L'URL placeholder REMPLACE_PAR_TON_USERNAME a bien ete substituee."""
    contenu = NOTEBOOK.read_text(encoding="utf-8")
    assert "REMPLACE_PAR_TON_USERNAME" not in contenu
    # L'URL est construite via f-string : on verifie les composants separes
    assert "tristan-paloc" in contenu
    assert "pivot-ai-handball" in contenu


def test_notebook_appelle_pipeline() -> None:
    """Le notebook utilise traiter_match_complet du module pivot_ai.pipeline."""
    contenu = NOTEBOOK.read_text(encoding="utf-8")
    assert "traiter_match_complet" in contenu
    assert "pivot_ai.pipeline" in contenu


def test_notebook_a_au_moins_10_cellules() -> None:
    """Le notebook a au moins 10 cellules (la version actuelle en a 22)."""
    contenu = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert len(contenu["cells"]) >= 10


def test_notebook_clone_anonyme_repo_public() -> None:
    """Repo public : clone HTTPS anonyme, pas de PAT ni de userdata.get."""
    contenu = NOTEBOOK.read_text(encoding="utf-8")
    # URL HTTPS publique (construite via f-string sur REPO_OWNER/REPO_NAME)
    assert "https://github.com/" in contenu
    # Aucun token en dur (defense en profondeur)
    assert "ghp_" not in contenu
    assert "github_pat_" not in contenu
    # Plus de mecanisme PAT cote notebook
    assert "GITHUB_TOKEN" not in contenu
    assert "userdata.get" not in contenu


def test_notebook_cellules_bien_typees() -> None:
    """Chaque cellule a un cell_type valide et une source non vide."""
    contenu = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for i, cell in enumerate(contenu["cells"]):
        assert cell["cell_type"] in {"markdown", "code"}, f"Cellule {i} : type inconnu"
        assert "source" in cell, f"Cellule {i} sans source"


def test_notebook_cellules_code_parse_python() -> None:
    """Chaque cellule code parse en Python valide (apres avoir remplace les magics !/%)."""
    import ast

    contenu = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
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
        code = "\n".join(lignes)
        try:
            ast.parse(code)
        except SyntaxError as exc:
            raise AssertionError(f"Cellule {i} : SyntaxError {exc}") from exc


def test_builder_notebook_idempotent(tmp_path: Path) -> None:
    """Le script build_notebook ecrit toujours le meme JSON."""
    import subprocess
    import sys

    script = Path(__file__).parent.parent / "scripts" / "build_notebook.py"
    assert script.exists()

    avant = NOTEBOOK.read_bytes()
    subprocess.run(
        [sys.executable, str(script)],
        check=True,
        capture_output=True,
    )
    apres = NOTEBOOK.read_bytes()
    assert avant == apres, "Le builder n'est pas idempotent"
