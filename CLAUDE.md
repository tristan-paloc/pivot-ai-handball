# CLAUDE.md - Configuration projet pivot-ai-handball

## Mission

Pipeline computer vision pour scouting handball : detection, tracking, stats joueur, decoupage automatique en actions. Le repo a ete pre-scaffole : structure de dossiers, `pyproject.toml`, modules avec signatures et docstrings, tests squelettes. Le code legacy d'origine est dans `legacy/pipeline_colab.py`.

## Contexte utilisateur

Tristan : data scientist freelance, code en Python avec conventions strictes :
- snake_case **francais** pour variables/fonctions (`traiter_clip`, `calculer_distance_parcourue`)
- Type hints obligatoires sur signatures publiques
- Docstrings courtes en francais, style Google
- Logging au lieu de print, pas de bare except
- Pas de TODO laisses dans le code final, pas d'inventions sur les libs

## Workflow git OBLIGATOIRE

- Branche `main` toujours stable
- Une branche par feature : `feat/inference-locale`, `feat/stats-joueur`, etc.
- Commits atomiques en francais
- PR ouverte a la fin de chaque feature

## Process

1. Lire `legacy/pipeline_colab.py` en entier AVANT de coder
2. Proposer un plan detaille a Tristan en quelques lignes
3. Attendre son GO
4. Coder une etape a la fois, commit, passer a la suite
5. A chaque etape, montrer un test concret qui prouve que ca marche
6. Verifier les imports et signatures des libs (Ultralytics 8.3+, supervision 0.25+)

## Etat actuel des modules

- `config.py` : COMPLET - constantes, TerrainConfig, points caracteristiques
- `detection.py` : COMPLET - DetecteurLocal avec Ultralytics YOLO
- `tracking.py` : COMPLET - ByteTrack + stabilisation classe
- `homographie.py` : COMPLET - calibrer_homographie avec fallback RANSAC
- `equipes.py` : COMPLET - extraire_couleur_torse + classifier_equipes
- `radar.py` : COMPLET - dessiner_radar
- `stats.py` : A IMPLEMENTER (skeletons avec NotImplementedError)
- `decoupage.py` : A IMPLEMENTER (skeletons avec NotImplementedError)
- `pipeline.py` : A IMPLEMENTER (orchestrateur principal)
- `cli.py` : A FINALISER

## Etapes du sprint

### Etape 1 : Valider l'existant (1h)
Faire tourner `pytest tests/` et verifier que les modules deja implementes passent.
Identifier les bugs eventuels dans les modules COMPLETS et les corriger.

### Etape 2 : Implementer stats.py (2h)
- `calculer_stats_joueur` : DataFrame Polars avec toutes les colonnes specifiees
- `generer_heatmap_joueur` : heatmap 2D avec lissage gaussien optionnel
- `calculer_largeur_bloc_defensif` : DataFrame temporel
- Tests : completer `tests/test_stats.py` (4 tests skippes a activer)

### Etape 3 : Implementer decoupage.py (2h)
- `detecter_actions` : machine a etats sur le nb de joueurs presents
- `decouper_clips_video` : appels ffmpeg via subprocess
- Tests : creer `tests/test_decoupage.py` avec sequence synthetique

### Etape 4 : Implementer pipeline.py (3h)
- `traiter_match_complet` : enchainer tous les modules
- Interpolation lineaire des positions sur frames non-echantillonnees (uniquement pour rendu visuel)
- Generation video SBS (broadcast + radar cote a cote)
- Sauvegarde stats CSV et Parquet

### Etape 5 : Finaliser cli.py (30 min)
- Charger correspondances homographie depuis fichier JSON
- Appeler `traiter_match_complet`
- Output : message recap avec chemins des artefacts produits

### Etape 6 : Notebook Colab end-to-end (1h)
- Remplacer URL placeholder du repo dans le notebook
- Tester sur clip court (3-10s) d'abord
- Valider sur clip 58s
- Documenter les pieges Colab dans le README (quota, T4, mount Drive)

## Contraintes techniques

- Python 3.11+
- T4 Colab gratuite = 15 Go VRAM. Verifier les batch sizes.
- Un match 60min @ 25fps = 90k frames. Avec subsample=3, ~1500 inferences/min sur T4.
  Estimer le temps total et l'afficher avant de lancer.
- Pas de torch.load arbitraire (verifier les checkpoints, eviter pickle non sigle)

## Hors scope (Tristan a dit NON)

- Detection automatique des lignes terrain pour homographie auto -> plus tard
- Identification individuelle des joueurs (numeros, noms) -> plus tard
- Tracking multi-plans avec recalage automatique -> plus tard
- Modele de classification d'evenements ML -> plus tard
- SAM3 ou tout autre modele experimental -> plus tard
- Interface web ou app -> plus tard

## Critere de fin de mission

Le notebook `notebooks/colab_pivot_ai.ipynb` s'execute end-to-end sur Colab T4 :
- Entree : clip MP4 dans Drive + correspondances homographie
- Sortie : DataFrame stats joueurs + clips d'actions decoupes + video radar SBS
- Pas d'intervention manuelle au-dela de la saisie homographie

Et `pytest tests/ -v` passe au vert.
