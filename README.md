# pivot-ai-handball

Pipeline computer vision pour scouting handball : detection, tracking, stats par joueur, decoupage automatique en actions.

## Etat du projet

Squelette de fondation. La logique metier complete est dans `legacy/pipeline_colab.py` et doit etre refactoree dans les modules `pivot_ai/`.

## Architecture

```
pivot-ai-handball/
├── pivot_ai/                # Module principal
│   ├── detection.py         # Detection YOLO locale (inference GPU)
│   ├── tracking.py          # ByteTrack + stabilisation classes
│   ├── homographie.py       # Projection pixels -> coordonnees terrain
│   ├── equipes.py           # Classification equipes par KMeans
│   ├── radar.py             # Rendu radar 2D
│   ├── stats.py             # Stats joueur (distance, vitesse, heatmap)
│   ├── decoupage.py         # Decoupage automatique en actions
│   ├── pipeline.py          # Orchestrateur end-to-end
│   ├── config.py            # Constantes et schemas Pydantic
│   └── cli.py               # CLI d'entree
├── notebooks/
│   └── colab_pivot_ai.ipynb # Notebook Colab "thin"
├── tests/                   # Tests unitaires
├── legacy/                  # Code monolithique d'origine (reference)
├── data/                    # Vidéos, caches, outputs (ignored par git)
├── scripts/                 # Scripts utilitaires
└── pyproject.toml
```

## Quickstart Colab

1. Ouvre le notebook directement depuis GitHub :
   `https://colab.research.google.com/github/<TON_USER>/pivot-ai-handball/blob/main/notebooks/colab_pivot_ai.ipynb`

2. Le notebook clone le repo, installe les deps, monte ton Drive et execute le pipeline.

3. Tes videos doivent etre dans `/MyDrive/PIVOT_AI/raw_clips/`.

## Quickstart local (CPU ou GPU)

```bash
git clone https://github.com/<TON_USER>/pivot-ai-handball.git
cd pivot-ai-handball
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate     # Windows
pip install -e ".[dev]"

# Tester sur un clip
pivot-ai traiter --video data/raw_clips/mon_clip.mp4 --output data/outputs/
```

## Conventions de code

- Python 3.11+, type hints obligatoires sur les signatures publiques
- Variables et fonctions en **snake_case francais** : `traiter_clip`, `calculer_distance_parcourue`, `dets_par_frame`
- Docstrings courtes en francais, style Google
- Logging au lieu de print
- Pas de bare except
- Tests pytest pour toute fonction critique

## Roadmap

- [x] Squelette repo
- [ ] Refactor du code legacy en modules
- [ ] Migration Roboflow Cloud -> inference locale GPU
- [ ] Module stats joueur (distance, vitesse, heatmap)
- [ ] Decoupage actions par heuristique simple
- [ ] Pipeline match complet (60 min)
- [ ] Detection automatique lignes terrain (homographie auto)
- [ ] Identification individuelle joueurs (numeros maillots)
- [ ] Classification d'evenements ML

## Licence

Proprietaire. Tous droits reserves.
