# pivot-ai-handball

Pipeline computer vision pour scouting handball : detection, tracking, stats par joueur, decoupage automatique en actions.

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
│   ├── config.py            # Constantes et schemas
│   └── cli.py               # CLI `pivot-ai`
├── notebooks/
│   └── colab_pivot_ai.ipynb # Notebook Colab "thin"
├── scripts/
│   └── build_notebook.py    # Generation du notebook depuis Python
├── tests/                   # Tests unitaires
├── docs/
│   └── correspondances_example.json  # Modele homographie 10 points
├── legacy/                  # Code monolithique d'origine (reference)
├── data/                    # Videos, caches, outputs (ignored par git)
└── pyproject.toml
```

## Quickstart Colab

1. Ouvre le notebook depuis GitHub :
   `https://colab.research.google.com/github/tristan-paloc/pivot-ai-handball/blob/main/notebooks/colab_pivot_ai.ipynb`

2. Le notebook clone le repo, installe les deps, monte ton Drive et execute le pipeline.

3. Tes videos doivent etre dans `/MyDrive/PIVOT_AI/raw_clips/`.

### Authentification GitHub

Le repo est **public** : le clone HTTPS depuis Colab fonctionne en anonyme, aucun token requis.

> **Si tu repasses le repo en prive** : la 1ere cellule du notebook doit alors lire un Personal Access Token via Colab Secrets. Le mecanisme complet (PAT fine-grained + secret `GITHUB_TOKEN`) etait implemente jusqu'au commit `b33dc2d`, recuperable au besoin via :
> ```bash
> git show b33dc2d:scripts/build_notebook.py
> ```

### Pieges Colab a connaitre

- **Quota T4 gratuit** : environ 12h cumulees sur 24h, degradation au-dela de 15h. Si tu travailles intensivement, planifie ta journee.
- **Mount Drive** : token unique par session. Si Colab te deconnecte, il faudra re-mount (cellule 3 du notebook).
- **Premier `pip install`** : 3-5 minutes (ultralytics + torch + opencv + supervision + scipy). Les installs suivants utilisent le cache pip de la VM.
- **Estimation temps de traitement** : un match 60min @ 25fps avec `subsample=3` (1 frame sur 3) genere ~30k inferences, soit ~20 min sur T4. Le notebook affiche une estimation avant lancement.
- **VRAM** : 15 Go sur T4 gratuit, suffisant pour YOLOv8m batch 1. Si OOM, passer en CPU (`device="cpu"` dans `ModeleConfig`, beaucoup plus lent).
- **Pin `supervision<0.30`** : `sv.ByteTrack` est deprecated depuis 0.28 et sera supprime en 0.30. Le `pyproject.toml` pin la version pour eviter les surprises. Migration ByteTrack -> nouveau tracker = dette technique pour un sprint futur.
- **Re-execution complete** : pour repartir d'un etat propre, "Execution > Tout reinitialiser" puis relancer les cellules dans l'ordre.

## Quickstart local

```bash
git clone https://github.com/tristan-paloc/pivot-ai-handball.git
cd pivot-ai-handball
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate     # Windows
pip install -e ".[dev]"
```

### Lancer le pipeline via CLI

```bash
pivot-ai traiter \
  --video data/raw_clips/mon_clip.mp4 \
  --output data/outputs/ \
  --homographie docs/correspondances_example.json \
  --subsample 2
```

Flags optionnels :
- `--no-video-radar` : ne genere pas la video SBS broadcast+radar
- `--no-decoupage` : ne decoupe pas les actions detectees

Codes de sortie :
- `0` : succes
- `1` : erreur runtime (video ou homographie invalide, exception pipeline)
- `2` : argument obligatoire manquant

### Format du fichier homographie

JSON, minimum 4 points (>= 5 active RANSAC). Voir `docs/correspondances_example.json` pour un modele.

```json
{
  "A": {"pixel": [120, 540], "terrain_m": [0.0, 0.0]},
  "B": {"pixel": [1800, 540], "terrain_m": [40.0, 0.0]},
  "C": {"pixel": [1700, 300], "terrain_m": [40.0, 20.0]},
  "D": {"pixel": [220, 300], "terrain_m": [0.0, 20.0]}
}
```

Les cles commencant par `_` sont traitees comme commentaires (utile pour documenter le JSON inline).

## Tests

```bash
pytest tests/ -v --cov=pivot_ai
```

Tests qui requierent `ffmpeg` (3 tests de decoupage video) : auto-skip si `ffmpeg` n'est pas dans le PATH.

## Conventions de code

- Python 3.11+, type hints obligatoires sur les signatures publiques
- Variables et fonctions en **snake_case francais** : `traiter_clip`, `calculer_distance_parcourue`, `dets_par_frame`
- Docstrings courtes en francais, style Google
- Logging au lieu de print
- Pas de bare except
- Tests pytest pour toute fonction critique
- Ruff active avec `select = ["E", "F", "I", "N", "W", "UP", "B", "SIM"]`

## Roadmap

- [x] Squelette repo
- [x] Refactor du code legacy en modules
- [x] Migration Roboflow Cloud -> inference locale GPU (`DetecteurLocal`)
- [x] Module stats joueur (distance, vitesse, heatmap, bloc defensif)
- [x] Decoupage actions par heuristique simple
- [x] Pipeline match complet (avec interpolation et video SBS)
- [x] CLI `pivot-ai traiter`
- [x] Notebook Colab end-to-end
- [ ] Migration `sv.ByteTrack` vers le nouveau tracker supervision (`>=0.30`)
- [ ] Determination automatique de l'equipe en attaque (sens de jeu)
- [ ] Detection automatique lignes terrain (homographie auto)
- [ ] Identification individuelle joueurs (numeros maillots)
- [ ] Classification d'evenements ML

## Licence

Proprietaire. Tous droits reserves.
