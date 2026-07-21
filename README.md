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

## Modele de detection : COCO generique vs fine-tune handball

Par defaut, le pipeline utilise **YOLOv8m pre-entraine COCO** : il ne connait que la classe
generique "person". Consequence : pas de distinction joueur / gardien / arbitre / ballon, et un
tracking fragmente (beaucoup d'IDs pour peu de joueurs reels) car les detections sont bruitees.

Pour un vrai saut de qualite, **entraine un modele specialise handball** :

1. Ouvre `notebooks/train_handball_yolo.ipynb` sur Colab (T4 GPU).
2. Cree un compte [Roboflow](https://roboflow.com) gratuit, recupere ta cle API
   ([app.roboflow.com/settings/api](https://app.roboflow.com/settings/api)) et ajoute-la aux
   Secrets Colab sous le nom `ROBOFLOW_API_KEY`.
3. Le notebook telecharge le dataset
   [handball-detection-fj8rc](https://universe.roboflow.com/handballdetectionvictorcollado/handball-detection-fj8rc)
   (4 classes players/goalkeeper/referees/ball), fine-tune un YOLOv8m, affiche les mAP et exporte
   `best.pt` vers `/MyDrive/PIVOT_AI/models/handball_yolov8m.pt`.
4. Dans le notebook d'inference, la cellule "Modele de detection" detecte automatiquement ce fichier
   et l'utilise via `ModeleConfig.pour_handball(...)`.

Le pipeline remappe les classes du modele vers `CLASSES_HANDBALL` **par nom** (pas par ordre), donc
peu importe l'ordre des classes dans le dataset d'entrainement.

En local / CLI, pointe le modele explicitement :

```python
from pivot_ai.config import ModeleConfig
from pivot_ai.pipeline import traiter_match_complet

config = ModeleConfig.pour_handball("models/handball_yolov8m.pt")
traiter_match_complet(..., modele_config=config)
```

### Authentification GitHub

Le repo est **public** : le clone HTTPS depuis Colab fonctionne en anonyme, aucun token requis.

> **Si tu repasses le repo en prive** : la 1ere cellule du notebook doit alors lire un Personal Access Token via Colab Secrets. Le mecanisme complet (PAT fine-grained + secret `GITHUB_TOKEN`) etait implemente jusqu'au commit `b33dc2d`, recuperable au besoin via :
> ```bash
> git show b33dc2d:scripts/build_notebook.py
> ```

### Pieges Colab a connaitre

- **Quota T4 gratuit** : environ 12h cumulees sur 24h, degradation au-dela de 15h. Si tu travailles intensivement, planifie ta journee.
- **Mount Drive** : token unique par session. Si Colab te deconnecte, il faudra re-mount (cellule 3 du notebook).
- **Premier `pip install`** : le notebook installe les deps runtime seulement (`pip install -e .`, sans `[dev]`) pour eviter les conflits de versions avec les paquets Colab. Compter 1-3 min la 1ere fois, quasi instantane ensuite (cache pip de la VM).
- **Lancement** : `Execution > Tout executer`. Le seul geste manuel est d'autoriser le mount Drive quand il le demande. La calibration homographie est persistee sur Drive (`correspondances.json`), donc a saisir une seule fois.
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
- `--tracker {bytetrack,botsort}` : backend de tracking. `bytetrack` (defaut,
  rapide, mouvement seul) ou `botsort` (ReID par apparence : maintient les IDs
  au contact/occlusion, plus lent). `botsort` necessite un vrai modele YOLO.
- `--min-frames-track N` : ecarte les trackers vus sur moins de N frames (fragments)

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
- [x] Fine-tune modele YOLO handball (4 classes) + notebook d'entrainement
- [x] Tracker BoT-SORT + ReID (maintient les IDs au contact) en option
- [ ] Detection ballon -> possession et passes
- [ ] Detection d'evenements (tirs, buts) par heuristique
- [ ] Migration `sv.ByteTrack` vers le nouveau tracker supervision (`>=0.30`)
- [ ] Determination automatique de l'equipe en attaque (sens de jeu)
- [ ] Detection automatique lignes terrain (homographie auto)
- [ ] Identification individuelle joueurs (numeros maillots)
- [ ] Classification d'evenements ML

## Licence

Proprietaire. Tous droits reserves.
