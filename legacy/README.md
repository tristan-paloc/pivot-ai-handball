# Code legacy

`pipeline_colab.py` contient le pipeline monolithique d'origine, qui sert de reference de specification pour le refactor.

**Ne pas executer ce fichier directement.** Toute la logique doit etre migree dans les modules `pivot_ai/`.

Points cles a conserver lors du refactor :
- Structure des detections via `supervision.Detections`
- Tracking ByteTrack avec stabilisation classe par majority voting
- Classification equipes KMeans sur couleur torse
- Homographie 4-7 points avec fallback RANSAC
- Filtre asymetrique `[0,40]x[2,18]`
- Rendu radar avec lignes 6m/9m et arcs 9m

Points a SUPPRIMER ou remplacer :
- Dependance `detect.roboflow.com` (API cloud) -> inference locale via `DetecteurLocal`
- Code Colab specifique (drive.mount, userdata) -> deporte dans le notebook thin
