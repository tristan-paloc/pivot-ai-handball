# Prompt pour Claude Code - Mission pivot-ai-handball

Copie le contenu ci-dessous quand tu lances Claude Code a la racine du repo.

---

Lis d'abord `CLAUDE.md` a la racine. Il contient le contexte complet de la mission, les conventions de code, les modules deja implementes et ceux a finir, et le workflow git attendu.

Lis ensuite `legacy/pipeline_colab.py` en entier. C'est la specification metier.

Lis enfin la structure du projet : `find . -type f -name "*.py" | head -30`.

Une fois ces 3 lectures faites :

1. Verifie que les modules COMPLETS marchent : lance `pytest tests/ -v`. Si des tests echouent, identifie pourquoi (probablement deps manquantes ou bugs mineurs dans mon scaffolding).

2. Propose-moi en quelques lignes ton plan d'execution pour les etapes 2 a 6 du CLAUDE.md. Quels modules d'abord, dans quel ordre, quels risques tu identifies, ou tu as des doutes sur la specification metier.

3. Attends mon GO avant de coder.

Une fois mon GO recu :
- Cree une branche par feature (`feat/stats-joueur`, `feat/decoupage`, etc.)
- Code une etape complete, commit, ouvre une PR de cette feature, attends mon validation, merge sur main, passe a la suivante
- A chaque etape, MONTRE-MOI un test concret qui prouve que ca marche
- Pas de TODO laisses, pas d'inventions sur les libs
- Si tu as un doute, demande au lieu d'inventer

Quand tu as besoin de tester sur GPU/Colab, on aura une iteration ou je lance le notebook sur Colab T4 et te remonte les logs/erreurs.

Demarre maintenant : lectures, verification tests, plan d'attaque.
