# Guide — lancer le projet en local

Fiche récapitulative des commandes pour lancer l'API, les tests, Docker, et les
analyses de monitoring (drift, latence, taux d'erreur).

## 1. API (FastAPI + démo Gradio)

```bash
pip install -r requirements-dev.txt
uvicorn main:app --app-dir app --reload
```

- API : http://localhost:8000
- Documentation interactive (Swagger) : http://localhost:8000/docs
- Démo Gradio (sans clé API) : http://localhost:8000/demo

Tester une prédiction (clé API définie dans `.env`) :
```bash
curl -X POST "http://localhost:8000/predict/100002" \
  -H "X-API-Key: $(grep '^API_KEY=' .env | cut -d= -f2)"
```

## 2. Tests

```bash
pytest -v
```

## 3. Docker

```bash
docker build -t projet8-api .
docker run -p 8080:7860 -e API_KEY=... projet8-api
```
- API : http://localhost:8080 (mêmes routes que ci-dessus)

## 4. Analyse de drift des données

Nécessite un environnement Python 3.12 dédié (`evidently` est incompatible avec le
Python 3.14 du projet). À faire une seule fois :
```bash
uv venv --python 3.12 .venv-monitoring
uv pip install -r requirements-monitoring.txt --python .venv-monitoring
```

Puis, à chaque analyse :
```bash
.venv-monitoring/bin/python3 scripts/analyse_drift.py
```
- Résumé texte : affiché directement dans le terminal.
- Rapport détaillé (graphiques) : fichier `drift_report.html` généré à la racine —
  pas d'URL/serveur, ouvrez-le directement dans un navigateur :
  ```bash
  xdg-open drift_report.html
  ```

⚠️ Avec peu de prédictions loguées (quelques unités), le rapport affichera du drift
sur toutes les colonnes — artefact de la petite taille d'échantillon, pas un vrai
signal. Devient fiable à partir de quelques centaines/milliers de prédictions réelles.

## 5. Analyse opérationnelle (latence + taux d'erreur)

Pas besoin d'environnement séparé — tourne dans l'environnement standard du projet :
```bash
python3 scripts/analyse_operationnel.py
```
Affiche la latence (moyenne, P95, P99, anomalies au-delà de `SEUIL_ANOMALIE_MS`,
200 ms par défaut) et le taux d'erreur (erreurs serveur 500 / total des requêtes,
table `error_events`).

## Récap des fichiers générés (non versionnés, gitignorés ou à titre d'exemple)

| Fichier | Généré par | Contenu |
|---|---|---|
| `predictions.db` | l'API, à chaque prédiction | historique des prédictions + inputs surveillés + durée |
| `drift_report.html` | `analyse_drift.py` | rapport visuel de dérive des données |
