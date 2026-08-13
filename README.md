---
title: Projet 8 - Scoring Credit
emoji: 💳
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Projet 8 — API de scoring crédit

API de prédiction de risque de défaut de paiement (dataset Home Credit Default Risk),
basée sur un modèle CatBoost.

## Routes

- `GET /` : health check
- `POST /predict/{customer_id}` : prédiction pour un client existant (header `X-API-Key` requis)
- `GET /demo` : interface Gradio de démonstration (pas de clé requise)

## Lancer en local

```
pip install -r requirements-dev.txt
python3 scripts/prepare_parquet.py   # génère customers_indexed.parquet à partir de df_final_clean.parquet
uvicorn main:app --app-dir app --reload
```

## Tests

```
pytest -v
```

## Docker

```
docker build -t projet8-api .
docker run -p 8080:7860 -e API_KEY=... projet8-api
```

## Déploiement en production

L'état actuel privilégie la simplicité (déploiement sans dépendance externe) :
modèle baké dans l'image, base de prédictions SQLite éphémère. Le code est déjà
prêt pour une version "production" plus robuste — voici ce qu'il faudrait changer,
sans réécrire de logique métier.

### Modèle : passer d'un modèle baké à un stockage S3 externe

Aujourd'hui `model.pkl` est copié dans l'image Docker (`Dockerfile`). Le code de
téléchargement à distance existe déjà (`minio_client.py`, appelé dans le `lifespan`
de `main.py`) mais ne fait rien tant que le fichier est présent localement.

Pour activer le vrai flux "modèle externalisé" (utile pour mettre à jour le modèle
sans reconstruire l'image) :
1. Retirer `model.pkl` du `COPY` dans `Dockerfile`.
2. Ajouter `model.pkl` à `.gitignore` et faire `git rm --cached model.pkl`.
3. Héberger le modèle sur un stockage S3-compatible joignable publiquement
   (ex. Cloudflare R2, free tier largement suffisant pour un fichier de cette taille) —
   MinIO en local ne fonctionne pas pour un déploiement cloud (non joignable depuis l'extérieur).
4. Configurer `S3_URL`, `S3_KEY`, `S3_SECRET` comme secrets sur la plateforme de déploiement.
5. Publier une version du modèle : `PYTHONPATH=app python3 -c "from minio_client import upload_model; import joblib; _,_,score = joblib.load('model.pkl'); upload_model(score=score)"` (ou directement via `scripts/train.py`).

Aucun changement de code requis au-delà de ça — `download_model()` est déjà appelé
au démarrage de l'API.

### Base de données : passer de SQLite à un serveur persistant

`predictions.db` (SQLite) est régénérée à chaque démarrage et n'est pas persistée —
acceptable pour une démo, pas pour un historique de prédictions en prod.

`database.py` supporte déjà un `DATABASE_URL` externe (Postgres, etc.) sans aucun
changement de code :
```
DATABASE_URL=postgresql://user:password@host:5432/dbname
```
Il suffit de définir cette variable comme secret sur la plateforme de déploiement.
