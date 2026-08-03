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
python3 prepare_parquet.py   # génère customers_indexed.parquet à partir de df_final_clean.parquet
uvicorn main:app --reload
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
