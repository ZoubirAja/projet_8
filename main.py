from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
import logging
import os
import joblib

from customer import get_customer, log_prediction

load_dotenv()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")


def verify_api_key(key: str = Depends(API_KEY_HEADER)):
    if key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=403,
            detail="API Key invalide"
        )


app = FastAPI(debug=DEBUG)


# Handler d'erreur global
@app.exception_handler(Exception)
async def global_exception_handler(_request, exc: Exception):
    if DEBUG:
        # En dev → stack trace complète
        raise exc
    else:
        # En prod → message générique, pas de détails
        logging.error(f"Erreur : {exc}")
        return JSONResponse(
            status_code=500,
            content={"message": "Erreur interne du serveur"}
        )


# pipeline : modèle CatBoost entraîné (train.py). seuil_optimal : seuil de décision calculé
# via le coût métier (10x plus coûteux de rater un défaut que d'avoir une fausse alerte),
# à utiliser à la place du seuil par défaut (0.5) de pipeline.predict().
pipeline, seuil_optimal, score = joblib.load('model.pkl')


@app.get("/", include_in_schema=False)
def home():
    return {"status": "ok", "model_score_f1": score}


@app.post("/predict/{customer_id}", dependencies=[Depends(verify_api_key)])
def predict_by_id(customer_id: int):
    customer_df = get_customer(customer_id)
    if customer_df is None:
        return JSONResponse(
            status_code=200,
            content={"Erreur": "Aucun client pour cet ID"}
        )

    # TARGET n'existe pas au moment de la prédiction réelle (c'est ce qu'on cherche à prédire) :
    # on la retire pour donner au modèle exactement les colonnes vues à l'entraînement (config.py).
    customer_df = customer_df.drop(columns=["TARGET"], errors="ignore")

    return run_prediction(customer_df, customer_id)


def run_prediction(customer_df, customer_id=None):
    proba = pipeline.predict_proba(customer_df)[:, 1]
    prediction = int(proba[0] >= seuil_optimal)

    resultat = (
        "Le client aura du mal à rembourser son prêt"
        if prediction == 1
        else "N'aura pas de mal à rembourser son prêt"
    )
    probabilite = round(proba[0] * 100)

    try:
        log_prediction(
            customer_id=customer_id,
            prediction=prediction,
            probabilite=probabilite,
            resultat=resultat
        )
    except Exception as e:
        logging.warning(f"log_prediction échoué (BDD down probable) : {e}")

    return {
        "prediction": prediction,
        "probabilite_de_defaut": f"{probabilite}%",
        "resultat": resultat
    }
