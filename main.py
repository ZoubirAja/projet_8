from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
import gradio as gr
import logging
import os
import time
import joblib

from customer import get_customer, log_prediction
from minio_client import download_model
from monitoring import extraire_inputs_surveilles
from contextlib import asynccontextmanager

load_dotenv()

DEBUG = os.getenv("DEBUG", "false").lower() == "true"

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")


def verify_api_key(key: str = Depends(API_KEY_HEADER)):
    if key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=403,
            detail="API Key invalide"
        )


# pipeline : modèle CatBoost entraîné (train.py). seuil_optimal : seuil de décision calculé
# via le coût métier (10x plus coûteux de rater un défaut que d'avoir une fausse alerte),
# à utiliser à la place du seuil par défaut (0.5) de pipeline.predict().
pipeline = seuil_optimal = score = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, seuil_optimal, score
    # download_model() est un no-op ici : model.pkl est baké dans l'image (Dockerfile),
    # donc le fichier existe déjà et le téléchargement S3 est sauté. Code laissé en place
    # pour le jour où on repasse en mode "modèle externalisé" — voir README.
    download_model()
    pipeline, seuil_optimal, score = joblib.load('model.pkl')
    yield

app = FastAPI(
    lifespan=lifespan,
    debug=DEBUG
)


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
    debut = time.perf_counter()
    proba = pipeline.predict_proba(customer_df)[:, 1]
    duree_ms = (time.perf_counter() - debut) * 1000

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
            resultat=resultat,
            inputs=extraire_inputs_surveilles(customer_df),
            duree_ms=duree_ms,
        )
    except Exception as e:
        logging.warning(f"log_prediction échoué (BDD down probable) : {e}")

    return {
        "prediction": prediction,
        "probabilite_de_defaut": f"{probabilite}%",
        "resultat": resultat
    }


def predict_for_demo(customer_id: float):
    """Fonction appelée par l'interface Gradio (pas de clé API : démo publique)."""
    if customer_id is None:
        return "Renseigne un ID client (ex : 100002).", ""

    customer_df = get_customer(int(customer_id))
    if customer_df is None:
        return "Aucun client trouvé pour cet ID.", ""

    customer_df = customer_df.drop(columns=["TARGET"], errors="ignore")
    result = run_prediction(customer_df, int(customer_id))
    return result["resultat"], result["probabilite_de_defaut"]


demo = gr.Interface(
    fn=predict_for_demo,
    inputs=gr.Number(label="ID client (SK_ID_CURR)", precision=0),
    outputs=[gr.Textbox(label="Résultat"), gr.Textbox(label="Probabilité de défaut")],
    title="Scoring crédit — démo",
    description="Entre un ID client existant (ex : 100002) pour voir la prédiction du modèle. "
                "Cette interface n'utilise pas de clé API, contrairement à /predict/{id}.",
)

# Démo accessible sur /demo, à côté des routes JSON protégées par clé API.
app = gr.mount_gradio_app(app, demo, path="/demo")
