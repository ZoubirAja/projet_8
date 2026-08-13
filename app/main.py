from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
import gradio as gr
import logging
import os
import time
import joblib

from customer import get_customer, log_prediction, log_error
from explain import get_top_influential_features, FEATURES_INTERPRETABLES, valider_bornes
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
        try:
            log_error(route=str(_request.url.path), message=str(exc))
        except Exception as e:
            logging.warning(f"log_error échoué (BDD down probable) : {e}")
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


@app.post("/predict/{customer_id}/simulate", dependencies=[Depends(verify_api_key)])
def simulate_prediction(customer_id: int, valeurs: dict[str, float] = Body(...)):
    """Rejoue la prédiction d'un client en remplaçant certaines valeurs (ex. les
    facteurs influents renvoyés par /predict/{id}) — pour un "what-if", pas une
    vraie prédiction : non journalisée dans predictions.db.
    """
    customer_df = get_customer(customer_id)
    if customer_df is None:
        return JSONResponse(
            status_code=200,
            content={"Erreur": "Aucun client pour cet ID"}
        )

    customer_df = customer_df.drop(columns=["TARGET"], errors="ignore").copy()

    colonnes_invalides = set(valeurs) - set(customer_df.columns)
    if colonnes_invalides:
        raise HTTPException(
            status_code=422,
            detail=f"Colonnes inconnues : {sorted(colonnes_invalides)}"
        )

    erreurs_bornes = valider_bornes(valeurs)
    if erreurs_bornes:
        raise HTTPException(
            status_code=422,
            detail=f"Valeurs hors bornes : {erreurs_bornes}"
        )

    for colonne, valeur in valeurs.items():
        customer_df[colonne] = valeur

    return run_prediction(customer_df, customer_id, log=False)


def run_prediction(customer_df, customer_id=None, log=True):
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
    facteurs_influents = get_top_influential_features(
        pipeline, customer_df, candidats=FEATURES_INTERPRETABLES
    )

    if log:
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
        "resultat": resultat,
        "facteurs_influents": facteurs_influents,
    }


def _facteurs_en_table(facteurs):
    return [[f["feature"], f["valeur"], f["contribution"]] for f in facteurs]


def predict_for_demo(customer_id):
    """Fonction appelée par l'interface Gradio (pas de clé API : démo publique)."""
    champs_vides = [gr.update(visible=False)] * 3

    if customer_id is None:
        return ("Renseigne un ID client (ex : 100002).", "", None, [], *champs_vides)

    customer_df = get_customer(int(customer_id))
    if customer_df is None:
        return ("Aucun client trouvé pour cet ID.", "", None, [], *champs_vides)

    customer_df = customer_df.drop(columns=["TARGET"], errors="ignore")
    result = run_prediction(customer_df, int(customer_id))
    facteurs = result["facteurs_influents"]
    noms = [f["feature"] for f in facteurs]

    # Un champ modifiable par facteur influent (label + valeur actuelle du client) ;
    # ceux en trop (si moins de 3 facteurs) restent masqués.
    champs = [gr.update(label=f["feature"], value=f["valeur"], visible=True) for f in facteurs]
    champs += [gr.update(visible=False)] * (3 - len(champs))

    return (result["resultat"], result["probabilite_de_defaut"], _facteurs_en_table(facteurs), noms, *champs)


def simuler_demo(customer_id, noms_facteurs, valeur1, valeur2, valeur3):
    """Rejoue la prédiction avec les 3 facteurs influents modifiés (what-if, non journalisé)."""
    if not noms_facteurs or customer_id is None:
        return "Faites d'abord une prédiction ci-dessus.", "", None

    customer_df = get_customer(int(customer_id))
    if customer_df is None:
        return "Aucun client trouvé pour cet ID.", "", None

    customer_df = customer_df.drop(columns=["TARGET"], errors="ignore").copy()
    for colonne, valeur in zip(noms_facteurs, [valeur1, valeur2, valeur3]):
        customer_df[colonne] = valeur

    result = run_prediction(customer_df, int(customer_id), log=False)
    return result["resultat"], result["probabilite_de_defaut"], _facteurs_en_table(result["facteurs_influents"])


with gr.Blocks(title="Scoring crédit — démo") as demo:
    gr.Markdown(
        "## Scoring crédit — démo\n"
        "Entre un ID client existant (ex : 100002) pour voir la prédiction du modèle. "
        "Cette interface n'utilise pas de clé API, contrairement à `/predict/{id}`."
    )

    customer_id_input = gr.Number(label="ID client (SK_ID_CURR)", precision=0)
    predict_btn = gr.Button("Prédire")

    resultat_output = gr.Textbox(label="Résultat")
    proba_output = gr.Textbox(label="Probabilité de défaut")
    facteurs_output = gr.Dataframe(
        headers=["Facteur", "Valeur actuelle", "Contribution"],
        label="Facteurs les plus influents pour ce client",
    )

    noms_facteurs_state = gr.State([])

    gr.Markdown("### Simuler une nouvelle prédiction en modifiant ces facteurs")
    with gr.Row():
        champ1 = gr.Number(label="—", visible=False)
        champ2 = gr.Number(label="—", visible=False)
        champ3 = gr.Number(label="—", visible=False)
    simulate_btn = gr.Button("Simuler avec ces valeurs")

    predict_btn.click(
        predict_for_demo,
        inputs=customer_id_input,
        outputs=[resultat_output, proba_output, facteurs_output, noms_facteurs_state, champ1, champ2, champ3],
    )
    simulate_btn.click(
        simuler_demo,
        inputs=[customer_id_input, noms_facteurs_state, champ1, champ2, champ3],
        outputs=[resultat_output, proba_output, facteurs_output],
    )

# Démo accessible sur /demo, à côté des routes JSON protégées par clé API.
app = gr.mount_gradio_app(app, demo, path="/demo")
