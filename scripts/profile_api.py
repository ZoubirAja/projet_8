"""Profile de run_prediction() (le coeur de /predict) — identifie où passe le temps
avant de choisir une piste d'optimisation. cProfile + pstats, aucune dépendance
nouvelle.
"""
import os
import sys
import cProfile
import pstats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import joblib
from dotenv import load_dotenv

load_dotenv()

import main
from customer import get_customer

# Reproduit ce que fait lifespan() au démarrage de l'API (un simple import de main
# ne déclenche pas le lifespan, réservé au cycle de vie FastAPI/ASGI).
main.pipeline, main.seuil_optimal, main.score = joblib.load("model.pkl")

CUSTOMER_ID = 100002
customer_df = get_customer(CUSTOMER_ID).drop(columns=["TARGET"], errors="ignore")

N_APPELS = 200


def boucle():
    for _ in range(N_APPELS):
        main.run_prediction(customer_df, CUSTOMER_ID, log=False)


profiler = cProfile.Profile()
profiler.enable()
boucle()
profiler.disable()

print(f"--- Profil sur {N_APPELS} appels à run_prediction() ---\n")
stats = pstats.Stats(profiler).sort_stats("cumulative")
stats.print_stats(20)
