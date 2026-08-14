"""Compare la latence de prédiction CatBoost natif vs ONNX Runtime, sur la cause
identifiée par profile_api.py (coût de construction du Pool CatBoost à partir d'un
DataFrame large, ~79% du temps de run_prediction()). Vérifie la non-régression des
prédictions avant toute comparaison de temps.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import joblib
import numpy as np
import onnxruntime as ort

from customer import get_customer

CUSTOMER_ID = 100002
N_APPELS = 200

pipeline, seuil_optimal, score = joblib.load("model.pkl")
modele = pipeline.named_steps["model"]
customer_df = get_customer(CUSTOMER_ID).drop(columns=["TARGET"], errors="ignore")

modele.save_model("model.onnx", format="onnx")
session = ort.InferenceSession("model.onnx")
entree_nom = session.get_inputs()[0].name


def predire_catboost():
    return pipeline.predict_proba(customer_df)[0, 1]


def predire_onnx():
    entree = customer_df.to_numpy(dtype=np.float32)
    sortie = session.run(None, {entree_nom: entree})
    return sortie[1][0][1]


# --- Non-régression AVANT toute comparaison de temps ---
proba_catboost = predire_catboost()
proba_onnx = predire_onnx()
ecart = abs(proba_catboost - proba_onnx)
print(f"CatBoost : {proba_catboost:.6f}")
print(f"ONNX     : {proba_onnx:.6f}")
print(f"Écart    : {ecart:.8f}")
assert ecart < 0.01, "Écart de prédiction trop important entre CatBoost et ONNX"
print("OK — prédictions équivalentes\n")


def chronometrer(fn, n=N_APPELS):
    debut = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - debut) / n * 1000  # ms/appel


ms_catboost = chronometrer(predire_catboost)
ms_onnx = chronometrer(predire_onnx)

print(f"CatBoost natif (predict_proba) : {ms_catboost:.2f} ms/appel")
print(f"ONNX Runtime                   : {ms_onnx:.2f} ms/appel")
print(f"Gain                           : {(1 - ms_onnx / ms_catboost) * 100:.1f}%")
