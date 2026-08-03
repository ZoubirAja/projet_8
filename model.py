from catboost import CatBoostClassifier
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, recall_score
import joblib


# ── Fonction de coût métier (reprise du projet home-credit) ──────────
def cout_metier(y_true, y_proba, seuil, cout_fn=10, cout_fp=1):
    """
    FN = ne pas détecter un cas positif → coût cout_fn
    FP = fausse alerte sur un cas négatif → coût cout_fp
    """
    y_pred = (y_proba >= seuil).astype(int)
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    return fn * cout_fn + fp * cout_fp


def chercher_seuil_optimal(y_true, y_proba, cout_fn=10, cout_fp=1):
    seuils = np.arange(0.05, 0.95, 0.01)
    couts = [cout_metier(y_true, y_proba, s, cout_fn, cout_fp) for s in seuils]
    idx_min = np.argmin(couts)
    return seuils[idx_min], couts[idx_min]


#------------------------------------ Modele de prédiction-------------------------
def predict_model(X, y, cout_fn=10, cout_fp=1):
    # ---------------- SPLIT ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    catboost = CatBoostClassifier(
        iterations=200, learning_rate=0.1,
        depth=6, random_state=42, verbose=0,
        scale_pos_weight=1,       # le rééquilibrage se fait via le seuil, pas à l'entraînement
        thread_count=8,           # garde 4 cœurs libres pour le système/UI
        used_ram_limit='6gb'      # force CatBoost à respecter une limite RAM
    )

    pipeline = Pipeline([
        ('model', catboost)
    ])

    # ---------------- PIPELINE ----------------
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # ---------------- SEUIL OPTIMAL (rééquilibrage post-hoc, ratio 10:1) ----------------
    seuil_optimal, cout_minimal = chercher_seuil_optimal(y_test.values, y_proba)
    y_pred_optimal = (y_proba >= seuil_optimal).astype(int)

    score = f1_score(y_test, y_pred_optimal, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    recall = recall_score(y_test, y_pred_optimal, zero_division=0)

    print(f"Seuil optimal calculé : {seuil_optimal:.2f} (coût minimal : {cout_minimal})")

    # On sauvegarde le modèle + le seuil pour l'appeler via l'API
    joblib.dump((pipeline, seuil_optimal, score), 'model.pkl')
    #upload_model()

    return score, auc, recall, seuil_optimal
