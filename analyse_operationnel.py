"""Analyse opérationnelle : latence anormale + taux d'erreur, à partir de predictions.db
et error_events. Ne nécessite pas evidently (contrairement à analyse_drift.py) — tourne
dans l'environnement Python standard du projet : `python3 analyse_operationnel.py`.
"""
import pandas as pd
from sqlalchemy import create_engine

# Seuil fixe (SLA), à recalibrer une fois plus de données réelles accumulées.
# Valeur de départ : les prédictions observées tournent à 40-55 ms (predict_proba sur
# un modèle déjà en mémoire) ; 200 ms est donc déjà ~4x la normale, un signal clair.
SEUIL_ANOMALIE_MS = 200


def charger_predictions(database_url="sqlite:///./predictions.db"):
    engine = create_engine(database_url)
    return pd.read_sql("SELECT customer_id, duree_ms, created_at FROM predictions", engine)


def charger_erreurs(database_url="sqlite:///./predictions.db"):
    engine = create_engine(database_url)
    return pd.read_sql("SELECT route, message, created_at FROM error_events", engine)


def analyser_latence(predictions):
    if predictions.empty:
        print("Aucune prédiction loguée à analyser.")
        return

    print(f"{len(predictions)} prédictions analysées")
    print(f"Durée moyenne : {predictions['duree_ms'].mean():.1f} ms")
    print(f"P95 : {predictions['duree_ms'].quantile(0.95):.1f} ms")
    print(f"P99 : {predictions['duree_ms'].quantile(0.99):.1f} ms")

    anomalies = predictions[predictions["duree_ms"] > SEUIL_ANOMALIE_MS]
    print(f"\n{len(anomalies)}/{len(predictions)} prédictions au-dessus du seuil ({SEUIL_ANOMALIE_MS} ms) :")
    if anomalies.empty:
        print("Aucune anomalie détectée.")
    else:
        print(anomalies[["customer_id", "duree_ms", "created_at"]].to_string(index=False))


def analyser_taux_erreur(predictions, erreurs):
    total = len(predictions) + len(erreurs)
    if total == 0:
        print("Aucune donnée (ni prédiction, ni erreur) à analyser.")
        return

    taux = len(erreurs) / total * 100
    print(f"\n{len(erreurs)} erreurs serveur (500) sur {total} requêtes — taux d'erreur : {taux:.1f}%")
    if not erreurs.empty:
        print(erreurs[["route", "message", "created_at"]].to_string(index=False))


def main():
    predictions = charger_predictions()
    erreurs = charger_erreurs()

    print("=== Latence ===")
    analyser_latence(predictions)

    print("\n=== Taux d'erreur ===")
    analyser_taux_erreur(predictions, erreurs)


if __name__ == "__main__":
    main()
