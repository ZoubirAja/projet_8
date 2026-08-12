"""Analyse de drift : compare les colonnes surveillées (monitoring.FEATURES_MONITOREES)
entre la référence (données d'entraînement) et les prédictions récentes loguées en
production (predictions.db).

Tourne dans un venv Python 3.12 dédié : evidently est incompatible avec Python 3.14
(celui de l'API/Dockerfile) — voir requirements-monitoring.txt.
    .venv-monitoring/bin/python3 scripts/analyse_drift.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import pandas as pd
from sqlalchemy import create_engine

from evidently import Report
from evidently.presets import DataDriftPreset

from monitoring import FEATURES_MONITOREES
from config import X as X_entrainement  # même preprocessing que le training (config.py)


def charger_predictions_production(database_url="sqlite:///./predictions.db"):
    """Reconstruit un DataFrame des inputs loggués (une colonne par feature surveillée)."""
    engine = create_engine(database_url)
    predictions = pd.read_sql("SELECT inputs FROM predictions", engine)
    lignes = [
        json.loads(inputs) if isinstance(inputs, str) else inputs
        for inputs in predictions["inputs"]
    ]
    return pd.DataFrame(lignes)


def main():
    reference = X_entrainement[FEATURES_MONITOREES]
    courant = charger_predictions_production()

    if courant.empty:
        print("Aucune prédiction en production à analyser — lancez d'abord quelques prédictions.")
        return

    print(f"Référence (entraînement) : {len(reference)} lignes")
    print(f"Courant (production loguée) : {len(courant)} lignes")

    rapport = Report(metrics=[DataDriftPreset()])
    resultat = rapport.run(current_data=courant, reference_data=reference)
    resultat.save_html("drift_report.html")

    print("\nRésumé par colonne (méthode, distance à la référence, seuil) :")
    for metrique in resultat.dict()["metrics"]:
        config = metrique.get("config", {})
        colonne = config.get("column")
        if colonne is None:
            continue  # métriques globales (ex. nombre de colonnes driftées), pas par-colonne
        valeur, seuil = metrique["value"], config["threshold"]
        statut = "DRIFT" if valeur > seuil else "ok"
        print(f"  [{statut:5s}] {colonne:35s} {config['method']:30s} {valeur:.3f} (seuil {seuil})")

    print("\nRapport détaillé (graphiques) : drift_report.html")


if __name__ == "__main__":
    main()
