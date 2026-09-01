"""Analyse opérationnelle : latence anormale + taux d'erreur, à partir de predictions.db
et error_events. Ne nécessite pas evidently (contrairement à analyse_drift.py) — tourne
dans l'environnement Python standard du projet : `python3 scripts/analyse_operationnel.py`.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
from sqlalchemy import create_engine, text

# Seuil fixe (SLA), à recalibrer une fois plus de données réelles accumulées.
# Valeur de départ : les prédictions observées tournent à 40-55 ms (predict_proba sur
# un modèle déjà en mémoire) ; 200 ms est donc déjà ~4x la normale, un signal clair.
SEUIL_ANOMALIE_MS = 200

# "Dead man's switch" : au-delà de cette inactivité, on alerte — qu'elle soit due à
# une vraie absence de trafic ou à un logging cassé silencieusement, ce script ne peut
# pas trancher lequel (voir analyser_activite_recente). À recalibrer selon le trafic
# réel attendu une fois en production.
SEUIL_INACTIVITE_HEURES = 24


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


def _derniere_activite(database_url="sqlite:///./predictions.db"):
    """Timestamp de la dernière ligne loguée, toutes tables confondues (une prédiction
    ou une erreur comptent toutes les deux comme "l'app est vivante et logue")."""
    engine = create_engine(database_url)
    with engine.connect() as conn:
        dates = [
            conn.execute(text(f"SELECT MAX(created_at) FROM {table}")).scalar()
            for table in ("predictions", "error_events")
        ]
    dates = [datetime.fromisoformat(d).replace(tzinfo=timezone.utc) for d in dates if d is not None]
    return max(dates) if dates else None


def analyser_activite_recente(database_url="sqlite:///./predictions.db", seuil_heures=SEUIL_INACTIVITE_HEURES):
    """"Dead man's switch" : alerte si rien n'a été logué récemment.

    Ne distingue PAS "pas de trafic" de "le logging s'est cassé silencieusement" — les
    deux ont la même signature ici (rien dans la base). Pour trancher, croiser avec les
    logs applicatifs (main.py::compteur_echecs_logging, exposé aussi par GET /). Ce
    script signale juste l'absence de preuve d'activité, quelle qu'en soit la cause.
    """
    derniere = _derniere_activite(database_url)

    if derniere is None:
        print("⚠️  Aucune activité n'a jamais été loguée (base vide).")
        return False

    ecart = datetime.now(timezone.utc) - derniere
    if ecart > timedelta(hours=seuil_heures):
        print(
            f"⚠️  Dernière activité loguée il y a {ecart} — au-delà du seuil de "
            f"{seuil_heures}h. Vérifier s'il s'agit d'un manque de trafic ou d'un "
            f"problème de logging (voir GET / et les logs applicatifs)."
        )
        return False

    print(f"OK — dernière activité il y a {ecart}.")
    return True


def main():
    predictions = charger_predictions()
    erreurs = charger_erreurs()

    print("=== Activité récente (dead man's switch) ===")
    analyser_activite_recente()

    print("\n=== Latence ===")
    analyser_latence(predictions)

    print("\n=== Taux d'erreur ===")
    analyser_taux_erreur(predictions, erreurs)


if __name__ == "__main__":
    main()
