"""Contrôle mensuel du drift : compare la référence (entraînement) uniquement aux
prédictions des 30 derniers jours — pas tout l'historique accumulé (contrairement à
analyse_drift.py, pensé pour un run manuel ponctuel). Pensé pour être planifié.

Lancer manuellement :
    .venv-monitoring/bin/python3 scripts/drift_mensuel.py

Planifier avec cron sur un serveur qui tourne en continu (le 1er de chaque mois, 3h) :
    0 3 1 * * cd /chemin/vers/le/projet && .venv-monitoring/bin/python3 scripts/drift_mensuel.py >> rapports_drift/cron.log 2>&1

Alternative si l'API tourne sur une plateforme sans serveur permanent à vous (le cas
ici) : un workflow GitHub Actions avec un déclencheur `schedule` (cron) — tourne sur
l'infra GitHub, pas besoin que votre machine soit allumée. Demandez si vous voulez que
je le mette en place.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from evidently import Report
from evidently.presets import DataDriftPreset

from monitoring import FEATURES_MONITOREES
from config import X as X_entrainement  # même preprocessing que le training (app/config.py)
from analyse_drift import charger_predictions_production, generer_page_resume

FENETRE_JOURS = 30
DOSSIER_SORTIE = "rapports_drift"


def main():
    depuis = datetime.now(timezone.utc) - timedelta(days=FENETRE_JOURS)
    reference = X_entrainement[FEATURES_MONITOREES]
    courant = charger_predictions_production(depuis=depuis)

    if courant.empty:
        print(f"Aucune prédiction depuis le {depuis.date()} — rien à analyser ce mois-ci.")
        return

    print(f"Fenêtre analysée : depuis le {depuis.date()} ({len(courant)} prédictions)")
    print(f"Référence (entraînement) : {len(reference)} lignes")

    rapport = Report(metrics=[DataDriftPreset()])
    resultat = rapport.run(current_data=courant, reference_data=reference)

    os.makedirs(DOSSIER_SORTIE, exist_ok=True)
    suffixe = datetime.now(timezone.utc).strftime("%Y-%m")  # un rapport par mois, jamais écrasé
    chemin_detail = os.path.join(DOSSIER_SORTIE, f"drift_report_{suffixe}.html")
    chemin_resume = os.path.join(DOSSIER_SORTIE, f"drift_summary_{suffixe}.html")
    resultat.save_html(chemin_detail)

    print("\nRésumé par colonne (méthode, distance à la référence, seuil) :")
    lignes_resume = []
    for metrique in resultat.dict()["metrics"]:
        config = metrique.get("config", {})
        colonne = config.get("column")
        if colonne is None:
            continue  # métriques globales (ex. nombre de colonnes driftées), pas par-colonne
        valeur, seuil = metrique["value"], config["threshold"]
        statut = "DRIFT" if valeur > seuil else "ok"
        lignes_resume.append({
            "colonne": colonne, "methode": config["method"], "valeur": valeur,
            "seuil": seuil, "statut": statut,
        })
        print(f"  [{statut:5s}] {colonne:35s} {config['method']:30s} {valeur:.3f} (seuil {seuil})")

    generer_page_resume(lignes_resume, len(reference), len(courant), chemin=chemin_resume, chemin_detail=chemin_detail)

    n_drift = sum(1 for l in lignes_resume if l["statut"] == "DRIFT")
    print(f"\nRapports : {chemin_resume} / {chemin_detail}")
    if n_drift > 0:
        print(f"⚠️  {n_drift}/{len(lignes_resume)} colonnes en drift ce mois-ci — à investiguer.")
    else:
        print("Aucun drift détecté ce mois-ci.")


if __name__ == "__main__":
    main()
