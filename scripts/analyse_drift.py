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
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import pandas as pd
from sqlalchemy import create_engine

from evidently import Report
from evidently.presets import DataDriftPreset

from monitoring import FEATURES_MONITOREES, DESCRIPTIONS_FEATURES
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


def generer_page_resume(lignes_resume, n_reference, n_courant):
    """Page HTML lisible (titres, descriptions des variables, méthodologie) — écrite par
    nous, donc éditable. Contrairement à drift_report.html (bundle généré par evidently,
    réécrit intégralement à chaque run), c'est ici qu'il faut ajouter du contexte/des
    commentaires : ce fichier est régénéré à partir de ce code, pas modifié à la main.
    """
    lignes_html = "\n".join(
        f"""
        <tr class="{'drift' if l['statut'] == 'DRIFT' else 'ok'}">
          <td><code>{l['colonne']}</code></td>
          <td class="desc">{DESCRIPTIONS_FEATURES.get(l['colonne'], '—')}</td>
          <td>{l['methode']}</td>
          <td class="num">{l['valeur']:.3f}</td>
          <td class="num">{l['seuil']}</td>
          <td><span class="badge">{l['statut']}</span></td>
        </tr>"""
        for l in lignes_resume
    )
    n_drift = sum(1 for l in lignes_resume if l["statut"] == "DRIFT")
    horodatage = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Résumé — analyse de drift</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Arial, sans-serif; max-width: 900px;
          margin: 2.5rem auto; padding: 0 1.5rem; color: #17202A; line-height: 1.55; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 0.2rem; }}
  .meta {{ color: #56636D; font-size: 0.85rem; margin-bottom: 1.5rem; }}
  .methodo {{ background: #F2F4F3; border-radius: 8px; padding: 1rem 1.2rem; font-size: 0.92rem; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 1.5rem; font-size: 0.88rem; }}
  th {{ text-align: left; border-bottom: 2px solid #17202A; padding: 0.5rem 0.6rem; }}
  td {{ padding: 0.55rem 0.6rem; border-bottom: 1px solid #DADFDE; vertical-align: top; }}
  td.desc {{ color: #56636D; }}
  td.num {{ font-variant-numeric: tabular-nums; }}
  code {{ background: #E4F1F1; padding: 0.1em 0.4em; border-radius: 4px; font-size: 0.85em; }}
  tr.drift .badge {{ background: #A8571C; color: white; }}
  tr.ok .badge {{ background: #4E7A4A; color: white; }}
  .badge {{ padding: 0.15em 0.55em; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }}
  .note {{ font-size: 0.85rem; color: #56636D; margin-top: 1.2rem; }}
</style>
</head>
<body>
  <h1>Analyse de drift — résumé</h1>
  <p class="meta">Généré le {horodatage} · Référence (entraînement) : {n_reference:,} lignes ·
  Courant (production loguée) : {n_courant:,} lignes · {n_drift}/{len(lignes_resume)} colonnes en drift</p>

  <div class="methodo">
    <p><b>Méthode</b> : chaque variable surveillée est comparée entre les données
    d'entraînement du modèle (référence) et les prédictions réellement loguées par l'API
    (courant), via <code>evidently</code>. Distance de <b>Wasserstein</b> pour les
    variables numériques, <b>Jensen-Shannon</b> pour les catégorielles. Une colonne est
    marquée <b>DRIFT</b> si sa distance dépasse 0.1 (seuil par défaut d'evidently, non
    recalibré pour ce projet).</p>
  </div>

  <table>
    <thead>
      <tr><th>Variable</th><th>Description</th><th>Méthode</th><th>Distance</th><th>Seuil</th><th>Statut</th></tr>
    </thead>
    <tbody>{lignes_html}
    </tbody>
  </table>

  <p class="note">Rapport détaillé avec graphiques interactifs : <a href="drift_report.html">drift_report.html</a>
  (généré automatiquement par evidently, ne pas éditer à la main).</p>
</body>
</html>"""

    with open("drift_summary.html", "w", encoding="utf-8") as f:
        f.write(html)


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

    generer_page_resume(lignes_resume, len(reference), len(courant))

    print("\nRésumé clair (titres, descriptions des variables) : drift_summary.html")
    print("Rapport détaillé (graphiques interactifs, généré par evidently) : drift_report.html")


if __name__ == "__main__":
    main()
