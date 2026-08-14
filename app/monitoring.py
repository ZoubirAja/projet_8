# Colonnes les plus influentes pour le modèle (importance globale CatBoost, calculée sur
# model.pkl le 2026-08-07 — top 10). Servent de base au logging
# des prédictions et à l'analyse de drift : contrairement à `explain.FEATURES_INTERPRETABLES`
# (choisies pour être lisibles par un humain), cette liste est choisie pour représenter
# fidèlement ce qui pèse vraiment sur les décisions du modèle.
FEATURES_MONITOREES = [
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "EXT_SOURCE_1",
    "DAYS_BIRTH",
    "AMT_GOODS_PRICE",
    "CODE_GENDER",
    "AMT_ANNUITY",
    "NAME_EDUCATION_TYPE",
    "DAYS_LAST_DUE_1ST_VERSION_max",
    "AMT_CREDIT",
]

# Description en langage clair, pour les rapports (drift_summary.html) — pas de quoi
# reconstituer la sémantique exacte d'un simple nom de colonne sans ça.
DESCRIPTIONS_FEATURES = {
    "EXT_SOURCE_2": "Score de solvabilité calculé par un organisme externe (bureau de crédit). Normalisé entre 0 et 1 ; la méthode de calcul exacte n'est pas connue de Home Credit.",
    "EXT_SOURCE_3": "Idem EXT_SOURCE_2, deuxième organisme externe.",
    "EXT_SOURCE_1": "Idem EXT_SOURCE_2, troisième organisme externe.",
    "DAYS_BIRTH": "Âge du client, en jours négatifs depuis aujourd'hui (convention du dataset).",
    "AMT_GOODS_PRICE": "Prix du bien financé par le crédit.",
    "CODE_GENDER": "Genre du client (encodé numériquement).",
    "AMT_ANNUITY": "Montant de la mensualité de remboursement.",
    "NAME_EDUCATION_TYPE": "Niveau d'éducation du client (encodé numériquement).",
    "DAYS_LAST_DUE_1ST_VERSION_max": "Délai avant l'échéance prévue du crédit précédent le plus récent (agrégé depuis l'historique bureau).",
    "AMT_CREDIT": "Montant du crédit demandé.",
}


def extraire_inputs_surveilles(customer_df):
    """Valeurs des colonnes FEATURES_MONITOREES pour ce client, sous forme de dict
    JSON-sérialisable (types numpy convertis en types Python natifs)."""
    row = customer_df.iloc[0]
    inputs = {}
    for colonne in FEATURES_MONITOREES:
        if colonne in row.index:
            valeur = row[colonne]
            inputs[colonne] = valeur.item() if hasattr(valeur, "item") else valeur
    return inputs
