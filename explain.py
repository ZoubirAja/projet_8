from catboost import Pool

# Colonnes "concrètes" (compréhensibles sans expertise data) parmi les 1264 du dataset —
# le reste est soit du bureau/previous_application agrégé (peu lisible), soit des scores
# externes opaques (EXT_SOURCE_*). Pas de "montant en banque" : Home Credit est un dossier
# de crédit/bureau, pas un relevé bancaire, cette donnée n'existe pas dans ce dataset.
FEATURES_INTERPRETABLES = [
    "AMT_INCOME_TOTAL",  # revenu total déclaré
    "AMT_CREDIT",        # montant du crédit demandé
    "AMT_ANNUITY",       # mensualité de remboursement
    "AMT_GOODS_PRICE",   # prix du bien financé par le crédit
    "DAYS_EMPLOYED",     # ancienneté dans l'emploi actuel, en jours (valeur négative)
    "DAYS_BIRTH",        # âge, en jours (valeur négative)
    "CNT_CHILDREN",      # nombre d'enfants à charge
]


def get_top_influential_features(pipeline, customer_df, top_n=3, candidats=None):
    """Renvoie les `top_n` colonnes qui ont le plus pesé sur CETTE prédiction précise.

    Valeurs de Shapley (calculées nativement par CatBoost), pas l'importance globale
    du modèle : elles changent d'un client à l'autre, contrairement à
    `get_feature_importance()` sans `type='ShapValues'` qui donne un classement fixe.
    Contribution positive = pousse vers "risque de défaut", négative = pousse vers
    "pas de risque".

    `candidats` restreint le classement à une liste de colonnes données (ex.
    FEATURES_INTERPRETABLES) — le SHAP est calculé sur tout le modèle, seul le
    classement final est filtré, donc le top reste bien "le plus influent parmi
    des variables compréhensibles", pas juste les 3 premières concrètes trouvées.
    """
    model = pipeline.named_steps['model']
    shap_values = model.get_feature_importance(Pool(customer_df), type='ShapValues')[0]
    contributions = shap_values[:-1]  # dernière valeur = biais du modèle, pas une feature

    items = list(zip(customer_df.columns, contributions, customer_df.iloc[0]))
    if candidats is not None:
        items = [item for item in items if item[0] in candidats]

    ranked = sorted(items, key=lambda item: abs(item[1]), reverse=True)

    return [
        {
            "feature": name,
            "contribution": round(float(contrib), 4),
            "valeur": value.item() if hasattr(value, "item") else value,
        }
        for name, contrib, value in ranked[:top_n]
    ]
