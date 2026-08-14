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

# Bornes plausibles pour /simulate — évite qu'un revenu négatif, un âge de -5 ans ou
# 200 enfants ne soient acceptés silencieusement. (min, max), None = pas de borne de ce côté.
# DAYS_BIRTH/DAYS_EMPLOYED : jours négatifs (convention du dataset) → bornés entre
# 18 et 100 ans pour l'âge, et "pas dans le futur" pour l'emploi.
BORNES_SIMULATION = {
    "AMT_INCOME_TOTAL": (0, None),
    "AMT_CREDIT": (0, None),
    "AMT_ANNUITY": (0, None),
    "AMT_GOODS_PRICE": (0, None),
    "CNT_CHILDREN": (0, 20),
    "DAYS_EMPLOYED": (None, 0),
    "DAYS_BIRTH": (-100 * 365, -18 * 365),
}


def valider_bornes(valeurs: dict) -> list[str]:
    """Renvoie la liste des violations de bornes (vide si tout est valide)."""
    erreurs = []
    for colonne, valeur in valeurs.items():
        bornes = BORNES_SIMULATION.get(colonne)
        if bornes is None:
            continue
        minimum, maximum = bornes
        if minimum is not None and valeur < minimum:
            erreurs.append(f"{colonne}={valeur} : inférieur au minimum ({minimum})")
        if maximum is not None and valeur > maximum:
            erreurs.append(f"{colonne}={valeur} : supérieur au maximum ({maximum})")
    return erreurs


def get_top_influential_features(modele, customer_df, top_n=3, candidats=None, pool=None):
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

    `pool` : Pool CatBoost déjà construit, à réutiliser tel quel plutôt que d'en
    reconstruire un depuis `customer_df` (~11ms économisées par appel — voir
    notebooks/analyse_performance.ipynb). Si absent, un Pool est construit ici,
    pour rester utilisable indépendamment (notebooks d'analyse, etc.).
    """
    if pool is None:
        pool = Pool(customer_df)
    shap_values = modele.get_feature_importance(pool, type='ShapValues')[0]
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
