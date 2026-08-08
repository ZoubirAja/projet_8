import pandas as pd
from database import SessionLocal, Prediction, ErrorEvent

# Écrit une fois par prepare_parquet.py : mêmes clean_col_names/downcast_floats qu'à
# l'entraînement (config.py), row groups de 5000 lignes triées par SK_ID_CURR.
PARQUET_PATH = "customers_indexed.parquet"
ID_COL = "SK_ID_CURR"


def get_customer(customer_id: int) -> pd.DataFrame | None:
    """Retourne les données d'un client sous forme de DataFrame à une ligne, ou None si l'id est inconnu.

    Ne charge pas tout le fichier en mémoire (307 511 lignes x 1265 colonnes ≈ 8 Go une fois
    décompressé, trop pour lancer via `fastapi dev --reload` sur une machine à RAM limitée) :
    le filtre "==" est appliqué par pyarrow au niveau des row groups (grâce au tri par id lors
    de la préparation), donc un seul petit groupe de ~5000 lignes est décompressé par appel.
    """
    result = pd.read_parquet(
        PARQUET_PATH,
        engine="pyarrow",
        filters=[(ID_COL, "==", customer_id)],
    )
    if result.empty:
        return None
    return result.set_index(ID_COL, drop=False)


def log_prediction(
    customer_id: int,
    prediction: int,
    probabilite: float,
    resultat: str,
    inputs: dict,
    duree_ms: float,
) -> None:
    """Enregistre le résultat d'une prédiction. Une session par appel (courte durée de vie)."""
    db = SessionLocal()
    try:
        db.add(Prediction(
            customer_id=customer_id,
            prediction=prediction,
            probabilite=probabilite,
            resultat=resultat,
            inputs=inputs,
            duree_ms=duree_ms,
        ))
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def log_error(route: str, message: str) -> None:
    """Enregistre une erreur serveur (500). Appelé depuis le handler global —
    doit rester silencieux en cas d'échec (on ne veut pas qu'un souci de BDD
    masque l'erreur d'origine qu'on est justement en train de logguer)."""
    db = SessionLocal()
    try:
        db.add(ErrorEvent(route=route, message=message))
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()