from conftest import KNOWN_CUSTOMER_ID, UNKNOWN_CUSTOMER_ID
from customer import get_customer, log_prediction


def test_get_customer_id_connu_renvoie_une_ligne():
    result = get_customer(KNOWN_CUSTOMER_ID)

    assert result is not None
    assert result.shape[0] == 1
    assert result.index[0] == KNOWN_CUSTOMER_ID


def test_get_customer_id_inconnu_renvoie_none():
    assert get_customer(UNKNOWN_CUSTOMER_ID) is None


def test_get_customer_id_negatif_renvoie_none():
    # Valeur hors plage : aucun SK_ID_CURR n'est négatif dans les données.
    assert get_customer(-1) is None


def test_log_prediction_ne_leve_pas_d_erreur():
    # Ne vérifie pas le contenu de la BDD (détail d'implémentation),
    # juste que l'écriture ne plante pas avec des valeurs valides.
    log_prediction(
        customer_id=KNOWN_CUSTOMER_ID,
        prediction=1,
        probabilite=42.0,
        resultat="Le client aura du mal à rembourser son prêt",
    )
