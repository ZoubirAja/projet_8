from conftest import KNOWN_CUSTOMER_ID, UNKNOWN_CUSTOMER_ID


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200


def test_predict_sans_cle_api_est_refuse(client):
    # Champ obligatoire manquant : pas de header X-API-Key.
    # 401 (pas 403) : FastAPI rejette l'absence du header au niveau de APIKeyHeader,
    # avant même d'exécuter verify_api_key (qui, lui, renvoie 403 pour une clé fausse mais présente).
    response = client.post(f"/predict/{KNOWN_CUSTOMER_ID}")
    assert response.status_code == 401


def test_predict_avec_mauvaise_cle_api_est_refuse(client):
    response = client.post(
        f"/predict/{KNOWN_CUSTOMER_ID}",
        headers={"X-API-Key": "mauvaise-cle"},
    )
    assert response.status_code == 403


def test_predict_id_connu_renvoie_une_prediction(client, auth_headers):
    response = client.post(f"/predict/{KNOWN_CUSTOMER_ID}", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert body["probabilite_de_defaut"].endswith("%")
    assert isinstance(body["resultat"], str) and body["resultat"]


def test_predict_id_inconnu_renvoie_une_erreur_explicite(client, auth_headers):
    response = client.post(f"/predict/{UNKNOWN_CUSTOMER_ID}", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == {"Erreur": "Aucun client pour cet ID"}


def test_predict_id_negatif_ne_plante_pas(client, auth_headers):
    # Valeur hors plage attendue (aucun SK_ID_CURR négatif) : ne doit pas faire planter l'API.
    response = client.post("/predict/-1", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == {"Erreur": "Aucun client pour cet ID"}


def test_predict_type_invalide_est_rejete(client, auth_headers):
    # Type de donnée incorrect : du texte au lieu d'un entier -> FastAPI valide le path automatiquement.
    response = client.post("/predict/abc", headers=auth_headers)

    assert response.status_code == 422
