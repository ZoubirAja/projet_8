import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import customer
import database
import main

SAMPLE_PARQUET = "tests/fixtures/sample_customers.parquet"
TEST_API_KEY = "test-api-key"

# Ids réellement présents dans tests/fixtures/sample_customers.parquet
KNOWN_CUSTOMER_ID = 100002
UNKNOWN_CUSTOMER_ID = 999999999


@pytest.fixture(autouse=True)
def _use_sample_parquet(monkeypatch):
    """Toutes les requêtes de test lisent le petit fixture versionné, jamais
    customers_indexed.parquet (275 Mo, gitignoré, absent en CI)."""
    monkeypatch.setattr(customer, "PARQUET_PATH", SAMPLE_PARQUET)


@pytest.fixture(autouse=True)
def _use_test_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", TEST_API_KEY)


@pytest.fixture(autouse=True)
def _use_in_memory_database(monkeypatch):
    """Évite que les tests écrivent dans predictions.db (la vraie base locale)."""
    test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    database.Base.metadata.create_all(bind=test_engine)
    TestSessionLocal = sessionmaker(bind=test_engine)
    monkeypatch.setattr(customer, "SessionLocal", TestSessionLocal)


@pytest.fixture
def client():
    # `with` déclenche le lifespan (download_model + joblib.load) ; sans ça,
    # `pipeline` reste None et les prédictions plantent.
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def auth_headers():
    return {"X-API-Key": TEST_API_KEY}
