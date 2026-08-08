from dotenv import load_dotenv
import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

# Fallback sur SQLite local si DATABASE_URL n'est pas défini (pas de serveur à installer).
# Pour passer sur Postgres plus tard : définir DATABASE_URL dans .env, aucun code à changer ici.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")

# check_same_thread=False : nécessaire uniquement pour SQLite, ignoré par les autres moteurs.
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(Integer, index=True, nullable=False)
    prediction = Column(Integer, nullable=False)
    probabilite = Column(Float, nullable=False)
    resultat = Column(String, nullable=False)
    # Valeurs des colonnes les plus influentes (monitoring.FEATURES_MONITOREES) au moment
    # de la prédiction — sert de base à l'analyse de drift (comparaison à la distribution
    # d'entraînement). JSON plutôt qu'une colonne par feature : évite une migration de
    # schéma si la liste surveillée change.
    inputs = Column(JSON, nullable=False)
    duree_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Crée la table predictions si elle n'existe pas encore (pas d'Alembic pour ce projet).
Base.metadata.create_all(bind=engine)
