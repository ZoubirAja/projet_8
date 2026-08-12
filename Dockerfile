FROM python:3.14-slim

WORKDIR /app

# libgomp1 : catboost en a besoin au runtime (parallélisation OpenMP), absent de l'image slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Modèle + données avant le code : ils changent moins souvent, le cache Docker
# reconstruit rarement cette couche même si le code applicatif change.
# model.pkl est baké dans l'image ici pour simplifier le déploiement (pas de dépendance
# réseau/S3 au démarrage). En production, remplacer par un téléchargement depuis un
# stockage S3 (MinIO/R2) — voir la section "Déploiement en production" du README.
COPY model.pkl customers_indexed.parquet ./
COPY app/ ./app/

# Port 7860 : convention Hugging Face Spaces (SDK Docker). En local : docker run -p 8000:7860 ...
EXPOSE 7860

# --app-dir app : main.py et les modules qu'il importe (customer, minio_client, monitoring,
# explain...) vivent dans app/, sans réécrire leurs imports internes en imports de package.
CMD ["uvicorn", "main:app", "--app-dir", "app", "--host", "0.0.0.0", "--port", "7860"]
