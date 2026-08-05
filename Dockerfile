FROM python:3.14-slim

WORKDIR /app

# libgomp1 : catboost en a besoin au runtime (parallélisation OpenMP), absent de l'image slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Modèle + données avant le code : ils changent moins souvent, le cache Docker
# reconstruit rarement cette couche même si main.py/customer.py changent.
# model.pkl est baké dans l'image ici pour simplifier le déploiement (pas de dépendance
# réseau/S3 au démarrage). En production, remplacer par un téléchargement depuis un
# stockage S3 (MinIO/R2) — voir la section "Déploiement en production" du README.
COPY model.pkl customers_indexed.parquet ./
COPY main.py customer.py database.py preprocessing.py ./

# Port 7860 : convention Hugging Face Spaces (SDK Docker). En local : docker run -p 8000:7860 ...
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
