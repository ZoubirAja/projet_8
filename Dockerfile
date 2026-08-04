FROM python:3.14-slim

WORKDIR /app

# libgomp1 : catboost en a besoin au runtime (parallélisation OpenMP), absent de l'image slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Données avant le code : elles changent moins souvent, le cache Docker
# reconstruit rarement cette couche même si main.py/customer.py changent.
# model.pkl n'est plus copié dans l'image : il est téléchargé depuis MinIO au démarrage
# (voir main.py:lifespan / minio_client.download_model), ça évite de bake un binaire
# de plusieurs centaines de Mo à chaque build et de devoir rebuild pour changer de modèle.
COPY customers_indexed.parquet ./
COPY main.py customer.py database.py preprocessing.py ./

# Port 7860 : convention Hugging Face Spaces (SDK Docker). En local : docker run -p 8000:7860 ...
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
