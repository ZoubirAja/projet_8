import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from config import X, y
from model import predict_model

score, auc, recall, seuil_optimal = predict_model(X, y)
print(f"Modèle entraîné — F1 score : {score}")
print(f"Modèle entraîné — recall score : {recall}")
print(f"Modèle entraîné — auc score : {auc}")