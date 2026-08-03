from config import X, y
from model import predict_model

score, auc, recall, seuil_optimal = predict_model(X, y)
print(f"Modèle entraîné — F1 score : {score}")
print(f"Modèle entraîné — recall score : {recall}")
print(f"Modèle entraîné — auc score : {auc}")