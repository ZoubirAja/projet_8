"""Verrou de qualité CI/CD : bloque la publication de l'image Docker si le modèle
inclus (model.pkl) est en dessous d'un seuil minimum de performance.

Contrairement à un vrai model registry (MLflow, SageMaker...), il n'y a pas de
comparaison automatique à la version précédente ici — juste un seuil absolu, simple
à vérifier sans historique de runs à maintenir. Voir la fiche de révision, section
"Cycle de vie MLOps", pour la nuance avec un vrai registre.

Lancé en CI avant le build Docker (.github/workflows/ci-cd.yml) :
    python3 scripts/verifier_qualite_modele.py
Sort avec un code non-nul (échec du job) si le score est sous le seuil.
"""
import sys

import joblib

# F1 réel du modèle actuel : 0.30. Seuil fixé nettement en dessous (assez de marge pour
# ne pas bloquer sur du bruit d'entraînement normal) mais assez haut pour attraper un
# modèle vraiment cassé (ex. F1 proche de 0 après un bug d'entraînement).
SEUIL_MINIMUM_F1 = 0.20


def main():
    pipeline, seuil_optimal, score = joblib.load("model.pkl")

    print(f"F1 score du modèle : {score:.4f} (seuil minimum : {SEUIL_MINIMUM_F1})")

    if score < SEUIL_MINIMUM_F1:
        print(f"❌ Échec : F1 ({score:.4f}) sous le seuil minimum ({SEUIL_MINIMUM_F1}).")
        print("   Publication de l'image bloquée — modèle probablement défectueux.")
        sys.exit(1)

    print("✅ Modèle au-dessus du seuil minimum — publication autorisée.")


if __name__ == "__main__":
    main()
