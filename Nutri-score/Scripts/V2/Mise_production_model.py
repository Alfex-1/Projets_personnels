import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from pydantic import BaseModel

# Sérialisation du modèle
joblib.dump(model,'model_nutriscore.pkl')

# Charger le modèle
modele = joblib.load('model_nutriscore.pkl')

# Définition des données d'entrée
class DonneesEntree(BaseModel):
    Energie_kcal : float
    Graisses : float
    Dont_graisses_saturees : float
    Glucides : float
    Dont_sucres : float
    Fibres : float
    Proteines : float
    Sel : float

# Création de l'instance de l'application Flask
app = Flask(__name__)

# Définition de la route racine qui retourne un message de bienvenue
@app.route("/", methods=["GET"])
def accueil():
    """ Endpoint racine qui fournit un message de bienvenue. """
    return jsonify({"message": "Bienvenue sur l'API de prédiction pour le diagnostic du diabète"})

# Définition de la route pour les prédictions de diabète
@app.route("/predire", methods=["POST"])
def predire():
    """
    Endpoint pour les prédictions en utilisant le modèle chargé.
    Les données d'entrée sont validées et transformées en DataFrame pour le traitement par le modèle.
    """
    if not request.json:
        return jsonify({"erreur": "Aucun JSON fourni"}), 400
    
    
    try:
        # Extraction et validation des données d'entrée en utilisant Pydantic
        donnees = DonneesEntree(**request.json)
        donnees_df = pd.DataFrame([donnees.dict()])  # Conversion en DataFrame

        # Utilisation du modèle pour prédire et obtenir les probabilités
        predictions = modele.predict(donnees_df)
        probabilities = modele.predict_proba(donnees_df)

        # Compilation des résultats dans un dictionnaire
        resultats = donnees.dict()
        resultats['prediction'] = int(predictions[0])
        resultats['probabilite_diabete'] = probabilities[0]

        # Renvoie les résultats sous forme de JSON
        return jsonify({"resultats": resultats})
    except Exception as e:
        # Gestion des erreurs et renvoi d'une réponse d'erreur
        return jsonify({"erreur": str(e)}), 400

# Point d'entrée pour exécuter l'application
if __name__ == "__main__":
    app.run(debug=True, port=8000)