import requests

# URL de base de l'API
url_base = 'http://127.0.0.1:8000'

# Test du endpoint d'accueil
response = requests.get(f"{url_base}/")
print("Réponse du endpoint d'accueil:", response.text)
# Données d'exemple pour la prédiction
donnees_predire = {
    "Energie_kcal" : 264,
    "Graisses" : 22.1,
    "Dont_graisses_saturees" : 3.7,
    "Glucides" : 9.5,
    "Dont_sucres" : 2.5,
    "Fibres" : 28,
    "Proteines" : 8.1,
    "Sel" : 0.9,
}

# Test du endpoint de prédiction
response = requests.post(f"{url_base}/predire", json=donnees_predire)
print("Réponse du endpoint de prédiction:", response.text)