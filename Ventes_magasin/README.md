# Analyse des ventes de produits (2023)

Pour construire une stratégie marketing fiable, il est essentiel de comprendre les habitudes de consommation des clients. Cette étude analyse statistiquement les tendances de consommation des clients. Il est donc demande de répondre à deux questions majeures:

1. Existe-t-il des tendances significatives dans la vente de produits ?
2. La fluctuation du chiffre d’affaires de 2024 restera en moyenne similaire à celle de 2023 ?

Par ces questions, il faut tirer des besoins. D’abord le premier besoin est d’analyser les achats des clients et voir si certains clients privilégies certains produits ou non et en quelle quantité. La seconde c’est de faire des prévisions du chiffre d’affaires pour voir si la tendance de 2023 concernant le chiffre d’affaires tant à se répéter ou non.


## Prérequis
Les conditions préalables pour exploiter efficacement ce projet varient selon l'utilisation que vous comptez en faire. Voici les recommandations spécifiques :

### Utilisation de l'Algorithme de Calcul du Nutri-Score :

1. **Installation de Python :** Veuillez installer Python dans sa version 3.11 Vous pouvez la télécharger  sur [python.org](https://www.python.org/).
1. **Installation de R :** Veuillez installer Python dans sa version 4.3.2 Vous pouvez la télécharger  sur [rstudio.com](https://cran.rstudio.com/bin/windows//base/old/).

   
## Structure du dépôt 

- __docs__ : Le support business de présentation.      
- __src__     
    - **`\data`** : Dossier où on retrouve le les fichier .csv étant la base de données utilisées.      
    - **`\tools`** : Ttous les codes Python et R dont un script Python dédié aux fonctions utilisées par les autres scripts       
- __tests__ : Tests unitaires effectués sur les fonctions.       
- __README.md__ : Le message qui décrit le projet         
- __requirements_Pyton.txt__ : Liste des modules nécessaires à l'éxecution des codes Python.  
- __requirements_R.txt__ : Liste des modules nécessaires à l'éxecution des codes R.      

## Installation

1. **Clonez le dépôt GitHub sur votre machine locale:** 
```bash
git clone https://https://github.com/Alfex-1/Projets_personnels.git
```

2. **Installez les dépendances requises:**

Pour Python, insérez cette ligne de commande dans le terminal :
```bash
pip install -r requirements_Pyton.txt
```
Pour R, ouvrez le script `requirements_R.R`

## Utilisation

Deux modèles RandomForest sont déjà entrainés et à disposition au format pickle (**`random_forest_prod.pickle`** et **`random_forest_conso.pickle`**) dans le répertoire **`src\tools`**. L'un est à disposition des **consommateurs** et l'autre à l'attention des **producteurs**.

Si vous voulez générer un nouveau modèle, il vous suffit de lancer dans un terminal le script `make_random_forest.py` disponible dans ce même répertoire. Il vous permettra de créer un nouveau modèle RandomForest cosommateur ou producteur. Avant de l'exécuter, assurez vous d'être dans le répertoire `src\tools`.  

**Exécutez le script:** 
```bash
python main_random_forest.py  
```
