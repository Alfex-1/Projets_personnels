# Modélisation du prix des maisons de l'Etat de l'Iowa (USA)

Ce projet traite de la modélisation du prix de près de 3000 maisons de l'Etat de l'Iowa (USA). Par cette modélisation, l'objectif était double.
D'abord, l'exploration et le traitement des données à disposition pour sélectionner les variables pertinentes après avoir traité les valeurs manquantes et détecter les anomalies.
Ensuite, avec la base de données nettoyée et pour établir des prévisions sur le prix des maisons grâce à leurs caractéristiques, deux régressions linéaires ont été étudiées comparées (avec et sans les anomalies). Cela s'est fait par le biais de leurs performances et de leur respect des hypothèses statistiques.
Les performances d'une régression polynomiale ont été étudiées pour voir si la régression linéaire suffirait ou non.
Tout s'est fait grâce au logiciel Python.

## Prérequis

### Téléchargement de la base de données

Vous trouverez la base de données utilisée en cliquant sur ce lien qui mène directement au site de Kaggle [kaggle.com](https://www.kaggle.com/datasets/marcopale/housing/data). Le fichier au format csv est celui qui est traité tout au long du projet. La création de nouveaux fichiers csv sera explicitement mentionnés dans les scripts Python.

### Utilisation des scripts

**Installation de Python** : Veuillez installer Python dans sa version 3.11 Vous pouvez la télécharger  sur [python.org](https://www.python.org/).
   
## Structure du dépôt 
   
- __src__      
    - **`\tools`** : Tous les scripts Python dont un dédié aux fonctions utilisées par les autres scripts
    - **`\data`** : Toutes les bases utilisées lors de ce projet (la base originale et les bases construites au cours du projet)
- __README.md__ : Le message qui décrit le projet         
- __requirements_Pyton.txt__ : Liste des modules nécessaires à l'exécution des codes Python.      

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

## Objectifs de chaque script

1. **`Developpement_modeles.py`** évalue l'impact des anomalies dans les performances d'une régression linéaire et évalue les performances d'une régression polynomiale d'ordre 2.
2. **`Fonctions.py`** répertorie toutes les fonctions Python qui servent à l'exécution des autres scripts.
3. **`Traitement.py`** effectue une exploration des données puis impute les données manquantes, sélectionne les variables pertinentes et détecte les anomalies. Il enregistre également la base de données traitées en y incluant la variable indicatrice désignant les observations qui sont des anomalies (i.e. des observations extrêmes et/ou aberrantes sur le plan multivarié).

Avant d'utiliser les scripts Python, veuillez d'abord exécuter le script des fonctions :

**Exécutez le script:** 
```bash
python Fonctions.py  
```

## Stratégie de traitement

Étant donné le grand nombre de variables (82 variables), l'objectif principal est de réduire la dimensionnalité (le nombre de variables). Pour cela, plusieurs filtres ont été appliqué et le premier d'entre eux était la proportion de données manquantes au sein de chaque variable. En effet, si elles ont 50 % des valeurs qui sont manquantes, ou plus, elles sont automatiquement évincées.
Le deuxième filtre repose sur la multicolinéarité, qui est un problème majeur en modélisation statistique. Une méthode itérative a été effectuée sur les variables quantitatives (numériques continues seulement) pour supprimer la variable qui a un VIF le plus élevé (indicateur de la multicolinéarité) jusqu'à toutes les variables ont un vif au maximum de 5.

Concernant les variables qualitatives et variables quantitatives discrètes, deux filtres ont été appliqués. Le premier s'appuie sur le contrôle des modalités. Plus précisément, les variables supprimées sont celles qui ont des modalités qui sont très déséquilibrées (si une ou deux reviennent beaucoup plus que les autres) : les données ne varient que très peu donc les intégrer dans le modèle est inutile. Ensuite, un test du Khi-Deux a été appliqué pour tester l'indépendance de chaque variable avec la variable cible. Les variables supprimées sont celles qui sont dépendantes de la variable cible.

Une fois que la pré-sélection est effectuée, les valeurs manquantes ont été imputées par estimation du K plus proche voisins (KNN) avec K=3 et pondérés par la distance entre chaque valeur manquante et leurs 3 plus proches voisins. En temps normal, ce ne sont pas les KNN qui sont utilisées pour imputer TOUTES les valeurs manquantes. Cependant, étant donné que l'algorithme qui complète les KNN est en phase expérimentale, ce sont les KNN qui se chargent de faire tout le travail.

L'étape suivante était d'explorer les données (quantitatives continues) pour se donner une idée de la quantité de données extrêmes existantes au sein des données. Cette proportion s'est visuellement estimée à 5 %. Un algorithme a ensuite détecté, sur le plan multivarié, quel étaient les individus qui constituaient les 5 % : quelles observations sont des anomalies ?
Ces anomalies ne seront pas traitées, mais seront tout de même utilisées pour évaluer leur impact sur les performances du modèle développé.

Ensuite, comme les variables qualitatives restantes ne sont pas numériques, il a fallu les encoder. Pour cela, des variables ont été créées autant qu'il existe de modalités dans chaque variable. L'idée est que pour chaque observation, une nouvelle colonne est exprimée avec la valeur "1" si la modalité correspondante est présente, et "0" sinon. Cela permet de transformer des variables catégorielles en un format compréhensible par les algorithmes de machine learning tout en préservant l'information de chaque modalité.

Maintenant, les données sont entièrement "compréhensibles" par un modèle. Même si une pré-sélection a eu lieu, ce n'est pas suffisant, car il reste encore beaucoup de variables. L'idéal sera d'en avoir maximum 15, pour éviter le sur-ajustement. Donc, une sélection unvariée a été faite, en choissiant de garder les 20 meilleurs variables. Puis, pour prendre en compte des relations entre les variables. Une sélection multivariée se fait à partir de l'évaluation des performances d'un Random Forest par validation croisée avec l'élimination itérative de variable. Ainsi, les variables conservées minimisent d'une régression linéaire (modèle utilisé par la suite).

Ainsi pour la base avec et sans les anomalies, seules 11 variables ont été conservées, cependant 10 d'entre elles sont communes au 2 bases.

## Stratégie de modélisation et résultats

L'objectif était de construire un modèle simple puis d'aller vers un modèle un peu plus complexe. Dans cette optique, une régression linéaire a été développée d'une part pour l'ensemble des données (anomalies comprises) et d'autre part sans les anomalies. À première vue, le modèle sans les anomalies affichait de meilleures performances en termes de critère d'information (AIC, BIC) et de variance expliquée (R² ajusté). Pour les deux modèles, une vérification du respect des hypothèses était de rigueur. Il semblerait encore une fois que le modèle sans anomalies respecte un peu mieux les hypothèses que l'autre, mais pas de manière significative. De même, lorsque les erreurs de prévision sont comparées.

Ensuite, les performances d'une régression polynomiale d'ordre 2 ont été évaluées. Cela semblait être une bonne idée, car il n'existe pas de relation entre les variables explicatives et la variable cible. Au final, les deux types de modèles (linéaire et polynomiale) affichent globalement les mêmes performances. Pour aller dans la simplicité, la régression linéaire est retenue pour faire de futures estimations.