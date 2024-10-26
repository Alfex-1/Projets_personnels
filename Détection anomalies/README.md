# Détection et prédiction des fraudes financières

Ce  projet est inspiré du projet déjà réalisé par [Aman Kharwal](https://thecleverprogrammer.com/2023/08/21/anomaly-detection-in-transactions-using-python/). Il voulait détecter la présence d'anomalies dans des transactions financières sous *Python*, afin de développer un modèle permettant de prédire les transactions qui seront, à l'avenir, hautement suspectées d'être frauduleuses. Dans ce cas, une anomalie est une transaction pour laquelle la valeur en jeu est considérablement éloignée des montants moyens habituels.

Le présent projet n'a pas pour but de reprendre tout ce qui a déjà fait, mais plutôt apporter :
En effet, deux objectifs ont été fixés au début de ce projet :
 1. Des améliorations des travaux déjà réalisés, notamment l'exploration qui est plus poussée.
 2. Mais aussi et surtout des éléments supplémentaires avec une comparaison de plusieurs modèles de Machine Learning (simples comme complexes) et une détection des anomalies moins arbitraire et reposant plus sur les données elles-mêmes. Enfin, un sous-échantillonnage et sur-échantillonnage pour améliorer les résultats des modèles de Machine Larning.
 
## Prérequis

### Téléchargement et description de la base de données

Veuillez trouver la base de données utilisée en cliquant sur ce lien qui mène au site de [statso.io](https://statso.io/anomaly-detection-case-study/). Le fichier au format csv est celui qui est traité tout au long du projet. La création de nouveaux fichiers csv sera explicitement mentionnés dans les scripts Python.

- Transaction_ID : identifiant unique pour chaque transaction.
- Transaction_Amount : valeur monétaire de la transaction.
- Transaction_Volume : quantité ou le nombre d’éléments/actions impliqués dans la transaction.
- Average_Transaction_Amoun : montant moyen historique des transactions pour le compte.
- Frequency_of_Transactions : fréquence à laquelle les transactions sont généralement effectuées par le compte.
- Time_Since_Last_Transaction : temps écoulé depuis la dernière transaction.
- Day_of_Week : le jour de la semaine où la transaction a eu lieu.
- Time_of_Da y: l’heure de la journée à laquelle la transaction a eu lieu.
- Age : âge du titulaire du compte.
- Gender : genre du titulaire du compte.
- Income : revenu du titulaire du compte.
- Account_Type : type de compte (compte courant ou compte épargne).

### Utilisation des scripts

1. **Installation de Python** : Veuillez installer Python dans sa version 3.12.6 Vous pouvez la télécharger  sur [python.org](https://www.python.org/).
2. **Ordre à suivre lors de la naviguation des scripts** :
    a. **`Fonctions.py`**
    b. **`Etude exploratoire.py`**
    c. **`Detection_anomalie.py`**
    d. **`Modelisations.py`**
   
## Structure du dépôt 

- __docs__ : Les éléments graphiques illustrant l'exploration des données ainsi que les traitements (sous et sur-échantillonnage).  
- __src__ :  
    - **`\tools`** : Les scripts Python pour le projet.
        - **`\Detection_anomalie`** : Détection des anomalies sur le plan multivarié, sous-échantillonnage de la classe majoritaire et sur-échantllonnage de la classe minoritaire.
        - **`\Etude exploratoire`** : Etude exploratoire des données brutes.
        - **`\Fonctions`** : Importations des modules nécessaires et création de fonctions utiles pour le projet.
        - **`\Modelisation`** : Comparaison des algorithmes de Machine Learning pour la prévision des futures transactions potentiellement frauduleuses.
- __README.md__ : Le message qui décrit le projet         
- __requirements.txt__ : Liste des modules nécessaires à l'exécution des codes Python.      

## Installation

1. **Clonez le dépôt GitHub sur votre machine locale :** 
```bash
git clone https://https://github.com/Alfex-1/Projets_personnels.git
```

2. **Installez les dépendances requises:**

Pour Python, insérez cette ligne de commande dans le terminal :
```bash
pip install -r requirements.txt
```

3. **Avant d'utiliser les scripts Python, veuillez d'abord exécuter le script des fonctions :**

```bash
python Fonctions.py  
```

## Exploration des données

D'abord,

## Stratégie de traitement et de modélisation

Après l'exploration des données et avoir définit la proportion des montants qui sont vraisemblablement des anomalies, j'ai détecté sur le plan multivarié les anomalies. Les observations sont donc étiquetées en conséquence. Par cette détection, des transactions considérées comme normales ont été supprimées, cela représente un sous-échantillonnage (écarter et donc supprimer des observations de la classe majoritaire : la classe des transactions "normales"). Dans la continuité de ce sous-échantillonnage, les 10 % des observations les moins "normales" ont été également supprimées. 
Enfin, la classe minoritaire (les transactions potentiellement frauduleuses) a été gonflée de 30 observations. Cela semblait suffisant pour ne pas trop remplir cette classe d'observations artificielles. Par ces deux traitements, la classe majoritaire est moins importante qu'auparavant, contrairement à la classe minoritaire. Les données sont ensuite enregistrées pour les utiliser pour la modélisation.

Pour la modélisation, le sous et le sur-échantillonnage sont très importants dans le développement de modèles. Idéalement, les deux classes doivent être équilibrées (même nombre d'observation), mais dans ce cas, cela pourrait représenter un très grand problème de représentativité des données. Alors, l'objectif était seulement d'équilibrer légèrement les classes, sans les rendre totalement équilibrées.

3 modèles ont été comparés pour modéliser au mieux la relation entre les caractéristiques numériques des transactions et la possibilité qu'elles soient frauduleuses. Un modèle simple d'interprétation (la régression logistique), un modèle ayant un niveau intermédiaire d'interprétabilité (les K-NearNeighboors) puis un modèle complexe (le XGBoost). Leurs hyperparamètres ont été sélectionnés par validation croisée de sorte que ces hyperparamètres permettent de maximiser la capacité du modèle à distinguer les 2 classes que nous avons. La validation croisée permet de faire ce choix en minimisant le risque de sur-apprentissage du modèle, et donc de minimiser le risque que le modèle ne puisse pas se généraliser sur d'autres données.

Pour finir, pour savoir quel était le meilleur modèle, 3 métriques d'évaluation ont été calculées par validation croisée :
    - La précision : proportion de vrai positifs parmi tous les positifs (vrai comme faux) ;
    - Le recall : proportion de vrai positif parmi les vrai positif et les faux négatifs ;
    - Le F1 : combinaison du recall et de la précision pour évaluer entièrement les performances du modèle.

Pour finir, une matrice de confusion est établie pour chaque modèle pour visualiser s'il existe des faux négatif et/ou des faux positifs.

## Résultats

Pour détecter les anomalies, l'algorithme IsolationForest est le plus pertient, car à partir des caractéristiques de chaque individu, il peut évaluer lesquels s'écartent le plus de la moyenne. Cet algorithme a besoin d'un taux de contamination (la proportion d'anomalies présentes), et comme avec l'exploration, ce taux était estimé à 2 %, alors l'algorithme s'applique avec ce taux-ci. Cependant, avec une contamination de 2 %, toutes les anomalies sont assez mal détectées (beaucoup de valeurs "normales" sont considérées comme étant des anomalies alors que toutes les vraies anomalies ne sont pas considérées comme telles). De ce fait, l'évaluation s'est effectuée qu'avec les données numériques. Le problème est presque totalement réglé. Pour en finir, la contamination a augmenté, détectant ainsi toutes les vraies anomalies et 5 fausses anomalies, qui seront ensuite supprimées. Ensuite, 98 observations de la classe des transactions "normales" ont été supprimées puis 30 observations artificielles de la classe des anomalies ont été ajoutées. Au final, la base finale contient 877 observations de la classe normale et 50 observations de la classe des anomalies.

Pour finir, vous trouverez les métriques ainsi que les matrices de confusion dans la section suivante. Il apparaît que la régression logistique soit un modèle très peu adapté pour prédire les anomalies. Ensuite, les deux autres modèles ont de très bonnes performances, surtout le XGBoost. Pourtant, les KNN seront privilégiés, car il est beaucoup plus intuitif et facile à expliquer. Néanmoins, si l'interprétabilité n'est pas un critère dans le choix du modèle, alors c'est le XGBoost qui sera privilégié.

*Recommandation* : Si de nouvelles données apparaissent, le XGBoost ainsi que le choix des hyperparamètres, prendront beaucoup plus de temps que les KNN, surtout si ces hyperparamètres sont recherchées de manière exhaustive. Cela sera d'autant plus vrai avec l'augmentation du nombre de données.

### Évaluation des modèles de classification

#### Métriques de performance

| Métriques       | Régression logistique | K-NearNeighboors | XGBoost          |
|:----------------:|:---------------------:|:-----------------:|:----------------:|
| Précision       | 89,53%                | 93,26%            | 99,9%            |
| Recall          | 94,6%                 | 95,58%            | 99,89%           |
| F1-Score        | 92%                   | 93,96%            | 100%             |


#### Matrices de confusion

| **Régression logistique** | Prédit Normal | Prédit Anomalie |
|:-------------------------:|:-------------:|:---------------:|
| Réel Normal              | 294           | 0               |
| Réel Anomalie            | 50            | 0               |


| **K-NearNeighboors**     | Prédit Normal | Prédit Anomalie |
|:-------------------------:|:-------------:|:---------------:|
| Réel Normal              | 294           | 0               |
| Réel Anomalie            | 0             | 50              |


| **XGBoost**              | Prédit Normal | Prédit Anomalie |
|:-------------------------:|:-------------:|:---------------:|
| Réel Normal              | 294           | 0               |
| Réel Anomalie            | 0             | 50              |


