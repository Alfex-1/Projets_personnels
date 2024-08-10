# Prédiction du Nutri-Score (version 2)

Ce  projet reprend le [projet digital](https://github.com/Alfex-1/Projet_digital) mené au sein de ma formation de Master 2.
Cette version 2 ne revoit seulement que la partie consacrée au traitement des données et construction de modèles. Une toute nouvelle approche est présentée ici, notamment concernant le traitement des données.
En effet, deux objectifs ont été fixés au début de ce projet :
 1. Optimisation du traitement des données pour **minimiser le temps de calcul des différents modèles** tout en conservant la représentativité de la base de données à disposition.
 2. Comparaison de 4 modèles de **boosting** pour choisir le meilleur.

## Prérequis

### Téléchargement de la base de données

Du fait de sa grande volumétrie (9 Go), il n'est pas possible de les stocker sur GitHub. De ce fait, veuillez trouver la base de données utilisée en cliquant sur ce lien qui mène directement au site du gouvernement [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/open-food-facts-produits-alimentaires-ingredients-nutrition-labels/). Le fichier au format csv est celui qui est traité tout au long du projet. La création de nouveaux fichiers csv sera explicitement mentionnés dans les scripts Python.

### Utilisation des scripts

1. **Installation de Python** : Veuillez installer Python dans sa version 3.11 Vous pouvez la télécharger  sur [python.org](https://www.python.org/).
2. **Ordre à suivre lors de la naviguation des scripts** :
    - Le premier dossier vers lequel se diriger est le dossier "**Traitement de données**". A l'intérieur chaque script est numéroté désignant leur ordre auquel les scripts doivent être exécutés.
    - Le second dossier est "**Développement modèles**". La logique est la même que celle utilisée précédemment.
   
## Structure du dépôt 

- __docs__ : Le support business de présentation.      
- __src__      
    - **`\tools`** : Tous les scripts Python dont un dédié aux fonctions utilisées par les autres scripts       
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

1. **`Lecture_et_pretraitement.py`** lis la base de données et effectue des premiers traitements
2. **`Analyse_classe_d.py`** effectue un sous-échantillonage en supprimant les valeurs extrêmes de la classe majoritaire (classe D) dans l'objectif de minimiser significiativement le temps de calcul des modèles
3. **`Traitement_val_manq.py`** impute méthodiquement les données manquantes selon les caractéristiques de chaque variable
4. **`Rééquilibrage_données.py`** rééquilibre les classes en ajoutant autant d'observations "fictives" que possibles pour que toutes les classes soient aussi bien représentées que le classe majoritaire : ajout de données augmentant le temps de calcul des modèles.
5. **`Choix_meilleur_modèle.py`** choisist le meilleur de boosting grâce notamment à la validation croisée et à une vérification du sur-apprentissage
6. **`Fonctions.py`** répertorie toutes les fonctions Python qui servent à l'exécution des autres scripts.

Avant d'utiliser les scripts Python, veuillez d'abord exécuter le script des fonctions :

**Exécutez le script:** 
```bash
python Fonctions.py  
```

## Stratégie de traitement

D'abord, étant donné la grande quantité de variable dans la base, je me suis rapporté à ce que je connaissais sur le Nutri-Score : il est calculé à partir des valeurs nutritionnels exprimées par 100g. Donc toutes les variables en rapport avec ceci (en plus du nutri-score en lui même) sont les seuls variables conservées.
Les données n'ayant pas un nutri-score correct ont été également supprimés.
Après ces traitement, une base de données a été enregistrée avec 5% de données avant de les supprimer de la base pré-traitée

En temps normal, les données manquantes devraient être traitées en les supprimant ou en les imputant. Dans ce cas ce traitement sera effectué après la suppression des certaines observations de la classe D (la classe ayant le plus d'observations). L'utilité réelle de ce traitement sera expliquée lorsque viendra le temps du rééquilibrage des données. Au début, l'idée était de supprimer un certain nombre d'observations de la classe D totalement hasard. Mais étant donné que beaucoup de modèles sont sensibles aux valeurs extrêmes, autant supprimer ces données là. La détection des valeurs extrêmes s'est faite sur le plan multivarié.

L'étape suivante consistait à imputer les données manquantes. L'idée était d'étudier si les données manquantes des variables dépendaient des données manquantes des autres variables. Dans les cas où les variables sont MAR ou MCAR, alors les données manquantes sont imputées par KNNImputer, sinon par IterativeImputer. CE traitement arrive après la suppression des observations de la classe D car, l'évaluation des valeurs extrêmes devait se faire sur des vraies valeurs. De plus, imputer des valeurs qui vont être supprimées ensuite est une perte de temps et d'énergie.

Enfin, étant donné le désquilibre des classes évident, le rééquilibrage peut s'avérer utile pour des meilleures performances. Pour ce faire, un sur-échantillonnage a été effectué. Ce dernier créé des observations "fictives" ou synthétiques dans chaque classe afin que ces classes aient le même nombre d'observations que la classe ayant le plus (dans ce cas, la classe D). C'est pour cela qu'en premier lieu, j'ai supprimé au maximum les observations de la classe majoritaire, afin de ne pas trop gonfler artificiellement le nombre d'observations des autres classes et pour économiser les ressources lors de la construction des modèles. Le sur-échantillonnage utilise le concept de K plus proches voisins. Dans mon cas chaque observation synthétique résulte du calcul d'une interpolation entre une observation réelle de la classe associée et ses 4 plus proches voisines.

## Résultats