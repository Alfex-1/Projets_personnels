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

## Description de la base

- Transaction ID : le numéro de la transaction
- Customer ID : l'idendifiant du client ayant fait son achat à la date T
- Date : date à laquelle s'est réalisée la transaction. Les dates vont du 01/01/2023 au 01/01/2023
- Gender : genre du client (Male ou Female) ayant réalisé la transaction
- Age : âge du client ayant réalisé la transaction
- Product Category : catégorie du produit acheté (Beauty, Clothing ou Electronics)
- Quantity : quantité achetée par transaction
- Price per Unit : prix unitaire du produit acheté
- Total_Amount : bénéfice tiré de la transaction

## Résultats

### Analyse factorielle de données 

Grâce à cette analyse, des conclusions ont pu être tirées sur l'impact qu'on le genre et l'âge des clients sur les types de produits qu'ils achètent et sur leur quantité.
L'objectif est d'éclairer les décisions en isolant des comportements d'achat des clients, s'il en existe.

Globalement, ce sont les vêtements qui sont les plus achetés et les produits électroniques qui sont les moins achetés. Ce sont d'ailleurs les clients les plus âgés qui achètent ces deux types de produits, mais les produits de beauté sont plutôt privilégiés par les femmes, contrairement aux hommes qui préfèrent les deux autres types de produits. Ensuite, les clients les plus âgés semblent privilégier les vêtements et l’électronique et les plus jeunes les produits de beauté.

Enfin, les quantités vendues ne semblent être en dépendants ni de l'âge ni du genre des clients.

Malheureusement, ces analyses sont peu robustes en raison de la faible représentativité des différentes variables sur le plan factorielle. D'où la nécessité de passer par des tests statistiques pour avoir des preuves des liens qui peuvent potentiellement exister.

### Tests statistiques

Ces tests ont pour seul objectif de confirmer ou non les observations effectuées grâce à l'analyse factorielle précédente. Cette confirmation (ou réfutation) se fait par le biais des P-Value des tests. Lorsque cette P-Value est inférieure à un certain seuil (posons 5 % : 0,05), le test démontre qu'il y a un effet. Cet effet change en fonction du test effectué. Des précisions seront faites au moment voulu.

Toutes les observations précédentes ont été réfutes par les tests statistiques. Autrement dit, il n'y a pas assez de preuve statistique pour affirmer que certains produits (comme les vêtements) sont significativement privilégiés par rapport aux produits électroniques.
Il y a néanmoins une observation qui est validée statistiquement : les quantités vendues n'ont réellement rien à voir avec l'âge des clients ni leur genre.

Conclusion : même si des actions marketing et commerciales sont menées, le chiffre d'affaires peut sûrement augmenter, mais cela ne restera que marginal.

## Modélisation du chiffre d'affaires

Le deuxième enjeu de cette étude est d'estimer le niveau du chiffre d'affaires pour l'année 2024, au du moins le début de l'année 2024.
Pour cela, deux modèles ont été utilisés : ARMA et GARCH. Ces modèles ont été ensuite passés dans la phase de validation pour confirmer si la modélisation était correcte mathématiquement.
Étant donné que l'évolution du chiffre d'affaires avait une moyenne constante au fil du temps, il n'y avait aucune raison de différencier la série (technique pour rendre stationnaire une série temporelle).

Cependant, même le meilleur modèle ARMA (ARMA(0,0)) n'arrive pas à capter la structure temporelle dans les données. Autrement dit, les observations sont indépendantes et identiquement distribuées autour de la moyenne qui est une constante : ce n'est que du bruit blanc. Les prévisions se font donc seulement à partir de cette moyenne constante. De plus, les résidus présentent une hétéroscédasticité : la variance des résidus change au fil du temps. Pour remédier à ce problème, le modèle GARCH permet de mieux capter cette volatilité qui peut changer selon les périodes.

En développement un modèle GARCH(6,0) - donc un modèle ARCH(6) - il apparaît que des prévisions sur le long terme ne sont pas possibles car le modèle converge vers une seule valeur au bout de quelques jours. De plus, le modèle ne valide pas l'hypothèse d'hétéroscédasticité conditionnelle qui doit être présente. Autrement dit, le modèle ARCH (ou GARCH) est approprié lorsque la variance des résidus dépend de l'information passée. Ce n'est pas le cas ici.
En conclusion : ni le modèle ARMA, ni le modèle ARCH ne sont appropriés pour cette série temporelle. L'explication peut-être que la base de données est simulée et que les valeurs sont générées aléatoirement.