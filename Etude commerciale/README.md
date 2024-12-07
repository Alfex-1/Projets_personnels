# Analyse des ventes de produits (2023)

Pour construire une stratégie marketing fiable, il est essentiel de comprendre les habitudes de consommation des clients. Cette étude analyse statistiquement les tendances de consommation des clients. Il est donc demande de répondre à deux questions majeures :

1. Existe-t-il des tendances significatives dans la vente de produits ?
2. La fluctuation du chiffre d’affaires de 2024 restera en moyenne similaire à celle de 2023 ?

Par ces questions, il faut tirer des besoins. D’abord, le premier besoin est d’analyser les achats des clients et voir si certains clients privilégies certains produits ou non et en quelle quantité. La seconde, c’est de faire des prévisions du chiffre d’affaires pour voir si la tendance de 2023 concernant le chiffre d’affaires se répète ou non.


## Prérequis
Les conditions préalables pour exploiter efficacement ce projet varient selon l'utilisation que vous comptez en faire. Voici les recommandations spécifiques :

### Utilisation des scripts utilisés dans cette étude :

1. **Installation de Python :** Veuillez installer Python dans sa version 3.11 Vous pouvez la télécharger  sur [python.org](https://www.python.org/).
1. **Installation de R :** Veuillez installer Python dans sa version 4.3.2 Vous pouvez la télécharger  sur [rstudio.com](https://cran.rstudio.com/bin/windows//base/old/).

   
## Structure du dépôt 

- __docs__ : Le support business de présentation.      
- __src__     
    - **`\data`** : Dossier où on retrouve le fichier .csv étant la base de données utilisées.      
    - **`\tools`** : Tous les codes Python et R dont un script Python dédié aux fonctions utilisées par les autres scripts       
- __README.md__ : Le message qui décrit le projet         
- __requirements_PytHon.txt__ : Liste des modules nécessaires à l'exécution des codes Python.  
- __requirements_R.txt__ : Liste des modules nécessaires à l'exécution des codes R.      

## Installation

1. **Clonez le dépôt GitHub sur votre machine locale:** 
```bash
git clone https://https://github.com/Alfex-1/Projets_personnels.git
```

2. **Installez les dépendances requises:**

Pour Python, insérez cette ligne de commande dans le terminal :
```bash
pip install -r requirements_Python.txt
```
Pour R, ouvrez et exécutez le script suivant:
```bash
requirements_R.R
```

## Utilisation

Le script **`Analyse_factorielle.R`** effectue une Analyse Factorielle de Données Mixtes (AFDM) pour analyser s'il existe des tendances dans le comportement des clients.
Le script **`Analyse_statistique.py`** est utilisé pour confirmer ou réfuter les hypothèses avancées avec les résultats de l'AFDM avec l'aide d'une série de tests statistiques.
Le script **`Prévisions_chiffre_affaires.py`** fournit les codes pour la modélisation et la prévision du chiffre d'affaires.
Le script **`Fonction.py`** répertorie toutes les fonctions Python qui servent à l'exécution des deux autres scripts Python.

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

Toutes les observations précédentes ont été réfutées par les tests statistiques. Autrement dit, il n'y a pas assez de preuve statistique pour affirmer que certains produits (comme les vêtements) sont significativement privilégiés par rapport aux produits électroniques.
Il y a néanmoins une observation qui est validée statistiquement : les quantités vendues n'ont réellement rien à voir avec l'âge des clients ni leur genre.

Conclusion : même si des actions marketing et commerciales sont menées, le chiffre d'affaires peut sûrement augmenter, mais cela ne restera que marginal.

## Modélisation du chiffre d'affaires

Le deuxième enjeu de cette étude est d'estimer le niveau du chiffre d'affaires pour l'année 2024, au du moins le début de l'année 2024.
Pour cela, deux modèles ont été utilisés : ARMA et GARCH. Ces modèles ont été ensuite passés dans la phase de validation pour confirmer si la modélisation était correcte mathématiquement.
Étant donné que l'évolution du chiffre d'affaires avait une moyenne constante au fil du temps, il n'y avait aucune raison de différencier la série (technique pour rendre stationnaire une série temporelle et modélisable par la suite).

Cependant, en choisissant le modèle ARMA pour modéliser l'évolution du chiffre d'affaire, il s'avère que c'est un modèle d'ordre 0 (ou ARMA(0,0)), ce qui signifie que la variance des données (l'information contenue dans les données) n'est pas modélisable. Cela se manifeste par le fait que les observations sont indépendantes et identiquement distribuées autour de la moyenne qui est une constante : ce n'est que du bruit blanc. Les prévisions se font donc seulement à partir de cette moyenne constante. Cela été observé en développement le modèle sous Python, mais le cas est similaire sous SAS. En termes d'hypothèses, les résidus présentent une hétéroscédasticité : la variance des résidus change au fil du temps. Pour remédier à ce problème, le modèle GARCH permet de mieux capter cette volatilité qui peut changer selon les périodes.

En développement un modèle GARCH(6,0) - donc un modèle ARCH(6) - il apparaît que des prévisions sur le long terme ne sont pas possibles car le modèle converge vers une seule valeur au bout de quelques jours. De plus, le modèle ne valide pas l'hypothèse d'hétéroscédasticité conditionnelle qui est fondamentale pour ce modèle. Autrement dit, le modèle ARCH (ou GARCH) est approprié lorsque la variance des résidus dépend de l'information passée. Ce n'est pas le cas ici.
En conclusion : ni le modèle ARMA, ni le modèle ARCH ne sont appropriés pour cette série temporelle, tout simplement car les données sont bruyantes et n'ont pas de struture temporelle modélisable.