# Prédiction du Nutri-Score (version 2)

Ce  projet reprend le [projet digital](https://github.com/Alfex-1/Projet_digital) mené au sein de ma formation de Master 2.
Cette version 2 ne revoit seulement que la partie consacrée au traitement des données et construction de modèles. Une toute nouvelle approche est présentée ici.
En effet, deux objectifs ont été fixés au début de ce projet :
 1. Optimisation du traitement des données pour **minimiser le temps de calcul des différents modèles** tout en conservant la représentativité de la base de données initiale
 2. Comparaison de deux modèles de régression logistique ordinale pour  évaluer l'impact des classes déséquilibrées, puis pour choisir le meilleur modèle.
 
 *Important* : Il était prévu initialement d'utiliser des modèles de **Boosting** et des **réseaux de neurones (MLP)**. Malheureusement, étant donné la très grande quantité de données et les contraintes techniques, l'idée a été abandonné, bien que des codes ont été développés. Des modèles très simples (de régression logistique ordinale) ont été exploités à la place.

## Prérequis

### Téléchargement de la base de données

Du fait de sa grande volumétrie (9 Go), il n'est pas possible de les stocker sur GitHub. De ce fait, veuillez trouver la base de données utilisée en cliquant sur ce lien qui mène directement au site du gouvernement [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/open-food-facts-produits-alimentaires-ingredients-nutrition-labels/). Le fichier au format csv est celui qui est traité tout au long du projet. La création de nouveaux fichiers csv sera explicitement mentionnés dans les scripts Python.

### Utilisation des scripts

1. **Installation de Python** : Veuillez installer Python dans sa version 3.11 Vous pouvez la télécharger  sur [python.org](https://www.python.org/).
2. **Ordre à suivre lors de la naviguation des scripts** :
    - Le premier dossier vers lequel se diriger est le dossier "**Traitement de données**". A l'intérieur chaque script est numéroté désignant leur ordre auquel les scripts doivent être exécutés.
    - Le second dossier est "**Développement modèles**". La logique est la même que celle utilisée précédemment.
   
## Structure du dépôt 

- __docs__ : Les éléments graphiques illustrant l'exploration des données et les résultats de la modélisation.  
- __src__ :  
    - **`\tools`** : Les scripts Python pour le projet dont un script dédié aux fonctions utilisés par les autres scripts.
        - **`\Développement de modèles`** : Script Python utilisé pour développer et comparer les régression logistiques. Effectue avant une sélection de variable basée sur la corrélation entre les variables explicatives.
        - **`\Traitement des données`** : Scripts Python qui permettent de rendre les données utilisables dans un modèle statistique : gestion des valeurs manquantes et extrêmes. Effectue également un sous-échantillonnage sur la classe majoritaire et un sur-échantillonnage sur les autres classes.
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

1. **`1. Lecture_et_pretraitement.py`** lis la base de données et effectue des premiers traitements.
2. **`2. Analyse_classe_d.py`** effectue un sous-échantillonage en supprimant les valeurs extrêmes de la classe majoritaire (classe D) dans l'objectif de minimiser significiativement le temps de calcul des modèles.
3. **`3. Traitement_val_extremes.py`** détecte les valeurs des observations qui sont anormalement très élevés et les remplace par des valeurs manquantes.
4. **`4. Traitement_val_manq.py`** impute méthodiquement les données manquantes selon les caractéristiques de chaque variable
5. **`5. Rééquilibrage_données.py`** rééquilibre les classes en ajoutant autant d'observations "fictives" que possibles pour que toutes les classes soient aussi bien représentées que le classe majoritaire : ajout de données augmentant le temps de calcul des modèles.
6. **`Choix_meilleur_modèle.py`** choisit la meilleure régression logistique ordinale et affichage de graphiques permettant de comprendre la relation entre les valeurs explicatives (nutriments : sel, graisses, etc.) et la variable cible (Nutri-Score).
7. **`Fonctions.py`** répertorie toutes les fonctions Python qui servent à l'exécution des autres scripts, ainsi que les modules qui permettent le fonctionnement des fonctions et des scripts.

Avant d'utiliser les scripts Python, veuillez d'abord exécuter le script des fonctions :

**Exécutez le script:** 
```bash
python Fonctions.py  
```

## Stratégie de traitement et de modélisation

D'abord, étant donné la grande quantité de variable dans la base, je me suis rapporté à ce que je connaissais sur le Nutri-Score : il est calculé à partir des valeurs nutritionnelles exprimées par 100 g. Donc toutes les variables en rapport avec ceci (en plus du Nutri-Score en lui-même) sont les seules variables conservées.
Les données n'ayant pas un Nutri-Score correct ont été également supprimées.
Après ces traitements, une base de données a été enregistrée avec 5 % de données avant de les supprimer de la base pré-traitée.

En temps normal, les données manquantes devraient être traitées en les supprimant ou en les imputant. Dans ce cas, ce traitement sera effectué après la suppression de certaines observations de la classe D (la classe ayant le plus d'observations). L'utilité réelle de ce traitement sera expliquée lorsque viendra le temps du rééquilibrage des données. Au début, l'idée était de supprimer un certain nombre d'observations de la classe D totalement hasard. Mais étant donné que beaucoup de modèles sont sensibles aux valeurs extrêmes, autant supprimer ces données-là. La détection des valeurs extrêmes s'est faite sur le plan multivarié.


En explorant les données, des valeurs anormalement élevées sont visuellement. Dans ce cas, l'écart inter-quartile a été utilisée pour cibler ces valeurs de manière individuelle, puis les remplacer par des valeurs manquantes. Ce changement se justifiait par l'étape suivante qui consistait à imputer les données manquantes. Ces dernières (dont celles qui ont été rendues manquantes) ont été imputées en faisant la moyenne des valeurs prises par leurs 3 voisins les plus proches. Les valeurs moyennées ont été pondérées par la distance de la valeur à imputer.


Enfin, étant donné le déséquilibre des classes évident, le rééquilibrage peut s'avérer utile pour des meilleures performances. Pour ce faire, un sur-échantillonnage a été effectué. Ce dernier créé des observations synthétiques dans chaque classe afin qu'elles aient le même nombre d'observations que la classe ayant le plus (dans ce cas, la classe D). C'est pour cela qu'en premier lieu, j'ai supprimé au maximum les observations de la classe majoritaire, afin de ne pas trop gonfler artificiellement le nombre d'observations des autres classes et pour économiser les ressources lors de la construction des modèles. Le sur-échantillonnage utilise le concept de K plus proches voisins. Dans mon cas, chaque observation synthétique résulte du calcul d'une interpolation entre une observation réelle de la classe associée et ses 4 plus proches voisines.


Concernant la modélisation, une régression logistique ordinale a été utilisée, pour faire face aux contraintes techniques. Deux modèles ont été entraînés : d'une part un modèle ajusté sur les données ayant les classes équilibrées et d'autre part un modèle ajusté sur les données ayant les classes déséquilibrées. En outre, l'accuracy était choisis pour évaluer les performances du "modèle équilibré" et pour le "modèle déséquilibré", c'est le F1-Score qui était choisis (car il prend en compte le déséquilibre des classes). Pour chaque modèle, les hypothèses statistiques ont été vérifiées ainsi que les rapports de côtes (*odds ratios* en anglais).

## Conclusion de la modélisation

D'abord, la variable concernant l'apport en kilocalories (pour 100g) a été supprimée du jeu de données car elle est très corrélée à d'autres variables. Si elle avait été conservée, de la multicolinéarité aurait affecté négativement les modèles. D'ailleurs, il semblerait à première vue que le "modèle déséquilibré" soit plus performant, car ses performances sont estimées à 0.1229, alors que pour le "modèle équilibré" elles sont estimées à 0.1131. Cependant, ce ne sont pas les mêmes métriques qui sont utilisées (F1-Score VS Accuracy), donc ça ne peut être directement comparé. Pour éclaircir ce point, une matrice de confusion a été dressée pour chaque modèle. Ces matrices se lisent en ligne (profils lignes) de sorte à mettre en évidence les bonnes et mauvaises prédictions par classe. Avec ces matrices, il apparaît que les bonnes prédictions du modèle déséquilibré sont meilleures globalement que celles du modèle équilibré (d'où un meilleur score), sauf pour la classe D où les prédictions sont très mauvaises. Donc le modèle équilibré est préférable. De manière global, les deux modèles réussissent à bien classer 50 à 52% des observations totales.

Un dernier résultat tient aux rapports de côtes des variables (qui sont très similaires entre les deux modèles). Ces rapports montrent l'impact de la variation d'une variable sur la probabilité d'appartenir à la classe supérieure. Si ce rapport est inférieur à 1, alors cette probabilité diminue, et inversement si ce rapport est supérieur à 1.
Grâce à eux, les relations entre les valeurs nutritionnelles et le Nutri-Score sont décelées. Ainsi, les deux seuls rapports significativement différents de 1 sont inférieurs à ce dernier (c'est le cas du Sel et des graisses saturées), cela signifie que "l'amélioration" du Nutri-Score ne passe pas forcément par l'augmentation des quantités de bons nutriments (comme les Protéines, les seuls ayant un rapport supérieur à 1), cela passe par la diminution des mauvais nutriments, surtout le Sel et les graisses saturées.

Bien que la régression logistique ne soit pas bien adapté à ce problème, c'est un très bon début pour étudier la force du lien entre les valeurs nutritionnelles des produits et le Nutri-Score. Autrement dit, elle permet de donner une explication potentielle sur la mnière dont les aliments sont classés. Pour finir, d'autres modèles peuvent mieux modéliser cette relation, comme le Random Forest qui avait été développé lors de la première version de ce projet.