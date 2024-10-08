import pandas as pd
import csv
csv.field_size_limit(500 * 1024 * 1024)

# Spécifiez le chemin vers votre fichier CSV
chemin_fichier = r"C:\Données_nutriscore_v1\1Data_brut.csv"

# Paramètres optionnels pour gérer la taille du fichier
chunksize = 100000
iterator = True  # Permet la lecture en morceaux
sep = "\t"

pieces = []

# Lecture du fichier CSV en morceaux avec gestion des lignes mal formées
chunks = pd.read_csv(chemin_fichier, chunksize=chunksize, iterator=iterator,
                     sep=sep, engine='python', quoting=csv.QUOTE_NONE,
                     usecols=lambda column: column.endswith('_100g') or column in ['nutriscore_grade'])

# Parcourir chaque morceau et le stocker dans la liste
for chunk in chunks:
    pieces.append(chunk)

# Concaténer les morceaux en un seul DataFrame
df = pd.concat(pieces, ignore_index=True)

# Affichage des premières lignes du DataFrame
print(df.head())

# Base de données qu'avec les données (variables) utiles (aucun traitement encore)
chemin_nvlles_donnnees = r"C:\Données_nutriscore_v1\2Data_usefull_feature_brut.csv"
df.to_csv(chemin_nvlles_donnnees, index=False)

# Suppression des données manquantes
miss = pd.DataFrame(round((df.isnull().sum() / len(df)) * 100, 2))
miss = miss.sort_values(by=[0], ascending=False)


# Suppression des colonnes qui ont trop de données manquantes (70% ou +)
missing_percentage = df.isnull().mean() * 100
columns_to_keep = missing_percentage[missing_percentage < 70].index
df = df[columns_to_keep]

# regarder le taux de données manquantes par variable
round((df.isnull().sum() / len(df)) * 100, 2)

"""
product_name                4.97
nutriscore_grade            0.41
energy-kcal_100g           25.44
energy_100g                23.89
fat_100g                   24.52
saturated-fat_100g         26.38
carbohydrates_100g         24.48
sugars_100g                25.54
fiber_100g                 65.54
proteins_100g              24.41
salt_100g                  33.24
sodium_100g                33.24
nutrition-score-fr_100g    65.36
"""

# Suppression de sodium qui n'est pas utilisé dans le calcul du nutri-score
# Suppression de product_name qui n'est pas utile à la modélisation
# Suppression de nutrition-score-fr_100g qui n'a pas de sens
# Suppression de energy_100g qui est exprimé en kj (ce qui n'est pas utilisé communément)
del df['sodium_100g']
del df['nutrition-score-fr_100g']
del df['energy_100g']

# Contrôle de la variable cible
unique = df['nutriscore_grade'].unique()
# Résultat :
# on doit se débarasser des "unknow",
# des "not applicable"
# des nan

# Conserver autre part les obs les modalités atypiques ailleurs pour possible réutilisation
df_unknown = df.loc[df['nutriscore_grade'] == 'unknown']  # 62 248
df_not_applicable = df.loc[df['nutriscore_grade']
                           == 'not-applicable']  # 1 970 451

# Supprimer les obs qui n'ont pas un score comme nutri-score (not applicale ou unknown)
df2 = df.loc[df['nutriscore_grade'] != 'not-applicable']
df2 = df2.loc[df['nutriscore_grade'] != 'unknown']

# Supprimer les obs qui n'ont rien en score
df2 = df2.dropna(subset=['nutriscore_grade'])

# Regarder à nouveau l'état des variables
round((df2.isnull().sum() / len(df2)) * 100, 1)

"""
nutriscore_grade       0.0
energy-kcal_100g       4.0
fat_100g               0.6
saturated-fat_100g     3.1
carbohydrates_100g     0.7
sugars_100g            1.4
fiber_100g            46.4
proteins_100g          0.6
salt_100g              0.6
"""

# Renommage des variables
rename_dict = {
    'nutriscore_grade': 'NutriScore',
    'energy-kcal_100g': 'Energie_kcal',
    'fat_100g': 'Graisses',
    'saturated-fat_100g': 'Dont_graisse_saturées',
    'carbohydrates_100g': 'Glucides',
    'sugars_100g': 'Dont_sucres',
    'fiber_100g': 'Fibres',
    'proteins_100g': 'Protéines',
    'salt_100g': 'Sel'
}

df2 = df2.rename(columns=rename_dict)

# Enregistrer 5% des observatiosn dans une nouvelle base et les enlever de la base initiale
nvlle_base = df2.sample(frac=0.05)

percentages = round(
    nvlle_base['NutriScore'].value_counts(normalize=True) * 100, 1)
print(percentages)
nb = nvlle_base['NutriScore'].value_counts()
print(nb)

"""
NutriScore
d    30.3
c    21.5
e    18.1
a    16.2
b    13.8
Name: proportion, dtype: float64
NutriScore
d    16432
c    11679
e     9806
a     8792
b     7487
Name: count, dtype: int64
"""

df2_rest = df2.drop(nvlle_base.index)

# Base de données pré-traité avec un minimum d'obs manquantes
df2_rest.to_csv(r"C:\Données_nutriscore_v2\3Data_usefull_feature_treat.csv",
                index=False)

# Suppression des indvidus qui n'ont pas toutes les informations
nvlle_base2 = nvlle_base.dropna()

percentages2 = round(
    nvlle_base2['NutriScore'].value_counts(normalize=True) * 100, 1)
print(percentages2)
nb2 = nvlle_base2['NutriScore'].value_counts()
print(nb2)

"""
NutriScore
d    26.6
a    22.0
c    21.5
e    15.4
b    14.5
Name: proportion, dtype: float64
NutriScore
d    7392
a    6102
c    5976
e    4281
b    4013
Name: count, dtype: int64
"""

# Enregistrement
nvlle_base2.to_csv(r"C:\Données_nutriscore_v3\Nouvelles_donnees.csv",
                   index=False)
