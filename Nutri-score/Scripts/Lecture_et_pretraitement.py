import pandas as pd
import csv
csv.field_size_limit(500 * 1024 * 1024)

# Spécifiez le chemin vers votre fichier CSV
chemin_fichier = r"C:\Projets-personnels\Nutri-score\Données\1Data_brut.csv" 

# Paramètres optionnels pour gérer la taille du fichier
chunksize = 100000  # Nombre de lignes à lire à la fois (vous pouvez ajuster cela en fonction de la mémoire disponible)
iterator = True  # Permet la lecture en morceaux
sep = "\t"

pieces = []

# Lecture du fichier CSV en morceaux avec gestion des lignes mal formées
chunks = pd.read_csv(chemin_fichier, chunksize=chunksize, iterator=iterator,
                     sep=sep, engine='python', quoting=csv.QUOTE_NONE,
                     usecols=lambda column: column.endswith('_100g') or column in ['nutriscore_grade', 'product_name'])

# Parcourir chaque morceau et le stocker dans la liste
for chunk in chunks:
    pieces.append(chunk)

# Concaténer les morceaux en un seul DataFrame
df = pd.concat(pieces, ignore_index=True)

# Affichage des premières lignes du DataFrame
print(df.head())

# Base de dnnées qu'avec les données (variables) utiles (aucun traitement encore)
df.to_csv(r"C:\Projets-personnels\Nutri-score\Données\2Data_usefull_feature_brut.csv", index=False)

### Suppression des données manquantes

# Suppression des colonnes qui ont trop de données manquantes (70% ou +)
nombre_minimal_non_nan = int(len(df) * 0.7)
df = df.dropna(axis=1, thresh=nombre_minimal_non_nan)

# regarder le taux de données manquantes par variable
round((df.isnull().sum() / len(df)) * 100,2)

"""
nutriscore_grade       0.41
energy-kcal_100g      25.44
energy_100g           23.89
fat_100g              24.52
saturated-fat_100g    26.38
carbohydrates_100g    24.48
sugars_100g           25.54
proteins_100g         24.41
salt_100g             33.24
sodium_100g           33.24
"""

# Contrôle de la variable cible
unique = df['nutriscore_grade'].unique()

# Conserver autre part les obs les modalités atypiques ailleurs pour possible réutilisation
df_unknown = df.loc[df['nutriscore_grade'] == 'unknown']
df_not_applicable = df.loc[df['nutriscore_grade'] == 'not-applicable']

# Supprimer les obs qui n'ont pas un score comme nutri-score (not applicale ou unknown)
df2 = df.loc[df['nutriscore_grade'] != 'not-applicable']
df2 = df2.loc[df['nutriscore_grade'] != 'unknown']

# Supprimer les obs qui n'ont rien en score
df2 = df2.dropna(subset=['nutriscore_grade'])

# Regarder à nouveau l'état des variables
round((df2.isnull().sum() / len(df2)) * 100,1)

"""
product_name          1.0
nutriscore_grade      0.0
energy-kcal_100g      4.0
energy_100g           0.6
fat_100g              0.6
saturated-fat_100g    3.1
carbohydrates_100g    0.7
sugars_100g           1.4
proteins_100g         0.6
salt_100g             0.6
sodium_100g           0.6
"""

# Base de données pré-traité avec un minimum d'obs manquantes
df2.to_csv(r"C:\Projets-personnels\Nutri-score\Données\3Data_usefull_feature_treat.csv", index=False)