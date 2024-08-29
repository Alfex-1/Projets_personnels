# Packages
import pandas as pd
from imblearn.over_sampling import SMOTE

# Importation de la base
chemin_fichier = r"\\172.20.3.5\vol_modelisation_001\modelisation\MOD_DONNEES_SATELLITAIRES\Stage\Alex\Autres\Traitement des données\Données_nutriscore_v3\6Data_no_miss_noextrem_unbalanced.csv"

df = pd.read_csv(chemin_fichier, sep=',')

percentages = round(
    df['NutriScore'].value_counts(normalize=True) * 100, 1)
print(percentages)
nb = df['NutriScore'].value_counts()
print(nb)

"""
NutriScore
d    25.2
c    23.0
e    19.5
a    17.2
b    15.1
Name: proportion, dtype: float64
NutriScore
d    239731
c    218306
e    185096
a    163022
b    143892
Name: count, dtype: int64

Nombre d'observation sur la base brute : 950 047
"""

# Appliquer SMOTE
X = df.drop(columns=['NutriScore'])
y = df['NutriScore']

smote = SMOTE(sampling_strategy='not majority',  # 'not majority' : on ajoute des observations sur toutes les classes sauf la majoritaire
              k_neighbors=4,  # Combien de voisins utilisés pour créer les nouvelles observations
              random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)

# Convertir en DataFrame
df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame(
    {'NutriScore': y_resampled})], axis=1)

# Vérification des nouvelles proportions
percentages_resampled = round(
    df_resampled['NutriScore'].value_counts(normalize=True) * 100, 1)
print(percentages_resampled)

nb_resampled = df_resampled['NutriScore'].value_counts()
print(nb_resampled)

"""
NutriScore
a    20.0
d    20.0
b    20.0
c    20.0
e    20.0
Name: proportion, dtype: float64

NutriScore
a    239731   + 76 709 (+47.05%)
d    239731   + 0
b    239731   + 95 839 (+66.6%)
c    239731   + 21 425 (+9.8%)
e    239731   + 54 635 (+29.5%)
Name: count, dtype: int64

Après resampling :
    - 1 198 655 observations,
    - Dont 248 608 observations ajoutées (resampling),
    - Soit 20.7% des observations sont "artificielles".

"""

# Enregistrer les données
df_resampled.to_csv(
    r"\\172.20.3.5\vol_modelisation_001\modelisation\MOD_DONNEES_SATELLITAIRES\Stage\Alex\Autres\Traitement des données\Données_nutriscore_v3\7Data_no_miss_balanced.csv", index=False)
