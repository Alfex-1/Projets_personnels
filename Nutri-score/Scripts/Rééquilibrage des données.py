# Packages
import pandas as pd
from imblearn.over_sampling import SMOTE

# Importation de la base
chemin_fichier = r"C:\4Data_no_miss_unbalanced.csv"

df = pd.read_csv(chemin_fichier, sep=',')

percentages = round(df['nutriscore_grade'].value_counts(normalize=True) * 100,1)
print(percentages)
nb = df['nutriscore_grade'].value_counts()
print(nb)

"""
nutriscore_grade
d    30.4
c    21.3
e    18.0
a    16.2
b    14.0
Name: proportion, dtype: float64

nutriscore_grade
d    330031
c    231091
e    195344
a    175194
b    152269
Name: count, dtype: int64

Nombre d'observation sur la base brute : 1 083 929
"""

# Appliquer SMOTE
X = df.drop(columns=['nutriscore_grade'])
y = df['nutriscore_grade']

smote = SMOTE(sampling_strategy = 'not majority', # 'not majority' : on ajoute des observations sur toutes les classes sauf la majoritaire
              k_neighbors = 3, # Combien de voisins utilisés pour créer les nouvelles observations
              n_jobs = 5, # Pour la parallélisation 
              random_state = 42)

X_resampled, y_resampled = smote.fit_resample(X, y)

# Convertir en DataFrame si nécessaire
df_resampled = pd.concat([pd.DataFrame(X_resampled), pd.DataFrame({'nutriscore_grade': y_resampled})], axis=1)

# Vérification des nouveaux pourcentages
percentages_resampled = round(df_resampled['nutriscore_grade'].value_counts(normalize=True) * 100,1)
print(percentages_resampled)

nb_resampled = df_resampled['nutriscore_grade'].value_counts()
print(nb_resampled)

"""
nutriscore_grade
a    20.0
d    20.0
b    20.0
c    20.0
e    20.0
Name: proportion, dtype: float64
nutriscore_grade
a    330031   + 154837 (154,8K : 46.9%)
d    330031   + 0
b    330031   + 177762 (177,8K : 53.9%)
c    330031   + 98940 (98,9K : 30%)
e    330031   + 134687 (134,7K : 40.8%)
Name: count, dtype: int64

Après avec resampling :
    - 1 650 155 observations,
    - Dont 566 226 observations ajoutées (resampling),
    - Soit 34.3% des observations sont "artificielles".

"""

# Enregistrer les données
df_resampled.to_csv(r"C:\Projets-personnels\Nutri-score\Données\5Data_no_miss_balanced.csv", index=False)
