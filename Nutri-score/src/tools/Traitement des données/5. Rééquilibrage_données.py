# Importation de la base
chemin_fichier = r"C:\Données_nutriscore\6Data_no_miss_unbalanced.csv"

df = pd.read_csv(chemin_fichier, sep=',')

percentages = round(
    df['NutriScore'].value_counts(normalize=True) * 100, 1)
print(percentages)
nb = df['NutriScore'].value_counts()
print(nb)

"""
NutriScore
d    25.4
c    23.0
e    19.4
a    17.1
b    15.1
Name: proportion, dtype: float64
NutriScore
d    241933
c    218796
e    185374
a    163051
b    144073
Name: count, dtype: int64

Nombre d'observation sur la base brute : 953 227
"""

# Appliquer SMOTE
X = df.drop(columns=['NutriScore'])
y = df['NutriScore']

smote = SMOTE(sampling_strategy='not majority',  # 'not majority' : on ajoute des observations sur toutes les classes sauf la majoritaire
              k_neighbors=4,
              random_state=42,
              n_jobs=-1)

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
a    241933   + 78 882 (+48.38%)
d    241933   + 0
b    241933   + 97 860 (+67.92%)
c    241933   + 23 137 (+10.57%)
e    241933   + 56 559 (+30.51%)
Name: count, dtype: int64

Après resampling :
    - 1 209 665 observations,
    - Dont 256 438 observations ajoutées (resampling),
    - Soit 21.2% des observations sont "artificielles".

"""

# Enregistrer les données
df_resampled.to_csv(r"C:\Données_nutriscore\7Data_no_miss_balanced.csv", index=False)
