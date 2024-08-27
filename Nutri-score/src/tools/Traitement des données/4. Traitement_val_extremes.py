import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import MissingIndicator, KNNImputer, SimpleImputer
import seaborn as sns
from sklearn.ensemble import IsolationForest

df = pd.read_csv(r"C:\Données_nutriscore_v3\5Data_no_miss_unbalanced.csv")

df_no_nutri = df.drop(columns=['NutriScore'])

for i in df_no_nutri.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="NutriScore", y=i)
    plt.title(f'Distribution de {i} en fonction du Nutri-score')
    plt.xlabel('Nutri-score')
    plt.ylabel(i)
    plt.show()
    
# Initialiser un compteur pour les valeurs aberrantes
total_aberrantes = 0

# Boucle à travers les colonnes numériques
for column in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    threshold = Q3 + 1.5 * IQR
    
    # Identifier les valeurs aberrantes
    aberrantes = df[column] > threshold
    
    # Compter le nombre d'aberrations pour cette colonne
    nombre_aberrantes_colonne = aberrantes.sum()
    total_aberrantes += nombre_aberrantes_colonne
    
    # Remplacer les valeurs aberrantes par np.nan
    df[column] = df[column].apply(lambda x: np.nan if x > threshold else x)

# Il existe ici 72 392 valeurs aberrantes, ce qui représentent moins de 10% des observations de la base

# Imputation par KNN des valeurs manquantes
knn_imputer = KNNImputer(n_neighbors=3)

col_knn = ['Glucides', 'Graisses', 'Dont_sucres',
           'Fibres','Energie_kcal','Dont_graisse_saturées',
           'Sel','Protéines']

df[col_knn] = knn_imputer.fit_transform(df[col_knn])

# Enregistrer les donnees
df.to_csv(r"C:\Données_nutriscore_v3\6Data_no_miss_noextrem_unbalanced.csv", index=False)