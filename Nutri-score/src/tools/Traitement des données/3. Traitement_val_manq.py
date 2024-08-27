# Importation des données
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import MissingIndicator, KNNImputer, SimpleImputer
from sklearn.impute import IterativeImputer
import seaborn as sns

# Importation de la base
chemin_fichier = r"\\172.20.3.5\vol_modelisation_001\modelisation\MOD_DONNEES_SATELLITAIRES\Stage\Alex\Autres\Traitement des données\Données_nutriscore_v3\4Data_dclass_treat.csv"

df = pd.read_csv(chemin_fichier, sep=',')

# Informations sur les données
df.describe()  # On voit qu'il existe des données manquantes

miss = pd.DataFrame(round((df.isnull().sum() / len(df)) * 100, 2))
miss = miss.sort_values(by=[0], ascending=False)

# Imputation des données manquantes

# On regarde quel est le type de nos données manquantes de nos variables explicatives
df2 = df.drop(columns=['NutriScore'])

msno.heatmap(df2)
plt.show()

msno.dendrogram(df2)
plt.show()

# Imputation par KNN pour les variables qui sont MAR et MCAR
knn_imputer = KNNImputer(n_neighbors=3)

col_knn = ['Energie_kcal', 'Graisses', 'Dont_graisse_saturées',
           'Glucides', 'Dont_sucres', 'Fibres', 'Protéines', 'Sel']

df[col_knn] = knn_imputer.fit_transform(df[col_knn])


miss2 = pd.DataFrame(round((df.isnull().sum() / len(df)) * 100, 2))
miss2 = miss2.sort_values(by=[0], ascending=False)
print(miss2)

test = df.describe()

# Détection des valeurs négatives : seulement 8 lignes concernées
df_neg = df[(df['Dont_sucres'] < 0) | (df['Fibres'] < 0)]

# Remplacer les valeurs négatives par 0 dans les colonnes 'Dont_sucres' et 'Fibres'
df['Dont_sucres'] = df['Dont_sucres'].map(lambda x: 0 if x < 0 else x)
df['Fibres'] = df['Fibres'].map(lambda x: 0 if x < 0 else x)

# Enregistrer les données complètes
df.to_csv(r"\\172.20.3.5\vol_modelisation_001\modelisation\MOD_DONNEES_SATELLITAIRES\Stage\Alex\Autres\Traitement des données\Données_nutriscore_v3\5Data_no_miss_unbalanced.csv", index=False)
