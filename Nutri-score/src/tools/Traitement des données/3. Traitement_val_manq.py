# Importation des donn�es
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import MissingIndicator, KNNImputer, SimpleImputer
from sklearn.impute import IterativeImputer

# Importation de la base
chemin_fichier = r"C:\Donn�es_nutriscore_v3\4Data_dclass_treat.csv"

df = pd.read_csv(chemin_fichier, sep=',')

# Informations sur les donn�es
df.describe()  # On voit qu'il existe des donn�es manquantes

miss = pd.DataFrame(round((df.isnull().sum() / len(df)) * 100, 2))
miss = miss.sort_values(by=[0], ascending=False)

# Imputation des donn�es manquantes

# On regarde quel est le type de nos donn�es manquantes de nos variables explicatives
df2 = df.drop(columns=['NutriScore'])

msno.heatmap(df2)
plt.show()

msno.dendrogram(df2)
plt.show()

# Imputation par KNN pour les variables qui sont MAR et MCAR
knn_imputer = KNNImputer(n_neighbors=4)

col_knn = ['Glucides', 'Graisses', 'Dont_sucres']

df[col_knn] = knn_imputer.fit_transform(df[col_knn])

miss2 = pd.DataFrame(round((df.isnull().sum() / len(df)) * 100, 2))
miss2 = miss2.sort_values(by=[0], ascending=False)
print(miss2)

# Imputation pour les donn�es MNAR avec IterativeImputer
simple_imputer = IterativeImputer(
    max_iter=20, random_state=42, sample_posterior=True)

col_it = ['Energie_kcal', 'Dont_graisse_satur�es',
          'Fibres', 'Prot�ines', 'Sel']

# Imputer les colonnes
df[col_it] = simple_imputer.fit_transform(df[col_it])

miss3 = pd.DataFrame(round((df.isnull().sum() / len(df)) * 100, 2))
miss3 = miss3.sort_values(by=[0], ascending=False)
print(miss3)

# Enregistrer les donn�es compl�tes
df.to_csv(r"C:\Donn�es_nutriscore_v3\5Data_no_miss_unbalanced.csv", index=False)
