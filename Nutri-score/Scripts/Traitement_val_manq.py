# Importation des donn�es
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import MissingIndicator, KNNImputer,SimpleImputer
from sklearn.impute import IterativeImputer

# Importation de la base
chemin_fichier = r"C:\Projets-personnels\Nutri-score\Donn�es\3Data_usefull_feature_treat.csv"

df = pd.read_csv(chemin_fichier, sep=',')

# On retire la premi�re colonne qui est inutile
df = df.iloc[:,1:11]
del df['energy_100g']

#Informations sur les donn�es
df.describe()

miss = pd.DataFrame(round((df.isnull().sum() / len(df)) * 100,2))
miss = miss.sort_values(by=[0], ascending=False)

# Imputation des donn�es manquantes

# On regarde quel est le type de nos donn�es manquantes
df2 = df.iloc[:,1:10]

msno.heatmap(df2)
plt.show()

msno.dendrogram(df)
plt.show()

# Imputation par KNN pour les variables qui sont MAR et MCAR
knn_imputer = KNNImputer(n_neighbors=3)

df[['sugars_100g','carbohydrates_100g',
    'fat_100g','proteins_100g','salt_100g',
    'sodium_100g']] = knn_imputer.fit_transform(df[['sugars_100g',
                                                 'carbohydrates_100g',
                                                 'fat_100g',
                                                 'proteins_100g',
                                                 'salt_100g',
                                                 'sodium_100g']])

miss2 = pd.DataFrame(round((df.isnull().sum() / len(df)) * 100,2))
miss2 = miss2.sort_values(by=[0], ascending=False)
print(miss2)

# Imputation pour les donn�es MNAR avec IterativeImputer
simple_imputer = IterativeImputer(max_iter=20, random_state=42, sample_posterior=True)

# Imputer les colonnes 'total_protein', 'rectal_temp' et 'pulse' de la DataFrame df
df[['energy-kcal_100g',
    'saturated-fat_100g']] = simple_imputer.fit_transform(df[['energy-kcal_100g',
                                                             'saturated-fat_100g']])
                                                             
miss3 = pd.DataFrame(round((df.isnull().sum() / len(df)) * 100,2))
miss3 = miss3.sort_values(by=[0], ascending=False)
print(miss3)

# Augmentation de la quantit� de donn�es de 52,24% (potentiellement dangereux !)

# Enregistrer les donn�es compl�tes
df.to_csv(r"C:\Projets-personnels\Nutri-score\Donn�es\4Data_no_miss_unbalanced.csv", index=False)