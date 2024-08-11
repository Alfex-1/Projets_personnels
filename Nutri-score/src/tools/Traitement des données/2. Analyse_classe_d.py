# Importation des donn�es
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Importation de la base
chemin_fichier = r"C:\Donn�es_nutriscore_v3\3Data_usefull_feature_treat.csv"

df = pd.read_csv(chemin_fichier, sep=',')

# Supprimer les individus qui n'ont aucune caract�ristique
df2 = df[~df.drop(columns=['NutriScore']).isnull().all(axis=1)]
print(len(df)-len(df2))  # 5 485 individus supprim�s
df = df2.copy()
del df2

# On stock les observations qui appartiennent � la classe D
df_d = df[df['NutriScore'] == 'd']
df_d_no_na = df_d.dropna()

# On stock les variables explicatives
df_d_no_na_ns = df_d_no_na.drop(columns='NutriScore')

# On stock les donn�es qui ont au moins 1 donn� manquante
df_d_na = df_d[df_d.isnull().any(axis=1)]

# 179 906 soit une diminution du nombre d'obs de 50% pour l'analyse (� cause des na)
round((len(df_d_no_na)-len(df_d))/len(df_d), 1)*100

# D�terminer la meilleure contamination pour ne pas trop ou pas assez retir� d'individus
nb = df['NutriScore'].value_counts()
print(nb)  # 218 796 individus dans la classe C

optimal_contamination = find_optimal_contamination(
    data=df_d_no_na_ns, target_count=218796-len(df_d_na), tol=5)

# Appliquer l'Isolation Forest
iso_forest = IsolationForest(
    contamination=0.5, random_state=42)
iso_forest.fit_predict(df_d_no_na_ns)

# Pr�dire les anomalies (1 pour normal, -1 pour anomalie)
anomaly_predictions = iso_forest.predict(df_d_no_na_ns)

# Ajouter les pr�dictions au DataFrame original
df_d_no_na['anomaly'] = anomaly_predictions

# Supprimer les anomalies de la base de donn�es
d_cleaned = df_d_no_na[df_d_no_na['anomaly'] != -1]

# Supprimer la colonne 'anomaly' des DataFrames
d_cleaned = d_cleaned.drop(columns=['anomaly'])
df_d_no_na = df_d_no_na.drop(columns=['anomaly'])

print(len(df_d_no_na)-len(d_cleaned))  # 71 021 individus supprim�s

# On reprend la base initiale en faisant les changement pour les individus de classe d
df_no_d = df[df['NutriScore'] != 'd']

# On rassemble tous les individus de la classe D
new_df = pd.concat([d_cleaned, df_d_na])

# On y ajoute les indivdius de toute les autres classes
new_df = pd.concat([df_no_d, new_df])

# On v�rifie s'il n'y a pas eu de probl�me dans le traitement
percentages = round(
    new_df['NutriScore'].value_counts(normalize=True) * 100, 1)
print(percentages)
nb = new_df['NutriScore'].value_counts()
print(nb)
# len(class_c) != len(classe_d), cela s'explique que contamination doit �tre <= 0.5
# Et que la moiti� des donn�es ont �t� exclues

# On peut enregistrer les nouvelles donn�es
new_df.to_csv(r"C:\Donn�es_nutriscore_v3\4Data_dclass_treat.csv", index=False)
