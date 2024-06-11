# Importation des données
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# Importation de la base
chemin_fichier = r"C:\Données_nutriscore_v3\3Data_usefull_feature_treat.csv"

df = pd.read_csv(chemin_fichier, sep=',')

# Supprimer les individus qui n'ont aucune caractéristique
df2 = df[~df.drop(columns=['NutriScore']).isnull().all(axis=1)]
print(len(df)-len(df2))  # 5 485 individus supprimés
df = df2.copy()
del df2

# On stock les observations qui appartiennent à la classe D
df_d = df[df['NutriScore'] == 'd']
df_d_no_na = df_d.dropna()

# On stock les variables explicatives
df_d_no_na_ns = df_d_no_na.drop(columns='NutriScore')

# On stock les données qui ont au moins 1 donné manquante
df_d_na = df_d[df_d.isnull().any(axis=1)]

# 179 906 soit une diminution du nombre d'obs de 50% pour l'analyse (à cause des na)
round((len(df_d_no_na)-len(df_d))/len(df_d), 1)*100

# Déterminer la meilleure contamination pour ne pas trop ou pas assez retiré d'individus


def find_optimal_contamination(data, target_count, tol=1):
    """
    Trouve la contamination optimale pour obtenir un nombre précis d'individus après nettoyage.

    Parameters
    ----------
    data : DataFrame
        DataFrame contenant les données à nettoyer.
    target_count : int
        Nombre souhaité d'individus après nettoyage.
    tol : int or float, optional
        Tolérance pour le nombre d'individus (par défaut 1).

    Returns
    -------
    best_contamination : float
        contamination optimale.

    """

    low, high = 0.0, 0.5  # Les valeurs limites pour la contamination
    best_contamination = 0.0
    best_diff = float('inf')

    while low <= high:
        contamination = (low + high) / 2
        iso_forest = IsolationForest(
            contamination=contamination, random_state=42)
        iso_forest.fit(data)
        predictions = iso_forest.predict(data)

        cleaned_data = data[predictions == 1]
        current_count = len(cleaned_data)
        diff = abs(current_count - target_count)

        if diff < best_diff:
            best_diff = diff
            best_contamination = contamination

        if current_count < target_count:
            high = contamination - tol / len(data)
        else:
            low = contamination + tol / len(data)

    return best_contamination


nb = df['NutriScore'].value_counts()
print(nb)  # 218 796 individus dans la classe C

optimal_contamination = find_optimal_contamination(
    data=df_d_no_na_ns, target_count=218796-len(df_d_na), tol=5)

# Appliquer l'Isolation Forest
iso_forest = IsolationForest(
    contamination=0.5, random_state=42)
iso_forest.fit_predict(df_d_no_na_ns)

# Prédire les anomalies (1 pour normal, -1 pour anomalie)
anomaly_predictions = iso_forest.predict(df_d_no_na_ns)

# Ajouter les prédictions au DataFrame original
df_d_no_na['anomaly'] = anomaly_predictions

# Supprimer les anomalies de la base de données
d_cleaned = df_d_no_na[df_d_no_na['anomaly'] != -1]

# Supprimer la colonne 'anomaly' des DataFrames
d_cleaned = d_cleaned.drop(columns=['anomaly'])
df_d_no_na = df_d_no_na.drop(columns=['anomaly'])

print(len(df_d_no_na)-len(d_cleaned))  # 71 021 individus supprimés

# On reprend la base initiale en faisant les changement pour les individus de classe d
df_no_d = df[df['NutriScore'] != 'd']

# On rassemble tous les individus de la classe D
new_df = pd.concat([d_cleaned, df_d_na])

# On y ajoute les indivdius de toute les autres classes
new_df = pd.concat([df_no_d, new_df])

# On vérifie s'il n'y a pas eu de problème dans le traitement
percentages = round(
    new_df['NutriScore'].value_counts(normalize=True) * 100, 1)
print(percentages)
nb = new_df['NutriScore'].value_counts()
print(nb)
# len(class_c) != len(classe_d), cela s'explique que contamination doit être <= 0.5
# Et que la moitié des données ont été exclues

# On peut enregistrer les nouvelles données
new_df.to_csv(r"C:\Données_nutriscore_v3\4Data_dclass_treat.csv", index=False)
