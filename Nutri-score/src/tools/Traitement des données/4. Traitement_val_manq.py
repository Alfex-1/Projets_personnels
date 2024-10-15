# Importation de la base
chemin_fichier = r"C:\Données_nutriscore\5Data_noextrem.csv"

df = pd.read_csv(chemin_fichier, sep=',')

# Informations sur les données manquantes

miss = pd.DataFrame(round((df.isnull().sum() / len(df)) * 100, 2))
miss = miss.sort_values(by=[0], ascending=False)
print(miss)

""""
Fibres                 52.88
Dont_sucres            12.11
Dont_graisse_saturées   9.66
Sel                     5.17
Energie_kcal            4.71
Graisses                4.10
Protéines               3.07
Glucides                0.16
NutriScore	            0.00
""""
# Suppression de la variable "Fibres" car trop de données manquantes
del df['Fibres']

# Imputation des données manquantes

# On regarde quel est le type de nos données manquantes de nos variables explicatives
df2 = df.drop(columns=['NutriScore'])

msno.heatmap(df2)
plt.show()

msno.dendrogram(df2)
plt.show()

# Imputation par IterativeImputer
estimator = KNeighborsRegressor(n_neighbors=3, weights='distance', n_jobs=6)

imputer = IterativeImputer(estimator=estimator, max_iter=100, random_state=42)

col_impute = ['Energie_kcal', 'Graisses', 'Dont_graisse_saturées', 'Glucides', 'Dont_sucres','Protéines', 'Sel']

df[col_impute] = imputer.fit_transform(df[col_impute])


miss2 = pd.DataFrame(round((df.isnull().sum() / len(df)) * 100, 2))
miss2 = miss2.sort_values(by=[0], ascending=False)
print(miss2)

# Nouvelle exploration des données
df.describe()

# Détection des valeurs négatives : 2 valeur concernées
df_neg = df[df['Dont_sucres'] < 0]

# Remplacer les valeurs négatives par des valeurs manquantes dans les colonnes 'Dont_sucres' et 'Fibres'
df['Dont_sucres'] = df['Dont_sucres'].map(lambda x: np.nan if x < 0 else x)

# Imputation des nouvelles données manquantes
new_col = ['Dont_sucres']
df[new_col] = imputer.fit_transform(df[new_col])

# Vérification
df.describe()

# Enregistrer les données complètes (mais non-équilibrées pour l'instant)
df.to_csv(r"C:\Données_nutriscore\6Data_no_miss_unbalanced.csv", index=False)