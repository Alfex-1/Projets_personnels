# Importation
df = pd.read_csv(r"C:\Données_nutriscore_v3\5Data_no_miss_unbalanced.csv")

# Base dans le nutri-score
df_no_nutri = df.drop(columns=['NutriScore'])

# Visualisation de l'existance de valeurs aberrantes et/ou extrêmes
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
    
    # Identifier les valeurs aberrantes et/ou extrêmes
    aberrantes = df[column] > threshold
    
    # Compter le nombre d'aberrations pour cette colonne
    nombre_aberrantes_colonne = aberrantes.sum()
    total_aberrantes += nombre_aberrantes_colonne
    
    # Remplacer les valeurs aberrantes par np.nan
    df[column] = df[column].apply(lambda x: np.nan if x > threshold else x)

# Il existe ici 383 890 valeurs aberrantes, ce qui représentent environ 40,3% des observations de la base

# Imputation par KNN des valeurs manquantes
knn_imputer = KNNImputer(n_neighbors=3)

col_knn = ['Glucides', 'Graisses', 'Dont_sucres',
           'Fibres','Energie_kcal','Dont_graisse_saturées',
           'Sel','Protéines']

df[col_knn] = knn_imputer.fit_transform(df[col_knn])

# Vérification s'il n'y a pas des données aberrantes et/ou extrêmes
for i in df_no_nutri.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="NutriScore", y=i)
    plt.title(f'Distribution de {i} en fonction du Nutri-score')
    plt.xlabel('Nutri-score')
    plt.ylabel(i)
    plt.show()

# Suppression des données illogiques (impossible d'avoir une masse de glucides/100g dépassant les 100g)
df2 = df[df['Glucides'] < 100]

# Vérification
df2_no_nutri = df2.drop(columns=['NutriScore'])
for i in df2_no_nutri.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df2, x="NutriScore", y=i,color='red')
    plt.title(f'Distribution de {i} en fonction du Nutri-score')
    plt.xlabel('Nutri-score')
    plt.ylabel(i)
    plt.show()


# Enregistrer les donnees
df2.to_csv(r"C:\Données_nutriscore_v3\6Data_no_miss_noextrem_unbalanced.csv", index=False)
