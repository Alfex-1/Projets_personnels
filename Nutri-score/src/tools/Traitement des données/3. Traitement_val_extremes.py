# Importation
df = pd.read_csv(r"C:\Données_nutriscore\4Data_dclass_treat.csv")

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

# Il existe ici 329 387 valeurs aberrantes, ce qui représentent environ 34,6% des observations de la base

# Nouvelles exploration des données pour vérification
for i in df.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="NutriScore", y=i)
    plt.title(f'Distribution de {i} en fonction du Nutri-score')
    plt.xlabel('Nutri-score')
    plt.ylabel(i)
    plt.show()
    
# Il ne semble plus avoir des données anormalement élevées, donc enregistrement
df.to_csv(r"C:\Données_nutriscore\5Data_noextrem.csv", index=False)
