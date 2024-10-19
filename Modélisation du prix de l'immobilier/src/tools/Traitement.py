# =============================================================================
# Premiers traitements
# =============================================================================

# Importation de la base de données
df = pd.read_csv(r"C:\Projets_personnels\Prix_immobilier_Iowa\src\data\Price_data.csv",sep=";")

# Analayse des données manquantes
miss = pd.DataFrame(round((df.isnull().sum() / len(df)) * 100, 2))
miss = miss.sort_values(by=[0], ascending=False) # Certaines ont trop de valeurs manquantes

# Suppression es variables qui ont plus de 50% des observations qui sont manquantes
missing_percentage = df.isnull().mean() * 100
columns_to_keep = missing_percentage[missing_percentage <= 50].index
df = df[columns_to_keep] # 82-77 = 5 variables supprimées

# Suppression des 3 prmières variables qui sont inutiles car ce sont des identifiant et nombre d'observation
df = df.iloc[:,3:]

# =============================================================================
# Traitement des valeurs manquantes (quantitatives)
# =============================================================================

# Données quantitatives (en enlevant les variables discrètes)
df_quant = df.select_dtypes(include=['int', 'float'])
discrete_vars = [col for col in df_quant.columns
                     if pd.api.types.is_integer_dtype(df_quant[col]) 
                     and df_quant[col].nunique() <= 10]
df_quant = df_quant.drop(columns=discrete_vars)

# Suppression de la multicolinéarité
rejected_variables = vif_selection(data=df_quant.dropna(),target='SalePrice',vif_value=5)

df = df.drop(columns=rejected_variables) # 74 - 63 = 11 variables rejettées

# Identification du type de valeurs manquantes
df_quant = df.select_dtypes(include=['int', 'float']).drop(columns='SalePrice')
msno.heatmap(df_quant)
plt.show()

# Liste des variables MAR
liste_knn = ['BsmtFin SF 2','Bsmt Unf SF',
             'Bsmt Full Bath', 'Bsmt Half Bath']

# Liste des variables des variables MCAR, MNAR
liste_iter = ['Mas Vnr Area']

# POINT IMPORTANT à avoir en tête :
# La 2e liste de variables doit être utiliser pour effectuer une imputation univariée (IterativeImputer)
# Cependant, actuellement cet algorithme est expériemental et présente des anomalies de résultats
# De ce fait, toutes les valeurs manquantes vont être imputées par KNN

liste_complete = liste_knn+liste_iter

# Imputation par KNN
knn_imputer = KNNImputer(n_neighbors=3, weights='distance')
df[liste_complete] = knn_imputer.fit_transform(df[liste_complete])

# Pour arrondir les nombres décimaux alors qu'elles doivent être des nombre entiers
df[['Bsmt Full Bath', 'Bsmt Half Bath']] = df[['Bsmt Full Bath', 'Bsmt Half Bath']].round().astype(int)

# Verification des valeurs
df[liste_complete].describe()
# Il n'y a pas de valeurs négataives (anomalies de résultats)
# En utilisant IterativeImputer, des valeurs négatives seraient apparues (affirmation vérifiée)

# Verification que toutes les valeurs manquantes ont été imputées (variables quantitatives)
df_quant = df.select_dtypes(include=['int', 'float']).drop(columns=['SalePrice'])
miss_quant = pd.DataFrame(round((df_quant.isnull().sum() / len(df_quant)) * 100, 2))
miss_quant = miss_quant.sort_values(by=[0], ascending=False)
miss_quant # Toutes les variables quantitatives ont été imputées correctement

# =============================================================================
# Traitement des valeurs manquantes (qualitatives)
# =============================================================================

# Idée : imputer les variables qualitatives avec les KNN APRES encodage des variables

# Pour éviter d'encoder des variables inutilement, deux pré-sélections
# 1 : fréquences des modalités
df_qual_col = df.select_dtypes(include=['object', 'category']).columns
discrete_vars_freq = [col for col in df.columns
                     if pd.api.types.is_integer_dtype(df[col]) 
                     and df[col].nunique() <= 10]
tot_col_freq = list(df_qual_col) + discrete_vars_freq

df_freq = df[tot_col_freq].dropna()

for column in df_freq.columns:
    # Calculer les fréquences des modalités
    frequencies = df_freq[column].value_counts()

    # Créer un diagramme circulaire
    plt.figure(figsize=(6, 6))
    plt.pie(frequencies, labels=frequencies.index, autopct='%1.1f%%', startangle=90)
    plt.title(f'Diagramme circulaire des modalités pour {column}')
    plt.axis('equal')  # Assure que le cercle est bien formé
    plt.show()
    
# Supprimer les variables ayant des modalités qui change très peu
col_drop = ['Lot Shape','Bldg Type','House Style','Heating','Garage Type',
            'Bsmt Half Bath','Bsmt Full Bath','Kitchen AbvGr'] # 8 variables rejettées (55 restantes)

df = df.drop(columns=col_drop) # Variables qualitatives supprimées

# 2 : indépendance avec la variable cible (analyse univariée)
rejected_variables = chi2_selection(data=df, target='SalePrice', threshold = 10, alpha=0.05)

# Suppression des variables dépendantes de la variable cible
df = df.drop(columns=rejected_variables) # 55-27 = 28 variables qualitatives supprimées

# Detection des valeurs manquantes
df_quali = df.select_dtypes(include=['object', 'category'])
miss_quali = pd.DataFrame(round((df_quali.isnull().sum() / len(df_quali)) * 100, 2))
miss_quali = miss_quali.sort_values(by=[0], ascending=False)
miss_quali

# Seule la première variable a des valeurs manquantes
miss_var = miss_quali.reset_index().iloc[0:1,]["index"].tolist()
df[miss_var].describe()

# Imputation de la variable qualitative
BsmtFin_Type_1 = knn_impute_categorical(df=df,column_name='BsmtFin Type 1')

# Vérification de la bonne imputation des valeurs manquantes sur la base complète
miss_tot = pd.DataFrame(round((df.isnull().sum() / len(df)) * 100, 2))
miss_tot = miss_tot.sort_values(by=[0], ascending=False)
miss_tot # Il n'y a plus aucune valeur manquante dans la base de données

# =============================================================================
# Traitement des valeurs aberrantes/extrêmes
# =============================================================================
df_quant = df.select_dtypes(include=['int', 'float']).drop(columns=['SalePrice'])

# Créer une grille de sous-graphes ajustée
n_cols = 3  # Nombre de colonnes dans la grille
n_rows = (len(df_quant.columns)) // n_cols + (len(df_quant.columns) % n_cols > 0)  # Calculer le nombre de lignes nécessaires

plt.figure(figsize=(25, n_rows * 5))  # Ajuster la taille de la figure

for i, column in enumerate(df_quant.columns):
    # Éviter de tracer un graphique entre SalePrice et lui-même
    if column != 'SalePrice':  # Cette condition est en fait redondante ici mais peut être ajoutée pour la clarté
        plt.subplot(n_rows, n_cols, i + 1)  # Créer la grille de sous-graphes
        plt.scatter(df_quant[column], df['SalePrice'], alpha=0.5)
        plt.xlabel(column)

plt.tight_layout()
plt.show()

# Après visualisation, une faible de valeurs extrêmes existe : moins de 5%


iso_forest = IsolationForest(n_estimators=500, max_features= 0.8,
                             contamination=0.05, bootstrap=True,
                             n_jobs=2, random_state=42)
iso_forest.fit(df_quant)
anomaly_predictions = iso_forest.predict(df_quant)

# Ajouter les prédictions au DataFrame original
df_quant['anomaly'] = anomaly_predictions

# On rajoute la variable cible
df_quant['SalePrice'] = df['SalePrice']

# Supprimer les anomalies de la base de données
df_no_anomaly = df_quant[df_quant['anomaly'] != -1]

# Supprimer la colonne 'anomaly' des DataFrames
df_no_anomaly = df_no_anomaly.drop(columns=['anomaly'])

print(len(df_quant)-len(df_no_anomaly)) # 147 des 2930 observations sont des valeurs extrêmes (soit 5,02%, comme prévu)

# Explorer les valeurs extrêmes
df_anomaly = df_quant[df_quant['anomaly'] == -1]

for column in df_quant.columns:
    if column not in ['SalePrice', 'anomaly']:  # Ignorer 'SalePrice' et 'anomaly'
        plt.figure(figsize=(8, 6))
        
        # Créer le scatter plot des points normaux
        plt.scatter(df_quant[df_quant['anomaly'] == 1][column], 
                    df_quant[df_quant['anomaly'] == 1]['SalePrice'], 
                    alpha=0.5, label='Normal')

        # Créer le scatter plot des anomalies
        plt.scatter(df_quant[df_quant['anomaly'] == -1][column], 
                    df_quant[df_quant['anomaly'] == -1]['SalePrice'], 
                    alpha=0.8, label='Anomalies')

        # Ajouter le titre et les légendes
        plt.title(f'Relation entre {column} et le prix de vente')
        plt.xlabel(column)
        plt.ylabel('Prix de vente')
        plt.legend()
        plt.show()
        
# Enregistrement des données complètes (avec indicatruce des anomalies)
# But : évaluer les performances du modèle avec et sans ces valeurs
df_quali = df.select_dtypes(include=['object', 'category'])
df_with_extrem = pd.concat([df_quant,df_quali], axis=1)

# df_with_extrem.to_csv(r"C:\Projets_personnels\Prix_immobilier_Iowa\src\data\Data_treat_with_extrem.csv",index=False)

# =============================================================================
# Encoding des variables qualitatives
# =============================================================================
# Lecture des nouvelles données si besoin
df = pd.read_csv(r"C:\Projets_personnels\Prix_immobilier_Iowa\src\data\Data_treat_with_extrem.csv")

# Variables qualitatives à encoder
col_a_encoder = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Créer l'encodeur
encoder = OneHotEncoder(drop='first',sparse_output=False)
onehot_encoded = encoder.fit_transform(df[col_a_encoder])
encoded_df = pd.DataFrame(onehot_encoded, columns=encoder.get_feature_names_out(col_a_encoder))

# Ajouter les colonnes encodées au DataFrame original
df = pd.concat([df.drop(columns=col_a_encoder).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Données avec anomalies VS données sans anomalies
df_no_anomaly = df[df['anomaly'] != -1]
del df['anomaly']
del df_no_anomaly['anomaly']

# =============================================================================
# Autres sélections des variables
# =============================================================================

# Elimination des variables avec une méthode univariée
rejected_variables = feature_elimination_kbest(data=df, target='SalePrice', test_size=1/3,k=20)
rejected_variables2 = feature_elimination_kbest(data=df_no_anomaly, target='SalePrice', test_size=1/3, k=20)

not_in_2 = [var for var in rejected_variables if var not in rejected_variables2]
not_in_1 = [var for var in rejected_variables2 if var not in rejected_variables]
# Même nombre de variables (69) supprimées mais 2 variables diffèrent

df_new_col = df.drop(columns=rejected_variables)
df_new_col_no_anom = df_no_anomaly.drop(columns=rejected_variables2)

# Elimination des variables par validation croisée
rejected_variables = feature_elimination_cv(data=df_new_col, target='SalePrice', model = LinearRegression(),
                                            scoring="rmse", test_size=1/3, cv=15, min_features_to_select=10)

rejected_variables2 = feature_elimination_cv(data=df_new_col_no_anom, target='SalePrice', model = LinearRegression(),
                                             scoring="rmse", test_size=1/3, cv=15, min_features_to_select=10)

df_new_col = df_new_col.drop(columns=rejected_variables)
df_new_col_no_anom = df_new_col_no_anom.drop(columns=rejected_variables2)

# Vérifier si les deux listes de variables supprimées sont similiares
len(rejected_variables) == len(rejected_variables2)
len(rejected_variables) # 10 Variables supprimées : 11 restantes
len(rejected_variables2) # 10 variables supprimées : 11 restantes

# Combien de variables sont communes entre les deux bases ?
not_in_2 = [var for var in df_new_col.columns if var not in df_new_col_no_anom.columns]
not_in_1 = [var for var in df_new_col_no_anom.columns if var not in df_new_col.columns]
# 10 variables sont communes

# Pour enregistrement
df_new_col.to_csv(r"C:\Projets_personnels\Prix_immobilier_Iowa\src\data\Data_selected_col_with_anom.csv",index=False)
df_new_col_no_anom.to_csv(r"C:\Projets_personnels\Prix_immobilier_Iowa\src\data\Data_selected_col_without_anom.csv",index=False)