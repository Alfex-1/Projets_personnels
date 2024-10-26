# Importation
df = pd.read_csv(r"C:\Projets_personnels\Détection anomalies\src\data\transaction_anomalies_dataset.csv")
df = df.drop(columns='Transaction_ID')
# =============================================================================
# Détection des anomalies
# =============================================================================

# Détection par Isolation Forest
df = df.select_dtypes(include=['int', 'float'])

# Détection des anomalies par une approche multivariée
# Rappel : la proportion d'anomalies est estimée à 2% par l'exploration de données
# la contamination est fixée légèrement plus haut que les 2% pour capter tous les vrais positifs (au déteriment de créer des faux positifs)
forest = IsolationForest(n_estimators=1000,contamination=0.0245, random_state=42)
forest.fit(df)
anomalies = forest.predict(df)
df['Anomaly'] = anomalies
df['Anomaly'] = df['Anomaly'].map({1: 'Non', -1: 'Oui'})

# Visualisation des anomalies
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Transaction_Amount", y="Average_Transaction_Amount", hue='Anomaly')
plt.title('\nPosition des anomalies dans la relation entre\nle montant des transactions et la moyenne des montants\n', fontsize=16)
plt.xlabel('Montant des transactions', fontsize=14)
plt.ylabel('Moyenne des montants par transactions', fontsize=14)
plt.legend(title='Anomalie', loc='best')
plt.show()

# Utiliser seulement les données qualitatives permettent de minimiser les faux positifs tout en captant tous les vrais positifs
df[df['Transaction_Amount'] < 2000]['Anomaly'].describe()
# Dans ce cas, il y a 5 faux positifs, soit 0,5% des observations totales
# Ces observations sont donc supprimées
df = df[~((df['Transaction_Amount'] < 2000) & (df['Anomaly'] == 'Oui'))]

# =============================================================================
# Sous-échantillonnage et sur-échantillonnage
# =============================================================================

# Encodage de la variable cible qui est binaire
df, infos = encoding_binary(data=df,list_variables=["Anomaly"])

# Sous-échantillonnage de la classe majoritaire via les valeurs extrêmes
df2 = df[df['Transaction_Amount'] < 2000].drop(columns='Anomaly')

# On souhaite retirer 10% des observations
forest = IsolationForest(n_estimators=1000,contamination=0.1, random_state=42)
forest.fit(df2)
df2['Anomaly'] = forest.predict(df2)
df2['Anomaly'] = df2['Anomaly'].map({1: False, -1: True})

anomalies_index = df2[df2['Anomaly'] == True].index
df = df.drop(index=anomalies_index).reset_index(drop=True)
# 98 observations ont été supprimées

# Nouvelle visualisation
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Transaction_Amount", y="Average_Transaction_Amount", hue='Anomaly')
plt.title('\nPosition des anomalies dans la relation entre\nle montant des transactions et la moyenne des montants\nsans sur-échantillonnage\n', fontsize=16)
plt.xlabel('Montant des transactions', fontsize=14)
plt.ylabel('Moyenne des montants par transactions', fontsize=14)
plt.legend(title='Anomalie', loc='best')
plt.show()

# Séparer les caractéristiques et la cible
X = df.drop(columns=['Anomaly'])
y = df['Anomaly']

# Configurer SMOTE pour générer 30 données supplémentaires de la classe minoritaire
smote = SMOTE(sampling_strategy={1: sum(y == 1) + 30}, k_neighbors=3, random_state=42)

# Appliquer SMOTE
X_resampled, y_resampled = smote.fit_resample(X, y)

# Reconstruire le DataFrame avec les nouvelles observations
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled['Anomaly'] = y_resampled

# Visualisation définitive
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_resampled, x="Transaction_Amount", y="Average_Transaction_Amount", hue='Anomaly')
plt.title('\nPosition des anomalies dans la relation entre\nle montant des transactions et la moyenne des montants\navec sur-échantillonnage\n', fontsize=16)
plt.xlabel('Montant des transactions', fontsize=14)
plt.ylabel('Moyenne des montants par transactions', fontsize=14)
plt.legend(title='Anomalie', loc='best')
plt.show()

# Enregistrement de la nouvelle base de données
df_resampled.to_csv(r"C:\Projets_personnels\Détection anomalies\src\data\New_data.csv",index=False)