# Importation
df = pd.read_csv(r"C:\Projets_personnels\Détection anomalies\src\data\transaction_anomalies_dataset.csv")

# Premières explorations
df.head(10)

df.describe()

# Est-ce qu'il existe des données manquantes ?
df.isnull().sum() # Non

# =============================================================================
# Analyse exploratoire
# =============================================================================

# Distribution des transactions
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Transaction_Amount", bins=30, kde=True, color='red', edgecolor='black')
plt.title("\nDistribution des montants des transactions\n", fontsize=16)
plt.xlabel("Montant des transactions", fontsize=14)
plt.ylabel("Fréquence", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
# Certaines transactions semblent se démarquer plus que d'autres

# Montants par genre et par type de compte
plt.figure(figsize=(8, 5))
sns.violinplot(data=df, x="Transaction_Amount", y="Account_Type", hue="Gender", split=True, gap=.05)
plt.title("\nDistribution des montants des transactions par type de compte et par genre des clients\n", fontsize=16)
plt.xlabel("Montant des transactions", fontsize=14)
plt.ylabel("Type de compte", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.legend(title='Genre',loc='best')
plt.show()
# Les montants semblent aller dans les comptes courant tout autant que dans les comptes épargnes.
# Il ne semble pas avoir de grandes différences entre les genres.
# Mais des montants sont clairement élevés comparés à la moyenne (comme précédement)

# Relation entre le montant et le revenu
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Transaction_Amount", y="Income")
plt.title('\nRelation entre le Montant des transactions et le revenu des clients\n', fontsize=16)
plt.xlabel('Montant des transactions', fontsize=14)
plt.ylabel('Revenu des clients', fontsize=14)
plt.ticklabel_format(style='plain', axis='y')
plt.show()

# Compter combien il y a des montants élevées
df['High']  = df['Transaction_Amount'] > 2500
nb_hauts = df['High'].sum() # Il y en a 20, soit 2%

# Indépendance montants-volume
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, y="Transaction_Amount", x="Transaction_Volume", hue="Gender")
plt.title('\nMontant des transactions par volume des transactions et par genre des clients\n', fontsize=16)
plt.xlabel('Volume des transactions', fontsize=14)
plt.ylabel('Montant des transactions', fontsize=14)
plt.legend(title="Genre")
plt.show()

high_df = df[df['High'] == 1]
transaction_count = high_df.groupby(['Transaction_Volume', 'Gender'])['Transaction_Amount'].count().reset_index()
transaction_count.columns = ['Volume', 'Genre', 'Nombre']
transaction_count = transaction_count.sort_values(by='Nombre', ascending=False)
print(transaction_count)

#  Volume   Genre  Nombre
#       3  Female       6
#       4  Female       4
#       3    Male       3
#       1  Female       2
#       2    Male       2
#       4    Male       2
#       2  Female       1

# 65 % des montants de transaction les plus importants sont réalisés par des femmes
# Globalement, plus le volume des transactions augmente, plus il y a de chance que le montant de la transaction soit anormalement élevée

## Réalisation de tests statistiques pour le prouver

### Relation Volume-Montant
correlation, pvalue= correlation(df, 'Transaction_Volume', 'Transaction_Amount', alpha=0.05)
# Corrélation de 4,57% et p-value de 0.1484 : il n'y a aucune relation statistiquement significative

### Relation Genre-Montants
quant_binary_ind(df, 'Gender', 'Transaction_Amount', alpha=0.05, graph=False)
# Les deux variables sont indépendantes : médianes similaires


# Montant par jours
days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, 
               x="Day_of_Week", 
               y="Transaction_Amount", 
               order=days_order)
plt.title('\nDistribution du montant des transactions par jour de la semaine\n', fontsize=16)
plt.xlabel('Jours de la semaine', fontsize=14)
plt.ylabel('Montant des transactions', fontsize=14)
plt.show()

# Relation l'âge et le montant + type de compte
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Age", y="Transaction_Amount", hue="Account_Type")
plt.title("\nRelation entre le montant des transactions et l'âge des clients\n", fontsize=16)
plt.xlabel('Âge des clients', fontsize=14)
plt.ylabel('Montant des transactions', fontsize=14)
plt.legend(title='Type de Compte')
plt.show()
# Le type de compte et l'âge des clients n'ont varisamblablement aucun lien avec le montant des transactions

# Matrice des corrélations (informations numériques)
plt.figure(figsize=(12, 8))
df_quant = df.select_dtypes(include=['int', 'float'])
correlation_matrix = df_quant.corr(method ='spearman')
sns.heatmap(correlation_matrix, annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5, annot_kws={"size": 15})
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.title('\nCorrélations entre les informations numériques avec les anomalies\n', fontsize=16)
plt.tight_layout()
plt.show()

# Matrice d'indépendance (informations qualitatives)
df_cat = df.select_dtypes(include=['object', 'category']).drop(columns='Transaction_ID')

# Calculer les p-values du test du khi-deux entre chaque paire de variables catégorielles
p_value_matrix = pd.DataFrame(np.zeros((len(df_cat.columns), len(df_cat.columns))), 
                              columns=df_cat.columns, index=df_cat.columns)

for col1 in df_cat.columns:
    for col2 in df_cat.columns:
        if col1 == col2:
            p_value_matrix.loc[col1, col2] = np.nan  # Pas de p-value pour la même variable
        else:
            # Créer une table de contingence
            contingency_table = pd.crosstab(df_cat[col1], df_cat[col2])
            # Calculer le test du khi-deux
            _, p_value, _, _ = chi2_contingency(contingency_table)
            # Ajouter la p-value dans la matrice
            p_value_matrix.loc[col1, col2] = p_value

# Afficher la heatmap des p-values
plt.figure(figsize=(12, 8))
sns.heatmap(p_value_matrix, annot=True, fmt=".4",
            linewidths=0.5, annot_kws={"size": 15})

plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.title('\nCarte thermique des p-values du test du Khi-Deux avec les anomalies\n', fontsize=16)
plt.tight_layout()
plt.show()

# Même chose en enlevant les anomalies observées
df_low = df[df['High'] != 1]

# Matrice des corrélations (informations numériques)
plt.figure(figsize=(12, 8))
df_quant = df_low.select_dtypes(include=['int', 'float'])
correlation_matrix = df_quant.corr(method ='spearman')
sns.heatmap(correlation_matrix, annot=True, fmt=".2f",
            cmap="coolwarm", linewidths=0.5, annot_kws={"size": 15})
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.title('\nCorrélations entre les informations numériques sans les anomalies\n', fontsize=16)
plt.tight_layout()
plt.show()

# Matrice d'indépendance (informations qualitatives)
df_cat = df_low.select_dtypes(include=['object', 'category']).drop(columns='Transaction_ID')

# Calculer les p-values du test du khi-deux entre chaque paire de variables catégorielles
p_value_matrix = pd.DataFrame(np.zeros((len(df_cat.columns), len(df_cat.columns))), 
                              columns=df_cat.columns, index=df_cat.columns)

for col1 in df_cat.columns:
    for col2 in df_cat.columns:
        if col1 == col2:
            p_value_matrix.loc[col1, col2] = np.nan  # Pas de p-value pour la même variable
        else:
            # Créer une table de contingence
            contingency_table = pd.crosstab(df_cat[col1], df_cat[col2])
            # Calculer le test du khi-deux
            _, p_value, _, _ = chi2_contingency(contingency_table)
            # Ajouter la p-value dans la matrice
            p_value_matrix.loc[col1, col2] = p_value

# Afficher la heatmap des p-values
plt.figure(figsize=(12, 8))
sns.heatmap(p_value_matrix, annot=True, fmt=".4",
            linewidths=0.5, annot_kws={"size": 15})

plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.title('\nCarte thermique des p-values du test du Khi-Deux sans les anomalies\n', fontsize=16)
plt.tight_layout()
plt.show()