# Fixer la graine pour la reproductibilite
np.random.seed(42)

# Importation de la base non-equilibree et de la base equilibree

df1 = pd.read_csv(r"C:\Données_nutriscore\6Data_no_miss_unbalanced.csv", sep=',')
df2 = pd.read_csv(r"C:\Données_nutriscore\7Data_no_miss_balanced.csv", sep=',')

# Exploration des données
df_graph, infos_graph = encoding_all_data(df1, reverse=True)
for i in df_graph.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_graph, x="NutriScore", y=i)
    plt.title(f'Distribution de {i} en fonction du Nutri-score')
    plt.xlabel('Nutri-score')
    plt.ylabel(i)
    plt.show()

# Encodage de la variable cible : 1 à 4
df1, infos1 = encoding_all_data(df1, reverse=False) # Reverse doit être False
df2, infos2 = encoding_all_data(df2, reverse=False)

# Suppression de la multicolinéarité
rejected_var1 = vif_selection(df1, 'NutriScore', vif_value=5)
rejected_var2 = vif_selection(df2, 'NutriScore', vif_value=5)

# Seule la variable 'Energie_kcal' est rejettée dans les 2 cas
df1 = df1.drop(columns=rejected_var1)
df2 = df2.drop(columns=rejected_var2)

# S'assurer que les modalités sont correctement ordonnées
df1['NutriScore'] = pd.Categorical(
    df1['NutriScore'],
    categories=[4, 3, 2, 1, 0],
    ordered=True)

df2['NutriScore'] = pd.Categorical(
    df2['NutriScore'],
    categories=[4, 3, 2, 1, 0],
    ordered=True)

# Division des donnees de la base non-equilibree
X1 = df1.drop('NutriScore', axis=1)  # Variables predictives
y1 = df1['NutriScore']  # Variable cible

# Division des donnees de la base equilibree
X2 = df2.drop('NutriScore', axis=1)  # Variables predictives
y2 = df2['NutriScore']  # Variable cible

# Définir les plages d'hyperparamètres pour la recherche
penalty_options = ['L1', 'L2', 'ElasticNet']
alpha_range = np.arange(0.05,1.05,0.05)
l1_ratio_range = np.arange(0.05,1.05,0.05)

best_params_eq = random_search_cv(X2, y2, penalty_options, alpha_range, 
                     l1_ratio_range, 50, n_splits=5, metric='accuracy')

best_params_non_eq = random_search_cv(X1, y1, penalty_options, alpha_range, 
                     l1_ratio_range, 50, n_splits=5, metric='f1')

# Construction du modèle non-équilibrée
model = OrderedModel(y1,X1,distr='logit').fit(method='bfgs', disp=True)
print(model.summary())

# Matrice de confusion
predicted_probs = model.predict()
predicted_classes = np.argmax(predicted_probs, axis=1)
y1_numeric = y1.cat.codes if hasattr(y1, 'cat') else y1
conf_matrix = confusion_matrix(y1_numeric, predicted_classes)
conf_matrix_df = pd.DataFrame(conf_matrix, 
                              index=[f'Vrai {i}' for i in range(conf_matrix.shape[0])],
                              columns=[f'Prédit {i}' for i in range(conf_matrix.shape[1])])

# Affichage graphique de la matrice de confusion avec les valeurs en pourcentage
classes = ['A', 'B', 'C', 'D', 'E']
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_normalized, annot=False, cmap='YlOrRd', 
            xticklabels=classes, yticklabels=classes, linewidths=0.1)
plt.xlabel('Nutri-Score prédit')
plt.ylabel('Nutri-Score réel')
plt.title('Matrice de confusion du modèle déséquilibré (en %)')
plt.show()

# Avoir le nombre d'observation correctement classé et le nombre relatif
sum_abs_diagonal = np.sum(np.diag(conf_matrix)) # 480 568
good_predictions = sum_abs_diagonal/len(X1) # 50,41%

# Obtenir les coefficients
params = model.params

# Calculer les rapports de cotes
odds_ratios = np.exp(params)[:len(X1.columns)]
sorted_indices = np.argsort(odds_ratios)
odds_ratios_sorted = odds_ratios[sorted_indices]
relevant_columns = X1.columns[:len(odds_ratios)]
relevant_columns_sorted = X1.columns[sorted_indices]

# Tracer le graphique
plt.figure(figsize=(8, 6))
bars = plt.barh(relevant_columns_sorted, odds_ratios_sorted, color='skyblue')
plt.xlabel('Rapport de Cotes (Odds Ratio)')
plt.title('Rapport de Cotes du modèle non-équilibré')
plt.axvline(x=1, color='red', linestyle='--')  # Ligne de référence à 1
# Ajouter les valeurs à droite des barres
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
             f'{bar.get_width():.2f}', va='center', ha='left', color='black')
plt.show()

# Construction du modèle équilibrée
model2 = OrderedModel(y2,X2,distr='logit').fit(method='bfgs', disp=True)
print(model2.summary())

# Matrice de confusion
predicted_probs2 = model2.predict()
predicted_classes2 = np.argmax(predicted_probs2, axis=1)
y2_numeric = y2.cat.codes if hasattr(y2, 'cat') else y2
conf_matrix2 = confusion_matrix(y2_numeric, predicted_classes2)
conf_matrix_df2 = pd.DataFrame(conf_matrix2, 
                              index=[f'Vrai {i}' for i in range(conf_matrix2.shape[0])],
                              columns=[f'Prédit {i}' for i in range(conf_matrix2.shape[1])])

# Affichage graphique de la matrice de confusion avec les valeurs en pourcentage
classes = ['A', 'B', 'C', 'D', 'E']
conf_matrix_normalized2 = conf_matrix2.astype('float') / conf_matrix2.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_normalized2, annot=False, cmap='YlOrRd', 
            xticklabels=classes, yticklabels=classes, linewidths=0.1)
plt.xlabel('Nutri-Score prédit')
plt.ylabel('Nutri-Score réel')
plt.title('Matrice de confusion du modèle équilibré (en %)')
plt.show()

# Avoir le nombre d'observation correctement classé et le nombre relatif
sum_abs_diagonal2 = np.sum(np.diag(conf_matrix2)) # 624 972
good_predictions2 = sum_abs_diagonal2/len(X2) # 51,66%

# Obtenir les coefficients
params2 = model2.params

# Calculer les rapports de cotes
odds_ratios2 = np.exp(params2)[:len(X2.columns)]
sorted_indices = np.argsort(odds_ratios2)
odds_ratios2_sorted = odds_ratios2[sorted_indices]
relevant_columns2 = X2.columns[:len(odds_ratios2)]
relevant_columns2_sorted = X2.columns[sorted_indices]

# Tracer le graphique
plt.figure(figsize=(8, 6))
bars2 = plt.barh(relevant_columns2_sorted, odds_ratios2_sorted, color='skyblue')
plt.xlabel('Rapport de Cotes (Odds Ratio)')
plt.title('Rapport de Cotes du modèle équilibré')
plt.axvline(x=1, color='red', linestyle='--')  # Ligne de référence à 1
# Ajouter les valeurs à droite des barres
for bar in bars2:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, 
             f'{bar.get_width():.2f}', va='center', ha='left', color='black')
plt.show()

# Construction des modèles et estimation de leurs erreurs théorique de prévision
f1_non_eq = convergence_error_OrderedModel(data=df1, target='NutriScore',
                               distr_logistic='logit', method_logistic='bfgs',
                               iterations=70, test_size=0.2, scoring='f1', average='weighted')
# F1 = 0.1229

accuracy_eq = convergence_error_OrderedModel(data=df2, target='NutriScore',
                               distr_logistic='logit', method_logistic='bfgs',
                               iterations=70, test_size=0.2, scoring='accuracy', average='macro')

# Accuracy : 0.1131