# Fixer la graine pour la reproductibilite
np.random.seed(42)

# Importation de la base non-equilibree et d ela base equilibree

df1 = pd.read_csv(r"C:\Donnees_nutriscore_v3\5Data_no_miss_unbalanced.csv", sep=',')
df2 = pd.read_csv(r"C:\Donnees_nutriscore_v3\5Data_no_miss_balanced.csv", sep=',')


# Encodage de la variable cible : 1 à 4
df1, infos1 = encoding_all_data(df1, reverse=True)
df2, infos2 = encoding_all_data(df2, reverse=True)

# Division des donnees de la base non-equilibree
X1 = df1.drop('NutriScore', axis=1)  # Variables predictives
y1 = df1['NutriScore']  # Variable cible

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Division des donnees de la base equilibree
X2 = df2.drop('NutriScore', axis=1)  # Variables predictives
y2 = df2['NutriScore']  # Variable cible

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)


# Initialiser tous les modeles
xg, ada, cat, lg = XGBClassifier(random_state=42), AdaBoostClassifier(random_state=42), CatBoostClassifier(
    random_state=42, verbose=False), lgb.LGBMClassifier(random_state=42, verbosity=-1)

# Chercher les modeles optimaux sur la base non-EQUILIBREE

# Initialiser le modèle
mlp = MLPClassifier(solver='adam',max_iter = 1000)

# Chercher le MLP optimal sur leux bases non-EQUILIBREE

hidden_layer_sizes = generate_layer_combinations(max_layers=4, max_neurons=4)
activation = ['tanh', 'relu']
alpha = np.arange(0.1,1.1,0.1)
learning_rate = np.arange(0.1, 1.1, 0.1)

# Pour la base equilibree
results_df_equ = CV_parameters_classif(mlp, hidden_layer_sizes, activation, alpha, learning_rate, selected_model=5,average='macro',X_train=X_train2,X_test=X_test2,y_train=y_train2,y_test=y_test2)

# Pour la base non-equilibree
results_df_non_eq = CV_parameters_classif(mlp, hidden_layer_sizes, activation, alpha, learning_rate, selected_model=5,average='weighted',X_train=X_train1,X_test=X_test1,y_train=y_train1,y_test=y_test1)

