# Importation
df = pd.read_csv(r"C:\Projets_personnels\Détection anomalies\src\data\New_data.csv")


# Division des données en apprentissage-validation
X = df.drop('Anomaly', axis=1)  # Les variables explicatives
y = df['Anomaly']  # La variable cible

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

# Création du modèle de régression logistique binaire
best_params_reglog = exhaustive_logistic_regression_search(
    X_train,
    y_train,
    ['l1', 'l2', 'elasticnet'],
    np.logspace(-4, 4, 10),
    np.arange(0, 1.1, 0.1),
    metric='roc_auc',
    solvers=['lbfgs', 'liblinear', 'saga', 'newton-cholesky'],
    average='weighted',
    cv=10
)
best_model_reglog = LogisticRegression(**best_params_reglog, max_iter=1000, random_state=42)

# Evaluation des performances
df_results_reglog, conf_matrix_reglog = evaluation_model(df, X, y, 'Anomaly', best_model_reglog, ['precision','recall','f1'], 10, 'weighted')




# Choix et évaluation d'un XGBoost
param_distributions = {
        'n_estimators': np.arange(10, 510, 10),
        'learning_rate': np.arange(0.1, 1.05, 0.05),
        'reg_alpha': np.arange(0, 2.1, 0.1),
        'reg_lambda': np.arange(0, 2.1, 0.1),
        'gamma': np.arange(0, 2.1, 0.1),
        'max_depth': np.arange(1, 11, 1)
    }
model_xgb = XGBClassifier(random_state=42)
grid_xgb = RandomizedSearchCV(estimator=model_xgb, param_distributions=param_distributions, cv=20,
                          scoring='roc_auc', n_jobs=8, n_iter=5000,
                          random_state=42)
grid_result_xgb = grid_xgb.fit(X_train, y_train)
best_params_xgb = grid_result_xgb.best_params_ # Meilleurs hyperparamètres
# {'reg_lambda': 0.7,
# 'reg_alpha': 0.0,
# 'n_estimators': 50,
# 'max_depth': 4,
#'learning_rate': 0.6,
# 'gamma': 0.8}

best_model_xgb = XGBClassifier(**best_params_xgb, random_state=42) # Initailisation du meilleur modèle

# Evaluation des performances
df_results_xgb, conf_matrix_xgb = evaluation_model(df, 'Anomaly', best_model_xgb, ['precision','recall','f1'], 10, 'weighted')



# Choix et évaluation d'un K-Nearest Neighbors
param_grid_knn = {
    'n_neighbors': np.arange(1, 21, 1),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

model_knn = KNeighborsClassifier()
grid_knn = GridSearchCV(estimator=model_knn, param_grid=param_grid_knn, cv=20, scoring='roc_auc', n_jobs=8)
grid_result_knn = grid_knn.fit(X, y)
best_params_knn = grid_result_knn.best_params_
# {'metric': 'manhattan', 'n_neighbors': 15, 'weights': 'distance'}

best_model_knn =  KNeighborsClassifier(**best_params_knn)

# Evaluation des performances
df_results_knn, conf_matrix_knn = evaluation_model(df, X, y, 'Anomaly', best_model_knn, ['precision','recall','f1'], 10, 'weighted')