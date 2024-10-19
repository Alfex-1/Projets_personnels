# =============================================================================
# Développement de premiers modèles simple de régression (avec et ans anomalies)
# =============================================================================

# Importation et division
df_with_anom = pd.read_csv(r"C:\Projets_personnels\Prix_immobilier_Iowa\src\data\Data_selected_col_with_anom.csv")
df_without_anom = pd.read_csv(r"C:\Projets_personnels\Prix_immobilier_Iowa\src\data\Data_selected_col_without_anom.csv")

############ Avec anomalies

# Division variables explicatives (X) et variable expliquée (y)
X = df_with_anom.drop('SalePrice', axis=1)
y = df_with_anom['SalePrice']

# Ajout de l'intercept (constante)
X = sm.add_constant(X)  

# Dvision apprentissage (train) et validation (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

############ Sans anomalies

# Division variables explicatives (X) et variable expliquée (y)
X2 = df_without_anom.drop('SalePrice', axis=1)
y2 = df_without_anom['SalePrice']

# Ajout de l'intercept (constante)
X2 = sm.add_constant(X2)  

# Dvision apprentissage (train) et validation (test)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Evaluation gloable sur la base entière

# Regression linéaire avec les anaomalies
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

# Verification des hypothèses
fitted_values = results.fittedvalues
residuals = results.resid

plt.figure(figsize=(8, 6))
sns.scatterplot(x=fitted_values, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Valeurs ajustées')
plt.ylabel('Résidus')
plt.title('Valeurs ajustées vs Résidus')
plt.show()

dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson Stat: {dw_stat:.2f}") # 1.54 : non-respect

# Test de Breusch-Pagan
test = sms.het_breuschpagan(residuals, X)
print(f"Breusch-Pagan Test - p-value: {test[1]:.4f}") # Rejet de H0

# Normalité
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True)
plt.title('Distribution des Résidus')
plt.xlabel('Résidus')
plt.show() # Normalité respectée

# Regression linéaire sans les anomalies
model2 = sm.OLS(y2, X2)
results2 = model2.fit()
print(results2.summary())
# Des performances légèrement améliorées

# Verification des hypothèses
fitted_values2 = results2.fittedvalues
residuals2 = results2.resid

# Homoscédascité et linéarité
plt.figure(figsize=(8, 6))
sns.scatterplot(x=fitted_values2, y=residuals2)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Valeurs ajustées')
plt.ylabel('Résidus')
plt.title('Valeurs ajustées vs Résidus')
plt.show() # Meilleure répartition que précédemment

# Indépendance
dw_stat2 = durbin_watson(residuals2)
print(f"Durbin-Watson Stat: {dw_stat2:.2f}") # 1.45

# Test de Breusch-Pagan (d'homoscedasticité)
test2 = sms.het_breuschpagan(residuals2, X2)
print(f"Breusch-Pagan Test - p-value: {test2[1]:.4f}") # Rejet de H0

# Normalité
plt.figure(figsize=(8, 6))
sns.histplot(residuals2, kde=True)
plt.title('Distribution des Résidus')
plt.xlabel('Résidus')
plt.show() # Normalité respectée

# Conclusion : Le modèle construit sans les anomalies semble être un meilleur choix

#### Etude des performances des modèles : estimation de l'erreur théorique de prévision

# Initier un modèle de régression linéaire pour les deux bases
model_with_anom = LinearRegression()
model_with_anom.fit(X_train,y_train)

model_without_anom = LinearRegression()
model_without_anom.fit(X2_train,y2_train)

# RMSE et MAPE sans les anomalies
rmse_without_anom = estimate_forecasting_error(model=model_without_anom, X=X2, y=y2, k=10, metric="rmse")
mape_without_anom = estimate_forecasting_error(model=model_without_anom, X=X2, y=y2, k=10, metric="mape")

# RMSE et MAPE avec les anomalies
rmse_with_anom = estimate_forecasting_error(model=model_with_anom, X=X, y=y, k=10, metric="rmse")
mape_with_anom = estimate_forecasting_error(model=model_with_anom, X=X, y=y, k=10, metric="mape")

print('RMSE sans anomalie :',rmse_without_anom)
print('MAPE sans anomalie :',mape_without_anom)
print('---------------------------------------')
print('RMSE avec anomalies :',rmse_with_anom)
print('MAPE avec anomalies :',mape_with_anom)

# Conclusion : la régression linéaire est plus performante sans les anomalies (sur tous les points : AIC, R², erreur, hypothèses),
# mais cela reste marginal.
# Alors les intégrer dans l'entraînement du modèle n'est pas une mauvaise idée

# =============================================================================
# Développement d'un modèle de régression polynomiale sans les anomalies
# =============================================================================

df = pd.read_csv(r"C:\Projets_personnels\Prix_immobilier_Iowa\src\data\Data_selected_col_without_anom.csv")

# Division variables explicatives (X) et variable expliquée (y)
X = df.drop(columns='SalePrice', axis=1)
y = df['SalePrice']

# Dvision apprentissage (train) et validation (test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle polynomial de degré 2
polynomial_features = PolynomialFeatures(degree=5)
X_train = polynomial_features.fit_transform(X_train)
X_test = polynomial_features.transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions sur les ensembles d'entraînement et de test
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluation du modèle
rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)
mape_scorer = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

rmse_poly2 = round(-np.mean(cross_val_score(estimator = model, X=X, y=y, cv=10, n_jobs=3, scoring = rmse_scorer)),4)
mape_poly2 = round(-np.mean(cross_val_score(estimator = model, X=X, y=y, cv=10, n_jobs=3, scoring = mape_scorer)),4)

print("RMSE du modèle polynomial d'ordre 2 :", rmse_poly2)
print("MAPE du modèle polynomial d'ordre 2 :", mape_poly2)

# Les deux modèles ont gloablement les mêmes performances,
# alors pour plus de simplicité, on grade la régression linéaire