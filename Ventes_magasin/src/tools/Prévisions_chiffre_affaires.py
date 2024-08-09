# =============================================================================
# Importation de la base puis traitement
# =============================================================================

df = pd.read_csv("Ventes.csv", sep=";")

# Conservation que des colonnes utiles pour la modélisation
df.columns  # Pour avoir la liste des variables
col_conserv = ['Date', 'Total_Amount']  # Liste des colonenes à conserver
df = df[col_conserv]
del col_conserv

# Transformation de la variable Date en Datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values(by='Date')


# Avoir la moyenne des ventes par jour
df = df.groupby('Date')['Total_Amount'].mean().reset_index()
df.set_index("Date", inplace=True)
df = df.dropna()

# =============================================================================
# Visualisation de l'évolution des ventes
# =============================================================================

# Visualisation graphique des ventes au fil du temps
plt.figure(figsize=(10, 6))
plt.plot(df.index, df.values)
plt.title('Evolution des ventes moyennes au cours du temps')
plt.xlabel('Date')
plt.ylabel('Ventes moyennes')
plt.grid(False)
plt.show()

# =============================================================================
# Evaluation de la stationnarité
# =============================================================================

# Division de la base en apprentissage/validation
train_data, test_data = temporal_train_test_split(df, test_size=1/3)

# Test de DickerFuller
DickeyFuller(train_data, 'Total_Amount', 0.05)
# La série est stationnaire, donc nous n'avons pas besoin de différencier

# =============================================================================
# Etude de l'autocorrélation
# =============================================================================

p, q = pq_param(lags=25, data=train_data)

# =============================================================================
# Construction du modèle et création des prévisions
# =============================================================================

# Création et ajustement du modèle ARMA sur les données d'entraînement
# Comme la série est stationnaire, d=0 donc le modèle développé est un ARMA(p,q)
model = ARIMA(train_data, order=(0, 0, 0))
model_fit = model.fit()

# Résumé du modèle
print(model_fit.summary())

# Prévisions pour l'ensemble de test
n_periods = len(test_data)
forecast = model_fit.forecast(steps=n_periods)

# Créez un index pour les prévisions basé sur l'index de test_data
forecast_index = pd.date_range(
    start=test_data.index[0], periods=n_periods, freq='D')

# Calcul de l'erreur (MAPE)
mape = mean_absolute_percentage_error(test_data, forecast)
print(f'Mean Absolute Percentage Error : {mape:.2f}%')

# Prédictions sur l'ensemble d'entraÃ®nement
train_predictions = model_fit.predict(
    start=train_data.index[0], end=train_data.index[-1])

# Tracer les prédictions et les données réelles avec les courbes collées
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data.values,
         label="Ensemble d'entraînement", color='blue')
plt.plot(test_data.index, test_data.values,
         label='Ensemble de test (réel)', color='blue', linestyle='--')
plt.plot(train_data.index, train_predictions,
         label='Prédictions (Entraînement)', color='red')
plt.plot(forecast_index, forecast, label='Prévisions (Test)', color='green')

# Définir les limites des axes x et y pour une continuité visuelle
plt.xlim(train_data.index[0], test_data.index[-1])
y_min = min(train_data.min().values[0], test_data.min().values[0])
y_max = max(train_data.max().values[0], test_data.max().values[0])
plt.ylim(y_min, y_max)

plt.xlabel('Date')
plt.ylabel('Valeur')
plt.title('Prédictions du modèle ARIMA')
plt.legend()
plt.show()

# =============================================================================
# Validation du modèle ou non
# =============================================================================

# Test de Ljung-Box
ljung_box_results = sm.stats.acorr_ljungbox(
    model_fit.resid, lags=[1], return_df=True)

# Test de Jarque-Bera
jb_test_stat, jb_test_p_value, skew, kurtosis = sm.stats.jarque_bera(
    model_fit.resid)

# Test de Hétéroscédasticité (Breusch-Pagan)
exog = add_constant(train_data)
bp_test_stat, bp_test_p_value, _, _ = het_breuschpagan(model_fit.resid, exog)

results_df = pd.DataFrame({
    'Test': ['Ljung-Box', 'Jarque-Bera', 'Breusch-Pagan'],
    'p-value': [ljung_box_results['lb_pvalue'].iloc[0], jb_test_p_value, bp_test_p_value],
    'Hypothèse': [
        'Pas d\'autocorrélation dans les résidus',
        'Les résidus sont distribués normalement',
        'Pas d\'hétéroscédasticité dans les résidus'],
    'Hypothèse validée': [ljung_box_results['lb_pvalue'].iloc[0] > 0.05, jb_test_p_value > 0.05, bp_test_p_value > 0.05]
})

# Afficher les résultats
print(results_df.iloc[:, -2:])

# Enseignements :
# Les résidus sont un bruit blanc
# Lé série n'est que du bruit blanc : p = q = 0
# Donc il n'y a rien à modéliser
# Les résidus sont hétéroscédastiques
# Développement d'un modèle ARCH (ou GARCH) pour capturer cette hétéroscédasticité des résidus

# =============================================================================
# Développement d'un modèle ARCH ou GARCH
# =============================================================================

df_results = GARCH_search(train_data, "Total_Amount", 7, 7)
print(df_results.head(3))

# Appliquer le meilleur model
model = arch_model(df["Total_Amount"], vol='Garch', p=6, q=0)
results = model.fit(disp='off')
summary = results.summary()

# =============================================================================
# Visualisation des prévisions
# =============================================================================

horizon_date = 30

pred = results.forecast(horizon=horizon_date)
future_dates = [df.index[-1] + timedelta(days=i)
                for i in range(1, horizon_date+1)]
pred = pd.Series(np.sqrt(pred.variance.values[-1, :]), index=future_dates)

plt.figure(figsize=(10, 4))
plt.plot(pred)
plt.title('Prédiction de Volatilité - 30 Prochains Jours', fontsize=20)

# =============================================================================
# Vérification des hypothèses
# =============================================================================

# Nous ne devons pas avoir d'hétéroscédasticité conditionnelle

# Residuals plot
plt.figure(figsize=(10, 4))
plt.plot(results.resid)
plt.title("Residuals of the GARCH model")

# ACF and PACF
plot_acf(results.resid, lags=20)
# S'il y a des pics significatifs, il y a hétéroscédasticité conditionnelle
plot_pacf(results.resid, lags=20)

# Shapiro-Wilk test for normality
stat, p = shapiro(results.resid)
print(f'Shapiro-Wilk statistic: {stat}, p-value: {p}')
# La normalité des résidus est rejettée

# LM test for ARCH effects pour vérifier la présence d'hétéroscédasticité conditionnelle
lm_test = het_arch(results.resid)
print('LM Statistical Test: %.3f, p-value: %.3f' % (lm_test[0], lm_test[1]))
# Le modèle ne capture pas correctement la structure de volatilité (car la P-Value est trop grande)

# Globalement, le modèle ARMA et ARCH ne sont pas de bons modèles pour
# modéliser les ventes de produits
