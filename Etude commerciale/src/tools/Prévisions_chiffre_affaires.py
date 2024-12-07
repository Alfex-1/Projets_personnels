# =============================================================================
# Importation de la base puis traitement
# =============================================================================

df = pd.read_csv(r"C:\Projets_personnels\Ventes_magasin\src\data\Ventes.csv", sep=";")

# Conservation que des colonnes utiles pour la mod�lisation
df.columns  # Pour avoir la liste des variables
col_conserv = ['Date', 'Total_Amount']  # Liste des colonenes � conserver
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
# Visualisation de l'�volution des ventes
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
# Evaluation de la stationnarit�
# =============================================================================

# Division de la base en apprentissage/validation
train_data, test_data = temporal_train_test_split(df, test_size=1/3)

# Test de DickerFuller
DickeyFuller(train_data, 'Total_Amount', 0.05)
# La s�rie est stationnaire, donc nous n'avons pas besoin de diff�rencier

# =============================================================================
# Etude de l'autocorr�lation
# =============================================================================

p, q = pq_param(lags=25, data=train_data)

# =============================================================================
# Construction du mod�le et cr�ation des pr�visions
# =============================================================================

# Cr�ation et ajustement du mod�le ARMA sur les donn�es d'entra�nement
# Comme la s�rie est stationnaire, d=0 donc le mod�le d�velopp� est un ARMA(p,q)
model = ARIMA(train_data, order=(0, 0, 0))
model_fit = model.fit()

# R�sum� du mod�le
print(model_fit.summary())

# Pr�visions pour l'ensemble de test
n_periods = len(test_data)
forecast = model_fit.forecast(steps=n_periods)

# Cr�ez un index pour les pr�visions bas� sur l'index de test_data
forecast_index = pd.date_range(
    start=test_data.index[0], periods=n_periods, freq='D')

# Calcul de l'erreur (MAPE)
mape = mean_absolute_percentage_error(test_data, forecast)
print(f'Mean Absolute Percentage Error : {mape:.2f}%')

# Pr�dictions sur l'ensemble d'entraînement
train_predictions = model_fit.predict(
    start=train_data.index[0], end=train_data.index[-1])

# Tracer les pr�dictions et les donn�es r�elles avec les courbes coll�es
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data.values,
         label="Ensemble d'entra�nement", color='blue')
plt.plot(test_data.index, test_data.values,
         label='Ensemble de test (r�el)', color='blue', linestyle='--')
plt.plot(train_data.index, train_predictions,
         label='Pr�dictions (Entra�nement)', color='red')
plt.plot(forecast_index, forecast, label='Pr�visions (Test)', color='green')

# D�finir les limites des axes x et y pour une continuit� visuelle
plt.xlim(train_data.index[0], test_data.index[-1])
y_min = min(train_data.min().values[0], test_data.min().values[0])
y_max = max(train_data.max().values[0], test_data.max().values[0])
plt.ylim(y_min, y_max)

plt.xlabel('Date')
plt.ylabel('Valeur')
plt.title('Pr�dictions du mod�le ARIMA')
plt.legend()
plt.show()

# =============================================================================
# Validation du mod�le ou non
# =============================================================================

# Test de Ljung-Box
ljung_box_results = sm.stats.acorr_ljungbox(
    model_fit.resid, lags=[1], return_df=True)

# Test de Jarque-Bera
jb_test_stat, jb_test_p_value, skew, kurtosis = sm.stats.jarque_bera(
    model_fit.resid)

# Test de H�t�rosc�dasticit� (Breusch-Pagan)
exog = add_constant(train_data)
bp_test_stat, bp_test_p_value, _, _ = het_breuschpagan(model_fit.resid, exog)

results_df = pd.DataFrame({
    'Test': ['Ljung-Box', 'Jarque-Bera', 'Breusch-Pagan'],
    'p-value': [ljung_box_results['lb_pvalue'].iloc[0], jb_test_p_value, bp_test_p_value],
    'Hypoth�se': [
        'Pas d\'autocorr�lation dans les r�sidus',
        'Les r�sidus sont distribu�s normalement',
        'Pas d\'h�t�rosc�dasticit� dans les r�sidus'],
    'Hypoth�se valid�e': [ljung_box_results['lb_pvalue'].iloc[0] > 0.05, jb_test_p_value > 0.05, bp_test_p_value > 0.05]
})

# Afficher les r�sultats
print(results_df.iloc[:, -2:])

# Enseignements :
# Les r�sidus sont un bruit blanc
# L� s�rie n'est que du bruit blanc : p = q = 0
# Donc il n'y a rien � mod�liser
# Les r�sidus sont h�t�rosc�dastiques
# D�veloppement d'un mod�le ARCH (ou GARCH) pour capturer cette h�t�rosc�dasticit� des r�sidus

# =============================================================================
# D�veloppement d'un mod�le ARCH ou GARCH
# =============================================================================

df_results = GARCH_search(train_data, "Total_Amount", 7, 7)
print(df_results.head(3))

# Appliquer le meilleur model
model = arch_model(df["Total_Amount"], vol='Garch', p=6, q=0)
results = model.fit(disp='off')
summary = results.summary()

# =============================================================================
# Visualisation des pr�visions
# =============================================================================

horizon_date = 30

pred = results.forecast(horizon=horizon_date)
future_dates = [df.index[-1] + timedelta(days=i)
                for i in range(1, horizon_date+1)]
pred = pd.Series(np.sqrt(pred.variance.values[-1, :]), index=future_dates)

plt.figure(figsize=(10, 4))
plt.plot(pred)
plt.title('Pr�diction de Volatilit� - 30 Prochains Jours', fontsize=20)

# =============================================================================
# V�rification des hypoth�ses
# =============================================================================

# Nous ne devons pas avoir d'h�t�rosc�dasticit� conditionnelle

# Residuals plot
plt.figure(figsize=(10, 4))
plt.plot(results.resid)
plt.title("Residuals of the GARCH model")

# ACF and PACF
plot_acf(results.resid, lags=20)
# S'il y a des pics significatifs, il y a h�t�rosc�dasticit� conditionnelle
plot_pacf(results.resid, lags=20)

# Shapiro-Wilk test for normality
stat, p = shapiro(results.resid)
print(f'Shapiro-Wilk statistic: {stat}, p-value: {p}')
# La normalit� des r�sidus est rejett�e

# LM test for ARCH effects pour v�rifier la pr�sence d'h�t�rosc�dasticit� conditionnelle
lm_test = het_arch(results.resid)
print('LM Statistical Test: %.3f, p-value: %.3f' % (lm_test[0], lm_test[1]))
# Le mod�le ne capture pas correctement la structure de volatilit� (car la P-Value est trop grande)

# Globalement, le mod�le ARMA et ARCH ne sont pas de bons mod�les pour
# mod�liser les ventes de produits