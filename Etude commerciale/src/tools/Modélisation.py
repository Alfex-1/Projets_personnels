import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.graph_objects as go
import pmdarima as pm
from collections import defaultdict
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def DickeyFuller(data, feature, pvalue, show_graph=False):
    data = data.dropna()
    # Perform the Augmented Dickey-Fuller test
    result = adfuller(data[feature], autolag='AIC')

    # Extract the test results
    test_statistic, p_value, lags, _, critical_values, _ = result

    # Display the time serie graphic if show_graph is True
    if show_graph:
        data[feature].plot(title='Time series')
        plt.ylabel(f'{feature} Values')
        plt.show()

    # Interpret the results
    if p_value <= pvalue:
        ccl = "stationary"
    else:
        ccl = "non-stationnary"
    return ccl

def seasonal_naive_forecast(series, horizon, seasonality):
    if len(series) < seasonality:
        raise ValueError("Not enough data for this seasonality.")
    last_season = series[-seasonality:]
    reps = int(np.ceil(horizon / seasonality))
    forecast_values = np.tile(last_season.values, reps)[:horizon]
    index = pd.date_range(start=series.index[-1] + pd.Timedelta(1, unit="D"), periods=horizon, freq=series.index.freq or pd.infer_freq(series.index))
    return pd.Series(data=forecast_values, index=index)

def evaluate_seasonal_naive(series, horizon, seasonality_list, n_splits=5, step=1):
    results = defaultdict(list)
    forecasts_store = {}

    max_len = len(series)

    for split in range(n_splits):
        train_end = max_len - horizon - (n_splits - split - 1) * step
        if train_end < max(seasonality_list):
            continue

        train = series.iloc[:train_end]
        test = series.iloc[train_end:train_end + horizon]

        for s in seasonality_list:
            if len(train) < s:
                continue
            try:
                forecast = seasonal_naive_forecast(train, horizon, s)
                score = pm.metrics.smape(test.values, forecast.values)
                results[s].append(score)
                # stocker la dernière prévision pour affichage
                if split == n_splits - 1:
                    forecasts_store[s] = forecast
            except Exception as e:
                print(f"Saut saisonnalité {s} (split {split}) — erreur : {e}")
                continue

    # Moyenne des SMAPE
    avg_smape = {s: np.mean(scores) for s, scores in results.items() if scores}
    sorted_results = sorted(avg_smape.items(), key=lambda x: x[1])
    
    best_s, best_smape = sorted_results[0]
    best_forecast = forecasts_store[best_s]

    print(f"✅ Meilleure saisonnalité : {best_s} — SMAPE moyenne = {best_smape:.2f}% sur {n_splits} splits")

    return best_s, best_smape, best_forecast

def seasonal_naive_forecast(series, horizon, seasonality):
    """
    Forecast future values using the seasonal naïve method.

    Parameters:
    - series: pd.Series, time series indexed by date
    - horizon: int, number of future steps to predict
    - seasonality: int, the periodicity (e.g., 12 for monthly with yearly cycle)

    Returns:
    - forecast: pd.Series with predicted values
    """
    if len(series) < seasonality:
        raise ValueError("Not enough data to use seasonal naïve forecast.")

    last_season = series[-seasonality:]  # extrait le dernier cycle complet
    reps = int(np.ceil(horizon / seasonality))  # combien de fois on doit répéter le dernier cycle
    forecast_values = np.tile(last_season.values, reps)[:horizon]  # découpe pile le bon nombre

    index = pd.date_range(start=series.index[-1] + pd.Timedelta(1, unit="D"), periods=horizon, freq='D')
    return pd.Series(data=forecast_values, index=index)

def stl_forecast(series, horizon, period):
    stl = STL(series, period=period, robust=True)
    res = stl.fit()

    trend = res.trend.dropna()
    delta = trend.diff().iloc[-1]
    last_trend = trend.iloc[-1]
    trend_forecast = [last_trend + delta * i for i in range(1, horizon + 1)]

    seasonal = res.seasonal.dropna()
    last_seasonal = seasonal.iloc[-period:]
    reps = int(np.ceil(horizon / period))
    seasonal_forecast = np.tile(last_seasonal.values, reps)[:horizon]

    forecast_values = np.array(trend_forecast) + seasonal_forecast
    index = pd.date_range(start=series.index[-1] + pd.Timedelta(1, unit="D"), periods=horizon, freq=series.index.freq or pd.infer_freq(series.index))
    
    return pd.Series(forecast_values, index=index)

def evaluate_stl_forecast(series, horizon, seasonality_list, n_splits=5, step=1):
    results = defaultdict(list)
    forecast_overlay = defaultdict(list)

    max_len = len(series)

    for split in range(n_splits):
        train_end = max_len - horizon - (n_splits - split - 1) * step
        if train_end < max(seasonality_list):
            continue

        train = series.iloc[:train_end]
        test = series.iloc[train_end:train_end + horizon]

        for s in seasonality_list:
            if len(train) < s:
                continue
            try:
                forecast = stl_forecast(train, horizon, period=s)
                score = pm.metrics.smape(test.values, forecast.values)
                results[s].append(score)
                forecast_overlay[s].append(forecast)
            except Exception as e:
                print(f"Saut STL saisonnalité {s} (split {split}) — erreur : {e}")
                continue

    avg_smape = {s: np.mean(scores) for s, scores in results.items() if scores}
    sorted_results = sorted(avg_smape.items(), key=lambda x: x[1])

    best_s, best_smape = sorted_results[0]
    best_forecasts = forecast_overlay[best_s]

    print(f"✅ Meilleure saisonnalité STL : {best_s} — SMAPE moyenne = {best_smape:.2f}% sur {n_splits} splits")

    return best_s, best_smape, best_forecasts[-1]

import matplotlib.pyplot as plt

def plot_seasonal_naive(series, forecast):
    plt.figure(figsize=(12, 5))

    # Tracer la série réelle
    plt.plot(series.index, series.values, label='Série réelle', color='blue')

    # Tracer la prévision
    plt.plot(forecast.index, forecast.values, label='Prévision (saisonnier naïf)',
             color='orange', linestyle='--')

    # Ligne verticale au début de la prévision
    plt.axvline(x=series.index[-1], color='gray', linestyle=':')

    # Mise en forme
    plt.title("Prévision saisonnière naïve")
    plt.xlabel("Date")
    plt.ylabel("Valeur")
    plt.legend(loc='upper left')
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def stl_forecast_naive(ts, period, horizon):
    """
    Prévision via STL avec extrapolation naïve du trend et de la saisonnalité.

    Args:
        ts (pd.Series): Série temporelle.
        period (int): Saison (ex: 7 pour hebdo).
        horizon (int): Nombre de points à prévoir.

    Returns:
        pd.Series: Prévisions futures.
    """
    stl = STL(ts, period=period, robust=True)
    res = stl.fit()
    
    # Trend: extrapolation naïve (last value)
    last_trend = res.trend.iloc[-1]
    trend_forecast = np.full(horizon, last_trend)
    
    # Saison: on boucle le dernier cycle complet
    seasonal_cycle = res.seasonal.iloc[-period:]
    seasonal_forecast = np.resize(seasonal_cycle.values, horizon)

    # Forecast = trend + saison
    forecast = trend_forecast + seasonal_forecast
    return pd.Series(forecast, 
                     index=pd.date_range(ts.index[-1], periods=horizon+1, freq=pd.infer_freq(ts.index))[1:])

# Importation et traitement
df = pd.read_csv(r"C:\Projets_personnels\Etude commerciale\src\data\Ventes.csv", sep=';')
df = df[['Date','Total_Amount']]

# Grouper par la somme les ventes par date
df.loc[:,'Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df = df.groupby(['Date']).sum().reset_index()
df.set_index('Date', inplace=True)

# Interpolation
new_dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq='D')
df = df.reindex(new_dates)
df = df.interpolate(method='quadratic', limit_direction='both')
df = abs(round(df,0))

# Test de stationnarité
DickeyFuller(df, 'Total_Amount', 0.05, show_graph=False) # Stationnaire

# ACF et PACF
plot_acf(df, lags=25)
plot_pacf(df, lags=25)

ts = df.iloc[:, 0]

# Tester différentes saisonnalités
max_period = 50  # Période max à tester, ici de 1 à 12 mois

# Dictionnaire pour stocker les résultats
results = []

# Comparer les modèles pour chaque saisonnalité
for period in range(2, max_period + 1):
    # Modèle additif
    decomposition_additive = seasonal_decompose(ts, model='additive', period=period)
    trend_additive = decomposition_additive.trend
    seasonal_additive = decomposition_additive.seasonal
    resid_additive = decomposition_additive.resid

    predicted_additive = trend_additive + seasonal_additive + resid_additive
    predicted_additive = predicted_additive.dropna()

    # Modèle multiplicatif
    decomposition_multiplicative = seasonal_decompose(ts, model='multiplicative', period=period)
    trend_multiplicative = decomposition_multiplicative.trend
    seasonal_multiplicative = decomposition_multiplicative.seasonal
    resid_multiplicative = decomposition_multiplicative.resid

    predicted_multiplicative = trend_multiplicative * seasonal_multiplicative * resid_multiplicative
    predicted_multiplicative = predicted_multiplicative.dropna()

    # Calcul des erreurs (SMAPE)
    actual_values_additive = ts.loc[predicted_additive.index]
    smape_additive = pm.metrics.smape(actual_values_additive, predicted_additive)

    actual_values_multiplicative = ts.loc[predicted_multiplicative.index]
    smape_multiplicative = pm.metrics.smape(actual_values_multiplicative, predicted_multiplicative)

    # Stockage des résultats
    results.append({
        'Period': period,
        'SMAPE Additive': smape_additive,
        'SMAPE Multiplicative': smape_multiplicative,
        'Best Model': 'Additive' if smape_additive < smape_multiplicative else 'Multiplicative'
    })

# Convertir en DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='SMAPE Additive')


# Affichage des résultats
print(results_df)

# Identifier la meilleure combinaison (minimum de SMAPE)
best_result = results_df.loc[results_df['SMAPE Additive'].idxmin()]
print(f"Meilleur modèle : {best_result['Best Model']} avec une saisonnalité de {best_result['Period']} mois")

# Tracer les prédictions avec la meilleure saisonnalité et modèle sans les résidus
best_period = best_result['Period']
if best_result['Best Model'] == 'Additive':
    decomposition_best = seasonal_decompose(ts, model='additive', period=best_period)
    predicted_best = decomposition_best.trend + decomposition_best.seasonal
else:
    decomposition_best = seasonal_decompose(ts, model='multiplicative', period=best_period)
    predicted_best = decomposition_best.trend * decomposition_best.seasonal

plt.figure(figsize=(12, 6))
plt.plot(ts, label='Série réelle', color='blue')
plt.plot(predicted_best, label=f'Prédiction {best_result["Best Model"]}', color='orange')
plt.title(f"Prédictions avec le meilleur modèle ({best_result['Best Model']}) et saisonnalité {best_period}")
plt.legend()
plt.show()

# Choisir la meilleure saisonnalité qui minimise l'erreur pour le modèle sausonnier naïf
best_s, best_score, best_forecast = evaluate_seasonal_naive(df.iloc[:, 0], horizon=12, seasonality_list=np.arange(1, 100), n_splits=5)

# Prévision sur 12 mois
forecast = seasonal_naive_forecast(df.iloc[:, 0], horizon=62, seasonality=best_s)
print(forecast)

# Visualisation   
plot_seasonal_naive(df.iloc[:, 0], forecast)

# Déterminer le cyle siaosnnier pour STL
best_s, best_smape, best_forecasts = evaluate_stl_forecast(ts, 30, np.arange(2, 50), n_splits=5, step=1)

# Prévision STL avec extrapolation naïve du trend et de la saisonnalité   
forecast_result = stl_forecast_naive(ts, period=best_s, horizon=30)

# Visualisation
plt.figure(figsize=(12, 5))
plt.plot(ts, label='Valeurs observées', color='blue')
plt.plot(forecast_result.index, forecast_result, label='Prévision STL', color='red', linestyle='--')
plt.axvline(ts.index[-1], color='gray', linestyle=':', label="Début de prévision")
plt.legend()
plt.title('Prévision STL avec extrapolation de tendance et saisonnalité')
plt.tight_layout()
plt.show()