# =============================================================================
# Importation des packages
# =============================================================================
import pandas as pd
import numpy as np
from sktime.forecasting.model_selection import temporal_train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from statsmodels.stats.diagnostic import het_breuschpagan, het_arch
from statsmodels.tools import add_constant
from datetime import timedelta
from arch import arch_model
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from scipy.stats import shapiro, ansari, mannwhitneyu, fligner, kruskal, spearmanr, pearsonr, chi2_contingency
import pingouin as pg
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# Définition des fonctions
# =============================================================================

# Fonction qui évalue si une série est stationnaire ou non avec un test de Dickey Fuller


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

# Fonction pour choisir les paramètres p et q du modèle ARMA fcailement à partir de l'ACF et la PACF


def pq_param(lags, data):
    print("Veuillez vous diriger vers l'onglet où apparaissent les graphiques !\n")
    # Créer le subplot pour l'ACF
    fig, ax1 = plt.subplots(figsize=(10, 5))
    plot_acf(data, lags=lags, zero=True, ax=ax1)
    ax1.set_title('ACF')
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Corrélation')
    ax1.grid(True)
    ax1.set_xticks(np.arange(0, lags+1, 1))
    plt.show()

    # Demander à l'utilisateur de saisir l'ordre auto-régressif p
    while True:
        try:
            p = int(input("Quel serait l'ordre auto-régressif p non-saisonnier ? "))
            confi_p = input(
                f"Confirmation : l'ordre auto-régressif non-saisonnier p serait = {p} ? (oui/non): ").lower()
            if confi_p == "oui":
                break
        except ValueError:
            print("Veuillez saisir un entier positif.")

    # Créer le subplot pour le PACF
    fig, ax2 = plt.subplots(figsize=(10, 5))
    plot_pacf(data, lags=lags, zero=True, ax=ax2)
    ax2.set_title('PACF')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Corrélation partielle')
    ax2.grid(True)
    ax2.set_xticks(np.arange(0, lags+1, 1))
    plt.show()

    #  Demander à l'utilisateur de saisir l'ordre de moyenne mobile q
    while True:
        try:
            q = int(
                input("Quel serait l'ordre de moyenne mobile q non-saisonnier ? "))
            confi_q = input(
                f"Confirmation : l'ordre de moyenne mobile non-saisonnier q serait = {q} ? (oui/non): ").lower()
            if confi_q == "oui":
                break
        except ValueError:
            print("Veuillez saisir un entier positif.")

    return p, q


def GARCH_search(data, feature, p_max, q_max):
    # Create a time series split for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Define the rank of p and q
    p_range = range(1, p_max)
    q_range = range(0, q_max)

    # Define the parameter grid
    param_grid = {'p': p_range, 'q': q_range}
    grid = ParameterGrid(param_grid)

    # Store the results
    results = []

    # Perform cross-validation
    for train_index, test_index in tscv.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        scores = []

        # Grid search
        for params in grid:
            p = params['p']
            q = params['q']

            # Fit the GARCH model
            model = arch_model(
                train_data[feature], mean='Zero', vol='GARCH', p=p, q=q)
            model_fit = model.fit(disp='off')

            # Evaluate on test data
            forecasts = model_fit.forecast(horizon=len(test_data))
            sigma = forecasts.variance.dropna().values[-1]
            score = np.mean(
                (test_data[feature] - model_fit.params['omega']) ** 2 / sigma)

            scores.append((params, score))

        # Sort scores by score value
        scores.sort(key=lambda x: x[1])

        # Compute ranks
        ranks = [i+1 for i in range(len(scores))]

        # Append results
        results.extend([(rank, params, score)
                       for rank, (params, score) in zip(ranks, scores)])

    # Convert results to DataFrame
    df_results = pd.DataFrame(
        results, columns=['Rank', 'Params', 'Std Test Score'])

    return df_results


def correlation(df, var1, var2, alpha=0.05):

    # Tester la normalité du couple d'observations
    multivariate_normality = pg.multivariate_normality(df[[var1, var2]])

    if multivariate_normality[1] > alpha:
        corr, p = pearsonr(df[var1], df[var2])
        print("\nLe test effectué est le test de Pearson")

    else:
        corr, p = spearmanr(df[var1], df[var2])
        print("\nLe test effectué est le test de Spearman")

    corr = round(corr*100, 2)
    p = round(p, 4)

    return corr, p, test


def chi2(df, var1, var2):
    # Table de contingence
    table_contin = pd.crosstab(df[var1], df[var2])

    # Test du Khi-Deux
    _, p, _, _ = chi2_contingency(table_contin)

    return p


p = chi2(df, 'Gender', 'Product Category')
