# =============================================================================
# Importation des packages
# =============================================================================
import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
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
from scipy.stats import shapiro, ansari, mannwhitneyu, fligner, kruskal, spearmanr, pearsonr, chi2_contingency, bartlett, f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg
import warnings
from itertools import combinations
import statsmodels.stats.multitest as smm
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
    # Création du nuage de points
    plt.figure(figsize=(8, 6))
    plt.scatter(df[var1], df[var2], alpha=0.5)
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.title('Nuage de points')
    plt.show()

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

    return corr, p


def chi2(df, var1, var2):
    # Table de contingence
    table_contin = pd.crosstab(df[var1], df[var2])

    # Test du Khi-Deux
    _, p, _, _ = chi2_contingency(table_contin)

    return table_contin, p


def quant_binary_ind(df, var_bin, var_quant, alpha=0.05, graph=True):
    # Création des sous-table
    binary = df[var_bin].unique()
    df1 = df[df[var_bin] == binary[0]]
    df2 = df[df[var_bin] == binary[1]]

    # Affichage des graphiques si graph=True
    if graph:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=var_bin, y=var_quant, data=df)
        plt.title('Boîtes à moustaches')

    # Tests de normalité
    _, p_value_shap1 = shapiro(df1[var_quant])
    _, p_value_shap2 = shapiro(df2[var_quant])

    if p_value_shap1 < alpha or p_value_shap2 < alpha:
        # Procédure non paramétrique

        # Egalité des variances
        _, p_value_ansari = ansari(df1[var_quant], df2[var_quant])

        if p_value_ansari >= alpha:

            # Egalité des médianes
            _, p_value_wilcox = mannwhitneyu(df1[var_quant], df2[var_quant])

            if p_value_wilcox >= alpha:
                print("Les deux variables sont indépendantes : médianes similaires")
            else:
                print("Un lien de dépendance existe : médianes différentes")

        else:
            print(
                "Les distributions conditionnelles de X connaissant Y différent d’un paramètre d’échelle.")

    else:
        # Procédure paramétrique

        # Egalité des variances
        var_test = stats.levene(df1[var_quant], df2[var_quant])

        # Egalité des moyennes
        if var_test.pvalue >= alpha:
            t_test = stats.ttest_ind(
                df1[var_quant], df2[var_quant], equal_var=True)
            if t_test.pvalue >= alpha:
                print("Les variables sont indépendantes : moyennes similaires")
            else:
                print(
                    "Il existe une dépebdance entre les variables : moyennes différentes")
        else:
            t_test = stats.ttest_ind(
                df1[var_quant], df2[var_quant], equal_var=False)
            if t_test.pvalue >= alpha:
                print("Les variables sont indépendantes : moyennes similaires")
            else:
                print(
                    "Il existe une dépebdance entre les variables : moyennes différentes")


def quant_multi_ind(df, var_multi, var_quant, alpha=0.05, graph=True):
    # Création des sous-tableaux
    multi = df[var_multi].unique()
    dataframes = {}

    for i, group in enumerate(multi):
        var_name = f"df{i + 1}"
        dataframes[var_name] = df[df[var_multi] == group]

    # Affichage des graphiques si graph=True
    if graph:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=var_multi, y=var_quant, data=df)
        plt.title('Boîtes à moustaches')
        plt.show()

    # Tests de normalité
    shapiro_results = {}
    for key, df in dataframes.items():
        group_data = df[var_quant]
        _, p_value = shapiro(group_data)
        shapiro_results[key] = {'p_value': p_value}
    all_normal = all(result['p_value'] >=
                     alpha for result in shapiro_results.values())

    # Extraire les valeurs des groupes
    group_data = [df[var_quant] for df in dataframes.values()]

    if all_normal:
        # Effectuer le test de Bartlett pour l'égalité des variances
        _, bartlett_p_value = bartlett(*group_data)

        if bartlett_p_value >= alpha:
            # Si les variances sont égales, effectuer une analyse de variance (ANOVA)
            _, f_p_value = f_oneway(*group_data)

            if f_p_value >= 0.05:
                print(
                    "Il y a une indépendance entre les deux variables étudiées : variances similaires")

            else:
                tukey = pairwise_tukeyhsd(
                    endog=df[var_quant], groups=df[var_multi], alpha=alpha)
                print(
                    "Il y a un lien de dépendance entre les deux variables étudiées : variances non similaires")
                print(tukey)
        else:
            tukey = pairwise_tukeyhsd(
                endog=df[var_quant], groups=df[var_multi], alpha=alpha)
            print(tukey)

            if all(tukey.pvalues >= alpha):
                print(
                    "\nLes variables sont indépendantes : moyenne des différentes paires sont similaires")
            else:
                print(
                    "\nLes variables sont pas indépendantes : moyenne des différentes paires ne sont pas similaires")

    else:
        # Effectuer le test de Fligner-Killeen pour l'égalité des variances
        # Correction pour la portée de group_data
        _, fligner_p_value = fligner(*group_data)

        if fligner_p_value >= alpha:
            # Si les variances sont égales, effectuer le test de Kruskal-Wallis
            _, kruskal_p_value = kruskal(*group_data)

            if kruskal_p_value < alpha:
                print(
                    "Les distributions conditionnelles de X connaissant Y différent d’un paramètre d’échelle.")
                group_combinations = list(
                    combinations(range(len(group_data)), 2))
                p_values = []
                for i, j in group_combinations:
                    data1 = group_data[i]
                    data2 = group_data[j]
                    _, p = mannwhitneyu(data1, data2, alternative='two-sided')
                    p_values.append(p)

                # Appliquer la correction de Bonferroni
                _, corrected_p_values, _, _ = smm.multipletests(
                    p_values, alpha=alpha, method='bonferroni')

                if all(corrected_p_values >= alpha):
                    print(
                        "\nLes variables sont indépendantes : moyenne des différentes paires sont similaires")
                else:
                    print(
                        "\nLes variables sont pas indépendantes : moyenne des différentes paires ne sont pas similaires")
            else:
                print(
                    "Les variables sont indépendantes : médianes similaires et variances similaires")
        else:
            print(
                "Les distributions conditionnelles de X connaissant Y différent d’un paramètre d’échelle.")
