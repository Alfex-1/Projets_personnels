import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, shapiro, ansari, mannwhitneyu, fligner, kruskal, spearmanr
from scipy.stats import pearsonr, bartlett, f_oneway
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg
import warnings
from itertools import combinations
import statsmodels.stats.multitest as smm
warnings.filterwarnings("ignore")

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
        print("\nLe test effectué est le test celui de Pearson")

    else:
        corr, p = spearmanr(df[var1], df[var2])
        print("\nLe test effectué est le test celui de Spearman")

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



df = pd.read_csv("Ventes.csv", sep=";")

del df['Transaction ID'], df['Customer ID']

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# =============================================================================
# Questions
# =============================================================================

# 1. Est-ce qu'il y a une catégorie de produit privilégiée par les hommes ou les femmes ?

table_contin, p_value_chi2 = chi2(df, 'Gender', 'Product Category')

# Réponse : Non, les hommes et les femmes ont tendance à acheter les mêmes catgéories de produits


# 2. Est-ce que les hommes achètent en moyenne plus que les femmes ou inversement ?

quant_binary_ind(df, 'Gender', 'Quantity', alpha=0.05, graph=True)

# Réponse : Aucun genre n'achète en moyenne plus que l'autre en quantité.


# 3. Est-ce que l'âge des clients a un impact sur leurs préférences, en quantité et en produit ?

quant_multi_ind(df, 'Product Category', 'Age', alpha=0.05, graph=True)


# Test de corrélation
corr, pvalue_corr = correlation(df, 'Age', 'Quantity', alpha=0.05)

# Réponse : L'âge n'impact en rien le choix des clients e quantité et en catégorie de produit


# 4. Quels sont les produits les plus achetés ?

quant_multi_ind(df, 'Product Category', 'Quantity', alpha=0.05, graph=True)

# Réponse : Aucun produit n'est privilégié autre par rapport aux autres
