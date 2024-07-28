import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import shapiro, ansari, mannwhitneyu, fligner, kruskal, spearmanr
import pingouin as pg

df = pd.read_csv("Ventes.csv", sep=";")

del df['Transaction ID'], df['Customer ID']

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# =============================================================================
# Questions
# =============================================================================

# 1. Est-ce qu'il y a une catgéorie de produit privilégiée par les hommes ou les femmes ?

table_contin, p_value_chi2 = chi2(df, 'Gender', 'Product Category')

# Réponse : Non, les hommes et les femmes ont tendance à acheter les mêmes catgéories de produits


# 2. Est-ce que les hommes achètent en moyenne plus que les femmes ou inversement ?

quant_binary_ind(df, 'Gender', 'Quantity', alpha=0.05, graph=True)

# Réponse : Les deux groupes ont la même médiane statistiquement.
# Donc aucun genre n'achète en moyenne plus que l'autre en quantité.


# 3. Est-ce que l'âge des clients a un impact sur leurs préférences, en uanitité et en produit ?

quant_multi_ind(df, 'Product Category', 'Age', alpha=0.05, graph=True)
# Les médianes sont statistiquement égales


# Test de corrélation
corr, pvalue_corr = correlation(df, 'Age', 'Quantity', alpha=0.05)

# Il n'existe donc aucune relation monotone entre l'âge des clients et les produits et quantités vendues


# 4. Quels sont les produits les plus achetés ?

quant_multi_ind(df, 'Product Category', 'Quantity', alpha=0.05, graph=True)
# Les médianes sont statistiquement égales
